import os, glob, random
import numpy as np, cv2, torch
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from sklearn.model_selection import KFold

from .config import PALETTE, SUFFIX_IN, SUFFIX_OUT

Image.MAX_IMAGE_PIXELS = None
PAL = np.array(PALETTE, dtype=np.uint8)


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # use_deterministic_algorithms(True)는 일부 upsample backward에서 예외를
    # 던지므로 켜지 않는다. 위 설정으로 재현성은 충분히 확보된다.


def rgb_to_label(rgb):
    """(H,W,3) uint8 -> (H,W) uint8. 팔레트 밖 색은 255로 남겨 검증에 걸리게 한다."""
    lab = np.full(rgb.shape[:2], 255, np.uint8)
    for i, c in enumerate(PAL):
        lab[(rgb == c).all(-1)] = i
    return lab


def load_split(root, split, with_mask=True, scale=1.0, target_width=0):
    """scale>1이면 모델 입력·학습 라벨만 확대한다.
    'label_eval'과 'orig_hw'는 항상 원본 해상도 — 채점은 원본에서 이뤄진다."""
    items = []
    for ip in sorted(glob.glob(f'{root}/{split}/images/*{SUFFIX_IN}')):
        stem = os.path.basename(ip)[:-len(SUFFIX_IN)]
        img0 = np.array(Image.open(ip).convert('L'))
        lab0 = None
        if with_mask:
            mp = f'{root}/{split}/masks/{stem}{SUFFIX_OUT}'
            lab0 = rgb_to_label(np.array(Image.open(mp).convert('RGB')))
            assert lab0.max() < 4, f'{stem}: 팔레트 밖 색상 존재'
            assert lab0.shape == img0.shape, f'{stem}: 이미지/마스크 크기 불일치'

        h0, w0 = img0.shape
        sc = (target_width / w0) if target_width else scale
        if abs(sc - 1.0) > 1e-6:
            h, w = img0.shape
            W, H = int(round(w * sc)), int(round(h * sc))
            img = cv2.resize(img0, (W, H), interpolation=cv2.INTER_CUBIC)
            lab = (cv2.resize(lab0, (W, H), interpolation=cv2.INTER_NEAREST)
                   if lab0 is not None else None)
        else:
            img, lab = img0, lab0

        items.append({
            'stem': stem, 'path': ip,
            'alloy': stem.split('_')[0], 'step': stem.split('_')[1],
            'image': img, 'label': lab,          # 모델 입력 해상도
            'label_eval': lab0, 'orig_hw': img0.shape,   # 원본 해상도 (채점용)
            'mean': float(img.mean()), 'std': float(img.std()) + 1e-6,
        })
    assert items, f'{root}/{split}/images 에 파일이 없다'
    return items


def make_folds(cfg):
    """반환: [(train_items, val_items), ...]"""
    tr = load_split(cfg.data_root, 'train', scale=cfg.scale, target_width=cfg.target_width)
    va = load_split(cfg.data_root, 'valid', scale=cfg.scale, target_width=cfg.target_width)
    if cfg.split == 'official':
        return [(tr, va)]
    allv = tr + va
    kf = KFold(cfg.n_folds, shuffle=True, random_state=cfg.seed)
    return [([allv[i] for i in a], [allv[i] for i in b]) for a, b in kf.split(allv)]


def to_train_label(lab, mode):
    """3class 모드: 초정(2)·공정(3) Si를 2로 합친다."""
    if mode == '3class':
        out = lab.copy(); out[out == 3] = 2
        return out
    return lab


def build_transform(cfg):
    # 버전 차이에 안전한 변환만 사용한다 (Affine/ShiftScaleRotate는 API가 자주 바뀜)
    tf = []
    if getattr(cfg, 'aug_scale', 0.0) > 0:
        tf.append(A.RandomScale(scale_limit=cfg.aug_scale, p=0.5))
    return A.Compose(tf + [
        A.PadIfNeeded(min_height=cfg.crop, min_width=cfg.crop,
                      border_mode=cv2.BORDER_REFLECT_101),
        A.RandomCrop(height=cfg.crop, width=cfg.crop),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.GaussNoise(p=0.2),
    ])


class PatchDataset(Dataset):
    """전체 이미지를 RAM에 올려두고 매 step 랜덤 크롭을 뽑는다."""
    def __init__(self, items, cfg, length):
        self.items, self.cfg, self.length = items, cfg, length
        self.tf = build_transform(cfg)

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        it = self.items[np.random.randint(len(self.items))]
        out = self.tf(image=it['image'], mask=to_train_label(it['label'], self.cfg.mode))
        x = (out['image'].astype(np.float32) - it['mean']) / it['std']
        return torch.from_numpy(x)[None], torch.from_numpy(out['mask'].astype(np.int64))


def normalize_full(it):
    """추론용. 전체 이미지를 1x1xHxW 텐서로."""
    x = (it['image'].astype(np.float32) - it['mean']) / it['std']
    return torch.from_numpy(x)[None, None]


def class_weights(items, mode, n_cls):
    cnt = np.zeros(n_cls, np.int64)
    for it in items:
        cnt += np.bincount(to_train_label(it['label'], mode).ravel(), minlength=n_cls)
    frac = cnt / cnt.sum()
    w = 1.0 / np.sqrt(frac)   # 역빈도를 그대로 쓰면 소수 클래스에 과한 가중이 붙는다
    return (w / w.mean()).astype(np.float32), frac
