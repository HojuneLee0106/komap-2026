import os, glob, zipfile
import numpy as np, torch
from PIL import Image

from .config import CFG, PALETTE, SUFFIX_IN, SUFFIX_OUT
from .data import load_split, normalize_full, set_seed
from .model import build_model
from .train import infer_logits, to_orig, DEVICE
from .postprocess import to_4class

PAL = np.array(PALETTE, np.uint8)


def save_mask(pred, src_path, out_dir, rgba=True):
    """제공된 정답 마스크와 동일한 팔레트로 저장.
    rgba=True는 정답 마스크의 채널 구성(RGBA, alpha 255)을 그대로 맞춘 것이다."""
    base = os.path.basename(src_path)
    assert base.endswith(SUFFIX_IN), f'예상 밖 파일명: {base}'
    name = base[:-len(SUFFIX_IN)] + SUFFIX_OUT
    w, h = Image.open(src_path).size
    assert pred.shape == (h, w), f'크기 불일치 {pred.shape} vs {(h, w)}'
    assert pred.max() < 4, f'클래스 범위 초과: {np.unique(pred)}'
    rgb = PAL[pred]
    if rgba:
        rgb = np.dstack([rgb, np.full((h, w, 1), 255, np.uint8)])
    Image.fromarray(rgb, 'RGBA' if rgba else 'RGB').save(os.path.join(out_dir, name))
    return name


def verify(out_dir, src_dir, n_expect=10):
    """제출 직전 반드시 통과시킬 것."""
    srcs = sorted(glob.glob(f'{src_dir}/*{SUFFIX_IN}'))
    allowed = {tuple(c) for c in PAL}
    errs = []
    if len(srcs) != n_expect:
        errs.append(f'입력 {len(srcs)}장 (기대 {n_expect})')
    for sp in srcs:
        name = os.path.basename(sp)[:-len(SUFFIX_IN)] + SUFFIX_OUT
        op = os.path.join(out_dir, name)
        if not os.path.exists(op):
            errs.append(f'누락: {name}'); continue
        si, oi = Image.open(sp), Image.open(op)
        if si.size != oi.size:
            errs.append(f'{name}: 크기 {oi.size} != 원본 {si.size}')
        bad = {tuple(c) for c in np.unique(
            np.array(oi.convert('RGB')).reshape(-1, 3), axis=0)} - allowed
        if bad:
            errs.append(f'{name}: 팔레트 밖 색상 {list(bad)[:3]}')
    print('검증 통과' if not errs else '검증 실패')
    for e in errs:
        print('  -', e)
    return not errs


def make_zip(out_dir, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(glob.glob(f'{out_dir}/*{SUFFIX_OUT}')):
            z.write(p, os.path.basename(p))     # 폴더 없이 평평하게
    n = len(zipfile.ZipFile(zip_path).namelist())
    print(f'{zip_path} 생성 ({n}개 파일)')


def main(cfg=None, ckpts=None):
    cfg = cfg or CFG()
    set_seed(cfg.seed)
    n_cls = 4 if cfg.mode == '4class' else 3
    run_dir = os.path.join(cfg.work_dir, cfg.exp)
    ckpts = ckpts or sorted(glob.glob(f'{run_dir}/fold*.pth'))
    assert ckpts, f'체크포인트 없음: {run_dir}'

    models = []
    for c in ckpts:
        sd = torch.load(c, map_location='cpu')
        m = build_model(cfg, n_cls).to(DEVICE).to(memory_format=torch.channels_last)
        m.load_state_dict(sd['model']); m.eval()
        models.append(m)
        print(f'  {os.path.basename(c)} (val mIoU {sd.get("mIoU", float("nan")):.4f})')

    test = load_split(cfg.data_root, 'test', with_mask=False, scale=cfg.scale)
    out_dir = os.path.join(run_dir, 'submission')
    os.makedirs(out_dir, exist_ok=True)

    for it in test:
        x = normalize_full(it)
        logit = sum(infer_logits(m, x, cfg, n_cls) for m in models) / len(models)
        pred = to_orig(logit.argmax(0).numpy().astype(np.uint8), it['orig_hw'])
        save_mask(to_4class(pred, cfg), it['path'], out_dir)
        print('  ->', it['stem'])

    if verify(out_dir, f'{cfg.data_root}/test/images'):
        make_zip(out_dir, os.path.join(run_dir, 'results.zip'))
    else:
        print('검증에 실패해 zip을 만들지 않았다.')


if __name__ == '__main__':
    main()
