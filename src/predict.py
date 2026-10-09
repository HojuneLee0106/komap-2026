import os, glob, zipfile
import numpy as np, torch
from PIL import Image

from .config import CFG, PALETTE, SUFFIX_IN, SUFFIX_OUT
from .data import load_split, normalize_full, set_seed
from .model import build_model
from .train import infer_logits, logit_to_orig, DEVICE
from .ensemble import _load
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


def main(cfg=None, ckpts=None, exps=None):
    cfg = cfg or CFG()
    set_seed(cfg.seed)
    n_cls = 4 if cfg.mode == '4class' else 3
    run_dir = os.path.join(cfg.work_dir, cfg.exp)
    if ckpts is None:
        if exps:   # 여러 실험을 섞는다
            ckpts = [p for e in exps
                     for p in sorted(glob.glob(f'{cfg.work_dir}/{e}/fold*.pth'))]
        else:
            ckpts = sorted(glob.glob(f'{run_dir}/fold*.pth'))
    assert ckpts, f'체크포인트 없음: {run_dir}'

    # 체크포인트마다 자기 설정(arch/encoder)을 들고 있으므로 그걸로 복원한다.
    # 데이터 해상도를 바꾸는 설정(scale/target_width/in_mode)은 전부 같아야 한다.
    models, biases, cfgs = [], [], []
    for c in ckpts:
        m, mc, bias, score = _load(c)
        models.append(m); biases.append(bias); cfgs.append(mc)
        print(f'  {os.path.basename(os.path.dirname(c))}/{os.path.basename(c)}  '
              f'{mc.arch}+{mc.encoder}  (val mIoU {score:.4f})')

    ref = cfgs[0]
    for mc in cfgs[1:]:
        assert mc.mode == ref.mode, f'3class와 4class는 섞을 수 없다: {mc.exp}'
    n_cls = 4 if ref.mode == '4class' else 3

    # 입력 해상도가 다른 모델도 함께 섞는다. 각자의 logit을 원본 격자로 되돌린
    # 뒤에 더하기 때문이다(logit_to_orig). 예전에는 argmax를 모델 해상도에서
    # 먼저 해버려서 scale이 같은 모델끼리만 섞을 수 있었다.
    feeds = {}
    for mc in cfgs:
        k = (mc.scale, mc.target_width, mc.in_mode)
        if k not in feeds:
            feeds[k] = load_split(ref.data_root, 'test', with_mask=False,
                                  scale=k[0], target_width=k[1], in_mode=k[2])
            print(f'  [입력] scale={k[0]} tw={k[1]} in={k[2]} — {len(feeds[k])}장')
    base = feeds[(ref.scale, ref.target_width, ref.in_mode)]
    out_dir = os.path.join(run_dir, 'submission')
    os.makedirs(out_dir, exist_ok=True)

    for i, it0 in enumerate(base):
        hw = it0['orig_hw']
        logit = None
        for m, mc, b in zip(models, cfgs, biases):
            it = feeds[(mc.scale, mc.target_width, mc.in_mode)][i]
            assert it['stem'] == it0['stem'], '해상도별 목록 순서가 어긋났다'
            l = logit_to_orig(infer_logits(m, normalize_full(it), mc, n_cls),
                              hw) + b[:, None, None]
            logit = l if logit is None else logit + l
        pred = (logit / len(models)).argmax(0).numpy().astype(np.uint8)
        save_mask(to_4class(pred, ref), it0['path'], out_dir)
        print('  ->', it0['stem'])

    if verify(out_dir, f'{ref.data_root}/test/images'):
        make_zip(out_dir, os.path.join(run_dir, 'results.zip'))
    else:
        print('검증에 실패해 zip을 만들지 않았다.')


if __name__ == '__main__':
    main()
