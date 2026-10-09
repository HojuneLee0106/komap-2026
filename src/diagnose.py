"""Vicki가 제안한 확인 항목을 수치로 답한다.

  1. 공정 Si를 Al3Ni로 잘못 분류하는가
  2. 공정 Si를 Al로 잘못 분류하는가
  3. 오류가 경계(가는 부분)에 몰려 있는가

처방이 완전히 갈리므로 추가 실험 전에 반드시 확인한다.
"""
import numpy as np, cv2, torch
from scipy import ndimage as ndi

from .config import CFG, CLASS_NAMES
from .data import make_folds, normalize_full, set_seed
from .ensemble import predict_logits, collect, _load
from .train import to_hard
from .postprocess import to_4class

K = np.ones((3, 3), np.uint8)


def run(work_dir=None, exps=None, fold=0, base_cfg=None):
    cfg = base_cfg or CFG()
    set_seed(cfg.seed)
    _, va = make_folds(cfg)[fold]
    ckpts = collect(work_dir or cfg.work_dir, exps, fold)
    assert ckpts, '체크포인트 없음'
    cfg0 = _load(ckpts[0])[1]

    L = predict_logits(ckpts, va)
    preds = [to_4class(to_hard(l, it['orig_hw']), cfg0) for l, it in zip(L, va)]
    gts = [it['label_eval'] for it in va]

    # ── 혼동행렬 (행=정답, 열=예측, 행 기준 %) ──
    C = np.zeros((4, 4), np.int64)
    for p, g in zip(preds, gts):
        C += np.bincount(g.ravel() * 4 + p.ravel(), minlength=16).reshape(4, 4)
    print('혼동행렬 — 행=정답, 열=예측 (행 합 100%)')
    head = '정답 / 예측'
    print(f'{head:>12} ' + ' '.join(f'{n:>8}' for n in CLASS_NAMES))
    for i, n in enumerate(CLASS_NAMES):
        r = 100 * C[i] / C[i].sum()
        print(f'{n:>12} ' + ' '.join(f'{v:7.2f}%' for v in r))

    # ── 공정 Si 오류가 어디로 가는가 ──
    i = 3
    miss = C[i].copy(); miss[i] = 0
    print(f'\n공정 Si를 놓친 픽셀의 행방 (총 {miss.sum():,}px)')
    for j in np.argsort(-miss):
        if miss[j] == 0: continue
        print(f'   -> {CLASS_NAMES[j]:6s} {100*miss[j]/miss.sum():5.1f}%')

    # ── 오류가 경계에 몰려 있는가 ──
    print('\n공정 Si 오류의 위치 (정답 영역 안쪽 거리별)')
    band = {1: [0, 0], 2: [0, 0], 3: [0, 0]}   # 거리: [맞음, 틀림]
    deep = [0, 0]
    for p, g in zip(preds, gts):
        s = (g == i)
        if not s.any(): continue
        dt = cv2.distanceTransform(s.astype(np.uint8), cv2.DIST_L2, 5)
        ok = (p == i)
        for d in (1, 2, 3):
            m = s & (dt > d - 1) & (dt <= d)
            band[d][0] += int((m & ok).sum()); band[d][1] += int((m & ~ok).sum())
        m = s & (dt > 3)
        deep[0] += int((m & ok).sum()); deep[1] += int((m & ~ok).sum())
    for d in (1, 2, 3):
        a, b = band[d]
        if a + b: print(f'   경계에서 {d}px 이내   정확도 {100*a/(a+b):5.1f}%  ({a+b:,}px)')
    a, b = deep
    if a + b: print(f'   3px 보다 안쪽      정확도 {100*a/(a+b):5.1f}%  ({a+b:,}px)')

    # ── Al3Ni도 같이 ──
    j = 1
    miss = C[j].copy(); miss[j] = 0
    print(f'\nAl3Ni를 놓친 픽셀의 행방 (총 {miss.sum():,}px)')
    for k in np.argsort(-miss):
        if miss[k] == 0: continue
        print(f'   -> {CLASS_NAMES[k]:6s} {100*miss[k]/miss.sum():5.1f}%')
    return C
