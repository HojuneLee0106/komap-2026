"""서로 다른 설정으로 학습한 모델들을 섞는다.

개별 설정 튜닝은 +0.005 수준에서 멈췄지만, 오차 패턴이 다른 모델을 합치면
보통 그보다 큰 이득이 난다. 다만 '보통'이 우리 데이터에서도 맞는지는
제출 전에 검증셋으로 직접 재야 한다 — 그게 eval_on_fold의 용도다.

체크포인트마다 자기 cfg를 들고 있으므로 encoder·crop이 달라도 섞인다.
단 데이터 해상도를 바꾸는 설정(scale, target_width)은 전부 같아야 한다.
"""
import os, glob, json
import numpy as np, torch

from .config import CFG
from .data import make_folds, normalize_full, set_seed
from .model import build_model
from .train import infer_logits, to_orig, DEVICE
from .postprocess import to_4class
from .metrics import evaluate


def _load(ckpt_path):
    sd = torch.load(ckpt_path, map_location='cpu')
    c = CFG(**{k: v for k, v in sd['cfg'].items() if k in CFG.__dataclass_fields__})
    n_cls = 4 if c.mode == '4class' else 3
    m = build_model(c, n_cls).to(DEVICE).to(memory_format=torch.channels_last)
    m.load_state_dict(sd['model']); m.eval()
    bias = torch.tensor(sd.get('bias', [0.0] * n_cls), dtype=torch.float32)
    return m, c, bias, sd.get('mIoU_calibrated', sd.get('mIoU', float('nan')))


def collect(work_dir, exps, fold):
    """exps 목록에서 해당 fold의 체크포인트 경로를 모은다."""
    out = []
    for e in exps:
        p = os.path.join(work_dir, e, f'fold{fold}.pth')
        if os.path.exists(p):
            out.append(p)
        else:
            print(f'  (없음) {e}/fold{fold}.pth')
    return out


@torch.no_grad()
def predict_logits(ckpts, items):
    """여러 모델의 logit을 원본 해상도에서 평균한다."""
    acc = None
    for p in ckpts:
        m, c, bias, _ = _load(p)
        n_cls = 4 if c.mode == '4class' else 3
        cur = []
        for it in items:
            l = infer_logits(m, normalize_full(it), c, n_cls) + bias[:, None, None]
            # 모델마다 입력 해상도가 달라도 원본 격자에서 더한다
            cur.append(l)
        acc = cur if acc is None else [a + b for a, b in zip(acc, cur)]
        del m; torch.cuda.empty_cache()
    return [a / len(ckpts) for a in acc]


def eval_on_fold(work_dir, exps, fold=0, base_cfg=None):
    """같은 fold의 검증셋에서 개별 성적과 앙상블 성적을 비교한다."""
    cfg = base_cfg or CFG()
    set_seed(cfg.seed)
    _, va = make_folds(cfg)[fold]
    ckpts = collect(cfg.work_dir, exps, fold)
    assert ckpts, '체크포인트가 하나도 없다'

    print(f'[fold {fold}] 검증 {len(va)}장 | 모델 {len(ckpts)}개')
    for p in ckpts:
        _, _, _, s = _load(p)
        print(f'   {os.path.basename(os.path.dirname(p)):16s} 단독 {s:.4f}')

    L = predict_logits(ckpts, va)
    cfg0 = _load(ckpts[0])[1]
    preds = [to_4class(to_orig(l.argmax(0).numpy().astype(np.uint8), it['orig_hw']), cfg0)
             for l, it in zip(L, va)]
    r = evaluate(preds, [it['label_eval'] for it in va], 4)
    print(f'   ─────────────────────────')
    print(f'   앙상블           {r["mIoU"]:.4f}  (±{r["std"]:.3f}, 최악 {r["worst"]:.3f})')
    print('   ' + '  '.join(f'{k} {v:.3f}' for k, v in r['per_class'].items()))
    return r
