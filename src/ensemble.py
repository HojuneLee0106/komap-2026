"""서로 다른 설정으로 학습한 모델들을 섞는다.

개별 설정 튜닝은 +0.005 수준에서 멈췄지만, 오차 패턴이 다른 모델을 합치면
보통 그보다 큰 이득이 난다. 다만 '보통'이 우리 데이터에서도 맞는지는
제출 전에 검증셋으로 직접 재야 한다 — 그게 eval_on_fold의 용도다.

체크포인트마다 자기 cfg를 들고 있으므로 encoder·crop이 달라도 섞인다.
입력 해상도(scale, target_width, in_mode)가 달라도 섞인다 — logit을 각자
원본 격자로 되돌린 뒤 더하기 때문이다.
"""
import os, glob, json
from dataclasses import replace
import numpy as np, torch

from .config import CFG
from .data import make_folds, normalize_full, set_seed
from .model import build_model
from .train import infer_logits, to_hard, logit_to_orig, DEVICE
from .postprocess import to_4class
from .metrics import evaluate
from .config import CLASS_NAMES


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
def predict_logits(ckpts, items, feed=None):
    """여러 모델의 logit을 원본 해상도로 되돌린 뒤 평균한다.

    feed(cfg) -> 그 모델의 입력 해상도로 읽은 items (items와 순서 동일).
    주지 않으면 모든 모델이 items를 그대로 쓴다."""
    hws = [it['orig_hw'] for it in items]
    acc = None
    for p in ckpts:
        m, c, bias, _ = _load(p)
        n_cls = 4 if c.mode == '4class' else 3
        cur = [logit_to_orig(infer_logits(m, normalize_full(it), c, n_cls), hw)
               + bias[:, None, None]
               for it, hw in zip(feed(c) if feed else items, hws)]
        acc = cur if acc is None else [a + b for a, b in zip(acc, cur)]
        del m; torch.cuda.empty_cache()
    return [a / len(ckpts) for a in acc]


@torch.no_grad()
def _logits_per_model(ckpts, items, stride=1, feed=None):
    """모델별 logit을 모은다. stride>1이면 픽셀을 솎아 메모리를 줄인다
    (가중치 탐색용 — 최종 예측은 stride=1로 다시 계산한다)."""
    hws = [it['orig_hw'] for it in items]
    out = []
    for p in ckpts:
        m, c, bias, _ = _load(p)
        n_cls = 4 if c.mode == '4class' else 3
        cur = []
        for it, hw in zip(feed(c) if feed else items, hws):
            l = (logit_to_orig(infer_logits(m, normalize_full(it), c, n_cls), hw)
                 + bias[:, None, None])
            cur.append(l[:, ::stride, ::stride].half())
        out.append(cur)
        del m; torch.cuda.empty_cache()
    return out


def fit_class_weights(ckpts, items, cfg, stride=2,
                      grid=(0.0, 0.5, 1.0, 1.5, 2.0), rounds=2, feed=None):
    """모델 x 클래스 가중치를 검증셋에서 좌표상승으로 찾는다.

    '어떤 모델은 Al3Ni를, 어떤 모델은 Si를 잘한다'를 이용하는 방식이다.
    덮어쓰기가 아니라 logit 합산이므로 클래스 배타성이 유지된다.

    주의: 18장에 20개 파라미터를 맞추므로 과적합 여지가 크다.
    반드시 다른 fold에서 검증하고, 단순 평균 대비 이득이 작으면 쓰지 말 것.
    """
    n_cls = 4
    L = _logits_per_model(ckpts, items, stride, feed)    # [M][N] 각 (C,h,w)
    GT = [torch.from_numpy(it['label_eval'][::stride, ::stride].astype(np.int64))
          for it in items]
    M = len(ckpts)

    def score(W):
        tot = []
        for i in range(len(items)):
            acc = sum(L[m][i].float() * torch.tensor(W[m], dtype=torch.float32)[:, None, None]
                      for m in range(M))
            p = acc.argmax(0).numpy().astype(np.uint8)
            tot.append(p)
        return evaluate(tot, [g.numpy() for g in GT], n_cls)['mIoU']

    W = np.ones((M, n_cls), np.float32)
    best = score(W)
    base = best
    for _ in range(rounds):
        for m in range(M):
            for c in range(n_cls):
                for v in grid:
                    W2 = W.copy(); W2[m, c] = v
                    if W2[:, c].sum() == 0:
                        continue
                    sc = score(W2)
                    if sc > best + 1e-5:
                        best, W = sc, W2
    return W, base, best


def eval_on_fold(work_dir, exps, fold=0, base_cfg=None, class_weights=False,
                 verify_fold=None):
    """같은 fold의 검증셋에서 개별 성적과 앙상블 성적을 비교한다."""
    cfg = base_cfg or CFG()
    set_seed(cfg.seed)
    _, va = make_folds(cfg)[fold]
    ckpts = collect(cfg.work_dir, exps, fold)
    assert ckpts, '체크포인트가 하나도 없다'

    # 입력 해상도가 다른 모델이 섞여 있으면 그 해상도로 검증셋을 다시 읽는다.
    # 같은 fold·같은 이미지 순서가 보장돼야 하므로 make_folds를 그대로 쓴다.
    feeds = {(cfg.scale, cfg.target_width, cfg.in_mode): va}

    def feed(c):
        k = (c.scale, c.target_width, c.in_mode)
        if k not in feeds:
            print(f'   [입력] scale={k[0]} tw={k[1]} in={k[2]} 로 다시 읽는다')
            feeds[k] = make_folds(replace(cfg, scale=k[0], target_width=k[1],
                                          in_mode=k[2]))[fold][1]
        return feeds[k]

    print(f'[fold {fold}] 검증 {len(va)}장 | 모델 {len(ckpts)}개')
    for p in ckpts:
        _, _, _, s = _load(p)
        print(f'   {os.path.basename(os.path.dirname(p)):16s} 단독 {s:.4f}')

    L = predict_logits(ckpts, va, feed)
    cfg0 = _load(ckpts[0])[1]
    preds = [to_4class(to_hard(l, it['orig_hw']), cfg0) for l, it in zip(L, va)]
    r = evaluate(preds, [it['label_eval'] for it in va], 4)
    print(f'   ─────────────────────────')
    print(f'   단순평균 앙상블  {r["mIoU"]:.4f}  (±{r["std"]:.3f}, 최악 {r["worst"]:.3f})')
    print('   ' + '  '.join(f'{k} {v:.3f}' for k, v in r['per_class'].items()))

    if class_weights:
        W, b, a = fit_class_weights(ckpts, va, cfg, feed=feed)
        print(f'\n   [클래스별 가중치] {b:.4f} -> {a:.4f} ({a - b:+.4f})')
        print(f'   {"모델":18s}  ' + '  '.join(f'{k:>6s}' for k in CLASS_NAMES))
        for p, w in zip(ckpts, W):
            print(f'   {os.path.basename(os.path.dirname(p)):18s}  '
                  + '  '.join(f'{x:6.1f}' for x in w))
        r['class_weights'] = W.tolist()
        if verify_fold is not None:
            print(f'\n   (fold {verify_fold}에서 과적합 검증은 같은 exps의 '
                  f'fold{verify_fold}.pth가 있어야 가능하다)')
    return r
