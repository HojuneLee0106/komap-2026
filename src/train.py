import os, json, time, math
import numpy as np, cv2, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import CFG
from .data import set_seed, make_folds, PatchDataset, normalize_full, class_weights
from .model import build_model
from .losses import build_loss
from .metrics import evaluate
from .postprocess import to_4class

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# T4(Turing)는 bf16을 지원하지 않는다. GPU에 맞춰 자동 선택한다.
#   bf16: A100, L4 등 -> GradScaler 불필요
#   fp16: T4          -> GradScaler 필요 (없으면 gradient underflow로 학습이 죽는다)
USE_BF16 = DEVICE == 'cuda' and torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
if DEVICE == 'cuda':
    print(f'[amp] {torch.cuda.get_device_name(0)} -> '
          f'{"bfloat16" if USE_BF16 else "float16 + GradScaler"}')


def pad_to_multiple(x, m=32):
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    return F.pad(x, (0, pw, 0, ph), mode='reflect'), h, w


def _tiles(h, w, c, overlap):
    if h <= c and w <= c:
        return [(0, 0, h, w)]
    step = max(1, int(c * (1 - overlap)))
    ys = list(range(0, max(h - c, 0) + 1, step)) or [0]
    xs = list(range(0, max(w - c, 0) + 1, step)) or [0]
    if ys[-1] + c < h: ys.append(max(h - c, 0))
    if xs[-1] + c < w: xs.append(max(w - c, 0))
    return [(y, x, min(c, h), min(c, w)) for y in ys for x in xs]


@torch.no_grad()
def infer_logits(model, x, cfg, n_cls):
    """슬라이딩 윈도우 + 겹침 평균 + (옵션) 90도 4방향 TTA.
    반환: (n_cls, H, W) float32 CPU 텐서. fold 앙상블은 이 logit을 평균한다."""
    model.eval()
    x = x.to(DEVICE)
    xp, H, W = pad_to_multiple(x)
    h, w = xp.shape[-2:]
    acc = torch.zeros(1, n_cls, h, w, device=DEVICE)
    cnt = torch.zeros(1, 1, h, w, device=DEVICE)
    rots = list(range(4)) if cfg.tta else [0]

    for (y, xx, th, tw) in _tiles(h, w, cfg.crop, cfg.overlap):
        patch = xp[..., y:y + th, xx:xx + tw]
        pp, ph, pw = pad_to_multiple(patch)
        pp = pp.to(memory_format=torch.channels_last)
        logit = 0
        for k in rots:
            with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=DEVICE == 'cuda'):
                o = model(torch.rot90(pp, k, (-2, -1)))
            logit = logit + torch.rot90(o.float(), -k, (-2, -1))
        acc[..., y:y + th, xx:xx + tw] += logit[..., :ph, :pw] / len(rots)
        cnt[..., y:y + th, xx:xx + tw] += 1

    return (acc / cnt)[0, :, :H, :W].cpu()


def predict_image(model, x, cfg, n_cls):
    return infer_logits(model, x, cfg, n_cls).argmax(0).numpy().astype(np.uint8)


def to_orig(pred, hw):
    """확대 입력으로 추론한 결과를 원본 해상도로 되돌린다.
    보간이 섞이면 팔레트 밖 값이 생기므로 반드시 NEAREST."""
    if pred.shape == tuple(hw):
        return pred
    return cv2.resize(pred, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)


@torch.no_grad()
def calibrate_bias(model, items, cfg, n_cls, grid=(-0.8, -0.6, -0.4, -0.2, 0.0,
                                                   0.2, 0.4, 0.6, 0.8), rounds=2):
    """클래스별 logit 보정값을 검증셋에서 좌표상승으로 탐색한다.

    argmax는 클래스 간 확률 크기만 비교하므로, 소수 클래스에 치우친 가중 학습 뒤에는
    결정 경계가 어긋나 있을 수 있다. 학습 없이 추론만 다시 하면 되는 보정이다.
    검증셋에 맞추는 것이므로 과적합 여지가 있다 — fold별로 따로 구하고,
    보정 전/후를 함께 기록해 실제 이득인지 확인할 것.
    """
    L = [infer_logits(model, normalize_full(it), cfg, n_cls) for it in items]
    G = [it['label_eval'] for it in items]
    HW = [it['orig_hw'] for it in items]

    def score(b):
        t = torch.tensor(b, dtype=torch.float32)[:, None, None]
        preds = [to_4class(to_orig((l + t).argmax(0).numpy().astype(np.uint8), hw), cfg)
                 for l, hw in zip(L, HW)]
        return evaluate(preds, G, 4)['mIoU']

    bias = np.zeros(n_cls, np.float32)
    best = score(bias)
    base = best
    for _ in range(rounds):
        for c in range(n_cls):
            for v in grid:
                b = bias.copy(); b[c] = v
                sc = score(b)
                if sc > best + 1e-5:
                    best, bias = sc, b
    return bias, base, best


def validate(model, items, cfg, n_cls):
    # 원본 해상도로 되돌린 뒤 후처리·채점한다 (두께 임계값도 원본 기준)
    preds = [to_4class(to_orig(predict_image(model, normalize_full(it), cfg, n_cls),
                               it['orig_hw']), cfg)
             for it in items]
    return evaluate(preds, [it['label_eval'] for it in items], 4)


def train_fold(cfg, fold, tr_items, va_items):
    n_cls = 4 if cfg.mode == '4class' else 3
    out_dir = os.path.join(cfg.work_dir, cfg.exp)
    os.makedirs(out_dir, exist_ok=True)

    ck_path = os.path.join(out_dir, f'fold{fold}.pth')
    if getattr(cfg, 'resume', True) and os.path.exists(ck_path):
        # 세션이 끊겨도 fold 단위로 이어서 돌 수 있게 한다.
        # 다시 학습하려면 cfg.resume=False.
        ck = torch.load(ck_path, map_location='cpu')
        sc = ck.get('mIoU_calibrated', ck.get('mIoU', 0.0))
        print(f'[fold {fold}] 건너뜀 — 체크포인트 있음 (mIoU {sc:.4f})')
        return sc, 0.0

    w, frac = class_weights(tr_items, cfg.mode, n_cls)
    print(f'[fold {fold}] train {len(tr_items)} / val {len(va_items)} | '
          f'class frac {np.round(frac * 100, 2)} | weight {np.round(w, 3)}')

    model = build_model(cfg, n_cls).to(DEVICE).to(memory_format=torch.channels_last)
    crit = build_loss(cfg.loss, w, DEVICE, cfg.loss_weights)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == 'cuda' and not USE_BF16))

    dl = DataLoader(PatchDataset(tr_items, cfg, cfg.steps_per_epoch * cfg.batch),
                    batch_size=cfg.batch, num_workers=cfg.num_workers,
                    pin_memory=True, drop_last=True,
                    persistent_workers=cfg.num_workers > 0)

    total = cfg.epochs * cfg.steps_per_epoch
    warm = cfg.warmup_epochs * cfg.steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s:
        s / max(warm, 1) if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))

    best, hist, t0 = -1.0, [], time.time()
    for ep in range(1, cfg.epochs + 1):
        model.train(); run = 0.0
        for xb, yb in tqdm(dl, desc=f'f{fold} ep{ep}', leave=False):
            xb = xb.to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
            yb = yb.to(DEVICE, non_blocking=True)
            with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=DEVICE == 'cuda'):
                logit = model(xb)
            loss = crit(logit.float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item()

        if ep % cfg.val_every == 0 or ep == cfg.epochs:
            r = validate(model, va_items, cfg, n_cls)
            hist.append({'epoch': ep, 'loss': run / len(dl),
                         **{k: r[k] for k in ('mIoU', 'std', 'worst')},
                         'per_class': r['per_class']})
            print(f'  ep{ep:3d} loss {run / len(dl):.4f} | mIoU {r["mIoU"]:.4f} '
                  f'(±{r["std"]:.3f}, 최악 {r["worst"]:.3f}) | '
                  + ' '.join(f'{k} {v:.3f}' for k, v in r['per_class'].items()))
            if r['mIoU'] > best or cfg.use_all:   # use_all이면 검증이 새므로 최신을 저장
                best = r['mIoU']
                torch.save({'model': model.state_dict(), 'cfg': vars(cfg),
                            'fold': fold, 'mIoU': best},
                           os.path.join(out_dir, f'fold{fold}.pth'))
            json.dump(hist, open(os.path.join(out_dir, f'fold{fold}_hist.json'), 'w'),
                      indent=1, ensure_ascii=False)

    src_exp = getattr(cfg, 'bias_from', '')
    if src_exp:
        # 90장 전부로 학습한 모델은 자체 검증이 새므로, 같은 설정의 k-fold
        # 실험에서 구한 보정값 평균을 그대로 쓴다.
        import glob as _g
        bs = [torch.load(q, map_location='cpu').get('bias')
              for q in sorted(_g.glob(os.path.join(cfg.work_dir, src_exp, 'fold*.pth')))]
        bs = [b for b in bs if b is not None]
        assert bs, f'보정값을 가져올 체크포인트가 없다: {src_exp}'
        bias = np.mean(np.array(bs, np.float32), 0)
        ck = torch.load(os.path.join(out_dir, f'fold{fold}.pth'), map_location='cpu')
        ck['bias'] = bias.tolist()
        torch.save(ck, os.path.join(out_dir, f'fold{fold}.pth'))
        print(f'  [보정] {src_exp} {len(bs)}개 fold 평균 {np.round(bias, 2)} 이식')
    elif cfg.calibrate:
        ck = torch.load(os.path.join(out_dir, f'fold{fold}.pth'), map_location='cpu')
        model.load_state_dict(ck['model'])
        bias, before, after = calibrate_bias(model, va_items, cfg, n_cls)
        print(f'  [보정] {np.round(bias, 2)} | {before:.4f} -> {after:.4f} '
              f'({after - before:+.4f})')
        ck['bias'] = bias.tolist(); ck['mIoU_calibrated'] = after
        torch.save(ck, os.path.join(out_dir, f'fold{fold}.pth'))
        best = after

    mins = (time.time() - t0) / 60
    print(f'[fold {fold}] best mIoU {best:.4f} | {mins:.1f}분')
    return best, mins


def main(cfg=None):
    cfg = cfg or CFG()
    set_seed(cfg.seed)
    folds = make_folds(cfg)
    scores, times = [], []
    for f in cfg.folds:
        if f >= len(folds):
            break
        s, m = train_fold(cfg, f, *folds[f])
        scores.append(s); times.append(m)
    print(f'\n=== {cfg.exp} | mode={cfg.mode} loss={cfg.loss} crop={cfg.crop} '
          f'scale={cfg.scale} tw={cfg.target_width} encoder={cfg.encoder} ===')
    print(f'fold별: {np.round(scores, 4)}')
    print(f'평균 {np.mean(scores):.4f} | 표준편차 {np.std(scores):.4f} | 총 {sum(times):.0f}분')
    summary = {'exp': cfg.exp, 'cfg': vars(cfg), 'scores': scores,
               'mean': float(np.mean(scores)), 'std': float(np.std(scores)),
               'minutes': float(sum(times))}
    json.dump(summary, open(os.path.join(cfg.work_dir, cfg.exp, 'summary.json'), 'w'),
              indent=1, ensure_ascii=False)
    return scores


if __name__ == '__main__':
    main()
