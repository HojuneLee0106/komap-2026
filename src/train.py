import os, json, time, math
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import CFG
from .data import set_seed, make_folds, PatchDataset, normalize_full, class_weights
from .model import build_model
from .losses import build_loss
from .metrics import evaluate
from .postprocess import to_4class

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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
            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == 'cuda'):
                o = model(torch.rot90(pp, k, (-2, -1)))
            logit = logit + torch.rot90(o.float(), -k, (-2, -1))
        acc[..., y:y + th, xx:xx + tw] += logit[..., :ph, :pw] / len(rots)
        cnt[..., y:y + th, xx:xx + tw] += 1

    return (acc / cnt)[0, :, :H, :W].cpu()


def predict_image(model, x, cfg, n_cls):
    return infer_logits(model, x, cfg, n_cls).argmax(0).numpy().astype(np.uint8)


def validate(model, items, cfg, n_cls):
    preds = [to_4class(predict_image(model, normalize_full(it), cfg, n_cls), cfg)
             for it in items]
    return evaluate(preds, [it['label'] for it in items], 4)


def train_fold(cfg, fold, tr_items, va_items):
    n_cls = 4 if cfg.mode == '4class' else 3
    out_dir = os.path.join(cfg.work_dir, cfg.exp)
    os.makedirs(out_dir, exist_ok=True)

    w, frac = class_weights(tr_items, cfg.mode, n_cls)
    print(f'[fold {fold}] train {len(tr_items)} / val {len(va_items)} | '
          f'class frac {np.round(frac * 100, 2)} | weight {np.round(w, 3)}')

    model = build_model(cfg, n_cls).to(DEVICE).to(memory_format=torch.channels_last)
    crit = build_loss(cfg.loss, w, DEVICE, cfg.loss_weights)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

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
            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == 'cuda'):
                logit = model(xb)
            loss = crit(logit.float(), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            run += loss.item()

        if ep % cfg.val_every == 0 or ep == cfg.epochs:
            r = validate(model, va_items, cfg, n_cls)
            hist.append({'epoch': ep, 'loss': run / len(dl),
                         **{k: r[k] for k in ('mIoU', 'std', 'worst')},
                         'per_class': r['per_class']})
            print(f'  ep{ep:3d} loss {run / len(dl):.4f} | mIoU {r["mIoU"]:.4f} '
                  f'(±{r["std"]:.3f}, 최악 {r["worst"]:.3f}) | '
                  + ' '.join(f'{k} {v:.3f}' for k, v in r['per_class'].items()))
            if r['mIoU'] > best:
                best = r['mIoU']
                torch.save({'model': model.state_dict(), 'cfg': vars(cfg),
                            'fold': fold, 'mIoU': best},
                           os.path.join(out_dir, f'fold{fold}.pth'))
            json.dump(hist, open(os.path.join(out_dir, f'fold{fold}_hist.json'), 'w'),
                      indent=1, ensure_ascii=False)

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
    print(f'\n=== {cfg.exp} | mode={cfg.mode} loss={cfg.loss} '
          f'crop={cfg.crop} encoder={cfg.encoder} ===')
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
