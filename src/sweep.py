"""여러 설정을 순서대로 돌리고 결과를 CSV에 누적한다.

한 설정이 터져도 나머지는 계속 진행하고, 이미 끝난 설정은 건너뛴다.
Colab 세션이 끊겨도 다시 실행하면 이어서 돈다.
"""
import os, csv, time, traceback
import numpy as np, torch

from .config import CFG
from . import train, prepare_weights


def _csv(work_dir):
    return os.path.join(work_dir, 'sweep.csv')


def done(work_dir):
    p = _csv(work_dir)
    if not os.path.exists(p):
        return set()
    with open(p) as f:
        return {r['exp'] for r in csv.DictReader(f)}


def log(work_dir, row):
    p = _csv(work_dir)
    new = not os.path.exists(p)
    with open(p, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def run(configs, resume=True):
    """configs: [(exp이름, dict(CFG 인자)), ...]"""
    wd = CFG().work_dir
    os.makedirs(wd, exist_ok=True)
    skip = done(wd) if resume else set()

    for name, kw in configs:
        if name in skip:
            print(f'[건너뜀] {name} (이미 완료)')
            continue
        cfg = CFG(exp=name, **kw)
        print(f'\n{"="*70}\n[{name}] {kw}\n{"="*70}')
        t0 = time.time()
        try:
            prepare_weights.ensure(cfg.encoder, cfg.weight_dir)
            scores = train.main(cfg)
            row = {'exp': name, 'mean': round(float(np.mean(scores)), 4),
                   'std': round(float(np.std(scores)), 4),
                   'folds': ' '.join(f'{s:.4f}' for s in scores),
                   'min': round(time.time() - t0) // 60, 'err': ''}
        except Exception as e:
            traceback.print_exc()
            row = {'exp': name, 'mean': '', 'std': '', 'folds': '',
                   'min': round(time.time() - t0) // 60, 'err': f'{type(e).__name__}: {e}'[:200]}
        finally:
            torch.cuda.empty_cache()
        log(wd, row)
        print(f'[{name}] -> {row["mean"] or row["err"]}')

    table(wd)


def table(work_dir=None):
    wd = work_dir or CFG().work_dir
    p = _csv(wd)
    if not os.path.exists(p):
        print('결과 없음'); return
    with open(p) as f:
        rows = [r for r in csv.DictReader(f)]
    ok = sorted([r for r in rows if r['mean']], key=lambda r: -float(r['mean']))
    print(f'\n{"실험":24s} {"평균":>8} {"편차":>7} {"분":>4}  fold별')
    for r in ok:
        print(f'{r["exp"]:24s} {float(r["mean"]):8.4f} {float(r["std"]):7.4f} {r["min"]:>4}  {r["folds"]}')
    for r in [r for r in rows if not r['mean']]:
        print(f'{r["exp"]:24s}   실패: {r["err"]}')
    return ok
