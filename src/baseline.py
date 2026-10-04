"""학습 없이 돌리는 기준선. 모델이 이 숫자를 못 넘으면 파이프라인에 문제가 있다.

밝기만 쓰는 multi-Otsu 4분할. GT로 이미지별 최적 임계값을 줬을 때조차
평균 mIoU가 0.67 근처였고, 공정 Si IoU는 0.30에 그쳤다.
"""
import numpy as np
from skimage.filters import threshold_multiotsu

from .config import CFG
from .data import load_split
from .metrics import evaluate


def multiotsu_predict(gray):
    t = threshold_multiotsu(gray, classes=4)
    b = np.digitize(gray, t)                       # 0 어두움 ~ 3 밝음
    # 밝기 순서: Al(밝음) > Al3Ni > 공정 Si > 초정 Si(어두움)
    return np.choose(b, [2, 3, 1, 0]).astype(np.uint8)


def main(cfg=None, split='valid'):
    cfg = cfg or CFG()
    items = load_split(cfg.data_root, split)
    preds = [multiotsu_predict(it['image']) for it in items]
    r = evaluate(preds, [it['label'] for it in items], 4)
    print(f'[multi-Otsu 기준선 / {split} {len(items)}장] '
          f'mIoU {r["mIoU"]:.4f} (±{r["std"]:.4f}, 최악 {r["worst"]:.4f})')
    for k, v in r['per_class'].items():
        print(f'   {k:6s} {v:.4f}')
    return r


if __name__ == '__main__':
    main()
