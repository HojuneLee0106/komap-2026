import numpy as np
from .config import CLASS_NAMES


def per_image_miou(pred, gt, n_cls=4):
    """대회 방식: 이미지 한 장의 클래스별 IoU를 평균."""
    ious = []
    for k in range(n_cls):
        p, g = (pred == k), (gt == k)
        u = (p | g).sum()
        if u == 0:                 # 예측·정답 모두에 없으면 제외
            continue
        ious.append((p & g).sum() / u)
    return float(np.mean(ious))


def evaluate(preds, gts, n_cls=4):
    """mIoU는 이미지별 평균 -> 전체 평균. 픽셀을 모두 합쳐 계산하면 큰 이미지
    쪽으로 치우쳐 대회 점수와 달라진다."""
    per_img = [per_image_miou(p, g, n_cls) for p, g in zip(preds, gts)]
    inter = np.zeros(n_cls); union = np.zeros(n_cls)
    for p, g in zip(preds, gts):
        for k in range(n_cls):
            pk, gk = (p == k), (g == k)
            inter[k] += (pk & gk).sum(); union[k] += (pk | gk).sum()
    return {
        'mIoU':  float(np.mean(per_img)),
        'std':   float(np.std(per_img)),
        'worst': float(np.min(per_img)),
        'per_class': {n: float(i / max(u, 1))
                      for n, i, u in zip(CLASS_NAMES, inter, union)},
        'per_image': per_img,
    }
