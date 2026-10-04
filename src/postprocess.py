import numpy as np, cv2
from scipy import ndimage as ndi

S8 = np.ones((3, 3), np.uint8)


def split_si_by_thickness(pred3, thresh=8.0):
    """3class 예측(0 Al / 1 Al3Ni / 2 Si) -> 4class(2 초정 / 3 공정).

    GT 90장 전수 측정 근거:
      - Si 연결성분 83,000개 중 초정·공정이 섞인 성분은 0.07%뿐 -> 성분 단위 분류 성립
      - 성분 최대 반두께 8.0px 단일 임계값으로 초정 IoU 0.919 / 공정 IoU 0.893
      - 밝기 기반 분리는 공정 Si IoU 0.30에 그침
    """
    out = pred3.astype(np.uint8).copy()
    si = (pred3 == 2)
    if not si.any():
        return out
    lab, n = ndi.label(si, structure=S8)
    dt = cv2.distanceTransform(si.astype(np.uint8), cv2.DIST_L2, 5)
    maxdt = np.asarray(ndi.maximum(dt, lab, range(1, n + 1)))
    cls = np.where(maxdt >= thresh, 2, 3).astype(np.uint8)   # 두꺼우면 초정
    out[si] = cls[lab[si] - 1]
    return out


def to_4class(pred, cfg):
    return split_si_by_thickness(pred, cfg.si_thresh) if cfg.mode == '3class' else pred
