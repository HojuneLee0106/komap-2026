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




def drop_tiny_class(pred, cls=2, into=3, min_frac=0.002):
    """면적이 너무 작은 클래스 예측을 지운다.

    대회 점수는 '이미지별 클래스 IoU의 평균'이고, 정답에 없는 클래스를 1픽셀이라도
    예측하면 그 클래스 IoU가 0으로 집계돼 그 이미지 점수가 1/4(=0.25)까지 깎인다.
    GT 90장 측정: 초정 Si가 완전히 없는 이미지가 2장(A57_step1, A71_step2)이고,
    있는 88장의 최소 면적비는 1.53%다. 0.2%는 그 1/7로, 진짜 있는 이미지를
    잘못 지울 위험은 거의 없고(그 경우 이미 IoU<0.13) 없는 이미지를 맞히면
    이미지당 최대 +0.25다. 지운 픽셀은 같은 Si 계열인 공정 Si로 돌린다.
    """
    m = (pred == cls)
    if 0 < m.mean() < min_frac:
        pred = pred.copy()
        pred[m] = into
    return pred


def to_4class(pred, cfg):
    pred = split_si_by_thickness(pred, cfg.si_thresh) if cfg.mode == '3class' else pred
    if getattr(cfg, 'min_sip_frac', 0) > 0:
        pred = drop_tiny_class(pred, 2, 3, cfg.min_sip_frac)
    return pred
