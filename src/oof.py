"""fold별 검증셋 예측(OOF)을 내보낸다 — 오분류 원인 분석용.

5-fold에서 각 이미지는 정확히 한 fold의 검증셋에만 들어간다. 그 fold의
모델은 그 이미지를 학습에 쓴 적이 없으므로, 5개 fold를 합치면 90장 전부가
'처음 보는 이미지'에 대한 예측을 갖는다 — 테스트셋과 같은 조건이다.
학습에 쓴 이미지의 예측을 보면 모델이 외운 것을 보는 셈이라 의미가 없다.

내보내는 것 (work_dir/oof/):
  {stem}_pred.png   예측 마스크. 정답 마스크와 동일한 팔레트·크기·파일 구성
  {stem}_err.png    오류 지도. 공정 Si<->Al3Ni 혼동만 눈에 띄게 칠한다
  per_image.csv     이미지별 fold·mIoU·클래스별 IoU·혼동 픽셀 수
  components.csv    GT 연결성분별 두께·밝기·주변 조직·예측 결과
  README.txt        어떤 모델·fold에서 나왔는지
"""
import os, csv, glob, zipfile
import numpy as np, cv2
from PIL import Image
from scipy import ndimage as ndi

from .config import CFG, PALETTE, CLASS_NAMES
from .data import make_folds, set_seed
from .ensemble import collect, predict_logits, _load
from .train import to_hard
from .postprocess import to_4class
from .metrics import per_image_miou

PAL = np.array(PALETTE, np.uint8)
S8 = np.ones((3, 3), np.uint8)

# 오류 지도 색(RGB). 공정 Si와 Al3Ni 사이의 혼동만 강한 색을 준다.
ERRCOL = {
    (3, 1): (235,  40,  40),   # 공정 Si -> Al3Ni   빨강  (분석 대상)
    (1, 3): ( 40, 120, 235),   # Al3Ni  -> 공정 Si  파랑  (반대 방향)
    (3, 0): (245, 205,   0),   # 공정 Si -> Al      노랑  (아예 놓침)
    (3, 2): ( 30, 185,  95),   # 공정 Si -> 초정 Si 초록  (두께 오판)
}
OTHER = (165, 70, 200)         # 그 밖의 모든 오류   보라


def _error_map(gray, gt, pred):
    vis = np.dstack([gray // 3 + 50] * 3).astype(np.uint8)
    wrong = (gt != pred)
    for (g, p), c in ERRCOL.items():
        vis[wrong & (gt == g) & (pred == p)] = c
    rest = wrong.copy()
    for (g, p) in ERRCOL:
        rest &= ~((gt == g) & (pred == p))
    vis[rest] = OTHER
    return vis


def gray_auc(a, b):
    """밝기 하나로 a와 b를 가를 수 있는 정도 (Mann-Whitney AUC).

    0.5면 밝기로 전혀 못 가르고, 0이나 1에 가까울수록 완전히 갈린다.
    90장 측정에서 공정 Si는 Al3Ni보다 항상 어두워 0.15 언저리가 나오는데,
    방향을 뒤집으면 0.85다 — 밝기는 생각보다 강한 신호다. 다만 '이 사진 안에서
    더 어둡다'가 신호지 절대 밝기값은 아무 예측력이 없다(이미지별 IoU와 r=-0.01).
    """
    ha = np.bincount(a.ravel(), minlength=256).astype(np.float64)
    hb = np.bincount(b.ravel(), minlength=256).astype(np.float64)
    na, nb = ha.sum(), hb.sum()
    if na == 0 or nb == 0:
        return None
    below = np.cumsum(hb) - hb                 # b 중 a보다 어두운 것
    return float((ha * (below + hb / 2)).sum() / (na * nb))


def adjacency_recall(gt, pred, near_px=3):
    """Al3Ni에 붙은 공정 Si와 떨어진 공정 Si의 재현율을 따로 잰다.

    90장 OOF 측정: 붙은 쪽 0.693 / 떨어진 쪽 0.769로 격차 +0.076,
    90장 중 77장에서 같은 방향이다. 방향은 분명하지만 크기는 크지 않다.
    이미지별 공정 Si IoU와의 상관도 떨어진 쪽 재현율(r=+0.68)이 붙은 쪽
    (r=+0.61)보다 높아서, 인접은 실재하되 주된 원인은 아니다.

    주된 것은 공정 Si와 Al3Ni 사이 결정 경계가 양방향으로 어긋난 것이다:
    헛본 공정 Si의 60%가 원래 Al3Ni이고, 놓친 공정 Si의 48.8%가 Al3Ni로 간다.
    정밀도 0.737 / 재현율 0.735로 양쪽이 거의 같고, 이미지별 IoU와는
    정밀도(r=+0.78)가 재현율(r=+0.70)보다 더 강하게 묶여 있다.
    """
    k = 2 * near_px + 1
    ni = cv2.dilate((gt == 1).astype(np.uint8), np.ones((k, k), np.uint8))
    e = (gt == 3)
    near, far = e & (ni > 0), e & (ni == 0)
    r = lambda m: round(float((pred[m] == 3).mean()), 4) if m.any() else ''
    return r(near), r(far), int(near.sum()), int(far.sum())


def _components(gt, pred, gray, cls, min_area=4):
    """GT에서 cls인 연결성분별 통계. 전부 벡터 연산이다 — 성분이 이미지당
    수천 개라 성분마다 루프를 돌면 끝나지 않는다."""
    m0 = (gt == cls)
    if not m0.any():
        return np.zeros((0, 0)), []
    lab, n = ndi.label(m0, structure=S8)
    dt = cv2.distanceTransform(m0.astype(np.uint8), cv2.DIST_L2, 5)
    idx = lab[m0] - 1
    g = gray[m0].astype(np.float64)

    area = np.bincount(idx, minlength=n).astype(np.float64)
    gsum = np.bincount(idx, weights=g, minlength=n)
    g2sum = np.bincount(idx, weights=g * g, minlength=n)
    dtsum = np.bincount(idx, weights=dt[m0].astype(np.float64), minlength=n)
    halfw = np.zeros(n, np.float32)
    np.maximum.at(halfw, idx, dt[m0])

    # 성분 픽셀이 무엇으로 예측됐는가
    pc = np.bincount(idx * 4 + pred[m0], minlength=n * 4).reshape(n, 4)

    # 성분 바로 바깥 이웃의 GT 구성 (8방향, 같은 성분은 제외)
    nb = np.zeros(n * 4, np.int64)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            g2 = np.roll(np.roll(gt, dy, 0), dx, 1)
            l2 = np.roll(np.roll(lab, dy, 0), dx, 1)
            mm = m0 & (l2 != lab)
            if mm.any():
                nb += np.bincount((lab[mm] - 1) * 4 + g2[mm], minlength=n * 4)
    nb = nb.reshape(n, 4).astype(np.float64)
    nbt = np.maximum(nb.sum(1), 1)

    keep = area >= min_area
    mean = gsum / np.maximum(area, 1)
    var = np.maximum(g2sum / np.maximum(area, 1) - mean ** 2, 0)
    out = np.column_stack([
        np.arange(1, n + 1), area, halfw, dtsum / np.maximum(area, 1),
        mean, np.sqrt(var),
        pc / np.maximum(area, 1)[:, None],
        nb / nbt[:, None],
    ])[keep]
    return out, keep


COMP_COLS = (['stem', 'fold', 'gt_class', 'comp_id', 'area_px',
              'max_halfwidth_px', 'mean_halfwidth_px', 'mean_gray', 'std_gray']
             + [f'pred_frac_{c}' for c in CLASS_NAMES]
             + [f'nbr_frac_{c}' for c in CLASS_NAMES]
             + ['img_w', 'img_h'])


def run(cfg=None, exps=('F1_upp_b5', 'F2_convnext', 'F3_mit_b2'),
        folds=(0, 1, 2, 3, 4), comp_classes=(3, 1), min_area=4, make_zip=True):
    cfg = cfg or CFG()
    set_seed(cfg.seed)
    out_dir = os.path.join(cfg.work_dir, 'oof')
    os.makedirs(out_dir, exist_ok=True)

    img_rows, comp_rows, used = [], [], []
    allf = make_folds(cfg)
    for f in folds:
        if f >= len(allf):
            continue
        _, va = allf[f]
        ckpts = collect(cfg.work_dir, exps, f)
        if not ckpts:
            print(f'[fold {f}] 체크포인트 없음 — 건너뜀')
            continue
        names = [f'{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}'
                 for p in ckpts]
        used.append((f, [it['stem'] for it in va], names))
        print(f'[fold {f}] 검증 {len(va)}장 | 모델 {len(ckpts)}개: {", ".join(names)}')

        feeds = {(cfg.scale, cfg.target_width, cfg.in_mode): va}

        def feed(c, _f=f):
            k = (c.scale, c.target_width, c.in_mode)
            if k not in feeds:
                from dataclasses import replace
                feeds[k] = make_folds(replace(cfg, scale=k[0], target_width=k[1],
                                              in_mode=k[2]))[_f][1]
            return feeds[k]

        L = predict_logits(ckpts, va, feed)
        cfg0 = _load(ckpts[0])[1]

        for l, it in zip(L, va):
            gt = it['label_eval'].astype(np.uint8)
            pred = to_4class(to_hard(l, it['orig_hw']), cfg0).astype(np.uint8)
            gray = cv2.imread(it['path'], cv2.IMREAD_GRAYSCALE)
            assert gray.shape == gt.shape, f"{it['stem']} 크기 불일치"

            Image.fromarray(np.dstack([PAL[pred],
                                       np.full(gt.shape + (1,), 255, np.uint8)]),
                            'RGBA').save(f'{out_dir}/{it["stem"]}_pred.png')
            Image.fromarray(_error_map(gray, gt, pred),
                            'RGB').save(f'{out_dir}/{it["stem"]}_err.png')

            C = np.bincount(gt.ravel() * 4 + pred.ravel(), minlength=16).reshape(4, 4)
            rn, rf, nn, nf = adjacency_recall(gt, pred)
            auc = gray_auc(gray[gt == 3], gray[gt == 1])
            row = {'stem': it['stem'], 'fold': f,
                   'mIoU': round(per_image_miou(pred, gt), 4),
                   'recall_Si_e_near_Al3Ni': rn, 'recall_Si_e_far_Al3Ni': rf,
                   'px_Si_e_near_Al3Ni': nn, 'px_Si_e_far_Al3Ni': nf,
                   'gray_auc_Si_e_vs_Al3Ni': round(auc, 4) if auc else '',
                   'img_w': gt.shape[1], 'img_h': gt.shape[0]}
            for k, nm in enumerate(CLASS_NAMES):
                p, g = (pred == k), (gt == k)
                u = (p | g).sum()
                row[f'IoU_{nm}'] = round(float((p & g).sum() / u), 4) if u else ''
                row[f'gt_frac_{nm}'] = round(float(g.mean()), 5)
            for gi, gn in enumerate(CLASS_NAMES):
                for pi, pn in enumerate(CLASS_NAMES):
                    if gi != pi:
                        row[f'{gn}->{pn}'] = int(C[gi, pi])
            img_rows.append(row)

            for cls in comp_classes:
                arr, _ = _components(gt, pred, gray, cls, min_area)
                for r in arr:
                    comp_rows.append([it['stem'], f, CLASS_NAMES[cls], int(r[0]),
                                      int(r[1]), round(float(r[2]), 2),
                                      round(float(r[3]), 2), round(float(r[4]), 1),
                                      round(float(r[5]), 1)]
                                     + [round(float(x), 4) for x in r[6:14]]
                                     + [gt.shape[1], gt.shape[0]])
            print(f'   {it["stem"]:28s} mIoU {row["mIoU"]:.4f}  '
                  f'공정Si->Al3Ni {row["Si_e->Al3Ni"]:,}px')

    assert img_rows, '예측이 하나도 안 나왔다 — exps 이름과 work_dir을 확인할 것'
    with open(f'{out_dir}/per_image.csv', 'w', newline='', encoding='utf-8-sig') as fp:
        w = csv.DictWriter(fp, fieldnames=list(img_rows[0].keys()))
        w.writeheader(); w.writerows(img_rows)
    with open(f'{out_dir}/components.csv', 'w', newline='', encoding='utf-8-sig') as fp:
        w = csv.writer(fp); w.writerow(COMP_COLS); w.writerows(comp_rows)
    with open(f'{out_dir}/README.txt', 'w', encoding='utf-8') as fp:
        fp.write(_readme(used, exps, len(img_rows), len(comp_rows)))

    print(f'\n이미지 {len(img_rows)}장 / 성분 {len(comp_rows):,}개 -> {out_dir}')
    if make_zip:
        z = os.path.join(cfg.work_dir, 'oof.zip')
        with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(glob.glob(f'{out_dir}/*')):
                zf.write(p, os.path.basename(p))
        print(f'{z} 생성 ({len(zipfile.ZipFile(z).namelist())}개 파일)')
    return img_rows


def _readme(used, exps, n_img, n_comp):
    L = ['OOF(out-of-fold) 예측 — 각 이미지는 그 이미지를 학습에 쓰지 않은',
         '모델이 예측한 것이다. 테스트셋과 같은 조건이라 오류 분석에 쓸 수 있다.',
         '', f'이미지 {n_img}장 / 연결성분 {n_comp:,}개', f'구성: {", ".join(exps)}', '']
    for f, stems, names in used:
        L += [f'[fold {f}] 모델 {len(names)}개 logit 평균',
              '  ' + ', '.join(names), f'  검증 {len(stems)}장: ' + ', '.join(stems), '']
    L += ['파일',
          '  {stem}_pred.png  예측 마스크 (정답 마스크와 같은 팔레트·크기)',
          '  {stem}_err.png   오류 지도',
          '      빨강 = 공정 Si를 Al3Ni로      파랑 = Al3Ni를 공정 Si로',
          '      노랑 = 공정 Si를 Al로         초록 = 공정 Si를 초정 Si로',
          '      보라 = 그 밖의 오류           회색 배경 = 맞힌 곳(원본 밝기)',
          '  per_image.csv    이미지별 mIoU·클래스별 IoU·혼동 픽셀 수',
          '  components.csv   GT 연결성분별 통계',
          '      max_halfwidth_px  성분 안에서 가장 두꺼운 지점의 반두께',
          '                        (폭 = 이 값의 약 2배)',
          '      mean_gray/std_gray  원본 흑백 이미지에서 그 성분의 밝기',
          '      pred_frac_*       그 성분의 픽셀이 무엇으로 예측됐는지 비율',
          '                        pred_frac_Si_e가 낮을수록 틀린 성분',
          '      nbr_frac_*        성분 바로 바깥 1px 이웃의 GT 구성',
          '                        nbr_frac_Al3Ni가 높으면 Al3Ni에 붙어 있다',
          '      img_w/img_h       이미지 크기 (675~1440px로 섞여 있어',
          '                        두께를 비교할 땐 폭으로 정규화해야 한다)']
    return '\n'.join(L) + '\n'


def summarize(cfg=None, path=None):
    """per_image.csv를 읽어 '무엇이 공정 Si를 망치는가'를 집계한다.

    oof.run() 직후에 부르면 된다. 새 구성을 돌린 뒤 같은 수치를 다시 재서
    인접 격차가 줄었는지 보는 것이 이 함수의 목적이다 — mIoU 한 줄로는
    고치려던 것을 고쳤는지 알 수 없다.
    """
    import statistics as st
    cfg = cfg or CFG()
    path = path or os.path.join(cfg.work_dir, 'oof', 'per_image.csv')
    rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
    num = lambda r, k: float(r[k]) if r.get(k) not in (None, '') else None

    L = [f'OOF 집계 — {len(rows)}장', '']
    pair = [(num(r, 'recall_Si_e_far_Al3Ni'), num(r, 'recall_Si_e_near_Al3Ni'))
            for r in rows]
    pair = [(a, b) for a, b in pair if a is not None and b is not None]
    if pair:
        far = [a for a, _ in pair]; near = [b for _, b in pair]
        gap = [a - b for a, b in pair]
        L += ['[인접] 공정 Si 재현율 — Al3Ni에서 3px 안/밖',
              f'  떨어진 것  평균 {st.mean(far):.3f}  중앙 {st.median(far):.3f}',
              f'  붙은 것    평균 {st.mean(near):.3f}  중앙 {st.median(near):.3f}',
              f'  격차       평균 {st.mean(gap):+.3f}  '
              f'붙은 쪽이 나쁜 이미지 {sum(1 for v in gap if v > 0)}/{len(gap)}장',
              '  (기준 — 원해상도 3구성 앙상블: 떨어진 것 0.769 / 붙은 것 0.693,',
              '   격차 +0.076, 77/90장. 비키가 잰 +0.162는 정의가 다르다 —',
              '   이웃이 50% 이상 Al3Ni인 성분 vs 전혀 없는 성분이라 양 극단만',
              '   비교한 값이고, 여기 것은 3px 안/밖 전체 픽셀이다.)', '']

    auc = [num(r, 'gray_auc_Si_e_vs_Al3Ni') for r in rows]
    auc = [max(v, 1 - v) for v in auc if v is not None]
    if auc:
        L += ['[밝기] 밝기만으로 공정 Si와 Al3Ni를 가르는 정도 (방향 보정 AUC)',
              f'  중앙 {st.median(auc):.3f}  최소 {min(auc):.3f}  '
              f'0.80 이상 {sum(1 for v in auc if v >= .8)}/{len(auc)}장', '']

    y = [num(r, 'IoU_Si_e') for r in rows]
    L += ['[상관] 이미지별 공정 Si IoU와의 상관']
    for k in ('gray_auc_Si_e_vs_Al3Ni', 'recall_Si_e_near_Al3Ni',
              'recall_Si_e_far_Al3Ni', 'gt_frac_Si_e', 'Si_e->Al3Ni', 'img_w'):
        xs = [num(r, k) for r in rows]
        p = [(x, t) for x, t in zip(xs, y) if x is not None and t is not None]
        if len(p) < 5:
            continue
        mx, my = st.mean(x for x, _ in p), st.mean(t for _, t in p)
        den = (sum((x - mx) ** 2 for x, _ in p) *
               sum((t - my) ** 2 for _, t in p)) ** 0.5
        if den:
            L.append(f'  r={sum((x-mx)*(t-my) for x,t in p)/den:+.3f}  {k}')

    # 성분 크기별 재현율 — '아주 작은 공정 Si가 더 틀리는가'
    cpath = os.path.join(os.path.dirname(path), 'components.csv')
    if os.path.exists(cpath):
        comps = [c for c in csv.DictReader(open(cpath, encoding='utf-8-sig'))
                 if c.get('gt_class') == 'Si_e']
        BUCKETS = [(0, 10), (10, 30), (30, 100), (100, 400), (400, 10 ** 9)]
        L += ['', '[성분 크기] 공정 Si 연결성분의 면적별 재현율',
              f'{"크기(px)":>14s} {"성분수":>8s} {"면적합":>11s} '
              f'{"재현율":>7s} {"->Al3Ni":>8s}']
        for lo, hi in BUCKETS:
            sel = [c for c in comps if lo <= float(c['area_px']) < hi]
            if not sel:
                continue
            a = sum(float(c['area_px']) for c in sel)
            rec = sum(float(c['area_px']) * float(c['pred_frac_Si_e']) for c in sel) / a
            ni = sum(float(c['area_px']) * float(c['pred_frac_Al3Ni']) for c in sel) / a
            tag = f'{lo}~{hi}' if hi < 10 ** 9 else f'{lo}+'
            L.append(f'{tag:>14s} {len(sel):8,d} {a:11,.0f} {rec:7.3f} {ni:8.3f}')

    worst = sorted((r for r in rows if num(r, 'IoU_Si_e') is not None),
                   key=lambda r: num(r, 'IoU_Si_e'))[:8]
    L += ['', '[최악 8장] 공정 Si IoU']
    for r in worst:
        L.append(f'  {r["stem"]:26s} {num(r,"IoU_Si_e"):.3f}  fold {r["fold"]}  '
                 f'붙은것 {r.get("recall_Si_e_near_Al3Ni","")}  '
                 f'떨어진것 {r.get("recall_Si_e_far_Al3Ni","")}')

    out = '\n'.join(L)
    print(out)
    with open(os.path.join(os.path.dirname(path), 'summary.txt'), 'w',
              encoding='utf-8') as fp:
        fp.write(out + '\n')
    return rows
