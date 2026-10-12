"""제출까지 남은 작업을 명령 하나씩으로 쪼갠 러너.

Colab이 1시간쯤에 끊기는 환경을 전제로, 한 번에 fold 하나만 돌린다.
끊기면 **같은 명령을 다시 실행**하면 된다 — 끊긴 epoch부터 이어 돈다.

    !python -m src.plan status                 # 지금 뭐가 됐고 뭐가 남았나
    !python -m src.plan photo 0                # 1단계: 광도 증강 테스트
    !python -m src.plan decide                 # 1단계 판정 (학습 안 함)
    !python -m src.plan fill E3_scale2_bw3 1   # 2단계: 빈 fold 채우기
    !python -m src.plan useall E3_scale2_bw3   # 3단계: 90장 전부로 1회
    !python -m src.plan tune                   # 5단계: 보정·겹침 (추론만)
    !python -m src.plan submit                 # 0/6단계: 제출물 생성

경로는 환경변수로 덮어쓸 수 있다 (기본은 CFG 값):
    KOMAP_WORK, KOMAP_DATA, KOMAP_WEIGHTS
"""
import os, sys, glob, json
from dataclasses import replace

from .config import CFG

# 2배 확대 공통 설정. 지금까지 확실하게 이득이 난 유일한 변경이다(+0.012~0.015).
BASE = dict(arch='unetpp', encoder='timm-efficientnet-b5', loss='ce_lovasz',
            aug_scale=0.0, sample_weight='fine', scale=2.0, crop=768, batch=4,
            steps_per_epoch=300)

EXPS = {
    'E1_scale2':     dict(BASE, boundary_weight=0.0),
    'E3_scale2_bw3': dict(BASE, boundary_weight=3.0),
    'E4_ridge':      dict(BASE, boundary_weight=3.0, in_mode='ridge'),
    # 1단계 후보: 광도 증강을 세게. 라벨을 안 건드리므로 얇은 조직에 안전하다.
    'P1_photo':      dict(BASE, boundary_weight=3.0, aug_photo=0.35),
}
SUBMIT_EXPS = ['E1_scale2', 'E3_scale2_bw3', 'E4_ridge']   # 측정된 최선 조합
BASELINE = {0: 0.8067, 3: 0.7472}        # E3 기준값. 1단계는 양쪽 다 넘어야 통과


def cfg_for(exp, **kw):
    c = CFG()
    for k, v in (('work_dir', 'KOMAP_WORK'), ('data_root', 'KOMAP_DATA'),
                 ('weight_dir', 'KOMAP_WEIGHTS')):
        if os.environ.get(v):
            c = replace(c, **{k: os.environ[v]})
    return replace(c, exp=exp, **EXPS[exp], **kw)


def _score(work, exp, fold):
    """끝난 fold의 점수. 안 끝났으면 None."""
    import torch
    p = os.path.join(work, exp, f'fold{fold}.pth')
    if not os.path.exists(p):
        return None
    ck = torch.load(p, map_location='cpu')
    if not (ck.get('done') or 'mIoU_calibrated' in ck):
        return None
    return ck.get('mIoU_calibrated', ck.get('mIoU'))


def status(argv=()):
    work = cfg_for('E3_scale2_bw3').work_dir
    print(f'작업 폴더: {work}\n')
    print(f'{"실험":16s} ' + ' '.join(f'{f"fold{f}":>8s}' for f in range(5)))
    for e in EXPS:
        row = []
        for f in range(5):
            s = _score(work, e, f)
            row.append(f'{s:8.4f}' if s else ('  도는중' if os.path.exists(
                os.path.join(work, e, f'fold{f}_last.pth')) else '       -'))
        print(f'{e:16s} ' + ' '.join(row))
    allx = os.path.join(work, 'A_all')
    print(f'\nuse_all(A_all): ' + ('있음' if _score(work, 'A_all', 0) else '없음'))
    zp = os.path.join(work, 'SUBMIT', 'results.zip')
    print(f'제출물: ' + (f'있음 — {zp}' if os.path.exists(zp) else '없음'))
    print('\n다음에 할 일은 HANDOFF.md 를 볼 것.')


def photo(argv):
    """1단계 — 광도 증강 테스트. fold 0과 3만 돌린다."""
    fold = int(argv[0])
    assert fold in (0, 3), '1단계는 fold 0과 3만 돌린다 (기준값이 그 둘뿐)'
    from . import train
    train.main(cfg_for('P1_photo', folds=(fold,)))


def decide(argv=()):
    """1단계 판정. 학습하지 않는다."""
    work = cfg_for('P1_photo').work_dir
    ok = True
    print(f'{"fold":>6s} {"기준(E3)":>10s} {"P1_photo":>10s} {"차이":>9s}')
    for f in (0, 3):
        s = _score(work, 'P1_photo', f)
        if s is None:
            print(f'{f:6d} {BASELINE[f]:10.4f} {"미완":>10s}')
            return print('\n아직 판정할 수 없다. 두 fold를 다 끝낼 것.')
        print(f'{f:6d} {BASELINE[f]:10.4f} {s:10.4f} {s - BASELINE[f]:+9.4f}')
        ok &= s > BASELINE[f]
    print('\n' + ('통과 — 2단계는 P1_photo 로 fold 1,2,4 를 채운다.'
                  if ok else
                  '탈락 — 한쪽이라도 못 넘었다. 2단계는 E3_scale2_bw3 로 채운다.\n'
                  '      (P1_photo fold 0,3 체크포인트는 앙상블에 그대로 쓴다)'))


def fill(argv):
    """2·4단계 — 빈 fold 하나를 채운다."""
    exp, fold = argv[0], int(argv[1])
    assert exp in EXPS, f'모르는 실험: {exp}. 쓸 수 있는 것: {list(EXPS)}'
    from . import train
    train.main(cfg_for(exp, folds=(int(fold),)))


def useall(argv):
    """3단계 — 90장 전부로 1회. 구성당 한 번뿐이라 시간당 가성비가 제일 좋다."""
    src = argv[0] if argv else 'E3_scale2_bw3'
    from . import train
    c = cfg_for(src, use_all=True, folds=(0,), bias_from=src)
    train.main(replace(c, exp='A_all'))


def tune(argv=()):
    """5단계 — 앙상블 보정과 추론 겹침. 추론만 한다."""
    from . import ensemble
    c = cfg_for('E3_scale2_bw3')
    exps = [e for e in list(EXPS) + ['A_all']
            if glob.glob(os.path.join(c.work_dir, e, 'fold*.pth'))]
    print(f'쓸 실험: {exps}\n')
    for ov in ({}, {'overlap': 0.5}):
        for f in (0, 3):
            print(f'\n===== 겹침 {ov.get("overlap", c.overlap)} | fold {f} =====')
            ensemble.eval_on_fold(None, exps, fold=f, base_cfg=replace(c, **ov))
    print('\n--- 앙상블 클래스별 보정 ---')
    b = ensemble.fit_ensemble_bias(exps, folds=(0, 3), cfg=c)
    json.dump({'bias': [float(x) for x in b], 'exps': exps},
              open(os.path.join(c.work_dir, 'tune.json'), 'w'))
    print(f'\n저장: {c.work_dir}/tune.json  (submit 이 자동으로 읽는다)')


def submit(argv=()):
    """0·6단계 — 제출물 생성. 끝난 fold가 있는 실험을 전부 쓴다."""
    from . import predict
    c = cfg_for('E3_scale2_bw3', exp='SUBMIT')
    exps = [e for e in SUBMIT_EXPS + ['P1_photo', 'A_all']
            if glob.glob(os.path.join(c.work_dir, e, 'fold*.pth'))]
    bias, over = None, {}
    tp = os.path.join(c.work_dir, 'tune.json')
    if os.path.exists(tp) and '--nobias' not in argv:
        t = json.load(open(tp))
        bias = t['bias']
        print(f'보정값 적용: {bias}')
    if '--overlap5' in argv:
        over = {'overlap': 0.5}
    print(f'쓸 실험: {exps}')
    predict.main(replace(c, **over), exps=exps, extra_bias=bias)


CMDS = {'status': status, 'photo': photo, 'decide': decide, 'fill': fill,
        'useall': useall, 'tune': tune, 'submit': submit}

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        sys.exit(f'사용법: python -m src.plan [{"|".join(CMDS)}] ...\n{__doc__}')
    CMDS[sys.argv[1]](sys.argv[2:])
