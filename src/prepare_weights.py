"""인터넷이 되는 환경에서 1회만 실행. 이후 학습·추론은 완전 오프라인으로 동작한다.
제출 시 weights/ 폴더를 함께 넣으면 심사 환경에 인터넷이 없어도 재현된다.

    python -m src.prepare_weights                     # 쓰는 인코더 전부, CFG.weight_dir로
    python -m src.prepare_weights timm-efficientnet-b5
    python -m src.prepare_weights mit_b2 ./weights

인자 없이 돌리는 것이 기본이다. 예전엔 기본값이 b3 하나에 './weights'여서,
정작 필요한 b5가 안 받아지고 Drive가 아닌 작업 디렉터리에 떨어졌다.
추론에는 이 파일들이 필요 없다(체크포인트가 전체 가중치를 들고 있다).
학습에만 필요하다.
"""
import os, sys, torch
import segmentation_models_pytorch as smp

from .config import CFG

# 이 프로젝트에서 실제로 쓰는 인코더
ENCODERS = ('timm-efficientnet-b5', 'tu-convnext_tiny', 'mit_b2',
            'timm-efficientnet-b3')


def path_for(encoder, out_dir):
    return os.path.join(out_dir, f'{encoder.replace("/", "_")}_encoder.pth')


def fetch(encoder, out_dir):
    """하나를 받아 저장한다. 이미 있으면 건너뛴다."""
    p = path_for(encoder, out_dir)
    if os.path.exists(p):
        print(f'있음:   {p} ({os.path.getsize(p) / 1e6:.1f} MB)')
        return p
    os.makedirs(out_dir, exist_ok=True)
    net = smp.Unet(encoder_name=encoder, encoder_weights='imagenet',
                   in_channels=3, classes=4)
    torch.save(net.encoder.state_dict(), p)
    print(f'저장:   {p} ({os.path.getsize(p) / 1e6:.1f} MB)')
    return p


def ensure(encoder, out_dir=None):
    return fetch(encoder, out_dir or CFG().weight_dir)


def main(encoders=None, out_dir=None):
    out_dir = out_dir or CFG().weight_dir
    if encoders is None:
        encoders = ENCODERS
    elif isinstance(encoders, str):
        encoders = [e.strip() for e in encoders.split(',') if e.strip()]
    print(f'받는 곳: {out_dir}')
    ok, bad = [], []
    for e in encoders:
        try:
            fetch(e, out_dir)
            ok.append(e)
        except Exception as ex:          # 한 개가 실패해도 나머지는 받는다
            bad.append((e, ex))
            print(f'실패:   {e} — {type(ex).__name__}: {ex}')
    print(f'\n완료 {len(ok)}개' + (f' / 실패 {len(bad)}개' if bad else ''))
    return ok, bad


if __name__ == '__main__':
    main(*sys.argv[1:3])
