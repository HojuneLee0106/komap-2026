"""인터넷이 되는 환경에서 1회만 실행. 이후 학습·추론은 완전 오프라인으로 동작한다.
제출 시 weights/ 폴더를 함께 넣으면 심사 환경에 인터넷이 없어도 재현된다.

    python -m src.prepare_weights timm-efficientnet-b3 ./weights
"""
import os, sys, torch
import segmentation_models_pytorch as smp


def ensure(encoder, out_dir):
    """가중치가 없으면 받아서 저장한다. 있으면 아무것도 안 한다."""
    p = os.path.join(out_dir, f'{encoder.replace("/", "_")}_encoder.pth')
    if not os.path.exists(p):
        main(encoder, out_dir)
    return p


def main(encoder='timm-efficientnet-b3', out_dir='./weights'):
    os.makedirs(out_dir, exist_ok=True)
    net = smp.Unet(encoder_name=encoder, encoder_weights='imagenet',
                   in_channels=3, classes=4)
    p = os.path.join(out_dir, f'{encoder.replace("/", "_")}_encoder.pth')
    torch.save(net.encoder.state_dict(), p)
    print('저장:', p, f'({os.path.getsize(p) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main(*sys.argv[1:3])
