import os, torch, torch.nn as nn
import segmentation_models_pytorch as smp


class GrayWrapper(nn.Module):
    """1채널 입력을 3채널로 복제해 ImageNet encoder에 그대로 먹인다.
    encoder 첫 conv를 수술하지 않으므로 가중치 호환 문제가 없다."""
    def __init__(self, net):
        super().__init__(); self.net = net

    def forward(self, x):
        return self.net(x.repeat(1, 3, 1, 1))


def build_model(cfg, n_classes):
    net = smp.Unet(
        encoder_name=cfg.encoder,
        encoder_weights=None,       # 항상 None. 가중치는 로컬 파일에서 읽는다
        in_channels=3,
        classes=n_classes,
    )
    if cfg.pretrained:
        p = os.path.join(cfg.weight_dir, f'{cfg.encoder.replace("/", "_")}_encoder.pth')
        assert os.path.exists(p), (
            f'사전학습 가중치 없음: {p}\n'
            'python -m src.prepare_weights 를 먼저 1회 실행할 것 (인터넷 필요).')
        sd = torch.load(p, map_location='cpu')
        missing, unexpected = net.encoder.load_state_dict(sd, strict=False)
        assert not missing, f'encoder 가중치 누락: {missing[:5]}'
        print(f'[model] encoder 가중치 로드: {os.path.basename(p)} '
              f'(unexpected {len(unexpected)}개)')
    return GrayWrapper(net)
