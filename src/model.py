import os, torch, torch.nn as nn
import segmentation_models_pytorch as smp


class GrayWrapper(nn.Module):
    """1채널 입력을 3채널로 복제해 ImageNet encoder에 그대로 먹인다.
    encoder 첫 conv를 수술하지 않으므로 가중치 호환 문제가 없다."""
    def __init__(self, net):
        super().__init__(); self.net = net

    def forward(self, x):
        return self.net(x.repeat(1, 3, 1, 1))


def _load_encoder_weights(encoder, path):
    """smp의 일부 encoder는 load_state_dict를 오버라이드하면서 반환값을 돌려주지
    않는다(예: EfficientNetEncoder). 그래서 키 비교를 직접 한 뒤 로드한다."""
    sd = torch.load(path, map_location='cpu')
    own = encoder.state_dict()
    # 분류 head 키는 encoder가 내부에서 버리므로 누락 판정에서 제외한다
    drop = ('_fc.', 'fc.', 'classifier.', 'head.')
    missing = [k for k in own
               if k not in sd and not k.startswith(drop)]
    unexpected = [k for k in sd if k not in own]

    # smp의 encoder마다 load_state_dict 시그니처가 다르다.
    #   EfficientNetEncoder: 반환값이 없다
    #   MixVisionTransformerEncoder: strict 인자를 받지 않는다
    try:
        encoder.load_state_dict(sd, strict=False)
    except TypeError:
        encoder.load_state_dict(sd)
    return missing, unexpected


ARCHS = {
    'unet': smp.Unet, 'unetpp': smp.UnetPlusPlus, 'deeplabv3plus': smp.DeepLabV3Plus,
    'manet': smp.MAnet, 'fpn': smp.FPN, 'pspnet': smp.PSPNet, 'linknet': smp.Linknet,
}


def build_model(cfg, n_classes):
    arch = getattr(cfg, 'arch', 'unet')
    assert arch in ARCHS, f'알 수 없는 arch: {arch} (가능: {list(ARCHS)})'
    net = ARCHS[arch](
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
        missing, unexpected = _load_encoder_weights(net.encoder, p)
        assert not missing, (
            f'encoder 가중치 {len(missing)}개 누락: {missing[:5]}\n'
            'prepare_weights를 지금 설치된 smp 버전으로 다시 실행할 것.')
        print(f'[model] encoder 가중치 로드: {os.path.basename(p)} '
              f'(unexpected {len(unexpected)}개)')
    return GrayWrapper(net)
