from dataclasses import dataclass

# 제공된 정답 마스크 팔레트 (순서 = 클래스 인덱스)
PALETTE = [
    (153, 127,  76),   # 0 Al 기지
    ( 25,  76, 153),   # 1 Al3Ni
    ( 76, 178,  76),   # 2 Primary Si (초정)
    (204, 204, 204),   # 3 Eutectic Si (공정)
]
CLASS_NAMES = ['Al', 'Al3Ni', 'Si_p', 'Si_e']
SUFFIX_IN, SUFFIX_OUT = '_image.png', '_mask.png'


@dataclass
class CFG:
    # 경로
    data_root:  str = '/content/data'
    work_dir:   str = '/content/drive/MyDrive/komap/runs'
    weight_dir: str = '/content/drive/MyDrive/komap/weights'

    # 과제 구성
    mode: str = '4class'       # '4class' | '3class' (Si 합쳐 학습 후 두께로 분리)
    si_thresh: float = 8.0     # 3class 전용. 성분 최대 반두께(px) 임계값

    # 모델
    encoder: str = 'timm-efficientnet-b3'
    # ConvNeXt는 'tu-convnext_tiny'. smp 버전에 따라 decoder stage가 안 맞을 수
    # 있으므로 첫 실행에서 forward 통과 여부를 반드시 확인할 것.
    pretrained: bool = True

    # 입력
    crop: int = 512
    overlap: float = 0.25

    # 학습
    batch: int = 8
    epochs: int = 60
    steps_per_epoch: int = 150
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    loss: str = 'wce_dice'     # 'ce' | 'wce' | 'wce_dice' | 'ce_lovasz'
    loss_weights: tuple = (0.5, 0.5)

    # 검증
    split: str = 'kfold'       # 'kfold' (train+valid 90장) | 'official' (70/20)
    n_folds: int = 5
    folds: tuple = (0, 1, 2, 3, 4)
    val_every: int = 5
    tta: bool = True

    seed: int = 42
    num_workers: int = 2
    exp: str = 'c1'
