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
    min_sip_frac: float = 0.002
    # 초정 Si 예측 면적이 이 비율 미만이면 공정 Si로 돌린다. 0이면 끈다.
    # 근거: GT 90장 중 초정 Si 부재 2장, 존재하는 88장의 최소 면적비 1.53%.
    # 부재 이미지에 1픽셀만 흘려도 그 이미지 점수가 0.25 깎인다.

    # 모델
    arch: str = 'unet'   # unet | unetpp | deeplabv3plus | manet | fpn | pspnet | linknet
    encoder: str = 'timm-efficientnet-b3'
    # ConvNeXt는 'tu-convnext_tiny'. smp 버전에 따라 decoder stage가 안 맞을 수
    # 있으므로 첫 실행에서 forward 통과 여부를 반드시 확인할 것.
    pretrained: bool = True

    # 입력
    scale: float = 1.0   # 입력 확대 배율. 공정 Si 폭이 약 2.5px라 1배에서는
                         # 경계 1px 오차만으로 IoU 상한이 0.50까지 떨어진다.
                         # 2배로 올리면 상한이 0.69로 오른다 (3배는 추가 이득 없음).
                         # 평가는 항상 원본 해상도에서 수행한다.
    in_mode: str = 'gray'    # 'gray' | 'ridge'
    # 'ridge': 3채널을 [원본, 얇은선 강조(sato), 국소대비]로 채운다.
    # 공정 Si와 Al3Ni는 같은 수지상간 공간에서 뒤엉켜 정출돼 명암만으로
    # 확정이 어려우므로 두께·분기 같은 구조 단서를 명시적으로 준다.
    target_width: int = 0    # >0이면 모든 이미지를 이 가로폭으로 맞춘다(축척 통일).
                             # 데이터가 675~1440px로 섞여 있고 종횡비는 모두 1.406으로
                             # 같다. 공정 Si 반두께도 1.4~3.0px로 폭에 거의 비례해,
                             # 같은 조직이 이미지마다 다른 픽셀 크기로 보이는 상태다.
                             # scale과 동시에 쓰지 않는다 (target_width가 우선).
    crop: int = 512
    overlap: float = 0.25
    aug_scale: float = 0.2   # RandomScale 폭. 0이면 끈다.
                             # 공정 Si 폭이 약 2.5px라 스케일 증강이 라벨을
                             # 뭉갤 수 있어 반드시 켠 경우/끈 경우를 비교할 것.

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
    use_all: bool = False   # True면 90장 전부로 학습하고 검증을 건너뛴다(최종 제출용).
    split: str = 'kfold'       # 'kfold' (train+valid 90장) | 'official' (70/20)
    n_folds: int = 5
    folds: tuple = (0, 1, 2, 3, 4)
    val_every: int = 5
    tta: bool = True
    calibrate: bool = True   # 검증셋에서 클래스별 logit 보정값을 탐색한다.
                             # 지금 모델은 공정 Si를 과예측 중(8.4% vs 정답 6.9%).

    sample_weight: str = 'uniform'   # 'uniform' | 'fine'
    # 'fine': Al3Ni가 잘게 쪼개진(둘레/면적이 큰) 이미지를 더 자주 뽑는다.
    # 90장 중 14장에서 Al3Ni 반두께가 1.0px(폭 2px)까지 얇아지는데 13장이
    # 675x480이다. 최악 이미지(A16, mIoU 0.50)가 전부 이 집단에 속한다.
    sample_power: float = 1.0        # 가중치 지수. 클수록 어려운 이미지에 치우친다.

    bias_from: str = ''
    # use_all 학습에서 쓴다. 90장 전부로 학습하면 검증셋이 학습에 포함돼 있어
    # 자체 보정값은 과적합이다. 같은 설정의 k-fold 실험 이름을 주면 그 fold들의
    # 보정값 평균을 그대로 가져다 쓰고 자체 보정은 건너뛴다.

    resume: bool = True      # fold{n}.pth가 이미 있으면 그 fold는 건너뛴다
    seed: int = 42
    num_workers: int = 2
    exp: str = 'c1'
