# KoMaP AI 경진대회 2026 — 알루미늄 미세조직 4상 분할

광학현미경 이미지에서 픽셀 단위로 4개 상을 분할한다.

| 인덱스 | 상 | 팔레트(RGB) | 면적 비중 |
|---|---|---|---|
| 0 | Al 기지 | 153,127,76 | 약 57% |
| 1 | Al3Ni | 25,76,153 | 약 27% |
| 2 | Primary Si (초정) | 76,178,76 | 약 8% |
| 3 | Eutectic Si (공정) | 204,204,204 | 약 8% |

- 데이터: train 70 / valid 20 / test 10 (test는 정답 없음)
- 평가: 이미지별 4개 상 IoU 평균 → 10장 전체 평균 (mIoU)
- 제출: 4색 PNG 10장 (`{원본파일명}_mask.png`) + 소스코드 + 패키지 정보

## 데이터 실측 요약

GT 90장 전수 측정 결과다. 모델 설계의 근거가 된다.

- **초정 Si와 공정 Si는 같은 물질(Si)** 이라 밝기로 구분되지 않는다. 밝기 기반
  분리 시 공정 Si IoU는 0.30에 그친다.
- 두 상이 **직접 맞닿는 경우가 거의 없다.** Si 연결성분 83,000개 중 섞인 성분
  0.07%, 초정 Si 경계의 이웃은 Al 65% / Al3Ni 35% / 공정 Si 0.03%.
- 따라서 **연결성분 단위 분류가 성립**하며, 성분 최대 반두께 8.0px 단일 임계값만으로
  초정 IoU 0.919 / 공정 IoU 0.893이 나온다.
- Si 경계에 10% 노이즈를 줘도 평균 0.836을 유지한다.

→ `mode='3class'`는 이 측정에 기반한 구조다. Al / Al3Ni / Si 3클래스로 학습한 뒤
두께로 Si를 초정·공정으로 가른다. `mode='4class'`(직접 분할)와 같은 fold에서
비교하는 것이 이 프로젝트의 핵심 실험이다.

## 구조

```
src/
  config.py        팔레트, 하이퍼파라미터
  data.py          로딩, 증강, Dataset, fold 분할
  model.py         U-Net + 로컬 사전학습 encoder
  losses.py        CE / Weighted CE / Dice / Lovász-Softmax
  metrics.py       대회 방식 mIoU (이미지별 평균)
  postprocess.py   두께 기반 Si 분리
  baseline.py      multi-Otsu 기준선 (학습 없음)
  train.py         fold 학습
  predict.py       추론 + 검증 + zip 생성
  prepare_weights.py  사전학습 가중치 1회 저장 (인터넷 필요)
notebooks/run_colab.ipynb
```

## 실행

```bash
pip install -r requirements.txt

# 1) 사전학습 가중치 저장 (인터넷 필요, 1회만)
python -m src.prepare_weights timm-efficientnet-b3 ./weights

# 2) 기준선 확인
python -m src.baseline

# 3) 학습
python -m src.train

# 4) 추론 + 제출 파일 생성
python -m src.predict
```

가중치는 `weights/`에 저장되고 학습·추론은 네트워크 없이 동작한다.

데이터 경로는 `src/config.py`의 `data_root`가 가리키며,
`<data_root>/{train,valid,test}/images`, `.../masks` 구조를 기대한다.

## 재현성

- `set_seed()`가 random / numpy / torch 시드와 cuDNN 결정론 플래그를 고정한다
- 사전학습 가중치는 항상 로컬 파일에서 읽는다 (실행 중 다운로드 없음)
- 패키지 버전은 `requirements.txt` / `environment.yml`에 고정

## 주의

- 출력 마스크를 **리사이즈하거나 JPEG로 저장하면 안 된다.** 보간이 섞여 팔레트 밖
  색이 생기고 `predict.verify()`에서 걸린다. 크기를 되돌릴 때는 `Image.NEAREST`.
- `crop=768` 이상은 쓰지 말 것. 675x480 이미지가 전체의 약 29%라 패딩 비율이
  12.6%까지 올라가고 크롭 다양성이 사라진다. 384 또는 512가 적절하다.
