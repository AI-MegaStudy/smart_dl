# 사과 DL 신선도 판별 전체 프로세스 정리

## 1. 목적과 서비스 내 역할

이 DL 파트의 목적은 점주가 수확 또는 발주 확인 과정에서 사과 이미지를 촬영했을 때, 모델이 품질 등급과 보조 점수를 계산해 점주의 최종 선별 판단을 돕는 것입니다.

중요한 전제는 다음과 같습니다.

- DL 결과는 자동 출고 확정값이 아닙니다.
- 모델은 `A/B/C` 등급, 신선도 점수, 색상 점수, 둥근 정도 점수, 멍 가능성, 모델 판정을 제공합니다.
- 최종 등급과 최종 출고 판단은 점주가 확정합니다.
- 앱과 DB에서는 모델 결과와 점주 확정 결과를 분리해서 저장해야 합니다.

Docs 기준으로는 아래 흐름과 연결됩니다.

```text
점주 앱 이미지 촬영
-> Multipart 업로드
-> FastAPI 저장
-> PyTorch inference
-> quality_inspections 저장
-> 점주 앱 결과 표시
-> 점주 최종 판정 저장
```

현재 최종 API 기준 endpoint는 다음입니다.

```text
POST /owner/quality-inspections
```

현재 DL 모델은 사과 전용입니다. 배, 감귤, 감까지 확장하려면 같은 구조로 과일별 모델 세트를 별도로 학습해야 합니다.

## 2. 사용한 데이터 개요

사용한 데이터는 AI-Hub 계열 농산물 품질 이미지 데이터 구조를 기준으로 합니다. 현재 Kaggle에는 사과 데이터만 연결한 상태입니다.

Kaggle 데이터 경로는 다음 형태로 맞췄습니다.

```text
/kaggle/input/datasets/hmw0320/data-fruits/
  Training/
    라벨링데이터 또는 Label...
    원천데이터 또는 Source...
  Validation/
    라벨링데이터 또는 Label...
    원천데이터 또는 Source...
```

처음 원본 데이터는 `Training`과 `Validation`으로 나뉘어 있었고, 각각 내부에 라벨링 데이터와 원천 이미지 데이터가 나뉘어 있었습니다. 또한 내부 데이터는 zip으로 제공되는 구조였고, 과일과 품종, 크기 또는 등급 단위로 폴더가 분리되어 있었습니다.

프로젝트에서 원래 검토한 대상 과일은 다음 4종입니다.

| 과일 | 영문 키워드 |
|---|---|
| 사과 | `apple` |
| 배 | `pear` |
| 감귤 | `mandarine` |
| 감 | `persimmon` |

현재 최종 학습 노트는 사과만 대상으로 합니다.

사과 데이터에서 확인한 주요 품종은 다음입니다.

| 품종 | 사용 표기 |
|---|---|
| 부사 | `부사`, `fuji` |
| 양광 | `양광`, `yanggwang` |

노트에서는 사과 데이터 탐색 시 아래 키워드를 사용합니다.

```python
APPLE_VARIETIES = ['부사', '양광', 'fuji', 'yanggwang']
APPLE_KEYWORDS = ['apple', '사과', '부사', '양광', 'fuji', 'yanggwang']
```

## 3. 라벨링 JSON 구조와 사용한 정보

원본 이미지에는 라벨링 JSON이 함께 제공됩니다. 예시 JSON에는 다음과 같은 정보가 들어 있습니다.

```json
{
  "group_no": 601031001000,
  "no": 601031001001,
  "img_no": 1,
  "cate1": "사과",
  "cate2": "부사",
  "cate3": "특",
  "width": "9.7",
  "height": "9.0",
  "weight": "450",
  "identifier": "/사과_부사_특_1_1TOP.png",
  "angle_direction": "top",
  "verticality_angle": 90,
  "horizontality_angle": 0,
  "bndbox": {
    "xmin": 0,
    "ymin": 0,
    "xmax": 1000,
    "ymax": 1000
  }
}
```

여기서 최종 학습에 직접 사용한 핵심 정보는 다음입니다.

| JSON 필드 | 사용 목적 |
|---|---|
| `cate1` | 과일 종류 확인 |
| `cate2` | 품종 확인, 예: 부사/양광 |
| `cate3` | 품질 등급 라벨 생성 |
| `angle_direction` | 촬영 시점 라벨 생성 |
| `group_no` | 같은 사과 개체를 묶는 group id |
| `bndbox` | 객체 영역 정보. 현재 최종 노트에서는 기본적으로 crop에 사용하지 않음 |

사용하지 않거나 보조 분석으로만 본 정보도 있습니다.

| JSON 필드 | 판단 |
|---|---|
| `width`, `height`, `weight` | 실제 앱 추론 시 사용자가 제공할 수 없는 정보라 모델 입력에는 넣지 않음 |
| 카메라 모델, ISO, 노출 시간 등 | 촬영 환경 분석에는 참고 가능하지만 실제 앱에서는 입력받기 어렵기 때문에 학습 입력에는 넣지 않음 |
| `img_height`, `img_width` | 이미지 로딩 및 전처리 확인용 |

중요한 점은 라벨링 JSON은 학습용 정답을 만들기 위해 사용했고, 실제 앱 추론에서는 JSON 없이 이미지만 입력받는다는 것입니다. 따라서 최종 모델은 실제 서비스에서 사용자 사진 1장만 받아도 동작하도록 구성했습니다.

## 4. 등급 라벨 처리

원본 품질 등급은 `cate3` 또는 폴더명에서 가져옵니다. 최종 학습에서는 이를 `A/B/C`로 통일했습니다.

| 원본 라벨 | 최종 라벨 | 의미 |
|---|---|---|
| `특`, `L`, `A` | `A` | 좋은 등급 |
| `상`, `M`, `B` | `B` | 중간 등급 |
| `보통`, `S`, `C` | `C` | 낮은 등급 |

노트 설정은 다음과 같습니다.

```python
GRADE_LABELS = ['A', 'B', 'C']
QC_FOLDER_LABEL_MAP = {'L': 'A', 'M': 'B', 'S': 'C'}
QC_CATE3_LABEL_MAP = {
    '특': 'A',
    '상': 'B',
    '보통': 'C',
    'L': 'A',
    'M': 'B',
    'S': 'C',
    'A': 'A',
    'B': 'B',
    'C': 'C',
}
```

## 5. 촬영 각도 라벨 처리

초기에는 원본 데이터의 세부 촬영 각도를 그대로 쓰는 방식을 검토했습니다.

원본 세부 각도는 다음 5개였습니다.

```text
top
front45
front90
diagonal45
diagonal90
```

하지만 실제 앱에서 사용자가 `front45`, `diagonal45`처럼 정확한 각도로 촬영하게 만드는 것은 어렵습니다. 또한 사과는 둥근 형태라 45도 계열 이미지가 시각적으로 명확히 구분되지 않는 문제가 있었습니다.

그래서 최종 구조에서는 세부 각도를 서비스 관점에 맞게 3개 view로 재분류했습니다.

| 원본 세부 각도 | 최종 view 라벨 | 이유 |
|---|---|---|
| `top` | `top` | 위에서 본 사진 |
| `front45` | `middle` | 위와 옆 사이의 중간 시점 |
| `diagonal45` | `middle` | 위와 옆 사이의 중간 시점 |
| `front90` | `side` | 옆면에 가까운 사진 |
| `diagonal90` | `side` | 옆면에 가까운 사진 |

최종 매핑은 노트에서 다음처럼 처리합니다.

```python
def view_label_from_detail(angle_detail):
    if angle_detail == 'top':
        return 'top'
    if angle_detail in {'front45', 'diagonal45'}:
        return 'middle'
    if angle_detail in {'front90', 'diagonal90'}:
        return 'side'
    return None
```

이 결정의 이유는 다음입니다.

- 앱 사용자는 사과 이미지 1장만 업로드합니다.
- 사진이 위쪽인지, 옆쪽인지, 중간인지 정도는 모델이 판단할 수 있습니다.
- 세부 각도 5개를 그대로 분류하면 `front45`와 `diagonal45`, `front90`과 `diagonal90` 사이의 의미 차이가 앱 서비스에서 크지 않습니다.
- 등급 모델을 view별로 나누면, top 사진과 side 사진의 시각적 차이를 하나의 등급 모델이 모두 떠안지 않아도 됩니다.

## 6. 데이터 누수 문제와 split 방식 변경

원본 데이터를 확인해보니 같은 번호 또는 같은 `group_no`는 같은 사과를 여러 각도에서 찍은 사진이었습니다.

예를 들면 다음과 같은 이미지들은 같은 사과일 가능성이 큽니다.

```text
1-39
1-39_4DI45
1-39_5DI90
```

처음에는 이미지 단위 random split도 검토했습니다. 이미지가 각도별로 다르기 때문에 학습 관점에서 완전히 다른 이미지라고 볼 수도 있었지만, 같은 사과의 색, 흠집, 형태, 표면 패턴이 train과 test에 동시에 들어가면 모델이 실제 일반화 성능보다 높게 평가될 위험이 있습니다.

특히 실제 서비스에서는 처음 보는 사과를 판단해야 하므로, 같은 사과의 다른 각도 이미지가 train과 test에 동시에 들어가는 것은 평가 기준으로 부적절하다고 판단했습니다.

최종 split 방식은 다음입니다.

```text
원본 Training + Validation 폴더를 먼저 합침
-> group_id 기준 train_valid:test = 7:3
-> train_valid 내부에서 train:valid = 8:2
```

최종 비율은 대략 다음입니다.

```text
train 56% / valid 14% / test 30%
```

분리 단위는 이미지 1장이 아니라 `group_id`입니다.

```text
같은 사과의 top/middle/side 사진 -> train/valid/test 중 하나에만 배치
```

`group_id`는 다음 순서로 만듭니다.

1. JSON의 `group_no`를 우선 사용
2. 없으면 파일명에서 사과 번호 추출
3. 둘 다 안 되면 파일 stem 사용

이 방식으로 평가하면 random split보다 점수는 낮아질 수 있지만, 실제 서비스 환경에 가까운 성능을 확인할 수 있습니다.

## 7. 데이터 불균형 문제

원본 사과 데이터는 top 이미지가 압도적으로 많고, middle/side는 훨씬 적었습니다.

실제 확인된 분포 예시는 다음과 같았습니다.

```text
angle_label  middle  side    top
split
test            142   142   1900
train           565   565  11369
valid            59    59   1194
```

이 분포에서는 모델이 top 위주로 학습되기 쉽습니다. 전체 accuracy는 높아 보여도, middle/side를 제대로 구분하지 못할 수 있습니다.

예를 들어 test에서 top이 4,000장 이상이고 middle/side가 각각 200장 수준이면, top을 middle/side로 잘못 예측한 수가 조금만 늘어나도 middle/side precision이 낮게 보입니다. 따라서 이 문제에서는 accuracy 하나만 보기 어렵고, class별 recall, macro f1, confusion matrix를 함께 봐야 합니다.

## 8. 최종 데이터 보정 방식

최종 노트에서는 train/valid/test 모두 모델별 균형 subset을 사용합니다. 원본 split 자체는 유지하지만, 실제 Dataset이 사용하는 row는 조합별 최소 개수에 맞춰 downsampling합니다.

### 8.1 view 모델

view 모델은 `top/middle/side`를 구분하는 모델입니다. 이 모델은 top 데이터가 너무 많으면 middle/side recall과 precision 해석이 흔들릴 수 있으므로, 각 split마다 `angle_label x grade_label x variety` 조합을 맞춥니다.

```text
view 모델 train/valid/test
-> angle_label x grade_label x variety 조합 균형
-> top/middle/side, A/B/C, 부사/양광 비율을 함께 맞춤
```

중요한 점은 부족한 데이터를 증강해서 늘리는 것이 아니라, 많은 쪽을 줄이는 downsampling 방식이라는 점입니다.

balanced split 설정은 다음입니다.

```python
USE_BALANCED_SPLIT_DATASETS = True
SPLIT_BALANCE_GROUP_COLUMNS = {
    'angle': ['angle_label', 'grade_label', 'variety'],
    'top_grade': ['grade_label', 'variety'],
    'middle_grade': ['grade_label', 'variety'],
    'side_grade': ['grade_label', 'variety'],
}
SPLIT_BALANCE_TARGET = 'min'
SPLIT_BALANCE_GROUP_AWARE = True
```

또한 단순 랜덤 샘플링이 아니라 `group_id`를 우선 순회합니다.

```text
같은 사과에서 여러 장을 뽑기보다
가능한 한 서로 다른 사과를 먼저 포함
```

이렇게 한 이유는 개수를 줄일 때 특정 사과만 많이 남는 문제를 피하기 위해서입니다.

### 8.2 grade 모델

등급 모델은 view별로 따로 학습합니다.

```text
top_grade 모델 -> top 이미지만 사용
middle_grade 모델 -> middle 이미지만 사용
side_grade 모델 -> side 이미지만 사용
```

grade 모델도 train/valid/test 모두 각 view 내부에서 `grade_label x variety` 조합을 맞춥니다.

```python
SPLIT_BALANCE_GROUP_COLUMNS = {
    'top_grade': ['grade_label', 'variety'],
    'middle_grade': ['grade_label', 'variety'],
    'side_grade': ['grade_label', 'variety'],
}
```

이 방식을 선택한 이유는 다음입니다.

- 등급 모델의 A/B/C support를 동일하게 맞춰 macro f1을 더 해석하기 쉽게 만듭니다.
- 부사/양광 품종 개수도 같이 맞춰 품종 쏠림을 줄입니다.
- train뿐 아니라 valid/test도 같은 기준으로 맞춰 모델 비교가 더 공정해집니다.

## 9. 이미지 전처리와 증강

입력 이미지는 ResNet18 입력에 맞게 224x224로 처리합니다.

공통 설정:

```python
DEFAULT_IMAGE_SIZE = 224
USE_BBOX_CROP = False
```

`USE_BBOX_CROP = False`로 둔 이유는 실제 서비스에서는 라벨링 JSON의 bbox를 받을 수 없기 때문입니다. 학습 때만 bbox crop을 쓰면 실제 앱 사진과 입력 분포가 달라질 수 있습니다.

train 증강은 task별로 약간 다릅니다.

view 모델 train:

```text
RandomResizedCrop(224, scale=(0.78, 1.0), ratio=(0.9, 1.1))
RandomHorizontalFlip(0.5)
RandomRotation(8)
ColorJitter(brightness=0.25, contrast=0.22, saturation=0.18)
ImageNet Normalize
```

grade 모델 train:

```text
RandomResizedCrop(224, scale=(0.82, 1.0), ratio=(0.92, 1.08))
RandomHorizontalFlip(0.5)
RandomRotation(12)
ColorJitter(brightness=0.18, contrast=0.18, saturation=0.15)
ImageNet Normalize
```

valid/test:

```text
Resize(224, 224)
ImageNet Normalize
```

증강을 너무 강하게 하지 않은 이유는 품질 등급 분류에서 색, 표면 질감, 멍 가능성 등이 중요할 수 있기 때문입니다. 색상과 밝기를 지나치게 바꾸면 실제 등급 신호가 흔들릴 수 있습니다.

## 10. 사용한 딥러닝 모델

최종 모델은 모두 PyTorch `ResNet18` 기반입니다.

```python
model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

학습 시에는 ImageNet pretrained weight를 사용했습니다.

```python
USE_PRETRAINED = True
```

ResNet18을 선택한 이유는 다음입니다.

- Kaggle GPU에서 학습 시간이 비교적 짧습니다.
- 데이터 수가 충분히 크지 않은 상황에서 전이학습을 적용하기 쉽습니다.
- MobileNetV3보다 구조가 직관적이고, 발표나 문서에서 설명하기 좋습니다.
- EfficientNet 계열보다 구현과 디버깅 부담이 낮습니다.
- 224x224 이미지 입력과 ImageNet normalization을 그대로 사용할 수 있습니다.

ResNet18의 큰 구조는 다음처럼 설명할 수 있습니다.

```text
ResNet18(
  입력: RGB 이미지, 3 x 224 x 224
  conv1: 7x7 convolution, stride 2
  bn1
  relu
  maxpool
  layer1: residual block x 2
  layer2: residual block x 2
  layer3: residual block x 2
  layer4: residual block x 2
  avgpool
  fc: Linear(512 -> class 수)
)
```

각 모델의 출력 class 수는 다음입니다.

| 모델 | class 수 | 라벨 |
|---|---:|---|
| view 모델 | 3 | `top`, `middle`, `side` |
| top grade 모델 | 3 | `A`, `B`, `C` |
| middle grade 모델 | 3 | `A`, `B`, `C` |
| side grade 모델 | 3 | `A`, `B`, `C` |

## 11. 학습 설정

최종 노트의 학습 설정은 다음입니다.

```python
TRAIN_CONFIG = {
    'image_size': 224,
    'batch_size': 64,
    'epochs': 10,
    'patience': 4,
    'learning_rate': 3e-4,
    'weight_decay': 1e-4,
    'num_workers': 2 if IS_KAGGLE else 0,
}
```

학습 기준:

- optimizer: AdamW 계열 설정
- loss: CrossEntropyLoss
- 평가 지표: accuracy, macro f1
- best model 저장 기준: validation macro f1
- early stopping: validation 지표가 개선되지 않으면 중단

분류 문제이므로 RMSE, MAE, R2보다 accuracy, precision, recall, f1이 더 적합합니다.

특히 class 불균형이 있기 때문에 accuracy만 보면 안 됩니다.

| 지표 | 의미 | 해석 |
|---|---|---|
| accuracy | 전체 중 맞춘 비율 | class 분포가 균형일 때 직관적 |
| precision | 해당 class라고 예측한 것 중 실제로 맞은 비율 | 특정 class로 과하게 예측하면 낮아짐 |
| recall | 실제 해당 class 중 모델이 찾아낸 비율 | 놓치지 않는 능력 |
| macro f1 | class별 f1을 동일 가중 평균 | 불균형 데이터에서 중요 |
| weighted f1 | support 비율을 반영한 f1 | 다수 class 영향이 큼 |

이 프로젝트에서는 view 모델의 middle/side를 놓치지 않는 것이 중요하므로, view 모델은 class별 recall과 macro f1을 같이 봤습니다.

## 12. 최종 추론 구조

실제 앱에서는 이미지 1장만 업로드합니다.

최종 추론 흐름은 다음입니다.

```text
사과 이미지 1장
-> view 모델로 top/middle/side 판단
-> view confidence 확인
-> confidence < 0.60이면 RETAKE
-> confidence >= 0.60이면 해당 view의 grade 모델 선택
-> A/B/C 등급 예측
-> OpenCV 보조 점수 계산
-> freshness_score 계산
-> model_decision 반환
```

최종 모델 파일은 총 4개입니다.

```text
models/apple_balanced/
  apple_view_top_middle_side_balanced_resnet18_best.pt
  apple_top_grade_resnet18_best.pt
  apple_middle_grade_resnet18_best.pt
  apple_side_grade_resnet18_best.pt
```

각 모델 역할:

| 파일 | 역할 |
|---|---|
| `apple_view_top_middle_side_balanced_resnet18_best.pt` | 입력 이미지가 `top/middle/side` 중 어디에 가까운지 판단 |
| `apple_top_grade_resnet18_best.pt` | top 이미지의 A/B/C 등급 예측 |
| `apple_middle_grade_resnet18_best.pt` | middle 이미지의 A/B/C 등급 예측 |
| `apple_side_grade_resnet18_best.pt` | side 이미지의 A/B/C 등급 예측 |

## 13. 신선도 점수 계산

모델이 A/B/C 등급만 반환하면 앱에서 쓰기에는 설명력이 부족합니다. 그래서 OpenCV 기반 보조 특징을 함께 계산합니다.

계산 항목:

| 항목 | 의미 |
|---|---|
| `color_score` | 색상과 채도 기반 보조 점수 |
| `roundness_score` | 배경 대비 사과 영역의 형태 기반 둥근 정도 |
| `bruise_probability` | 어두운 영역 기반 멍 가능성 |
| `image_quality` | 밝기, 과노출, 저노출, 채도, 흐림 정도 |

신선도 점수 공식은 다음입니다.

```python
grade_score = {'A': 90.0, 'B': 70.0, 'C': 45.0}

freshness_score =
    grade_score * 0.60
  + color_score * 0.20
  + roundness_score * 0.10
  + (1 - bruise_probability) * 100 * 0.10
```

이 점수는 실제 생물학적 신선도를 정확히 측정하는 값이라기보다, 모델 등급과 이미지 보조 특징을 합친 품질 보조 점수입니다.

## 14. 모델 판정 정책

현재 threshold는 다음입니다.

```python
VIEW_CONFIDENCE_THRESHOLD = 0.60
GRADE_CONFIDENCE_THRESHOLD = 0.55
```

판정 정책:

| 조건 | `model_decision` | `action_required` | 앱 처리 |
|---|---|---|---|
| view confidence < 0.60 | `RETAKE` | `RETAKE` | 재촬영 요청 |
| grade confidence < 0.55 | `REVIEW` | `OWNER_REVIEW` | 점주 확인 |
| freshness_score < 60 또는 bruise_probability >= 0.5 | `HOLD` | `OWNER_REVIEW` | 품질 확인 |
| freshness_score >= 80 그리고 A등급 | `PASS` | `NONE` | 출고 가능 참고 |
| 그 외 | `REVIEW` | `OWNER_REVIEW` | 점주 확인 |

여기서 confidence는 softmax 확률 중 가장 큰 값입니다.

```python
probs = torch.softmax(model(tensor), dim=1)
confidence = probs.max()
```

`angle_confidence`라는 필드명은 기존 코드 흐름 때문에 남아 있지만, 현재 의미는 angle 5분류가 아니라 `top/middle/side` view 모델의 confidence입니다.

## 15. 실패하거나 보류한 모델 흐름

### 15.1 과일 4종 통합 노트

처음에는 사과, 배, 감귤, 감을 모두 처리할 수 있는 구조를 검토했습니다.

하지만 실제 학습은 과일별로 따로 진행하는 것이 맞다고 판단했습니다.

이유:

- 과일마다 색, 형태, 품질 기준이 다릅니다.
- 품종과 등급 폴더 구조가 다릅니다.
- Kaggle 학습 시간과 GPU 메모리 부담이 커집니다.
- FastAPI 적용 시 과일별 모델 파일을 명확히 관리하는 편이 안정적입니다.

따라서 현재 최종 노트는 사과 전용으로 분리했습니다.

### 15.2 5개 각도 모델 + 단일 등급 모델

초기에는 다음 구조를 시도했습니다.

```text
각도 모델: top/front45/front90/diagonal45/diagonal90
등급 모델: 전체 각도를 하나의 grade 모델로 A/B/C 분류
```

대표 결과:

```text
각도 모델 test
accuracy: 0.8581
macro_f1: 0.6167

등급 모델 test
accuracy: 0.6319
macro_f1: 0.6071
```

문제:

- top 데이터가 압도적으로 많아 전체 accuracy는 높아 보여도 소수 각도 precision이 낮았습니다.
- `front45`, `front90` support가 매우 적었습니다.
- 하나의 등급 모델이 top, 45도, 90도 이미지를 모두 처리해야 해서 학습 난도가 높았습니다.
- 품종별 성능 차이가 있었습니다. 부사보다 양광에서 낮은 성능이 반복적으로 나타났습니다.

### 15.3 강한 oversampling/증강 실험

데이터 불균형을 해결하기 위해 부족한 각도를 top 수준까지 늘리는 방향을 검토했습니다.

실험 결과:

```text
각도 모델 test
accuracy: 0.7395
macro_f1: 0.5195

등급 모델 test
accuracy: 0.5769
macro_f1: 0.5499
```

문제:

- 부족한 데이터를 과하게 증강하면 실제 새로운 정보가 늘어나는 것이 아니라 비슷한 이미지가 반복됩니다.
- 소수 class recall은 높아질 수 있지만 precision이 크게 낮아졌습니다.
- top 이미지를 middle/side로 잘못 보내는 경우가 늘었습니다.
- 전체 등급 성능도 오히려 떨어졌습니다.

따라서 소수 class를 무작정 다수 class 수까지 늘리는 방식은 최종 방향에서 제외했습니다.

### 15.4 조정 후 5개 각도 + 등급 모델

증강과 sampler를 조정한 뒤에는 일부 개선이 있었습니다.

```text
각도 모델 test
accuracy: 0.8636
macro_f1: 0.6235

등급 모델 test
accuracy: 0.6589
macro_f1: 0.6307
```

개선은 있었지만, 여전히 5개 세부 각도를 앱에서 요구하기 어렵고 등급 모델 성능도 충분하지 않았습니다.

### 15.5 top/side 2분류 + view별 등급 모델

다음으로 top과 side 두 가지로 나누는 구조를 검토했습니다.

```text
view 모델: top/side
top grade 모델
side grade 모델
```

대표 결과:

```text
top/side 확인 모델
accuracy: 0.8773
macro_f1: 0.7834

top 등급 모델
accuracy: 0.6284
macro_f1: 0.6005

side 등급 모델
accuracy: 0.6901
macro_f1: 0.6778
```

문제:

- 45도 이미지를 top에 넣을지 side에 넣을지 애매했습니다.
- 실제 앱에서 45도 촬영을 따로 요구하기도 어렵고, 그렇다고 버리기도 어려웠습니다.
- 그래서 45도 계열을 `middle`로 두는 3분류가 더 자연스럽다고 판단했습니다.

### 15.6 random split 모델

이미지 단위 random split으로 학습하면 성능이 매우 높게 나왔습니다.

대표 결과:

```text
top/side 확인 모델
accuracy: 0.9272
macro_f1: 0.8201

top 등급 모델
accuracy: 0.9734
macro_f1: 0.9735

side 등급 모델
accuracy: 0.9477
macro_f1: 0.9470
```

하지만 이 결과는 최종 모델 선정에서 제외했습니다.

이유:

- 같은 사과의 다른 각도 사진이 train과 test에 동시에 들어갈 수 있습니다.
- 모델이 사과 개체의 색, 표면 패턴, 흠집 같은 정보를 간접적으로 기억할 수 있습니다.
- 실제 서비스에서는 처음 보는 사과를 판단해야 하므로 random split 점수는 과대평가 가능성이 큽니다.

따라서 random split 모델은 참고용 실험으로만 보관했습니다.

### 15.7 top/middle/side base 모델

group 기준 split을 적용하고 top/middle/side 구조로 바꾼 첫 모델은 다음 성능이었습니다.

```text
view 모델
accuracy: 0.8777
macro_f1: 0.7033

top grade 모델
accuracy: 0.6058
macro_f1: 0.5801

middle grade 모델
accuracy: 0.5986
macro_f1: 0.5885

side grade 모델
accuracy: 0.6268
macro_f1: 0.6540
```

이 모델의 문제는 grade 성능이 낮다는 점이었습니다. 특히 middle/side grade 모델은 데이터 수가 적고 품종/등급 조합도 불균형했습니다.

### 15.8 grade category balanced 실험

grade 모델에서도 A/B/C와 품종 개수를 더 강하게 맞추는 실험을 했습니다.

결과:

```text
view 모델
accuracy: 0.8800
macro_f1: 0.6924

top grade 모델
accuracy: 0.8301
macro_f1: 0.8306

middle grade 모델
accuracy: 0.9061
macro_f1: 0.9045

side grade 모델
accuracy: 0.9014
macro_f1: 0.9019
```

base 모델보다 grade 성능이 크게 좋아졌지만, 최종 선정 모델과 비교하면 top/middle/side grade 모두 조금씩 낮았습니다.

또한 view 모델의 middle/side precision은 여전히 낮았습니다. 이는 test 분포에서 top이 압도적으로 많아 top 이미지 일부가 middle/side로 흘러 들어갈 때 precision이 낮게 계산되는 영향도 있었습니다.

## 16. 최종 선정 모델

최종 선정 모델은 다음 폴더에 있습니다.

```text
models/apple_balanced/
```

선정된 모델 파일:

```text
apple_view_top_middle_side_balanced_resnet18_best.pt
apple_top_grade_resnet18_best.pt
apple_middle_grade_resnet18_best.pt
apple_side_grade_resnet18_best.pt
```

선정 이유:

- group 기준 split을 사용해 같은 사과 개체 누수를 줄였습니다.
- 앱에서 이미지 1장만 받는 실제 사용 흐름과 맞습니다.
- 5개 각도보다 `top/middle/side`가 사용자 촬영 상황을 더 자연스럽게 반영합니다.
- view 모델은 train에서만 group-aware balanced dataset을 적용해 top 쏠림을 줄였습니다.
- grade 모델은 view별로 분리해 각 시점에 맞는 등급 판단을 수행합니다.
- 최종 grade 성능이 이전 보관 모델보다 높았습니다.
- API 응답 구조와 FastAPI/ngrok 테스트 노트까지 연결되어 있습니다.

## 17. 최종 모델 평가 결과

최종 모델은 `models/apple_balanced/report.md` 기준으로 정리합니다. 이 결과는 원본 test 전체 분포가 아니라, 모델별 balanced test subset에서 평가한 값입니다.

평가 균형 기준:

| 모델 | balanced test 기준 |
|---|---|
| view 모델 | `angle_label x grade_label x variety` |
| top grade 모델 | `grade_label x variety` |
| middle grade 모델 | `grade_label x variety` |
| side grade 모델 | `grade_label x variety` |

### 17.1 view 모델

balanced test 구성:

```text
원본 test 샘플 수: 4765
balanced test 샘플 수: 486
조합별 target: 27
angle_label balanced: top 162 / middle 162 / side 162
grade_label balanced: A 162 / B 162 / C 162
variety balanced: 부사 243 / 양광 243
```

성능:

```text
accuracy: 0.9177
macro_f1: 0.9162
```

confusion matrix:

```text
        top  middle  side
top     132      14    16
middle    6     156     0
side      4       0   158
```

class별 결과:

| class | precision | recall | f1-score | support |
|---|---:|---:|---:|---:|
| top | 0.93 | 0.81 | 0.87 | 162 |
| middle | 0.92 | 0.96 | 0.94 | 162 |
| side | 0.91 | 0.98 | 0.94 | 162 |

해석:

- 기존 불균형 test에서는 top이 압도적으로 많아 middle/side precision이 낮게 보였지만, balanced test에서는 세 class가 동일 support로 평가됩니다.
- middle과 side recall이 각각 0.96, 0.98로 높아 실제 중간/옆면 이미지를 잘 찾아냅니다.
- top recall은 0.81로 상대적으로 낮아 top 일부가 middle/side로 분산됩니다. 앱에서는 confidence threshold와 재촬영 정책으로 보완합니다.

### 17.2 top grade 모델

balanced test 구성:

```text
원본 test 샘플 수: 4339
balanced test 샘플 수: 3792
조합별 target: 632
grade_label balanced: A 1264 / B 1264 / C 1264
variety balanced: 부사 1896 / 양광 1896
```

성능:

```text
accuracy: 0.8605
macro_f1: 0.8612
```

confusion matrix:

```text
      A     B     C
A  1107    90    67
B    64  1071   129
C    19   160  1085
```

class별 결과:

| class | precision | recall | f1-score | support |
|---|---:|---:|---:|---:|
| A | 0.93 | 0.88 | 0.90 | 1264 |
| B | 0.81 | 0.85 | 0.83 | 1264 |
| C | 0.85 | 0.86 | 0.85 | 1264 |

품종별 성능:

| 품종 | count | accuracy | macro_f1 |
|---|---:|---:|---:|
| 부사 | 1896 | 0.8729 | 0.8741 |
| 양광 | 1896 | 0.8481 | 0.8481 |

해석:

- A/B/C support가 완전히 같아진 상태에서도 accuracy 0.8605, macro_f1 0.8612로 안정적입니다.
- 부사가 양광보다 조금 높지만, 두 품종의 차이는 이전보다 해석하기 쉬운 상태입니다.

### 17.3 middle grade 모델

balanced test 구성:

```text
원본 test 샘플 수: 213
balanced test 샘플 수: 162
조합별 target: 27
grade_label balanced: A 54 / B 54 / C 54
variety balanced: 부사 81 / 양광 81
```

성능:

```text
accuracy: 0.9074
macro_f1: 0.9065
```

confusion matrix:

```text
    A   B   C
A  43   2   9
B   2  50   2
C   0   0  54
```

class별 결과:

| class | precision | recall | f1-score | support |
|---|---:|---:|---:|---:|
| A | 0.96 | 0.80 | 0.87 | 54 |
| B | 0.96 | 0.93 | 0.94 | 54 |
| C | 0.83 | 1.00 | 0.91 | 54 |

세부 각도별 성능:

| 세부 각도 | count | accuracy | macro_f1 |
|---|---:|---:|---:|
| diagonal45 | 123 | 0.8943 | 0.8961 |
| front45 | 39 | 0.9487 | 0.9365 |

품종별 성능:

| 품종 | count | accuracy | macro_f1 |
|---|---:|---:|---:|
| 부사 | 81 | 0.8519 | 0.8485 |
| 양광 | 81 | 0.9630 | 0.9629 |

해석:

- middle 모델은 balanced test에서도 0.90 이상의 macro_f1을 유지합니다.
- C recall이 1.00으로 매우 높고, A recall이 0.80으로 상대적으로 낮습니다.
- 부사보다 양광 성능이 높게 나타났습니다.

### 17.4 side grade 모델

balanced test 구성:

```text
원본 test 샘플 수: 213
balanced test 샘플 수: 162
조합별 target: 27
grade_label balanced: A 54 / B 54 / C 54
variety balanced: 부사 81 / 양광 81
```

성능:

```text
accuracy: 0.9136
macro_f1: 0.9136
```

confusion matrix:

```text
    A   B   C
A  53   1   0
B   5  49   0
C   1   7  46
```

class별 결과:

| class | precision | recall | f1-score | support |
|---|---:|---:|---:|---:|
| A | 0.90 | 0.98 | 0.94 | 54 |
| B | 0.86 | 0.91 | 0.88 | 54 |
| C | 1.00 | 0.85 | 0.92 | 54 |

세부 각도별 성능:

| 세부 각도 | count | accuracy | macro_f1 |
|---|---:|---:|---:|
| diagonal90 | 123 | 0.9024 | 0.8988 |
| front90 | 39 | 0.9487 | 0.9434 |

품종별 성능:

| 품종 | count | accuracy | macro_f1 |
|---|---:|---:|---:|
| 부사 | 81 | 0.9012 | 0.9017 |
| 양광 | 81 | 0.9259 | 0.9257 |

해석:

- side 모델은 balanced test에서 accuracy와 macro_f1 모두 0.9136입니다.
- A recall이 0.98로 높고, C precision이 1.00입니다.
- front90이 diagonal90보다 약간 높게 나왔습니다.

## 18. 최종 결과 종합 분석

최종 모델의 강점:

- 실제 서비스에 가까운 group split을 사용했습니다.
- 이미지 1장 입력만으로 동작합니다.
- view 판단 후 view별 grade 모델로 라우팅하므로, top과 side의 시각 차이를 한 모델이 모두 처리하지 않아도 됩니다.
- train/valid/test 모두 균형 subset을 사용해 A/B/C, view, 품종 쏠림을 줄였습니다.
- view 모델은 balanced test에서 accuracy 0.9177, macro_f1 0.9162로 나왔습니다.
- top grade는 0.8605 accuracy, middle/side grade는 각각 0.9074, 0.9136 accuracy로 나왔습니다.
- `RETAKE` 정책으로 confidence가 낮은 이미지는 다시 촬영하게 만들 수 있습니다.
- FastAPI/ngrok 노트가 있어 Kaggle에서 바로 임시 API 테스트가 가능합니다.

남은 한계:

- balanced test 기준으로는 middle/side precision이 개선되었지만, top recall은 0.81로 상대적으로 낮습니다.
- 원본 test 분포 자체는 여전히 top이 압도적으로 많습니다. 실제 운영 환경에서 촬영 분포가 어떻게 나오는지 추가 검증이 필요합니다.
- 과수원처럼 배경이 복잡한 실제 사진에서는 성능이 떨어질 수 있습니다.
- 현재 모델은 JSON bbox 없이 전체 이미지를 학습하므로, 사과가 작게 찍히거나 여러 개가 찍히면 불안정할 수 있습니다.
- 현재 데이터는 같은 촬영 환경에서 찍힌 정제 이미지가 많아 실제 스마트폰 사진과 차이가 있을 수 있습니다.
- 신선도 점수는 실제 저장 기간이나 당도 측정값이 아니라 이미지 품질 등급과 보조 특징을 합친 대리 점수입니다.

## 19. 최종 API 응답 구조

현재 응답 형태는 다음입니다.

```json
{
  "data": {
    "quality_inspection_id": null,
    "fruit_type": "apple",
    "image_url": "/kaggle/working/uploads/example.png",
    "angle_label": "top",
    "angle_confidence": 0.9697,
    "view_confidence_threshold": 0.6,
    "model_grade": "C",
    "grade_confidence": 0.8143,
    "grade_confidence_threshold": 0.55,
    "freshness_score": 55.35,
    "color_score": 52.47,
    "roundness_score": 80.65,
    "bruise_probability": 0.021,
    "model_decision": "HOLD",
    "action_required": "OWNER_REVIEW",
    "retake_reason": null,
    "model_version": "apple-single-image-top-middle-side-balanced-split-router-v1",
    "image_quality": {
      "brightness": 0.8302,
      "underexposed_ratio": 0.0004,
      "overexposed_ratio": 0.4624,
      "saturation_mean": 0.3602,
      "blur_score": 0.1896
    }
  },
  "message": "품질 확인이 필요합니다. 점주가 최종 판단해주세요.",
  "error": null
}
```

앱에서는 `action_required`를 기준으로 화면을 분기하는 것이 가장 단순합니다.

```text
RETAKE -> 재촬영 안내
OWNER_REVIEW -> 모델 결과를 보여주고 점주가 최종 판단
NONE -> 모델 기준 통과, 그래도 점주 최종 확인 버튼 제공
```

## 20. 현재 사용하는 파일

현재 최종 학습 노트:

```text
Note/DL_apple_balanced_training.ipynb
```

Kaggle FastAPI/ngrok 서빙 노트:

```text
Note/DL_apple_ngrok_serving.ipynb
```

앱/FastAPI 적용 가이드:

```text
Note/DL_model_fastapi_apply_guide.md
```

현재 모델:

```text
models/apple_balanced/
```

이전 실험 모델:

```text
models/archive/exp_apple_tms_base/
models/archive/exp_apple_tms_grade_category_balanced/
```

## 21. Kaggle에서 실행할 때 필요한 것

학습 노트 실행 시 필요한 것은 다음입니다.

```text
1. Note/DL_apple_balanced_training.ipynb
2. Kaggle Input에 연결된 사과 데이터셋
3. 데이터 경로: /kaggle/input/datasets/hmw0320/data-fruits
4. GPU 활성화
```

학습 후 `/kaggle/working/models/` 아래에 모델 4개가 생성됩니다.

서빙 노트 실행 시 필요한 것은 다음입니다.

```text
1. Note/DL_apple_ngrok_serving.ipynb
2. 모델 4개
3. Kaggle Secret: NGROK_AUTH_TOKEN
4. ngrok public URL
```

ngrok token은 literal 문자열 `NGROK_AUTH_TOKEN`을 넣는 것이 아니라, ngrok 대시보드에서 복사한 실제 authtoken 값을 Kaggle Secrets에 저장해야 합니다.

## 22. 앞으로 개선할 수 있는 방향

실서비스 수준으로 끌어올리려면 다음 개선이 필요합니다.

### 22.1 실제 스마트폰 촬영 데이터 추가

현재 데이터는 정제된 이미지입니다. 과수원, 창고, 실내 조명, 복잡한 배경 등 실제 환경 사진을 추가해야 합니다.

### 22.2 사과 검출 또는 배경 제거

현재는 전체 이미지를 입력합니다. 실제 앱에서는 사과가 작게 찍히거나 배경이 복잡할 수 있으므로 다음 중 하나가 필요할 수 있습니다.

- 촬영 가이드로 사과를 중앙에 크게 맞추기
- 간단한 segmentation 또는 detection 적용
- 배경 제거 모델 추가
- 앱에서 촬영 영역 가이드 제공

### 22.3 confidence 기반 UX 강화

현재 view confidence threshold는 0.60입니다. 운영 중에는 threshold별 재촬영 비율과 오분류율을 분석해 조정해야 합니다.

예:

```text
0.50 -> 재촬영 적음, 오분류 위험 증가
0.60 -> 현재 기준
0.70 -> 오분류 감소 가능, 재촬영 증가
```

### 22.4 과일별 모델 확장

배, 감귤, 감까지 확장할 경우 다음 형태가 좋습니다.

```text
models/
  apple_balanced/
  pear_balanced/
  mandarine_balanced/
  persimmon_balanced/
```

과일별로 품종, 등급, 촬영 각도 분포가 다르므로 모델은 따로 학습하는 것이 안전합니다.

### 22.5 평가 데이터 재구성

현재 test는 group 기준으로 분리되어 있지만, 실제 운영 성능을 더 정확히 보려면 다음 test set이 필요합니다.

- 실제 스마트폰 촬영 이미지
- 복잡한 배경 이미지
- 사과가 작게 찍힌 이미지
- 조명이 어둡거나 과노출된 이미지
- 품종별 균형 test set
- 등급별 균형 test set

## 23. 최종 결론

최종적으로 선택한 구조는 다음입니다.

```text
사과 이미지 1장
-> top/middle/side view 모델
-> confidence 낮으면 RETAKE
-> view별 grade 모델 선택
-> A/B/C 예측
-> OpenCV 보조 특징 계산
-> freshness_score 계산
-> PASS/REVIEW/HOLD/RETAKE 반환
```

이 구조가 최종 선정된 이유는 다음입니다.

- 앱에서 이미지 1장만 받는 실제 사용 방식과 맞습니다.
- 같은 사과 개체가 train/test에 동시에 들어가는 데이터 누수를 줄였습니다.
- top, middle, side의 시각 차이를 grade 모델에서 분리했습니다.
- view 모델의 top 쏠림 문제를 group-aware balanced train dataset으로 완화했습니다.
- random split처럼 과대평가된 점수가 아니라 group split 기반 결과를 사용했습니다.
- 최종 grade 모델 성능이 이전 실험보다 개선되었습니다.
- FastAPI와 ngrok 테스트 흐름까지 연결되어 앱 연동 가능성이 높습니다.

현재 모델은 MVP와 발표용으로는 충분히 설명 가능한 수준입니다. 다만 실제 서비스 적용 전에는 실제 촬영 이미지 기반 추가 검증, 배경/크기 문제 대응, confidence threshold 재조정이 필요합니다.
