# DL 신선도 판별 현재 계획

## 1. 현재 방향

현재 DL 기능은 **사과 이미지 1장**을 입력받아 점주의 품질 판단을 보조하는 구조입니다.

전체 흐름:

```text
사과 이미지 1장
-> view 모델: top / middle / side 판단
-> view confidence 확인
-> confidence < 0.60이면 RETAKE
-> confidence >= 0.60이면 해당 view의 grade 모델 선택
-> A / B / C 등급 예측
-> color_score, roundness_score, bruise_probability 계산
-> freshness_score 계산
-> model_decision 반환
```

DL 결과는 자동 출고 확정값이 아니라 **점주 보조 판단 자료**입니다. 최종 등급과 출고 여부는 점주가 확정합니다.

## 2. 모델 구성

현재 사용하는 모델은 총 4개입니다.

```text
apple_view_top_middle_side_balanced_resnet18_best.pt
apple_top_grade_resnet18_best.pt
apple_middle_grade_resnet18_best.pt
apple_side_grade_resnet18_best.pt
```

각 모델 역할:

| 모델 | 역할 |
|---|---|
| view 모델 | 입력 이미지가 `top`, `middle`, `side` 중 어디에 가까운지 판단 |
| top grade 모델 | top 이미지의 A/B/C 등급 판단 |
| middle grade 모델 | middle 이미지의 A/B/C 등급 판단 |
| side grade 모델 | side 이미지의 A/B/C 등급 판단 |

view 라벨 매핑:

```text
top -> top
front45, diagonal45 -> middle
front90, diagonal90 -> side
```

## 3. 학습 노트

현재 학습 노트:

```text
Note/DL_apple_view_balanced_training.ipynb
```

주요 특징:

- 원본 `Training`과 `Validation` 폴더를 먼저 합칩니다.
- `group_no` 또는 파일명 번호 기준으로 같은 사과 개체를 하나의 group으로 묶습니다.
- group 기준으로 `train_valid:test = 7:3` 분리합니다.
- train_valid 내부에서 `train:valid = 8:2`로 다시 분리합니다.
- 같은 사과의 여러 각도 이미지는 train/valid/test에 나뉘지 않도록 합니다.
- view 모델 train 데이터만 `top/middle/side` 균형 샘플로 구성합니다.
- grade 모델은 view별 전체 train 데이터를 사용하고, `grade_label x variety` 기준 sampler로 불균형을 완화합니다.

## 4. 추론 정책

초기 threshold:

```python
VIEW_CONFIDENCE_THRESHOLD = 0.60
GRADE_CONFIDENCE_THRESHOLD = 0.55
```

판단 정책:

| 조건 | model_decision | 앱 처리 |
|---|---|---|
| view confidence < 0.60 | `RETAKE` | 재촬영 요청 |
| grade confidence < 0.55 | `REVIEW` | 점주 확인 |
| freshness_score < 60 또는 bruise_probability >= 0.5 | `HOLD` | 품질 확인 |
| freshness_score >= 80 그리고 A등급 | `PASS` | 출고 가능 참고 |
| 그 외 | `REVIEW` | 점주 확인 |

`RETAKE`는 모델 오류가 아니라 **이미지 각도 판단이 애매하므로 다시 촬영하는 것이 안전하다**는 의미입니다.

API 응답은 다음 공통 형태를 사용합니다.

```json
{
  "data": {
    "fruit_type": "apple",
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
    "model_version": "apple-single-image-top-middle-side-view-balanced-router-v0",
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

여기서 `angle_confidence`는 이름은 angle이지만 실제 의미는 `top/middle/side` 시점 판단 모델의 confidence입니다.

## 5. FastAPI/ngrok 테스트

현재 서빙 노트:

```text
Note/DL_apple_ngrok_serving.ipynb
```

Kaggle에서 이 노트를 실행하면 FastAPI 서버와 ngrok 터널을 열 수 있습니다.

API:

```text
GET  /
POST /owner/quality-inspections
```

multipart 필드:

```text
image: 사과 이미지 파일
```

로컬 테스트 스크립트:

```text
scripts/test_ngrok_quality_api.py
```

예시:

```powershell
python scripts\test_ngrok_quality_api.py `
  --url "https://xxxx.ngrok-free.app" `
  --image "Data\images\ex1.png"
```

## 6. 현재 주의점

- 현재 모델은 사과 전용입니다.
- 과수원처럼 배경이 복잡한 사진은 성능이 떨어질 수 있습니다.
- 사과가 작게 나오거나 여러 개가 동시에 찍힌 이미지는 안정적이지 않을 수 있습니다.
- 앱에서는 사과 1개가 화면 중앙에 크게 보이도록 촬영 가이드를 제공하는 것이 좋습니다.
- `model_decision`은 점주 보조 판단이며, 최종 확정값은 점주가 입력해야 합니다.
