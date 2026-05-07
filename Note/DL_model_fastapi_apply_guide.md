# 사과 DL 모델 FastAPI 및 앱 적용 가이드

## 1. 현재 사용하는 모델

현재 최종 후보 모델 세트는 아래 폴더에 있습니다.

```text
models/apple_view_balanced/
```

필요한 파일:

```text
apple_view_top_middle_side_balanced_resnet18_best.pt
apple_top_grade_resnet18_best.pt
apple_middle_grade_resnet18_best.pt
apple_side_grade_resnet18_best.pt
```

각 모델 역할:

| 파일 | 역할 |
|---|---|
| `apple_view_top_middle_side_balanced_resnet18_best.pt` | 이미지가 top/middle/side 중 어디에 가까운지 판단 |
| `apple_top_grade_resnet18_best.pt` | top 이미지의 A/B/C 등급 예측 |
| `apple_middle_grade_resnet18_best.pt` | middle 이미지의 A/B/C 등급 예측 |
| `apple_side_grade_resnet18_best.pt` | side 이미지의 A/B/C 등급 예측 |

## 2. 서비스 추론 흐름

앱은 사과 사진 1장만 서버로 보냅니다.

```text
앱에서 사과 이미지 1장 업로드
-> FastAPI가 이미지 저장
-> view 모델로 top/middle/side 판단
-> view confidence가 0.60 미만이면 RETAKE
-> confidence가 충분하면 해당 grade 모델 선택
-> A/B/C 등급 예측
-> 보조 점수 계산
-> JSON 응답 반환
```

view 분류 기준:

```text
top: 위에서 찍은 사진
middle: top과 side 사이의 중간/사선 시점
side: 옆면에 가까운 사진
```

## 3. FastAPI endpoint

현재 노트와 앱 연동 기준 endpoint:

```text
POST /owner/quality-inspections
Content-Type: multipart/form-data
```

요청 필드:

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `image` | File | Y | 사과 이미지 1장 |

앱에서는 multipart form-data로 `image`라는 이름의 파일 필드를 보내야 합니다.

## 4. 응답 구조

공통 응답 형태:

```json
{
  "data": {},
  "message": "string",
  "error": null
}
```

성공 응답 예시:

```json
{
  "data": {
    "quality_inspection_id": null,
    "fruit_type": "apple",
    "image_url": "/kaggle/working/uploads/example.png",
    "angle_label": "top",
    "angle_confidence": 0.92,
    "view_confidence_threshold": 0.6,
    "model_grade": "B",
    "grade_confidence": 0.74,
    "grade_confidence_threshold": 0.55,
    "freshness_score": 72.41,
    "color_score": 61.8,
    "roundness_score": 83.2,
    "bruise_probability": 0.12,
    "model_decision": "REVIEW",
    "action_required": "OWNER_REVIEW",
    "retake_reason": null,
    "model_version": "apple-single-image-top-middle-side-view-balanced-router-v0",
    "image_quality": {
      "brightness": 0.61,
      "underexposed_ratio": 0.02,
      "overexposed_ratio": 0.01,
      "saturation_mean": 0.48,
      "blur_score": 0.08
    }
  },
  "message": "모델 판단을 점주가 확인해주세요.",
  "error": null
}
```

품질 확인 응답 예시:

```json
{
  "data": {
    "quality_inspection_id": null,
    "fruit_type": "apple",
    "image_url": "/kaggle/working/uploads/914716c94afa474b88c5d0d3f1228d46.png",
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

위 예시는 시점 판단 confidence가 충분하므로 재촬영은 요청하지 않고, 신선도 점수 기준으로 `HOLD`를 반환한 경우입니다. `HOLD`는 자동 폐기나 자동 거절이 아니라 점주 확인이 필요한 상태로 처리합니다.

재촬영 응답 예시:

```json
{
  "data": {
    "angle_label": "middle",
    "angle_confidence": 0.48,
    "view_confidence_threshold": 0.6,
    "model_decision": "RETAKE",
    "action_required": "RETAKE",
    "retake_reason": "VIEW_CONFIDENCE_LOW"
  },
  "message": "이미지 각도 판단이 불확실합니다. 사과가 화면 중앙에 오도록 다시 촬영해주세요.",
  "error": null
}
```

## 5. 앱에서 처리해야 하는 주요 필드

| 필드 | 설명 | 앱 처리 |
|---|---|---|
| `model_grade` | 모델 예측 등급 A/B/C | 결과 카드에 표시 |
| `freshness_score` | 0~100 신선도 보조 점수 | 결과 카드에 표시 |
| `model_decision` | `PASS`, `REVIEW`, `HOLD`, `RETAKE` | 화면 상태 분기 |
| `action_required` | `NONE`, `OWNER_REVIEW`, `RETAKE` | 사용자 행동 분기 |
| `retake_reason` | 재촬영 사유 | 재촬영 안내 |
| `angle_label` | top/middle/side | 디버그 또는 상세 정보 |
| `angle_confidence` | top/middle/side 시점 모델의 confidence. 즉 view confidence | 낮으면 재촬영 |
| `grade_confidence` | 등급 모델 confidence | 낮으면 점주 확인 강조 |
| `color_score` | 색상 보조 점수 | 상세 정보 |
| `roundness_score` | 둥근 정도 보조 점수 | 상세 정보 |
| `bruise_probability` | 멍/어두운 영역 가능성 | 상세 정보 |
| `image_quality` | 밝기, 과노출, 저노출, 채도, 흐림 정도를 담은 촬영 품질 보조 정보 | 상세 정보 또는 디버그 |

앱 분기 추천:

```text
action_required == "RETAKE"
-> 결과 확정 화면으로 가지 않고 재촬영 안내

action_required == "OWNER_REVIEW"
-> 모델 결과를 보여주고 점주가 최종 등급/판정 선택

action_required == "NONE"
-> 모델 결과를 보여주되 점주 최종 확인 버튼 제공
```

## 6. 앱 촬영 가이드

현재 모델은 단일 사과 이미지에 가장 적합합니다.

앱에서 촬영 화면에 아래 가이드를 제공하는 것을 권장합니다.

```text
사과 1개만 촬영해주세요.
사과가 화면 중앙에 크게 보이도록 촬영해주세요.
너무 어둡거나 밝은 곳은 피해주세요.
흔들린 사진은 피해주세요.
가능하면 배경이 복잡하지 않은 곳에서 촬영해주세요.
```

재촬영이 필요한 경우:

```text
view confidence < 0.60
```

앱 메시지 예시:

```text
이미지 각도 판단이 불확실합니다.
사과가 화면 중앙에 오도록 다시 촬영해주세요.
```

## 7. Kaggle ngrok 테스트 방법

Kaggle에서 실행할 노트:

```text
Note/DL_apple_ngrok_serving.ipynb
```

Kaggle Input 또는 `/kaggle/working/models`에 모델 4개가 있어야 합니다.

ngrok 실행 후 출력되는 URL:

```text
https://xxxx.ngrok-free.app
```

로컬 PC에서 테스트:

```powershell
python scripts\test_ngrok_quality_api.py `
  --url "https://xxxx.ngrok-free.app" `
  --image "Data\images\ex1.png"
```

두 장 테스트:

```powershell
python scripts\test_ngrok_quality_api.py --url "https://xxxx.ngrok-free.app" --image "Data\images\ex1.png"
python scripts\test_ngrok_quality_api.py --url "https://xxxx.ngrok-free.app" --image "Data\images\ex2.png"
```

Postman 테스트:

```text
Method: POST
URL: https://xxxx.ngrok-free.app/owner/quality-inspections
Body: form-data
Key: image
Type: File
Value: 테스트 이미지
```

## 8. FastAPI 구현 시 핵심 코드 구조

서버에서는 모델을 요청마다 다시 로드하지 않고, 서버 시작 시 4개 모델을 한 번 로드해 캐시에 보관하는 것이 좋습니다.

권장 구조:

```text
backend/
  app/
    main.py
    routers/
      quality_router.py
    services/
      dl_service.py
    ml_models/
      apple_view_top_middle_side_balanced_resnet18_best.pt
      apple_top_grade_resnet18_best.pt
      apple_middle_grade_resnet18_best.pt
      apple_side_grade_resnet18_best.pt
```

서버 내부 처리:

```text
1. multipart image 수신
2. 이미지 임시 저장
3. view 모델 추론
4. view confidence 확인
5. confidence 낮으면 RETAKE 응답
6. confidence 충분하면 grade 모델 선택
7. A/B/C 등급 추론
8. 보조 점수 계산
9. quality_inspections 저장
10. JSON 응답 반환
```

## 9. DB 저장 권장 필드

Docs의 `quality_inspections` 기준으로 아래 값을 저장하면 됩니다.

```text
image_url
model_grade
freshness_score
color_score
roundness_score
bruise_probability
model_decision
model_version
owner_confirmed_grade
owner_decision
```

추가 저장을 권장하는 값:

```text
angle_label
angle_confidence
grade_confidence
action_required
retake_reason
```

현재 DB 스키마에 추가 필드가 없다면, 우선 API 응답에서만 사용하고 DB 확장 시 반영하면 됩니다.

## 10. 운영 시 주의점

- 현재 모델은 사과 전용입니다.
- 과수원처럼 배경이 복잡한 사진에서는 성능이 떨어질 수 있습니다.
- 사과가 여러 개 찍히면 어떤 사과를 판단하는지 불명확합니다.
- `RETAKE`는 실패가 아니라 안전장치입니다.
- `PASS`도 자동 출고 확정이 아니라 점주 확인 전의 모델 추천입니다.
- 최종 출고 판단은 반드시 점주가 확정해야 합니다.
