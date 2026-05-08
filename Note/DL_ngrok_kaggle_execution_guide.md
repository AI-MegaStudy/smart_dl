# DL API ngrok 실행 방법 - Kaggle 기준

현재 DL API는 Kaggle Notebook에서 FastAPI 서버를 실행하고, ngrok으로 외부 접속 URL을 열어 백엔드가 호출하는 방식입니다.

즉, 백엔드가 DL 모델을 직접 실행하는 것이 아니라 아래처럼 호출합니다.

```text
앱/백엔드
  -> ngrok public URL
  -> Kaggle Notebook 안의 FastAPI
  -> apple_balanced 모델 4개 실행
  -> 분석 결과 JSON 반환
```

## 1. Kaggle에 필요한 파일

Kaggle Notebook에는 아래 파일들이 있어야 합니다.

```text
DL_apple_ngrok_serving.ipynb

models/apple_balanced/
  apple_view_top_middle_side_balanced_resnet18_best.pt
  apple_top_grade_resnet18_best.pt
  apple_middle_grade_resnet18_best.pt
  apple_side_grade_resnet18_best.pt
```

현재 모델은 사과 전용입니다.

구조는 다음과 같습니다.

```text
이미지 1장 입력
  -> view 모델이 top / middle / side 중 하나로 분류
  -> 분류된 view에 맞는 등급 모델 1개 선택
  -> A / B / C 등급 예측
  -> freshness_score, confidence, review 여부 반환
```

## 2. ngrok 토큰 설정

Kaggle에서 `Add-ons > Secrets`에 아래 값을 추가합니다.

```text
Label: NGROK_AUTH_TOKEN
Value: 본인 ngrok authtoken
```

Notebook 코드에서는 직접 토큰을 문자열로 넣지 않고 Kaggle Secrets에서 읽습니다.

```python
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
NGROK_AUTH_TOKEN = user_secrets.get_secret("NGROK_AUTH_TOKEN")
```

따라서 코드에 아래처럼 토큰 이름 자체가 들어가면 안 됩니다.

```text
NGROK_AUTH_TOKEN
```

이 값은 토큰이 아니라 Secret의 이름입니다.

## 3. Kaggle에서 실행 순서

Kaggle Notebook에서 `DL_apple_ngrok_serving.ipynb`를 열고 위에서부터 순서대로 실행합니다.

실행이 정상적으로 되면 아래와 같은 로그가 나옵니다.

```text
Uvicorn running on http://0.0.0.0:8000
ngrok public URL: https://xxxx-xxxx-xxxx.ngrok-free.app
```

백엔드 또는 테스트 클라이언트는 이 `https://...ngrok-free.app` 주소를 사용하면 됩니다.

## 4. 백엔드가 호출할 주소

Kaggle Notebook에서 출력된 ngrok 주소가 아래와 같다고 하면,

```text
https://abcd-1234.ngrok-free.app
```

백엔드 호출 주소는 다음입니다.

```text
POST https://abcd-1234.ngrok-free.app/owner/quality-inspections
```

이미지 필드명은 `image`입니다.

## 5. 요청 예시

Python 기준 테스트 코드는 다음과 같습니다.

```python
import requests

API_URL = "https://abcd-1234.ngrok-free.app/owner/quality-inspections"
IMAGE_PATH = "Data/images/apple_sample_1.png"

with open(IMAGE_PATH, "rb") as f:
    files = {
        "image": ("apple_sample_1.png", f, "image/png")
    }
    response = requests.post(API_URL, files=files, timeout=60)

print(response.status_code)
print(response.json())
```

이미지 확장자가 jpg이면 `"image/jpeg"`를 사용해도 됩니다.

## 6. 응답 예시

응답은 아래 형태입니다.

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

## 7. 백엔드에서 우선 보면 되는 필드

백엔드 연동에서 우선 필요한 필드는 아래입니다.

```text
model_grade
freshness_score
color_score
roundness_score
bruise_probability
model_decision
action_required
angle_label
angle_confidence
grade_confidence
retake_reason
model_version
message
error
```

추가로 현재 응답에는 아래 필드도 포함될 수 있습니다.

```text
quality_inspection_id
fruit_type
image_url
view_confidence_threshold
grade_confidence_threshold
image_quality
```

이 필드는 응답에는 포함되지만, 백엔드가 반드시 DB에 저장해야 한다는 뜻은 아닙니다.

## 8. confidence 처리 기준

현재 기준은 다음과 같습니다.

```text
angle_confidence < 0.60
  -> RETAKE
  -> 다시 촬영 요청

grade_confidence < 0.55
  -> OWNER_REVIEW
  -> 점주 확인 필요
```

`angle_confidence`는 입력 이미지가 top, middle, side 중 어떤 view인지 판단할 때 모델이 가진 확신도입니다.

예를 들어 `angle_label = top`, `angle_confidence = 0.9697`이면 모델이 이 이미지를 top으로 볼 확률을 약 96.97%로 판단했다는 뜻입니다.

## 9. 중요한 운영 조건

ngrok 방식은 Kaggle Notebook이 켜져 있어야만 동작합니다.

즉, 아래 조건이 모두 유지되어야 합니다.

```text
Kaggle Notebook 실행 중
FastAPI 서버 실행 중
ngrok 터널 실행 중
모델 파일 4개 로드 성공
```

Notebook이 꺼지거나 세션이 끊기면 ngrok 주소도 더 이상 사용할 수 없습니다.

또한 ngrok 무료 주소는 실행할 때마다 바뀔 수 있습니다.

따라서 백엔드 개발자에게는 매번 새로 출력된 URL을 전달해야 합니다.

## 10. ngrok을 못 켜는 경우 대안

ngrok은 개발 및 테스트용 임시 연결 방식입니다.

만약 Kaggle Notebook을 계속 켜두기 어렵다면 대안은 다음과 같습니다.

```text
1. models/apple_balanced 모델 파일 4개를 백엔드 서버 또는 별도 Python 서버로 옮김
2. FastAPI 코드를 서버에서 직접 실행
3. 백엔드는 localhost 또는 내부 서버 주소로 DL API 호출
```

운영 환경에서는 Kaggle + ngrok보다 별도 Python FastAPI 서버에 모델을 올리는 방식이 더 안정적입니다.

Kaggle + ngrok은 백엔드 연동 테스트, 시연, 임시 검증에 적합합니다.

