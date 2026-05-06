# Kaggle 학습 모델 확인 및 FastAPI 적용 가이드

## 1. Kaggle 산출물 확인 결과

현재 확인한 모델 파일:

```text
models/apple_resnet18_best.pt
```

파일은 정상적인 PyTorch checkpoint 구조입니다.

```text
fruit_type: apple
labels: ['A', 'B', 'C']
image_size: 224
model_name: resnet18
model_version: freshness-apple-resnet18-v0
fc.weight.shape: (3, 512)
```

즉, 사과 이미지를 입력하면 `A/B/C` 품질 등급을 예측하는 ResNet18 기반 모델로 저장된 상태입니다.

학습 설정과 평가 지표는 다음과 같습니다.

```text
model: resnet18
image_size: 224
batch_size: 64
epochs: 15
learning_rate: 0.0003
weight_decay: 0.0001

valid_accuracy: 0.9979
valid_f1: 0.9979
test_accuracy: 0.6343
test_f1: 0.5880
```

판단:

- 모델 파일 생성 자체는 잘 됐습니다.
- checkpoint 안에 FastAPI 추론에 필요한 `model_state_dict`, `labels`, `fruit_type`, `image_size`, `model_name`, `model_version`이 들어 있습니다.
- 다만 test 성능은 약 63.4%라서, 실서비스용으로 충분하다고 단정하기보다는 추가 데이터 확인, 품종별 분포 확인, train/valid/test 분리 방식 점검이 필요합니다.
- 현재 모델은 `apple` 전용입니다. 배, 감귤, 감은 각각 `pear_resnet18_best.pt`, `mandarine_resnet18_best.pt`, `persimmon_resnet18_best.pt`를 따로 학습해야 합니다.

## 2. FastAPI 적용 방식

앱에서 직접 `.pt` 모델을 쓰는 것이 아니라, 앱은 FastAPI 서버에 이미지를 업로드하고 FastAPI가 모델을 로드해 추론한 뒤 JSON을 반환하는 구조로 연결합니다.

```text
Flutter 앱
  -> 이미지 + fruit_type 업로드
FastAPI 서버
  -> apple_resnet18_best.pt 로드
  -> 이미지 전처리
  -> ResNet18 추론
  -> 색상/형태/멍 추정 점수 계산
  -> JSON 응답 반환
Flutter 앱
  -> 등급, 점수, 출고 보조 판단 표시
```

권장 API:

```text
POST /api/v1/dl/freshness-scan
Content-Type: multipart/form-data
```

요청 필드 예시:

```text
image: 업로드 이미지 파일
fruit_type: apple
quality_inspection_id: 선택값
```

응답 예시:

```json
{
  "data": {
    "quality_inspection_id": 31,
    "model_grade": "A",
    "freshness_score": 91.3,
    "color_score": 88.2,
    "roundness_score": 94.5,
    "bruise_probability": 0.07,
    "model_decision": "PASS",
    "fruit_type": "apple",
    "model_confidence": 0.86,
    "model_version": "freshness-apple-resnet18-v0"
  },
  "message": "success",
  "error": null
}
```

## 3. 모델 파일 배치

개발 서버에서는 다음처럼 두는 것을 권장합니다.

```text
backend/
  app/
    main.py
    routers/
      dl_router.py
    services/
      dl_service.py
    ml_models/
      apple_resnet18_best.pt
```

또는 현재 저장소 구조를 그대로 쓰면:

```text
models/
  apple_resnet18_best.pt
```

중요:

- `.pt` 파일은 용량이 크기 때문에 Git에 올리지 않는 것이 좋습니다.
- 배포 시에는 서버 디스크, S3, Google Drive, Kaggle output 다운로드 파일 등으로 별도 전달합니다.
- FastAPI 서버가 시작될 때 모델을 한 번만 로드하고, 요청마다 다시 로드하지 않는 구조가 좋습니다.

## 4. FastAPI 구현 예시

아래 코드는 현재 노트북의 구조를 FastAPI 서비스로 옮기는 최소 예시입니다.

### 4.1 requirements

```text
fastapi
uvicorn
python-multipart
torch
torchvision
pillow
numpy
```

### 4.2 `app/services/dl_service.py`

```python
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from fastapi import UploadFile
from PIL import Image
from torch import nn
from torchvision import models, transforms


MODEL_PATHS = {
    "apple": Path("app/ml_models/apple_resnet18_best.pt"),
    "pear": Path("app/ml_models/pear_resnet18_best.pt"),
    "mandarine": Path("app/ml_models/mandarine_resnet18_best.pt"),
    "persimmon": Path("app/ml_models/persimmon_resnet18_best.pt"),
}

GRADE_SCORES = {"A": 90.0, "B": 70.0, "C": 45.0}


@dataclass(frozen=True)
class FreshnessResult:
    fruit_type: str
    model_grade: str
    freshness_score: float
    color_score: float
    roundness_score: float
    bruise_probability: float
    model_decision: str
    model_confidence: float
    model_version: str


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    labels = checkpoint["labels"]
    model = build_model(num_classes=len(labels))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


class DLModelStore:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache = {}

    def get(self, fruit_type: str):
        if fruit_type not in MODEL_PATHS:
            raise ValueError(f"Unsupported fruit_type: {fruit_type}")
        if fruit_type not in self.cache:
            self.cache[fruit_type] = load_checkpoint(MODEL_PATHS[fruit_type], self.device)
        return self.cache[fruit_type]


model_store = DLModelStore()


def build_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def extract_features(image_path: Path, image_size: int) -> dict[str, float]:
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32)
    saturation = hsv[..., 1] / 255.0
    value = hsv[..., 2] / 255.0

    color_score = round(float(np.clip((saturation.mean() * 0.65 + value.mean() * 0.35) * 100.0, 0.0, 100.0)), 2)
    dark_ratio = np.logical_and(value < 0.33, saturation > 0.18).mean()
    bruise_probability = round(float(np.clip(min(1.0, dark_ratio / 0.18), 0.0, 1.0)), 4)

    arr = np.asarray(image, dtype=np.float32)
    border = np.concatenate([arr[:8].reshape(-1, 3), arr[-8:].reshape(-1, 3), arr[:, :8].reshape(-1, 3), arr[:, -8:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    distance = np.linalg.norm(arr - bg, axis=2)
    mask = distance > max(18.0, float(distance.mean()))
    ys, xs = np.where(mask)
    if len(xs) < 50:
        roundness_score = 50.0
    else:
        width = max(1, xs.max() - xs.min() + 1)
        height = max(1, ys.max() - ys.min() + 1)
        aspect_score = min(width, height) / max(width, height)
        fill_ratio = mask.sum() / float(width * height)
        roundness_score = round(float(np.clip((aspect_score * 0.65 + min(fill_ratio / 0.78, 1.0) * 0.35) * 100.0, 0.0, 100.0)), 2)

    return {
        "color_score": color_score,
        "roundness_score": roundness_score,
        "bruise_probability": bruise_probability,
    }


def calculate_freshness_score(model_grade: str, color_score: float, roundness_score: float, bruise_probability: float) -> float:
    grade_score = GRADE_SCORES.get(model_grade, GRADE_SCORES["C"])
    bruise_free_score = max(0.0, min(100.0, (1.0 - bruise_probability) * 100.0))
    score = grade_score * 0.60 + color_score * 0.20 + roundness_score * 0.10 + bruise_free_score * 0.10
    return round(max(0.0, min(100.0, score)), 2)


def make_model_decision(freshness_score: float, bruise_probability: float) -> str:
    if freshness_score < 60.0 or bruise_probability >= 0.5:
        return "HOLD"
    if freshness_score >= 80.0 and bruise_probability < 0.2:
        return "PASS"
    return "REVIEW"


async def predict_upload(file: UploadFile, fruit_type: str) -> FreshnessResult:
    model, checkpoint = model_store.get(fruit_type)
    labels = checkpoint["labels"]
    image_size = int(checkpoint.get("image_size", 224))
    model_version = checkpoint.get("model_version", f"freshness-{fruit_type}-resnet18-v0")

    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        image_path = Path(tmp.name)

    image = Image.open(image_path).convert("RGB")
    tensor = build_transform(image_size)(image).unsqueeze(0).to(model_store.device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    confidence, index = torch.max(probabilities, dim=0)
    model_grade = labels[int(index.item())]
    features = extract_features(image_path, image_size)
    freshness_score = calculate_freshness_score(
        model_grade=model_grade,
        color_score=features["color_score"],
        roundness_score=features["roundness_score"],
        bruise_probability=features["bruise_probability"],
    )

    image_path.unlink(missing_ok=True)

    return FreshnessResult(
        fruit_type=fruit_type,
        model_grade=model_grade,
        freshness_score=freshness_score,
        color_score=features["color_score"],
        roundness_score=features["roundness_score"],
        bruise_probability=features["bruise_probability"],
        model_decision=make_model_decision(freshness_score, features["bruise_probability"]),
        model_confidence=round(float(confidence.item()), 4),
        model_version=model_version,
    )


def to_api_response(result: FreshnessResult, quality_inspection_id: int | None = None):
    data = asdict(result)
    data["quality_inspection_id"] = quality_inspection_id
    return {"data": data, "message": "success", "error": None}
```

### 4.3 `app/routers/dl_router.py`

```python
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.dl_service import predict_upload, to_api_response


router = APIRouter(prefix="/api/v1/dl", tags=["DL"])


@router.post("/freshness-scan")
async def freshness_scan(
    image: UploadFile = File(...),
    fruit_type: str = Form("apple"),
    quality_inspection_id: int | None = Form(None),
):
    try:
        result = await predict_upload(image, fruit_type)
        return to_api_response(result, quality_inspection_id)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Model file not found for fruit_type={fruit_type}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
```

### 4.4 `app/main.py`

```python
from fastapi import FastAPI

from app.routers.dl_router import router as dl_router


app = FastAPI(title="Harvest Slot API")
app.include_router(dl_router)
```

실행:

```powershell
uvicorn app.main:app --reload
```

Swagger 테스트:

```text
http://127.0.0.1:8000/docs
```

## 5. Flutter 앱 연결 방식

앱에서는 multipart 요청으로 이미지를 보냅니다.

```text
POST http://서버주소/api/v1/dl/freshness-scan
image = 촬영 또는 갤러리 이미지
fruit_type = apple
```

서버 응답의 주요 필드:

```text
model_grade: A/B/C
freshness_score: 0~100
model_decision: PASS/REVIEW/HOLD
model_confidence: 모델 confidence
```

앱 화면에서는 `model_grade`, `freshness_score`, `model_decision`을 기본 표시값으로 쓰고, 최종 출고 여부는 점주가 확정하는 구조가 문서 방향과 맞습니다.

## 6. 현재 상태에서 해야 할 일

1. Kaggle output에서 `apple_resnet18_best.pt`를 다운로드합니다.
2. FastAPI 프로젝트의 `app/ml_models/apple_resnet18_best.pt` 위치에 둡니다.
3. 위 `dl_service.py`, `dl_router.py` 구조로 API를 만듭니다.
4. Swagger에서 사과 이미지를 업로드해 응답이 나오는지 확인합니다.
5. Flutter 앱에서 같은 endpoint로 multipart 업로드를 연결합니다.
6. 이후 과일별로 학습을 반복해 아래 파일을 추가합니다.

```text
app/ml_models/apple_resnet18_best.pt
app/ml_models/pear_resnet18_best.pt
app/ml_models/mandarine_resnet18_best.pt
app/ml_models/persimmon_resnet18_best.pt
```

## 7. 주의점

- 현재 확인된 파일은 사과 전용 모델입니다. `fruit_type=pear`, `mandarine`, `persimmon`은 해당 모델 파일이 없으면 처리하지 않아야 합니다.
- 테스트 성능이 높지 않으므로 앱 데모에서는 사용 가능하지만, 실제 서비스 판단 자동화에는 추가 검증이 필요합니다.
- 모델 결과는 최종 출고 확정이 아니라 점주 판단을 돕는 보조 정보로 사용해야 합니다.
- FastAPI 서버가 CPU만 써도 동작은 가능하지만, 요청이 많아지면 GPU 서버나 배치 최적화가 필요할 수 있습니다.
