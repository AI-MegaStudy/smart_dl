# DL Notes

현재 기준으로 사용하는 핵심 노트는 아래 두 개입니다.

- `DL_apple_balanced_training.ipynb`
  - 사과 단일 이미지 학습 노트입니다.
  - 입력 이미지 1장을 `top/middle/side` 중 하나로 먼저 분류합니다.
  - 분류된 view에 따라 `top_grade`, `middle_grade`, `side_grade` 모델 중 하나를 사용해 A/B/C 등급을 예측합니다.
  - view confidence가 `0.60` 미만이면 `RETAKE`로 처리해 앱에서 재촬영을 요청하도록 구성했습니다.

- `DL_apple_ngrok_serving.ipynb`
  - Kaggle에서 학습된 모델 4개를 FastAPI + ngrok으로 임시 서빙하는 노트입니다.
  - 앱 또는 로컬 PC에서 ngrok URL로 이미지를 업로드해 실제 API 연동을 테스트할 수 있습니다.

현재 기준 모델 폴더는 아래입니다.

```text
models/apple_balanced/
```

이전 실험 노트는 삭제하지 않고 아래에 보관했습니다.

```text
Note/archive/
```
