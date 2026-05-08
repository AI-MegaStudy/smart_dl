# DL Notes

현재 기준으로 사용하는 주요 노트와 문서는 아래와 같습니다.

- `DL_apple_balanced_training.ipynb`
  - 사과 단일 이미지 학습 노트입니다.
  - 입력 이미지 1장을 `top/middle/side` 중 하나로 먼저 분류합니다.
  - 분류된 view에 따라 `top_grade`, `middle_grade`, `side_grade` 모델 중 하나를 사용해 A/B/C 등급을 예측합니다.
  - train, valid, test 모두 view/등급/품종 기준으로 균형을 맞춘 데이터셋을 사용합니다.
  - view confidence가 `0.60` 미만이면 `RETAKE`로 처리해 앱에서 재촬영을 요청할 수 있도록 구성했습니다.

- `DL_apple_ngrok_serving.ipynb`
  - Kaggle에서 학습된 모델 4개를 FastAPI + ngrok으로 임시 서빙하는 노트입니다.
  - 백엔드 또는 로컬 PC에서 ngrok URL로 이미지를 업로드해 실제 API 연동을 테스트할 수 있습니다.

- `DL_ngrok_kaggle_execution_guide.md`
  - Kaggle Notebook에서 FastAPI와 ngrok을 실행하는 방법을 정리한 문서입니다.
  - 백엔드 개발자에게 전달할 ngrok URL, 요청 필드명, 응답 필드, confidence 기준을 포함합니다.

- `DL_model_fastapi_apply_guide.md`
  - 최종 모델 파일을 FastAPI 또는 백엔드에서 사용하는 방법을 정리한 문서입니다.
  - 현재 기준 모델 폴더는 `models/apple_balanced`입니다.

- `DL_apple_freshness_full_process_report.md`
  - 데이터 구조, 전처리, 모델 선정 과정, 실패한 접근, 최종 모델 결과를 포함한 전체 프로세스 보고서입니다.

현재 기준 모델 폴더는 아래입니다.

```text
models/apple_balanced/
```

이전 실험 노트와 모델은 삭제하지 않고 archive 폴더에 보관합니다.

```text
Note/archive/
models/archive/
```

