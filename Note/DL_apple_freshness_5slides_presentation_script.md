# 사과 DL 신선도 판별 5페이지 발표 대본

기준 자료:

- `Note/DL_apple_freshness_full_process_report.md`
- `Note/DL_model_fastapi_apply_guide.md`
- `models/apple_balanced/report.md`
- `Note/canva_upload/` 발표 이미지

## 1페이지. 데이터와 문제 정의

이 딥러닝 파트의 목표는 이미지 1장을 입력받아 품질 등급과 신선도 관련 보조 지표를 예측하는 모델을 만드는 것입니다. 여기서 예측은 사람이 직접 기준을 넣는 것이 아니라, 이미지와 정답 라벨의 패턴을 모델이 학습해서 판단하는 방식입니다.

데이터는 원천 이미지와 라벨링 JSON으로 구성되어 있습니다. JSON에서는 품종, 품질 등급, 촬영 각도, 같은 개체를 구분하는 `group_no`를 사용했습니다. 다만 JSON은 학습용 정답을 만들 때만 쓰고, 실제 앱에서는 이미지 1장만 입력으로 사용합니다.

## 2페이지. 데이터 분리와 카테고리화

원본 데이터에는 `top`, `front45`, `front90`, `diagonal45`, `diagonal90`처럼 세부 촬영 각도가 있습니다. 하지만 실제 앱에서는 사용자가 각도를 정확히 맞추기 어렵기 때문에, 최종적으로 `top`, `middle`, `side` 3가지 view로 단순화했습니다.

품질 등급도 여러 표기를 모두 `A/B/C` classification, 즉 3개 등급 분류 문제로 통일했습니다. 그리고 같은 개체의 다른 각도 사진이 train과 test에 동시에 들어가면 평가가 과대평가될 수 있어, 이미지 단위가 아니라 `group_id` 기준으로 데이터를 분리했습니다.

## 3페이지. 모델 구조와 학습 방식

최종 모델은 PyTorch 기반 ResNet18입니다. ResNet18은 이미지 특징을 단계적으로 추출하는 CNN 계열 모델이고, 여기서는 ImageNet으로 미리 학습된 pretrained weight를 사용했습니다. 그래서 처음부터 학습하는 것보다 적은 데이터에서도 안정적으로 시작할 수 있습니다.

구조는 그림처럼 이미지를 224x224로 전처리한 뒤, `Conv2d`, `BatchNorm`, `ReLU`, residual block을 거쳐 특징을 뽑습니다. 이후 view 모델이 먼저 `top/middle/side`를 분류하고, 그 결과에 맞는 grade 모델이 `A/B/C`를 예측합니다. 학습은 CrossEntropyLoss와 AdamW를 사용했고, validation macro F1이 가장 좋은 모델을 저장했습니다.

## 4페이지. 성능 비교와 최종 모델 선정

초기에는 5개 세부 각도를 그대로 쓰는 방식, top/side 2분류 방식, random split 방식도 실험했습니다. 특히 random split은 점수가 높게 나왔지만, 같은 개체의 다른 각도 사진이 train과 test에 동시에 들어갈 수 있어 실제 성능보다 좋게 보일 위험이 있었습니다.

최종 모델은 top/middle/side로 먼저 나누고, view별 등급 모델을 따로 사용하는 구조입니다. view 모델은 accuracy 0.9177, macro F1 0.9162를 기록했고, grade 모델도 top, middle, side 모두 0.86 이상의 macro F1을 보였습니다. 오른쪽 confusion matrix에서도 대각선 값이 크게 나타나, 대부분 정답 class로 잘 분류된 것을 확인할 수 있습니다.

## 5페이지. 신선도 점수와 앱 적용 값

최종 결과는 A/B/C 등급만 반환하지 않고, 앱에서 쓸 수 있는 보조 값도 함께 제공합니다. 대표적으로 `freshness_score`, `model_decision`, `action_required`, 그리고 모델의 확신도인 confidence가 있습니다.

신선도 점수는 등급 점수, 색상, 둥근 정도, 멍 가능성을 가중합으로 계산합니다. 그리고 confidence가 낮으면 바로 확정하지 않고 재촬영이나 점주 확인으로 넘깁니다. 앱에서는 JSON 응답 중 `action_required`를 기준으로 `RETAKE`, `OWNER_REVIEW`, `NONE` 화면을 분기하면 됩니다.

## 발표자 참고용 핵심 용어 설명

아래 내용은 발표 중 그대로 읽기 위한 문장이 아니라, 발표자가 의미를 이해하기 위한 보조 설명입니다.

| 용어 | 설명 |
|---|---|
| `label` | 모델이 맞춰야 하는 정답입니다. 여기서는 view의 `top/middle/side`, 등급의 `A/B/C`가 label입니다. |
| `classification` | 여러 선택지 중 하나를 고르는 분류 문제입니다. 이 프로젝트는 view 분류와 등급 분류로 나뉩니다. |
| `inference` | 학습이 끝난 모델에 새 이미지를 넣어 예측값을 얻는 단계입니다. 실제 앱에서 실행되는 과정입니다. |
| `data leakage` | test에 들어가면 안 되는 정보가 학습에 섞여 점수가 실제보다 높게 나오는 문제입니다. |
| `group split` | 같은 개체의 사진을 한 묶음으로 보고 train/test가 섞이지 않게 나누는 방식입니다. |
| `balanced dataset` | 특정 class가 너무 많거나 적지 않도록 개수를 맞춘 데이터셋입니다. |
| `ResNet18` | 이미지 특징을 뽑는 CNN 모델입니다. 비교적 가볍고 이미지 분류에서 많이 쓰입니다. |
| `pretrained weight` | 큰 이미지 데이터로 미리 학습된 가중치입니다. 처음부터 학습하는 것보다 안정적입니다. |
| `Conv2d` | 이미지에서 경계, 색, 질감 같은 작은 패턴을 찾는 convolution layer입니다. |
| `residual block` | 이전 정보를 건너뛰어 더해주는 구조입니다. 깊은 모델을 안정적으로 학습하게 돕습니다. |
| `CrossEntropyLoss` | 분류 문제에서 예측 확률과 정답이 얼마나 다른지 계산하는 loss입니다. |
| `AdamW` | 모델 가중치를 업데이트하는 optimizer입니다. 과적합을 줄이는 weight decay도 함께 사용합니다. |
| `accuracy` | 전체 중 맞춘 비율입니다. 단, 데이터가 불균형하면 이 값만으로 판단하기 어렵습니다. |
| `macro F1` | class별 F1을 동일 비중으로 평균한 값입니다. 불균형 데이터에서 더 중요하게 봤습니다. |
| `confusion matrix` | 실제 class와 예측 class를 표로 보여줍니다. 어떤 class끼리 헷갈렸는지 볼 수 있습니다. |
| `confidence` | 모델이 자기 예측을 얼마나 확신하는지 나타내는 값입니다. 낮으면 재촬영이나 점주 확인으로 넘깁니다. |
| `threshold` | 판단 기준값입니다. 예를 들어 view confidence가 0.60보다 낮으면 `RETAKE`입니다. |
| `freshness_score` | 등급, 색상, 둥근 정도, 멍 가능성을 조합한 0~100 품질 보조 점수입니다. |
| `action_required` | 앱이 어떤 화면으로 넘어갈지 정하는 값입니다. 예시는 `RETAKE`, `OWNER_REVIEW`, `NONE`입니다. |
