# 사과 DL 신선도 판별 5페이지 발표 대본

기준 자료:

- `Note/DL_apple_freshness_full_process_report.md`
- `Note/DL_model_fastapi_apply_guide.md`
- `models/apple_balanced/report.md`
- `Note/canva_upload/` 발표 이미지

## 1페이지. 데이터와 문제 정의

이 딥러닝 파트의 목표는 이미지 1장을 입력받아 품질 등급과 신선도 관련 보조 지표를 예측하는 모델을 만드는 것입니다.

사용한 데이터는 AI-Hub 계열 농산물 품질 이미지 데이터 구조를 따릅니다. 원본 데이터는 `Training`과 `Validation`으로 나뉘고, 각각 원천 이미지와 라벨링 JSON을 포함합니다.

라벨링 JSON에서는 품종, 품질 등급, 촬영 각도, 그리고 같은 사과 개체를 구분하는 `group_no`를 사용했습니다. 중요한 점은 JSON은 학습용 label 생성에만 사용하고, 실제 앱 사용 단계에서는 이미지 1장만 입력으로 사용한다는 것입니다.

## 2페이지. 데이터 분리와 카테고리화

원본 데이터에는 (`top`, `front45`, `front90`, `diagonal45`, `diagonal90`처럼) 다양한 세부 촬영 각도가 존재합니다. 하지만 앱 사용자가 항상 정확한 각도로 촬영하기는 어렵기 때문에, 최종적으로 `top`, `middle`, `side` 3가지로 재분류했습니다.

품질 등급도 (원본의 `특/상/보통`, `L/M/S`, `A/B/C`를) 모두 `A/B/C` 분류로 통일했습니다.

또한 같은 사과가 여러 각도에서 촬영되어 있기 때문에 이미지 단위 random split을 사용하면 data 누수가 발생할 수 있습니다. 그래서 최종 방식은 `group_id` 기준 split 하였고, 원본 Training과 Validation을 합친 뒤 train/test를 7:3으로 나누고, train 내부에서 다시 train/valid를 8:2로 나눴습니다.

## 3페이지. 모델 구조와 학습 방식

최종 모델은 PyTorch 기반 ResNet18 학습 모델입니다. ImageNet pretrained weight를 사용하고, 마지막 fc layer만 각 과정의 class 수(view 모델의 경우 top/middle/side, grade 모델의 경우 A/B/C)에 맞게 교체했습니다.

모델 구조는 먼저 입력 이미지를 224x224 RGB 이미지로 전처리하고, ResNet18 backbone을 통과시킵니다. 내부적으로는 `Conv2d`, `BatchNorm`, `ReLU`, `MaxPool`, residual block, average pooling, fc layer 순서로 진행됩니다.

최종 예측 구조는 2단계입니다. 먼저 view 모델이 이미지를 `top/middle/side` 중 하나로 분류하고, 그 결과에 따라 `top grade`, `middle grade`, `side grade` 모델 중 하나를 선택해 A/B/C 등급을 예측합니다. 학습에는 CrossEntropyLoss와 AdamW optimizer를 사용했고, validation macro F1을 기준으로 best model을 저장했습니다.

## 4페이지. 성능 비교와 최종 모델 선정

초기에는 5개 세부 각도를 그대로 분류하거나, top/side 2분류 방식도 실험했습니다. 또한 random split 모델은 점수가 높게 나왔지만, 같은 사과의 다른 각도가 train과 test에 동시에 들어갈 수 있어 일반화 성능을 과대평가할 위험이 있었습니다.

최종 모델은 `models/apple_balanced`의 top/middle/side 구조입니다. view 모델은 accuracy 0.9177, macro F1 0.9162를 기록했습니다. grade 모델은 top 0.8612, middle 0.9065, side 0.9136 수준의 macro F1을 보였습니다.

confusion matrix를 보면 view 모델과 각 grade 모델이 대부분 대각선 방향으로 잘 맞추고 있습니다. 따라서 최종 구조는 데이터 누수를 줄인 group 별 split, class 불균형을 줄인 balanced test, 그리고 view별 grade classification이 결합된 구조라고 볼 수 있습니다.

## 5페이지. 신선도 점수와 앱 적용 값

결과적으로 모델은 단순히 A/B/C 등급만 반환하지 않고, `freshness_score`, `color_score`, `roundness_score`, `bruise_probability`, confidence 값까지 함께 제공합니다.

신선도 점수는 등급 기반 점수를 60%, 색상 점수를 20%, 둥근 정도를 10%, 멍 가능성을 10% 반영해 계산합니다. 그리고 confidence threshold를 기준으로 최종 action을 결정합니다.

예를 들어 view confidence가 0.60보다 낮으면 `RETAKE`로 재촬영을 요청하고, grade confidence가 0.55보다 낮으면 `OWNER_REVIEW`로 점주 확인을 요청합니다. 최종 API 응답에는 `model_grade`, `freshness_score`, `model_decision`, `action_required` 같은 값이 포함되며, 앱은 `action_required`를 기준으로 화면을 구성하게 됩니다.

## 발표자 참고용 용어 설명

아래 내용은 발표 중 그대로 읽기 위한 문장이 아니라, 발표자가 의미를 이해하기 위한 보조 설명입니다.

| 용어 | 설명 |
|---|---|
| `label` | 모델이 맞춰야 하는 정답 값입니다. 여기서는 view 라벨인 `top/middle/side`와 등급 라벨인 `A/B/C`가 label입니다. |
| `classification` | 입력 데이터를 여러 class 중 하나로 분류하는 문제입니다. 이 프로젝트는 view classification과 grade classification으로 나눌 수 있습니다. |
| `inference` | 학습이 끝난 모델에 새 이미지를 넣고 예측 결과를 얻는 과정입니다. 실제 앱에서 사용자가 이미지를 업로드했을 때 수행되는 단계입니다. |
| `data leakage` | 학습 데이터에 test 데이터와 지나치게 비슷하거나 같은 정보가 들어가 평가 점수가 실제보다 높게 나오는 문제입니다. 같은 사과의 다른 각도 사진이 train/test에 동시에 들어가면 이 위험이 있습니다. |
| `group split` | 이미지 단위가 아니라 같은 사과 개체를 하나의 group으로 묶어 train/valid/test 중 한 곳에만 배치하는 split 방식입니다. 데이터 누수를 줄이기 위해 사용했습니다. |
| `balanced dataset` | 특정 class가 너무 많거나 적지 않도록 개수를 맞춘 데이터셋입니다. top 이미지가 너무 많아 view 모델이 top에 치우치는 문제를 줄이기 위해 사용했습니다. |
| `ResNet18` | residual block을 사용하는 CNN 모델입니다. 이미지 분류에서 많이 쓰이며, 비교적 가볍고 학습이 안정적입니다. |
| `pretrained weight` | ImageNet 같은 대규모 이미지 데이터로 미리 학습된 가중치입니다. 처음부터 학습하는 것보다 적은 데이터에서도 더 안정적으로 학습할 수 있습니다. |
| `Conv2d` | 이미지에서 지역적인 패턴을 추출하는 convolution layer입니다. 색, 경계, 질감 같은 특징을 단계적으로 잡아냅니다. |
| `BatchNorm` | layer 출력값의 분포를 안정화해 학습을 더 빠르고 안정적으로 만드는 기법입니다. |
| `ReLU` | 음수는 0으로 만들고 양수는 그대로 통과시키는 activation function입니다. 신경망이 비선형 패턴을 학습할 수 있게 돕습니다. |
| `MaxPool` | feature map의 크기를 줄이면서 중요한 특징을 남기는 연산입니다. 계산량을 줄이고 위치 변화에 조금 더 강하게 만듭니다. |
| `residual block` | 입력을 layer 출력에 더하는 skip connection 구조입니다. 깊은 모델에서도 gradient가 잘 전달되도록 도와줍니다. |
| `fully connected layer` 또는 `fc layer` | 마지막 feature를 class 수에 맞는 출력값으로 바꾸는 layer입니다. 여기서는 view 모델은 3개 class, grade 모델은 3개 class로 출력합니다. |
| `CrossEntropyLoss` | classification 문제에서 많이 쓰는 loss function입니다. 모델 예측 확률이 정답 class와 얼마나 다른지 계산합니다. |
| `AdamW` | learning rate를 적응적으로 조절하는 optimizer입니다. weight decay를 분리해서 적용하기 때문에 딥러닝 학습에서 자주 사용됩니다.(가중치 제한으로 과적합 방지) |
| `learning rate` | 모델 가중치를 한 번 업데이트할 때 얼마나 크게 움직일지 정하는 값입니다. 너무 크면 불안정하고, 너무 작으면 학습이 느립니다. |
| `weight decay` | 모델이 과하게 복잡해지는 것을 막기 위한 regularization 값입니다. 과적합을 줄이는 데 도움을 줍니다. |
| `early stopping` | validation 성능이 더 이상 좋아지지 않으면 학습을 중단하는 방식입니다. 불필요한 학습과 과적합을 줄이기 위해 사용합니다. |
| `accuracy` | 전체 샘플 중 맞춘 비율입니다. 데이터가 균형일 때는 직관적이지만, class 불균형이 심하면 단독으로 보기 어렵습니다. |
| `precision` | 모델이 어떤 class라고 예측한 것 중 실제로 맞은 비율입니다. 특정 class로 과하게 예측하면 precision이 낮아질 수 있습니다. |
| `recall` | 실제 특정 class인 샘플 중 모델이 해당 class로 찾아낸 비율입니다. 놓치지 않는 능력을 볼 때 중요합니다. |
| `F1 score` | precision과 recall의 조화 평균입니다. 둘 중 하나만 높고 하나가 낮으면 F1도 낮아집니다. |
| `macro F1` | class별 F1을 단순 평균한 값입니다. 각 class를 동일하게 보기 때문에 class 불균형이 있을 때 중요합니다. |
| `confusion matrix` | 실제 class와 예측 class가 어떻게 매칭됐는지 보여주는 표입니다. 어떤 class끼리 헷갈리는지 확인할 수 있습니다. |
| `confidence` | softmax 확률 중 가장 높은 값입니다. 모델이 자신의 예측을 얼마나 확신하는지 나타내며, 낮으면 재촬영이나 점주 확인을 요청합니다. |
| `threshold` | 특정 기준값입니다. 예를 들어 view confidence가 0.60보다 낮으면 `RETAKE`로 처리합니다. |
| `freshness_score` | 모델 등급과 색상, 둥근 정도, 멍 가능성을 조합해 만든 0~100 품질 보조 점수입니다. 실제 생물학적 신선도 측정값은 아니고 앱 판단을 돕기 위한 점수입니다. |
| `model_decision` | 모델이 계산한 최종 판정입니다. 예시는 `PASS`, `REVIEW`, `HOLD`, `RETAKE`입니다. |
| `action_required` | 앱이 어떤 화면 흐름을 보여줘야 하는지 나타내는 값입니다. 예시는 `RETAKE`, `OWNER_REVIEW`, `NONE`입니다. |
