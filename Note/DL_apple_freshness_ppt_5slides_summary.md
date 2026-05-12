# 사과 DL 신선도 판별 프로세스 5장 발표 요약

원본 문서: `Note/DL_apple_freshness_full_process_report.md`

발표 화면 파일: `Note/DL_apple_freshness_ppt_5slides.html`

Canva 업로드용 파일:

- `Note/canva_upload/DL_apple_freshness_canva_5pages.pdf`
- `Note/canva_upload/page_01_data_problem.png`
- `Note/canva_upload/page_02_split_category.png`
- `Note/canva_upload/page_03_model_training.png`
- `Note/canva_upload/page_04_performance_selection.png`
- `Note/canva_upload/page_05_api_app.png`

## 1장. 데이터와 문제 정의

핵심 설명:

- AI-Hub 계열 농산물 품질 이미지 데이터를 사용했다.
- Kaggle에는 사과 데이터만 연결했고, 현재 모델도 사과 전용이다.
- 원본은 `Training/Validation`으로 나뉘며, 각각 원천 이미지와 라벨링 JSON을 포함한다.
- JSON의 `cate2`, `cate3`, `angle_direction`, `group_no`를 사용해 품종, 등급, 촬영 view, 같은 사과 개체를 파악했다.

시각화 자료 자리:

- `01_data_folder_structure.png`
- `02_json_label_mapping.png`

## 2장. 데이터 분리와 카테고리화

핵심 설명:

- 품질 등급은 `특/상/보통`, `L/M/S`, `A/B/C`를 최종 `A/B/C`로 통일했다.
- 세부 각도 5개는 앱 사용성에 맞춰 `top/middle/side` 3개 view로 재분류했다.
- 같은 사과가 train/test에 동시에 들어가지 않도록 `group_id` 기준으로 분리했다.
- `Training + Validation`을 합친 뒤 `train_valid:test = 7:3`, 다시 `train:valid = 8:2`로 나눴다.

시각화 자료 자리:

- `03_label_mappings.png`
- `04_group_split_vs_random_split.png`
- `05_view_distribution_and_balancing.png`

## 3장. 모델 구조와 학습 방식

핵심 설명:

- 최종 모델은 PyTorch `ResNet18` 기반 전이학습 모델이다.
- ImageNet pretrained weight를 사용하고 마지막 `fc` layer만 class 수에 맞게 교체했다.
- view 모델 1개가 `top/middle/side`를 먼저 판단하고, 이후 view별 grade 모델 3개 중 하나가 `A/B/C`를 예측한다.
- 주요 설정은 `image_size=224`, `batch_size=64`, `epochs=10`, `learning_rate=3e-4`, `weight_decay=1e-4`이다.

시각화 자료 자리:

- ResNet18 구조 다이어그램
- `06_final_model_routing.png`

## 4장. 성능 비교와 최종 모델 선정

핵심 설명:

- random split 모델은 점수가 높았지만 같은 사과의 다른 각도가 train/test에 동시에 들어갈 수 있어 제외했다.
- 5개 각도 모델은 세부 각도 불균형과 앱 사용성 문제로 제외했다.
- 최종 모델은 `models/apple_balanced/`의 `top/middle/side view 모델 + view별 grade 모델 3개`이다.

최종 balanced test 결과:

| 모델 | accuracy | macro_f1 |
|---|---:|---:|
| view | 0.9177 | 0.9162 |
| top grade | 0.8605 | 0.8612 |
| middle grade | 0.9074 | 0.9065 |
| side grade | 0.9136 | 0.9136 |

시각화 자료 자리:

- `07_experiment_performance_comparison.png`
- `08_final_model_metrics.png`
- `09_final_confusion_matrices.png`

## 5장. 신선도 지표와 앱 적용

핵심 설명:

- API는 이미지 1장을 받아 view, grade, confidence, 신선도 보조 점수를 반환한다.
- `freshness_score`는 등급 점수, 색상 점수, 둥근 정도, 멍 가능성을 합산해 계산한다.
- 앱은 `action_required`를 기준으로 `RETAKE`, `OWNER_REVIEW`, `NONE` 화면을 분기한다.
- Kaggle 테스트는 ngrok을 사용하고, 운영은 별도 Python FastAPI 서버가 더 안정적이다.

신선도 점수:

```text
freshness_score =
  grade_score * 0.60
  + color_score * 0.20
  + roundness_score * 0.10
  + (1 - bruise_probability) * 100 * 0.10
```

시각화 자료 자리:

- `12_freshness_decision_flow.png`
- `13_api_app_branching.png`
- API 응답 예시 JSON 캡처
