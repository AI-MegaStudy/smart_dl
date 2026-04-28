# Harvest Slot DL 담당 정리

## 1. 프로젝트에서 DL의 역할

DL 파트는 수확량 예측이 아니라, 수확 당일 과일 이미지를 촬영하거나 업로드했을 때 신선도와 품질을 판별하는 보조 기능이다.

전체 흐름 안에서는 다음 단계에 해당한다.

1. 점주가 ML 예측을 참고해 예약 슬롯을 확정한다.
2. 고객이 확정된 슬롯으로 예약한다.
3. 수확 당일 점주가 과일을 촬영한다.
4. PyTorch DL 모델이 품질 등급, 신선도 점수, 멍 확률, 출고 보조 판단을 반환한다.
5. 점주가 최종 등급과 출고 여부를 확정한다.

중요한 원칙은 DL 결과도 자동 출고 결정이 아니라 점주 판단을 돕는 보조 지표라는 점이다.

## 2. DL 요구사항 요약

문서 기준 필수 요구사항은 다음과 같다.

- 점주는 과일을 촬영해 신선도 판별 보조를 요청할 수 있어야 한다.
- DL 추론은 이미지 1장 기준 5초 이내 반환되어야 한다.
- 결과는 MySQL에 저장되어야 한다.
- DL 결과와 점주 최종 선별 결과는 분리 저장해야 한다.
- 카메라 실패 시 갤러리 이미지 업로드 fallback이 있어야 한다.
- 데이터가 부족해도 팀 자체 촬영 이미지 또는 rule 기반 fallback으로 데모가 가능해야 한다.

## 3. 입력과 출력

입력:

- image: 이미지 1장
- fruit_type: 과일 종류, MVP는 apple 우선
- product_id: optional
- order_id: optional

출력:

- predicted_grade: A/B/C
- freshness_score: 신선도 점수
- color_score: 색 선명도
- roundness_score: 둥근 정도
- bruise_probability: 멍 가능성
- shipping_decision: PASS/REVIEW/HOLD
- model_confidence: 모델 참고도
- image_url: 저장 이미지 경로

## 4. 모델 전략

2주 MVP에서는 복잡한 모델보다 전이학습 기반 이미지 분류와 OpenCV 보조 특징을 결합하는 방식이 가장 현실적이다.

권장 구성:

- PyTorch CNN: A/B/C 품질 등급 분류
- OpenCV HSV: 색 선명도 계산
- OpenCV contour: 둥근 정도 계산
- 어두운 영역 분석: 멍 의심 확률 계산
- scoring rule: 최종 신선도 점수 및 출고 보조 판단

추천 모델 우선순위:

1. ResNet18: 구현이 가장 쉽고 데모 안정성이 좋음
2. MobileNetV3: 가볍고 빠름
3. EfficientNet-B0: 성능과 속도 균형

MVP에서는 ResNet18로 시작하는 것이 가장 안전하다.

현재 프로젝트의 기본 학습 구성은 다음으로 고정한다.

- 모델: ResNet18 전이학습
- 입력 크기: 224x224
- 클래스: A/B/C = 특/상/보통
- 데이터 분할: 객체 단위 train 70% / valid 10% / test 20%
- batch size: 64
- epoch: 15
- early stopping patience: 5
- optimizer: Adam 또는 AdamW
- learning rate: 1e-4 ~ 3e-4
- weight decay: 1e-4

Kaggle T4 환경에서는 먼저 batch size 64로 시작하고, GPU 메모리가 충분하면 128을 시도한다.

## 5. 점수 산정 기준

문서 기준 신선도 점수는 아래 구조다.

```text
freshness_score =
  CNN 등급 점수 * 0.50
+ 색 선명도 * 0.25
+ 둥근 정도 * 0.15
+ 멍 없음 점수 * 0.10
```

출고 보조 판단:

- PASS: score >= 80 and bruise_probability < 0.2
- REVIEW: score >= 60
- HOLD: score < 60 or bruise_probability >= 0.5

등급 점수 예시:

- A: 90
- B: 70
- C: 45

## 6. 데이터 준비

우선순위:

1. 팀 자체 촬영 이미지
2. AI-Hub 농산물 QC 이미지
3. 장수 사과 데이터
4. 고품질 과수작물 통합 데이터

MVP 자체 촬영 또는 QC 데이터 활용 기준:

- 품목: 사과 우선
- 수량: 최소 100장, 목표 300장
- 라벨: A/B/C
- QC 데이터 매핑: A=특, B=상, C=보통
- 배경: 단색
- 조명: 일정한 조명
- 각도: 정면, 측면, 상단

전처리:

- 손상 이미지 제거
- 품목 1개 우선 선택
- A/B/C 라벨 정리
- 224x224 resize
- train/valid/test = 7:1:2

현재 `Data/Sample`의 QC 사과 데이터는 설명서 기준 품질 등급 데이터이므로 다음처럼 사용한다.

- `apple_fuji_L` -> A -> 특
- `apple_fuji_M` -> B -> 상
- `apple_fuji_S` -> C -> 보통

프로젝트 화면에서는 신선도 판별이라고 표현하되, 발표/문서에서는 "QC 품질 등급을 신선도 판단의 대리 라벨로 사용"한다고 설명한다.

권장 폴더:

```text
ai/dl/freshness/
  dataset.py
  train.py
  infer.py
  opencv_features.py
  data/
    train/A
    train/B
    train/C
    valid/A
    valid/B
    valid/C
    test/A
    test/B
    test/C
```

## 7. FastAPI 연동

API:

```text
POST /api/v1/dl/freshness-scan
Content-Type: multipart/form-data
```

처리 순서:

1. multipart 이미지 수신
2. UUID 기반 파일 저장
3. PyTorch 모델 추론
4. OpenCV 특징 계산
5. 점수와 출고 판단 계산
6. quality_scans 테이블 저장
7. JSON 반환

응답 예시:

```json
{
  "success": true,
  "data": {
    "scan_id": 501,
    "fruit_type": "apple",
    "predicted_grade": "A",
    "freshness_score": 91.3,
    "color_score": 88.2,
    "roundness_score": 94.5,
    "bruise_probability": 0.07,
    "shipping_decision": "PASS",
    "model_confidence": 0.86,
    "image_url": "/static/scans/scan_501.jpg"
  },
  "message": "출고 가능한 품질로 판별되었습니다. 최종 출고 여부는 점주가 확정합니다."
}
```

## 8. DB 저장 항목

quality_scans 핵심 필드:

- image_url
- predicted_grade
- freshness_score
- color_score
- roundness_score
- bruise_probability
- shipping_decision
- owner_confirmed_grade
- owner_confirmed_decision

분리 원칙:

- predicted_grade, freshness_score, shipping_decision은 DL 모델의 보조 결과
- owner_confirmed_grade, owner_confirmed_decision은 점주의 최종 확정 결과

## 9. 테스트 기준

모델 평가 목표:

- Accuracy >= 70%
- F1-score >= 0.7
- 추론 시간 <= 5초
- 발표용 테스트 이미지 100% 성공

E2E 테스트:

1. 점주 주문 목록 조회
2. 주문 상태를 HARVEST_READY로 변경
3. 카메라 촬영 또는 이미지 업로드
4. DL 판별 요청
5. 등급과 점수 표시
6. 점주가 출고 확정
7. 상태를 QUALITY_CHECKED 또는 SHIPPED로 변경

발표 전 체크:

- DL API 성공
- 카메라 권한, 촬영, 결과 표시 확인
- 카메라 실패 시 이미지 업로드 확인
- 사전 테스트 이미지로 데모 가능 여부 확인

## 10. 2주 MVP 작업 순서

Day 1-2:

- DL 폴더 구조 생성
- 데이터 라벨 규칙 확정
- 자체 촬영 가이드 공유

Day 3-4:

- 샘플 이미지 수집
- train/valid/test 폴더 정리
- OpenCV 특징 계산 함수 구현

Day 5-6:

- ResNet18 baseline 학습
- infer.py 구현
- 테스트 이미지에서 A/B/C 결과 확인

Day 7:

- FastAPI DL API 1차 연결
- 모델 로딩과 multipart 업로드 테스트

Day 8-9:

- 점주 앱 카메라 업로드 연동 지원
- quality_scans 저장
- 결과 JSON 스펙 고정

Day 10:

- 카메라 -> DL -> 결과 표시 성공
- 실패 시 fallback 이미지 업로드 확인

Day 11-12:

- Swagger 정리
- QA 시나리오 테스트
- 발표용 테스트 이미지 세트 고정

Day 13-14:

- 기능 추가 중단
- 버그 수정과 발표 리허설만 진행

## 11. 우선 구현해야 할 최소 산출물

반드시 있어야 하는 것:

- `ai/dl/freshness/dataset.py`
- `ai/dl/freshness/train.py`
- `ai/dl/freshness/infer.py`
- `ai/dl/freshness/opencv_features.py`
- 학습 또는 데모용 모델 파일
- 테스트 이미지 세트
- `/api/v1/dl/freshness-scan` API
- Swagger에서 업로드 테스트 가능
- 결과를 저장하는 quality_scans 구조

성능이 부족할 때의 현실적인 대안:

- CNN 결과 비중을 낮추고 OpenCV rule 비중을 높인다.
- A/B/C를 엄밀한 품질 판정이 아니라 MVP용 품질 보조 등급으로 설명한다.
- 발표는 실시간 촬영 대신 검증된 테스트 이미지로 안정적으로 진행한다.

## 12. 발표 멘트 핵심

사용할 표현:

- "수확 당일 촬영한 과일 이미지를 PyTorch 모델이 분석해 품질 등급과 신선도 점수를 제공합니다."
- "DL 결과는 출고를 자동 결정하지 않고, 점주의 선별 판단을 돕는 보조 지표입니다."
- "최종 등급과 출고 여부는 점주가 확정합니다."

피해야 할 표현:

- "AI가 출고 가능 여부를 자동 결정합니다."
- "모델이 품질을 정확히 보장합니다."
- "PASS면 무조건 출고됩니다."
