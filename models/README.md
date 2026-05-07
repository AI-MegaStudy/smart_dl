# Models

현재 기준으로 사용하는 모델 세트는 아래 폴더입니다.

```text
models/apple_view_balanced/
```

포함 파일:

```text
apple_view_top_middle_side_balanced_resnet18_best.pt
apple_top_grade_resnet18_best.pt
apple_middle_grade_resnet18_best.pt
apple_side_grade_resnet18_best.pt
report.md
```

역할:

- `apple_view_top_middle_side_balanced_resnet18_best.pt`: 입력 이미지가 `top`, `middle`, `side` 중 어디에 가까운지 판단합니다.
- `apple_top_grade_resnet18_best.pt`: top 이미지의 A/B/C 등급을 예측합니다.
- `apple_middle_grade_resnet18_best.pt`: middle 이미지의 A/B/C 등급을 예측합니다.
- `apple_side_grade_resnet18_best.pt`: side 이미지의 A/B/C 등급을 예측합니다.

현재 앱/서빙 노트는 `apple_view_balanced` 모델 세트를 기준으로 사용합니다.
