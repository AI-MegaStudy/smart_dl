[angle/test] balanced dataset 기준 컬럼: ['angle_label', 'grade_label', 'variety']
[angle/test] 원본 샘플 수: 4765 balanced 샘플 수: 486 조합별 target: 27
[angle/test] angle_label 원본: {'middle': 213, 'side': 213, 'top': 4339}
[angle/test] angle_label balanced: {'middle': 162, 'side': 162, 'top': 162}
[angle/test] grade_label 원본: {'A': 1466, 'B': 1628, 'C': 1671}
[angle/test] grade_label balanced: {'A': 162, 'B': 162, 'C': 162}
[angle/test] variety 원본: {'부사': 2160, '양광': 2605}
[angle/test] variety balanced: {'부사': 243, '양광': 243}

= top/middle/side 확인 모델 balanced test 평가
accuracy: 0.9177
macro_f1: 0.9162
        top  middle  side
top     132      14    16
middle    6     156     0
side      4       0   158
              precision    recall  f1-score   support

         top       0.93      0.81      0.87       162
      middle       0.92      0.96      0.94       162
        side       0.91      0.98      0.94       162

    accuracy                           0.92       486
   macro avg       0.92      0.92      0.92       486
weighted avg       0.92      0.92      0.92       486

[top_grade/test] balanced dataset 기준 컬럼: ['grade_label', 'variety']
[top_grade/test] 원본 샘플 수: 4339 balanced 샘플 수: 3792 조합별 target: 632
[top_grade/test] grade_label 원본: {'A': 1354, 'B': 1456, 'C': 1529}
[top_grade/test] grade_label balanced: {'A': 1264, 'B': 1264, 'C': 1264}
[top_grade/test] variety 원본: {'부사': 1980, '양광': 2359}
[top_grade/test] variety balanced: {'부사': 1896, '양광': 1896}

= top 등급 모델 balanced test 평가
accuracy: 0.8605
macro_f1: 0.8612
      A     B     C
A  1107    90    67
B    64  1071   129
C    19   160  1085
              precision    recall  f1-score   support

           A       0.93      0.88      0.90      1264
           B       0.81      0.85      0.83      1264
           C       0.85      0.86      0.85      1264

    accuracy                           0.86      3792
   macro avg       0.86      0.86      0.86      3792
weighted avg       0.86      0.86      0.86      3792

top 등급 모델 품종별 성능
variety	count	accuracy	macro_f1
0	부사	1896	0.872890	0.874103
1	양광	1896	0.848101	0.848084
[middle_grade/test] balanced dataset 기준 컬럼: ['grade_label', 'variety']
[middle_grade/test] 원본 샘플 수: 213 balanced 샘플 수: 162 조합별 target: 27
[middle_grade/test] grade_label 원본: {'A': 56, 'B': 86, 'C': 71}
[middle_grade/test] grade_label balanced: {'A': 54, 'B': 54, 'C': 54}
[middle_grade/test] variety 원본: {'부사': 90, '양광': 123}
[middle_grade/test] variety balanced: {'부사': 81, '양광': 81}

= middle 등급 모델 balanced test 평가
accuracy: 0.9074
macro_f1: 0.9065
    A   B   C
A  43   2   9
B   2  50   2
C   0   0  54
              precision    recall  f1-score   support

           A       0.96      0.80      0.87        54
           B       0.96      0.93      0.94        54
           C       0.83      1.00      0.91        54

    accuracy                           0.91       162
   macro avg       0.92      0.91      0.91       162
weighted avg       0.92      0.91      0.91       162

middle 등급 모델 세부 각도별 성능
angle_detail_label	count	accuracy	macro_f1
0	diagonal45	123	0.894309	0.896117
1	front45	39	0.948718	0.936464
middle 등급 모델 품종별 성능
variety	count	accuracy	macro_f1
0	부사	81	0.851852	0.848471
1	양광	81	0.962963	0.962938
[side_grade/test] balanced dataset 기준 컬럼: ['grade_label', 'variety']
[side_grade/test] 원본 샘플 수: 213 balanced 샘플 수: 162 조합별 target: 27
[side_grade/test] grade_label 원본: {'A': 56, 'B': 86, 'C': 71}
[side_grade/test] grade_label balanced: {'A': 54, 'B': 54, 'C': 54}
[side_grade/test] variety 원본: {'부사': 90, '양광': 123}
[side_grade/test] variety balanced: {'부사': 81, '양광': 81}

= side 등급 모델 balanced test 평가
accuracy: 0.9136
macro_f1: 0.9136
    A   B   C
A  53   1   0
B   5  49   0
C   1   7  46
              precision    recall  f1-score   support

           A       0.90      0.98      0.94        54
           B       0.86      0.91      0.88        54
           C       1.00      0.85      0.92        54

    accuracy                           0.91       162
   macro avg       0.92      0.91      0.91       162
weighted avg       0.92      0.91      0.91       162

side 등급 모델 세부 각도별 성능
angle_detail_label	count	accuracy	macro_f1
0	diagonal90	123	0.902439	0.898801
1	front90	39	0.948718	0.943355
side 등급 모델 품종별 성능
variety	count	accuracy	macro_f1
0	부사	81	0.901235	0.901669
1	양광	81	0.925926	0.925714
