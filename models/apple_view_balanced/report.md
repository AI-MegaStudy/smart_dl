
= top/middle/side 확인 모델 test 평가
accuracy: 0.8919
macro_f1: 0.7095
         top  middle  side
top     3871     253   215
middle    15     198     0
side      32       0   181
              precision    recall  f1-score   support

         top       0.99      0.89      0.94      4339
      middle       0.44      0.93      0.60       213
        side       0.46      0.85      0.59       213

    accuracy                           0.89      4765
   macro avg       0.63      0.89      0.71      4765
weighted avg       0.94      0.89      0.91      4765


= top 등급 모델 test 평가
accuracy: 0.8566
macro_f1: 0.8574
      A     B     C
A  1156    55   143
B    68  1119   269
C    22    65  1442
              precision    recall  f1-score   support

           A       0.93      0.85      0.89      1354
           B       0.90      0.77      0.83      1456
           C       0.78      0.94      0.85      1529

    accuracy                           0.86      4339
   macro avg       0.87      0.86      0.86      4339
weighted avg       0.87      0.86      0.86      4339

top 등급 모델 품종별 성능
variety	count	accuracy	macro_f1
1	양광	2359	0.828741	0.827163
0	부사	1980	0.889899	0.889360

= middle 등급 모델 test 평가
accuracy: 0.9437
macro_f1: 0.9435
    A   B   C
A  54   2   0
B   3  83   0
C   2   5  64
              precision    recall  f1-score   support

           A       0.92      0.96      0.94        56
           B       0.92      0.97      0.94        86
           C       1.00      0.90      0.95        71

    accuracy                           0.94       213
   macro avg       0.95      0.94      0.94       213
weighted avg       0.95      0.94      0.94       213

middle 등급 모델 세부 각도별 성능
angle_detail_label	count	accuracy	macro_f1
0	diagonal45	163	0.95092	0.950521
1	front45	50	0.92000	0.917404
middle 등급 모델 품종별 성능
variety	count	accuracy	macro_f1
1	양광	123	0.943089	0.945125
0	부사	90	0.944444	0.942887

= side 등급 모델 test 평가
accuracy: 0.939
macro_f1: 0.9375
    A   B   C
A  56   0   0
B   7  79   0
C   4   2  65
              precision    recall  f1-score   support

           A       0.84      1.00      0.91        56
           B       0.98      0.92      0.95        86
           C       1.00      0.92      0.96        71

    accuracy                           0.94       213
   macro avg       0.94      0.94      0.94       213
weighted avg       0.95      0.94      0.94       213

side 등급 모델 세부 각도별 성능
angle_detail_label	count	accuracy	macro_f1
0	diagonal90	163	0.932515	0.932242
1	front90	50	0.960000	0.956142
side 등급 모델 품종별 성능
variety	count	accuracy	macro_f1
1	양광	123	0.975610	0.976492
0	부사	90	0.888889	0.889374
