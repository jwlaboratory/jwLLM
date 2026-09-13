import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np


text_data = """

Iteration 0 Loss: 277.778137 X: 4 Actual Y: 17.000000 Predicted Y: 0.333323
Iteration 1 Loss: 512.738892 X: 6 Actual Y: 23.000000 Predicted Y: 0.356262
Iteration 2 Loss: 267.754791 X: 4 Actual Y: 17.000000 Predicted Y: 0.636786
Iteration 3 Loss: 636.470276 X: 7 Actual Y: 26.000000 Predicted Y: 0.771638
Iteration 4 Loss: 14.528536 X: 0 Actual Y: 5.000000 Predicted Y: 1.188368
Iteration 5 Loss: 173.664917 X: 3 Actual Y: 14.000000 Predicted Y: 0.821801
Iteration 6 Loss: 620.955261 X: 7 Actual Y: 26.000000 Predicted Y: 1.081027
Iteration 7 Loss: 785.827637 X: 8 Actual Y: 29.000000 Predicted Y: 0.967382
Iteration 8 Loss: 953.803223 X: 9 Actual Y: 32.000000 Predicted Y: 1.116296
Iteration 9 Loss: 459.475952 X: 6 Actual Y: 23.000000 Predicted Y: 1.564609
Iteration 10 Loss: 448.764618 X: 6 Actual Y: 23.000000 Predicted Y: 1.815935
Iteration 11 Loss: 215.744064 X: 4 Actual Y: 17.000000 Predicted Y: 2.311771
Iteration 12 Loss: 907.117798 X: 9 Actual Y: 32.000000 Predicted Y: 1.881603
Iteration 13 Loss: 29.634584 X: 1 Actual Y: 8.000000 Predicted Y: 2.556234
Iteration 14 Loss: 128.641251 X: 3 Actual Y: 14.000000 Predicted Y: 2.657988
Iteration 15 Loss: 701.870056 X: 8 Actual Y: 29.000000 Predicted Y: 2.507170
Iteration 16 Loss: 678.465820 X: 8 Actual Y: 29.000000 Predicted Y: 2.952623
Iteration 17 Loss: 19.167963 X: 1 Actual Y: 8.000000 Predicted Y: 3.621877
Iteration 18 Loss: 165.419189 X: 4 Actual Y: 17.000000 Predicted Y: 4.138461
Iteration 19 Loss: 15.910165 X: 1 Actual Y: 8.000000 Predicted Y: 4.011245
Iteration 20 Loss: 97.162148 X: 3 Actual Y: 14.000000 Predicted Y: 4.142913
Iteration 21 Loss: 468.771881 X: 7 Actual Y: 26.000000 Predicted Y: 4.348859
Iteration 22 Loss: 596.807861 X: 8 Actual Y: 29.000000 Predicted Y: 4.570349
Iteration 23 Loss: 558.062378 X: 8 Actual Y: 29.000000 Predicted Y: 5.376657
Iteration 24 Loss: 24.962135 X: 2 Actual Y: 11.000000 Predicted Y: 6.003788
Iteration 25 Loss: 100.658585 X: 4 Actual Y: 17.000000 Predicted Y: 6.967125
Iteration 26 Loss: 366.745789 X: 7 Actual Y: 26.000000 Predicted Y: 6.849391
Iteration 27 Loss: 459.751984 X: 8 Actual Y: 29.000000 Predicted Y: 7.558173
Iteration 28 Loss: 295.764709 X: 7 Actual Y: 26.000000 Predicted Y: 8.802189
Iteration 29 Loss: 4.442103 X: 2 Actual Y: 11.000000 Predicted Y: 8.892370
Iteration 30 Loss: 367.703796 X: 8 Actual Y: 29.000000 Predicted Y: 9.824396
Iteration 31 Loss: 31.971958 X: 4 Actual Y: 17.000000 Predicted Y: 11.345625
Iteration 32 Loss: 294.162262 X: 8 Actual Y: 29.000000 Predicted Y: 11.848841
Iteration 33 Loss: 5.923056 X: 3 Actual Y: 14.000000 Predicted Y: 11.566267
Iteration 34 Loss: 4.729904 X: 3 Actual Y: 14.000000 Predicted Y: 11.825166
Iteration 35 Loss: 480.906830 X: 9 Actual Y: 32.000000 Predicted Y: 10.070413
Iteration 36 Loss: 133.033234 X: 7 Actual Y: 26.000000 Predicted Y: 14.465997
Iteration 37 Loss: 7.780051 X: 2 Actual Y: 11.000000 Predicted Y: 13.789274
Iteration 38 Loss: 62.129486 X: 5 Actual Y: 20.000000 Predicted Y: 12.117774
Iteration 39 Loss: 91.722794 X: 7 Actual Y: 26.000000 Predicted Y: 16.422798
Iteration 40 Loss: 38.670956 X: 5 Actual Y: 20.000000 Predicted Y: 13.781403
Iteration 41 Loss: 64.514626 X: 6 Actual Y: 23.000000 Predicted Y: 14.967900
Iteration 42 Loss: 46.302422 X: 6 Actual Y: 23.000000 Predicted Y: 16.195412
Iteration 43 Loss: 86.088326 X: 8 Actual Y: 29.000000 Predicted Y: 19.721621
Iteration 44 Loss: 53.428478 X: 8 Actual Y: 29.000000 Predicted Y: 21.690521
Iteration 45 Loss: 264.121277 X: 0 Actual Y: 5.000000 Predicted Y: 21.251808
Iteration 46 Loss: 16.294611 X: 5 Actual Y: 20.000000 Predicted Y: 15.963342
Iteration 47 Loss: 11.500477 X: 3 Actual Y: 14.000000 Predicted Y: 17.391235
Iteration 48 Loss: 266.192444 X: 9 Actual Y: 32.000000 Predicted Y: 15.684595
Iteration 49 Loss: 16.451227 X: 6 Actual Y: 23.000000 Predicted Y: 18.943989
Iteration 50 Loss: 20.673431 X: 3 Actual Y: 14.000000 Predicted Y: 18.546804
Iteration 51 Loss: 19.628029 X: 7 Actual Y: 26.000000 Predicted Y: 21.569647
Iteration 52 Loss: 16.625751 X: 4 Actual Y: 17.000000 Predicted Y: 21.077469
Iteration 53 Loss: 13.340437 X: 6 Actual Y: 23.000000 Predicted Y: 19.347544
Iteration 54 Loss: 16.183338 X: 3 Actual Y: 14.000000 Predicted Y: 18.022852
Iteration 55 Loss: 10.429755 X: 3 Actual Y: 14.000000 Predicted Y: 17.229513
Iteration 56 Loss: 18.215902 X: 7 Actual Y: 26.000000 Predicted Y: 21.731991
Iteration 57 Loss: 50.480694 X: 2 Actual Y: 11.000000 Predicted Y: 18.104977
Iteration 58 Loss: 194.629349 X: 9 Actual Y: 32.000000 Predicted Y: 18.049038
Iteration 59 Loss: 111.123543 X: 1 Actual Y: 8.000000 Predicted Y: 18.541515
Iteration 60 Loss: 149.378174 X: 9 Actual Y: 32.000000 Predicted Y: 19.777964
Iteration 61 Loss: 1.708646 X: 5 Actual Y: 20.000000 Predicted Y: 18.692848
Iteration 62 Loss: 16.113007 X: 3 Actual Y: 14.000000 Predicted Y: 18.014101
Iteration 63 Loss: 50.320316 X: 2 Actual Y: 11.000000 Predicted Y: 18.093681
Iteration 64 Loss: 10.017521 X: 7 Actual Y: 26.000000 Predicted Y: 22.834953
Iteration 65 Loss: 8.461684 X: 3 Actual Y: 14.000000 Predicted Y: 16.908897
Iteration 66 Loss: 9.182130 X: 6 Actual Y: 23.000000 Predicted Y: 19.969797
Iteration 67 Loss: 101.935081 X: 9 Actual Y: 32.000000 Predicted Y: 21.903709
Iteration 68 Loss: 57.461864 X: 9 Actual Y: 32.000000 Predicted Y: 24.419640
Iteration 69 Loss: 121.311638 X: 1 Actual Y: 8.000000 Predicted Y: 19.014156
Iteration 70 Loss: 1.509658 X: 6 Actual Y: 23.000000 Predicted Y: 21.771318
Iteration 71 Loss: 0.284238 X: 5 Actual Y: 20.000000 Predicted Y: 19.466860
Iteration 72 Loss: 13.703347 X: 3 Actual Y: 14.000000 Predicted Y: 17.701803
Iteration 73 Loss: 48.281967 X: 9 Actual Y: 32.000000 Predicted Y: 25.051477
Iteration 74 Loss: 13.836295 X: 3 Actual Y: 14.000000 Predicted Y: 17.719717
Iteration 75 Loss: 0.390113 X: 6 Actual Y: 23.000000 Predicted Y: 22.375410
Iteration 76 Loss: 0.019779 X: 5 Actual Y: 20.000000 Predicted Y: 19.859362
Iteration 77 Loss: 0.066706 X: 7 Actual Y: 26.000000 Predicted Y: 25.741726
Iteration 78 Loss: 30.619207 X: 4 Actual Y: 17.000000 Predicted Y: 22.533463
Iteration 79 Loss: 70.634453 X: 1 Actual Y: 8.000000 Predicted Y: 16.404430
Iteration 80 Loss: 39.661659 X: 2 Actual Y: 11.000000 Predicted Y: 17.297750
Iteration 81 Loss: 211.513580 X: 0 Actual Y: 5.000000 Predicted Y: 19.543507
Iteration 82 Loss: 1.377125 X: 4 Actual Y: 17.000000 Predicted Y: 18.173510
Iteration 83 Loss: 25.693380 X: 1 Actual Y: 8.000000 Predicted Y: 13.068864
Iteration 84 Loss: 103.492188 X: 9 Actual Y: 32.000000 Predicted Y: 21.826889
Iteration 85 Loss: 12.631315 X: 7 Actual Y: 26.000000 Predicted Y: 22.445944
Iteration 86 Loss: 7.059353 X: 7 Actual Y: 26.000000 Predicted Y: 23.343056
Iteration 87 Loss: 30.591583 X: 8 Actual Y: 29.000000 Predicted Y: 23.469034
Iteration 88 Loss: 2.996754 X: 6 Actual Y: 23.000000 Predicted Y: 21.268887
Iteration 89 Loss: 0.677674 X: 7 Actual Y: 26.000000 Predicted Y: 25.176790
Iteration 90 Loss: 41.282440 X: 1 Actual Y: 8.000000 Predicted Y: 14.425141
Iteration 91 Loss: 3.051915 X: 6 Actual Y: 23.000000 Predicted Y: 21.253027
Iteration 92 Loss: 0.896445 X: 7 Actual Y: 26.000000 Predicted Y: 25.053192
Iteration 93 Loss: 0.450115 X: 7 Actual Y: 26.000000 Predicted Y: 25.329094
Iteration 94 Loss: 1.265067 X: 5 Actual Y: 20.000000 Predicted Y: 18.875248
Iteration 95 Loss: 4.013943 X: 3 Actual Y: 14.000000 Predicted Y: 16.003483
Iteration 96 Loss: 2.558463 X: 3 Actual Y: 14.000000 Predicted Y: 15.599520
Iteration 97 Loss: 31.962576 X: 2 Actual Y: 11.000000 Predicted Y: 16.653545
Iteration 98 Loss: 24.792000 X: 1 Actual Y: 8.000000 Predicted Y: 12.979156
Iteration 99 Loss: 5.109377 X: 6 Actual Y: 23.000000 Predicted Y: 20.739607
Iteration 100 Loss: 3.140184 X: 5 Actual Y: 20.000000 Predicted Y: 18.227943
Iteration 101 Loss: 0.783864 X: 3 Actual Y: 14.000000 Predicted Y: 14.885361
Iteration 102 Loss: 7.838084 X: 4 Actual Y: 17.000000 Predicted Y: 19.799658
Iteration 103 Loss: 24.695604 X: 8 Actual Y: 29.000000 Predicted Y: 24.030533
Iteration 104 Loss: 173.779190 X: 0 Actual Y: 5.000000 Predicted Y: 18.182533
Iteration 105 Loss: 2.315867 X: 4 Actual Y: 17.000000 Predicted Y: 18.521797
Iteration 106 Loss: 5.879807 X: 5 Actual Y: 20.000000 Predicted Y: 17.575169
Iteration 107 Loss: 6.016241 X: 7 Actual Y: 26.000000 Predicted Y: 23.547197
Iteration 108 Loss: 15.422303 X: 1 Actual Y: 8.000000 Predicted Y: 11.927124
Iteration 109 Loss: 0.000038 X: 3 Actual Y: 14.000000 Predicted Y: 13.993797
Iteration 110 Loss: 0.000026 X: 3 Actual Y: 14.000000 Predicted Y: 13.994914
Iteration 111 Loss: 0.000017 X: 3 Actual Y: 14.000000 Predicted Y: 13.995829
Iteration 112 Loss: 3.618622 X: 5 Actual Y: 20.000000 Predicted Y: 18.097733
Iteration 113 Loss: 3.474106 X: 7 Actual Y: 26.000000 Predicted Y: 24.136105
Iteration 114 Loss: 3.502048 X: 6 Actual Y: 23.000000 Predicted Y: 21.128624
Iteration 115 Loss: 1.085605 X: 5 Actual Y: 20.000000 Predicted Y: 18.958076
Iteration 116 Loss: 15.900363 X: 8 Actual Y: 29.000000 Predicted Y: 25.012474
Iteration 117 Loss: 0.051515 X: 7 Actual Y: 26.000000 Predicted Y: 25.773031
Iteration 118 Loss: 16.776611 X: 1 Actual Y: 8.000000 Predicted Y: 12.095926
Iteration 119 Loss: 11.179089 X: 1 Actual Y: 8.000000 Predicted Y: 11.343514
Iteration 120 Loss: 1.397771 X: 6 Actual Y: 23.000000 Predicted Y: 21.817726
Iteration 121 Loss: 34.953106 X: 9 Actual Y: 32.000000 Predicted Y: 26.087885
Iteration 122 Loss: 10.754101 X: 4 Actual Y: 17.000000 Predicted Y: 20.279345
Iteration 123 Loss: 0.001489 X: 7 Actual Y: 26.000000 Predicted Y: 26.038593
Iteration 124 Loss: 20.238226 X: 9 Actual Y: 32.000000 Predicted Y: 27.501308
Iteration 125 Loss: 0.258352 X: 5 Actual Y: 20.000000 Predicted Y: 20.508284
Iteration 126 Loss: 0.679299 X: 7 Actual Y: 26.000000 Predicted Y: 26.824196
Iteration 127 Loss: 0.055478 X: 6 Actual Y: 23.000000 Predicted Y: 23.235538
Iteration 128 Loss: 1.899700 X: 3 Actual Y: 14.000000 Predicted Y: 15.378296
Iteration 129 Loss: 0.001026 X: 6 Actual Y: 23.000000 Predicted Y: 23.032034
Iteration 130 Loss: 136.389221 X: 0 Actual Y: 5.000000 Predicted Y: 16.678579
Iteration 131 Loss: 0.722397 X: 5 Actual Y: 20.000000 Predicted Y: 19.150061
Iteration 132 Loss: 82.109131 X: 0 Actual Y: 5.000000 Predicted Y: 14.061409
Iteration 133 Loss: 14.396915 X: 2 Actual Y: 11.000000 Predicted Y: 14.794327
Iteration 134 Loss: 36.447979 X: 9 Actual Y: 32.000000 Predicted Y: 25.962784
Iteration 135 Loss: 0.004988 X: 3 Actual Y: 14.000000 Predicted Y: 14.070622
Iteration 136 Loss: 11.326215 X: 8 Actual Y: 29.000000 Predicted Y: 25.634556
Iteration 137 Loss: 6.555126 X: 1 Actual Y: 8.000000 Predicted Y: 10.560298
Iteration 138 Loss: 58.131275 X: 0 Actual Y: 5.000000 Predicted Y: 12.624387
Iteration 139 Loss: 2.842116 X: 1 Actual Y: 8.000000 Predicted Y: 9.685858
Iteration 140 Loss: 1.605323 X: 4 Actual Y: 17.000000 Predicted Y: 18.267014
Iteration 141 Loss: 33.996105 X: 0 Actual Y: 5.000000 Predicted Y: 10.830618
Iteration 142 Loss: 25.543503 X: 9 Actual Y: 32.000000 Predicted Y: 26.945942
Iteration 143 Loss: 7.949365 X: 8 Actual Y: 29.000000 Predicted Y: 26.180538
Iteration 144 Loss: 7.100654 X: 9 Actual Y: 32.000000 Predicted Y: 29.335295
Iteration 145 Loss: 2.616974 X: 9 Actual Y: 32.000000 Predicted Y: 30.382294
Iteration 146 Loss: 0.047478 X: 7 Actual Y: 26.000000 Predicted Y: 26.217896
Iteration 147 Loss: 3.516905 X: 1 Actual Y: 8.000000 Predicted Y: 9.875341
Iteration 148 Loss: 0.041781 X: 5 Actual Y: 20.000000 Predicted Y: 20.204405
Iteration 149 Loss: 4.810206 X: 4 Actual Y: 17.000000 Predicted Y: 19.193218
Iteration 150 Loss: 0.008220 X: 6 Actual Y: 23.000000 Predicted Y: 22.909336
Iteration 151 Loss: 1.960196 X: 8 Actual Y: 29.000000 Predicted Y: 27.599930
Iteration 152 Loss: 0.000994 X: 7 Actual Y: 26.000000 Predicted Y: 25.968468
Iteration 153 Loss: 0.771414 X: 8 Actual Y: 29.000000 Predicted Y: 28.121698
Iteration 154 Loss: 29.774506 X: 0 Actual Y: 5.000000 Predicted Y: 10.456602
Iteration 155 Loss: 1.956033 X: 9 Actual Y: 32.000000 Predicted Y: 30.601418
Iteration 156 Loss: 0.371445 X: 8 Actual Y: 29.000000 Predicted Y: 28.390537
Iteration 157 Loss: 18.656384 X: 0 Actual Y: 5.000000 Predicted Y: 9.319304
Iteration 158 Loss: 10.872743 X: 0 Actual Y: 5.000000 Predicted Y: 8.297384
Iteration 159 Loss: 2.238736 X: 4 Actual Y: 17.000000 Predicted Y: 18.496241
Iteration 160 Loss: 0.000001 X: 3 Actual Y: 14.000000 Predicted Y: 13.998911
Iteration 161 Loss: 1.106026 X: 8 Actual Y: 29.000000 Predicted Y: 27.948322
Iteration 162 Loss: 1.570258 X: 4 Actual Y: 17.000000 Predicted Y: 18.253099
Iteration 163 Loss: 0.686208 X: 8 Actual Y: 29.000000 Predicted Y: 28.171623
Iteration 164 Loss: 6.146084 X: 0 Actual Y: 5.000000 Predicted Y: 7.479130
Iteration 165 Loss: 0.283040 X: 7 Actual Y: 26.000000 Predicted Y: 25.467985
Iteration 166 Loss: 0.122197 X: 7 Actual Y: 26.000000 Predicted Y: 25.650433
Iteration 167 Loss: 0.006556 X: 3 Actual Y: 14.000000 Predicted Y: 14.080967
Iteration 168 Loss: 0.056108 X: 7 Actual Y: 26.000000 Predicted Y: 25.763128
Iteration 169 Loss: 0.000821 X: 5 Actual Y: 20.000000 Predicted Y: 20.028654
Iteration 170 Loss: 0.182725 X: 8 Actual Y: 29.000000 Predicted Y: 28.572536
Iteration 171 Loss: 0.007576 X: 5 Actual Y: 20.000000 Predicted Y: 20.087040
Iteration 172 Loss: 15.032740 X: 2 Actual Y: 11.000000 Predicted Y: 14.877208
Iteration 173 Loss: 0.246217 X: 7 Actual Y: 26.000000 Predicted Y: 25.503798
Iteration 174 Loss: 0.008462 X: 6 Actual Y: 23.000000 Predicted Y: 22.908012
Iteration 175 Loss: 0.005855 X: 3 Actual Y: 14.000000 Predicted Y: 13.923480
Iteration 176 Loss: 0.003125 X: 6 Actual Y: 23.000000 Predicted Y: 22.944098
Iteration 177 Loss: 0.777063 X: 4 Actual Y: 17.000000 Predicted Y: 17.881512
Iteration 178 Loss: 0.166695 X: 7 Actual Y: 26.000000 Predicted Y: 25.591717
Iteration 179 Loss: 0.039813 X: 5 Actual Y: 20.000000 Predicted Y: 19.800468
Iteration 180 Loss: 0.002415 X: 6 Actual Y: 23.000000 Predicted Y: 22.950855
Iteration 181 Loss: 0.336553 X: 8 Actual Y: 29.000000 Predicted Y: 28.419868
Iteration 182 Loss: 0.005106 X: 6 Actual Y: 23.000000 Predicted Y: 23.071453
Iteration 183 Loss: 0.016480 X: 7 Actual Y: 26.000000 Predicted Y: 25.871624
Iteration 184 Loss: 1.339977 X: 9 Actual Y: 32.000000 Predicted Y: 30.842426
Iteration 185 Loss: 0.027791 X: 7 Actual Y: 26.000000 Predicted Y: 26.166706
Iteration 186 Loss: 3.655500 X: 0 Actual Y: 5.000000 Predicted Y: 6.911936
Iteration 187 Loss: 0.000002 X: 7 Actual Y: 26.000000 Predicted Y: 25.998690
Iteration 188 Loss: 0.040318 X: 6 Actual Y: 23.000000 Predicted Y: 23.200792
Iteration 189 Loss: 0.001265 X: 5 Actual Y: 20.000000 Predicted Y: 20.035561
Iteration 190 Loss: 2.079885 X: 0 Actual Y: 5.000000 Predicted Y: 6.442181
Iteration 191 Loss: 0.013472 X: 7 Actual Y: 26.000000 Predicted Y: 25.883932
Iteration 192 Loss: 0.776572 X: 9 Actual Y: 32.000000 Predicted Y: 31.118767
Iteration 193 Loss: 0.791556 X: 1 Actual Y: 8.000000 Predicted Y: 8.889694
Iteration 194 Loss: 0.310085 X: 9 Actual Y: 32.000000 Predicted Y: 31.443148
Iteration 195 Loss: 1.272206 X: 0 Actual Y: 5.000000 Predicted Y: 6.127921
Iteration 196 Loss: 9.845598 X: 2 Actual Y: 11.000000 Predicted Y: 14.137770
Iteration 197 Loss: 0.010770 X: 5 Actual Y: 20.000000 Predicted Y: 19.896223
Iteration 198 Loss: 0.000454 X: 6 Actual Y: 23.000000 Predicted Y: 23.021303
Iteration 199 Loss: 0.000209 X: 6 Actual Y: 23.000000 Predicted Y: 23.014442
Iteration 200 Loss: 0.006195 X: 5 Actual Y: 20.000000 Predicted Y: 19.921291
Iteration 201 Loss: 0.479164 X: 4 Actual Y: 17.000000 Predicted Y: 17.692217
Iteration 202 Loss: 0.485937 X: 0 Actual Y: 5.000000 Predicted Y: 5.697092
Iteration 203 Loss: 0.710820 X: 9 Actual Y: 32.000000 Predicted Y: 31.156898
Iteration 204 Loss: 0.013941 X: 7 Actual Y: 26.000000 Predicted Y: 25.881929
Iteration 205 Loss: 0.005775 X: 7 Actual Y: 26.000000 Predicted Y: 25.924004
Iteration 206 Loss: 0.042575 X: 8 Actual Y: 29.000000 Predicted Y: 28.793663
Iteration 207 Loss: 0.006239 X: 3 Actual Y: 14.000000 Predicted Y: 13.921011
Iteration 208 Loss: 0.133943 X: 9 Actual Y: 32.000000 Predicted Y: 31.634018
Iteration 209 Loss: 0.000469 X: 8 Actual Y: 29.000000 Predicted Y: 28.978340
Iteration 210 Loss: 0.007420 X: 7 Actual Y: 26.000000 Predicted Y: 26.086138
Iteration 211 Loss: 0.046087 X: 9 Actual Y: 32.000000 Predicted Y: 31.785320
Iteration 212 Loss: 0.023212 X: 5 Actual Y: 20.000000 Predicted Y: 20.152355
Iteration 213 Loss: 0.358662 X: 0 Actual Y: 5.000000 Predicted Y: 5.598884
Iteration 214 Loss: 0.028201 X: 9 Actual Y: 32.000000 Predicted Y: 31.832069
Iteration 215 Loss: 0.000133 X: 3 Actual Y: 14.000000 Predicted Y: 13.988452
Iteration 216 Loss: 0.209097 X: 0 Actual Y: 5.000000 Predicted Y: 5.457271
Iteration 217 Loss: 0.000758 X: 3 Actual Y: 14.000000 Predicted Y: 13.972469
Iteration 218 Loss: 0.066748 X: 6 Actual Y: 23.000000 Predicted Y: 23.258356
Iteration 219 Loss: 0.303431 X: 1 Actual Y: 8.000000 Predicted Y: 8.550846
Iteration 220 Loss: 0.001808 X: 5 Actual Y: 20.000000 Predicted Y: 20.042526
Iteration 221 Loss: 0.098516 X: 0 Actual Y: 5.000000 Predicted Y: 5.313873
Iteration 222 Loss: 0.000456 X: 5 Actual Y: 20.000000 Predicted Y: 20.021355
Iteration 223 Loss: 0.009733 X: 8 Actual Y: 29.000000 Predicted Y: 28.901342
Iteration 224 Loss: 0.037260 X: 9 Actual Y: 32.000000 Predicted Y: 31.806971
Iteration 225 Loss: 0.060546 X: 0 Actual Y: 5.000000 Predicted Y: 5.246061
Iteration 226 Loss: 0.003236 X: 5 Actual Y: 20.000000 Predicted Y: 20.056883
Iteration 227 Loss: 0.000732 X: 8 Actual Y: 29.000000 Predicted Y: 28.972952
Iteration 228 Loss: 0.013786 X: 9 Actual Y: 32.000000 Predicted Y: 31.882586
Iteration 229 Loss: 0.038026 X: 6 Actual Y: 23.000000 Predicted Y: 23.195002
Iteration 230 Loss: 0.437160 X: 4 Actual Y: 17.000000 Predicted Y: 17.661180
Iteration 231 Loss: 0.003033 X: 6 Actual Y: 23.000000 Predicted Y: 23.055073
Iteration 232 Loss: 0.023799 X: 0 Actual Y: 5.000000 Predicted Y: 5.154270
Iteration 233 Loss: 5.657970 X: 2 Actual Y: 11.000000 Predicted Y: 13.378649
Iteration 234 Loss: 0.030706 X: 6 Actual Y: 23.000000 Predicted Y: 22.824768
Iteration 235 Loss: 0.100496 X: 4 Actual Y: 17.000000 Predicted Y: 17.317011
Iteration 236 Loss: 0.051783 X: 5 Actual Y: 20.000000 Predicted Y: 19.772442
Iteration 237 Loss: 0.073664 X: 1 Actual Y: 8.000000 Predicted Y: 8.271412
Iteration 238 Loss: 0.000104 X: 0 Actual Y: 5.000000 Predicted Y: 5.010176
Iteration 239 Loss: 0.136778 X: 8 Actual Y: 29.000000 Predicted Y: 28.630165
Iteration 240 Loss: 0.153358 X: 9 Actual Y: 32.000000 Predicted Y: 31.608391
Iteration 241 Loss: 0.122923 X: 4 Actual Y: 17.000000 Predicted Y: 17.350603
Iteration 242 Loss: 0.007646 X: 5 Actual Y: 20.000000 Predicted Y: 19.912560
Iteration 243 Loss: 0.003836 X: 5 Actual Y: 20.000000 Predicted Y: 19.938066
Iteration 244 Loss: 0.000218 X: 6 Actual Y: 23.000000 Predicted Y: 22.985249
Iteration 245 Loss: 0.001755 X: 5 Actual Y: 20.000000 Predicted Y: 19.958109
Iteration 246 Loss: 0.043611 X: 3 Actual Y: 14.000000 Predicted Y: 13.791167
Iteration 247 Loss: 0.001118 X: 0 Actual Y: 5.000000 Predicted Y: 5.033442
Iteration 248 Loss: 0.000631 X: 0 Actual Y: 5.000000 Predicted Y: 5.025129
Iteration 249 Loss: 0.069290 X: 1 Actual Y: 8.000000 Predicted Y: 8.263229
Iteration 250 Loss: 0.050848 X: 9 Actual Y: 32.000000 Predicted Y: 31.774506
Iteration 251 Loss: 0.097102 X: 4 Actual Y: 17.000000 Predicted Y: 17.311611
Iteration 252 Loss: 0.025135 X: 7 Actual Y: 26.000000 Predicted Y: 25.841459
Iteration 253 Loss: 0.000001 X: 5 Actual Y: 20.000000 Predicted Y: 20.001038
Iteration 254 Loss: 0.023702 X: 3 Actual Y: 14.000000 Predicted Y: 13.846046
Iteration 255 Loss: 0.000181 X: 5 Actual Y: 20.000000 Predicted Y: 20.013447
Iteration 256 Loss: 0.002163 X: 6 Actual Y: 23.000000 Predicted Y: 23.046513
Iteration 257 Loss: 0.064449 X: 4 Actual Y: 17.000000 Predicted Y: 17.253868
Iteration 258 Loss: 3.550519 X: 2 Actual Y: 11.000000 Predicted Y: 12.884282
Iteration 259 Loss: 0.025769 X: 5 Actual Y: 20.000000 Predicted Y: 19.839472
Iteration 260 Loss: 0.004822 X: 4 Actual Y: 17.000000 Predicted Y: 17.069441
Iteration 261 Loss: 0.020559 X: 6 Actual Y: 23.000000 Predicted Y: 22.856617
Iteration 262 Loss: 0.010335 X: 5 Actual Y: 20.000000 Predicted Y: 19.898336
Iteration 263 Loss: 0.005187 X: 5 Actual Y: 20.000000 Predicted Y: 19.927982
Iteration 264 Loss: 2.165459 X: 2 Actual Y: 11.000000 Predicted Y: 12.471550
Iteration 265 Loss: 0.205807 X: 9 Actual Y: 32.000000 Predicted Y: 31.546341
Iteration 266 Loss: 0.063658 X: 3 Actual Y: 14.000000 Predicted Y: 13.747694
Iteration 267 Loss: 0.004903 X: 4 Actual Y: 17.000000 Predicted Y: 17.070021
Iteration 268 Loss: 0.002649 X: 4 Actual Y: 17.000000 Predicted Y: 17.051472
Iteration 269 Loss: 0.009145 X: 0 Actual Y: 5.000000 Predicted Y: 4.904368
Iteration 270 Loss: 0.008408 X: 6 Actual Y: 23.000000 Predicted Y: 22.908302
Iteration 271 Loss: 0.004852 X: 0 Actual Y: 5.000000 Predicted Y: 4.930341
Iteration 272 Loss: 0.002899 X: 5 Actual Y: 20.000000 Predicted Y: 19.946156
Iteration 273 Loss: 0.057633 X: 7 Actual Y: 26.000000 Predicted Y: 25.759932
Iteration 274 Loss: 0.024316 X: 3 Actual Y: 14.000000 Predicted Y: 13.844065
Iteration 275 Loss: 0.019103 X: 7 Actual Y: 26.000000 Predicted Y: 25.861788
Iteration 276 Loss: 0.000511 X: 6 Actual Y: 23.000000 Predicted Y: 23.022596
Iteration 277 Loss: 0.026124 X: 1 Actual Y: 8.000000 Predicted Y: 8.161628
Iteration 278 Loss: 0.000230 X: 5 Actual Y: 20.000000 Predicted Y: 20.015160
Iteration 279 Loss: 0.010830 X: 7 Actual Y: 26.000000 Predicted Y: 25.895933
Iteration 280 Loss: 0.008623 X: 9 Actual Y: 32.000000 Predicted Y: 31.907141
Iteration 281 Loss: 0.018020 X: 4 Actual Y: 17.000000 Predicted Y: 17.134239
Iteration 282 Loss: 0.010671 X: 3 Actual Y: 14.000000 Predicted Y: 13.896700
Iteration 283 Loss: 0.002719 X: 7 Actual Y: 26.000000 Predicted Y: 25.947857
Iteration 284 Loss: 0.012801 X: 4 Actual Y: 17.000000 Predicted Y: 17.113142
Iteration 285 Loss: 0.000956 X: 0 Actual Y: 5.000000 Predicted Y: 4.969082
Iteration 286 Loss: 0.001142 X: 5 Actual Y: 20.000000 Predicted Y: 20.033796
Iteration 287 Loss: 0.006859 X: 3 Actual Y: 14.000000 Predicted Y: 13.917179
Iteration 288 Loss: 1.563733 X: 2 Actual Y: 11.000000 Predicted Y: 12.250493
Iteration 289 Loss: 0.004512 X: 6 Actual Y: 23.000000 Predicted Y: 22.932831
Iteration 290 Loss: 0.004701 X: 0 Actual Y: 5.000000 Predicted Y: 4.931436
Iteration 291 Loss: 0.933347 X: 2 Actual Y: 11.000000 Predicted Y: 11.966099
Iteration 292 Loss: 0.008058 X: 0 Actual Y: 5.000000 Predicted Y: 4.910232
Iteration 293 Loss: 0.083407 X: 8 Actual Y: 29.000000 Predicted Y: 28.711197
Iteration 294 Loss: 0.004364 X: 5 Actual Y: 20.000000 Predicted Y: 19.933937
Iteration 295 Loss: 0.002179 X: 5 Actual Y: 20.000000 Predicted Y: 19.953318
Iteration 296 Loss: 0.023433 X: 8 Actual Y: 29.000000 Predicted Y: 28.846920
Iteration 297 Loss: 0.013983 X: 7 Actual Y: 26.000000 Predicted Y: 25.881748
Iteration 298 Loss: 0.001996 X: 0 Actual Y: 5.000000 Predicted Y: 4.955329
Iteration 299 Loss: 0.001683 X: 4 Actual Y: 17.000000 Predicted Y: 17.041027
Iteration 300 Loss: 0.000907 X: 4 Actual Y: 17.000000 Predicted Y: 17.030111
Iteration 301 Loss: 0.000004 X: 5 Actual Y: 20.000000 Predicted Y: 20.002018
Iteration 302 Loss: 0.641043 X: 2 Actual Y: 11.000000 Predicted Y: 11.800652
Iteration 303 Loss: 0.378093 X: 2 Actual Y: 11.000000 Predicted Y: 11.614893
Iteration 304 Loss: 0.009004 X: 5 Actual Y: 20.000000 Predicted Y: 19.905113
Iteration 305 Loss: 0.037213 X: 7 Actual Y: 26.000000 Predicted Y: 25.807093
Iteration 306 Loss: 0.002359 X: 1 Actual Y: 8.000000 Predicted Y: 8.048567
Iteration 307 Loss: 0.001848 X: 5 Actual Y: 20.000000 Predicted Y: 19.957006
Iteration 308 Loss: 0.246755 X: 2 Actual Y: 11.000000 Predicted Y: 11.496744
Iteration 309 Loss: 0.000371 X: 1 Actual Y: 8.000000 Predicted Y: 8.019249
Iteration 310 Loss: 0.004184 X: 5 Actual Y: 20.000000 Predicted Y: 19.935314
Iteration 311 Loss: 0.055837 X: 9 Actual Y: 32.000000 Predicted Y: 31.763702
Iteration 312 Loss: 0.016403 X: 9 Actual Y: 32.000000 Predicted Y: 31.871925
Iteration 313 Loss: 0.000293 X: 5 Actual Y: 20.000000 Predicted Y: 20.017115
Iteration 314 Loss: 0.007919 X: 0 Actual Y: 5.000000 Predicted Y: 4.911014
Iteration 315 Loss: 0.000207 X: 5 Actual Y: 20.000000 Predicted Y: 20.014376
Iteration 316 Loss: 0.000061 X: 4 Actual Y: 17.000000 Predicted Y: 16.992172
Iteration 317 Loss: 0.004462 X: 0 Actual Y: 5.000000 Predicted Y: 4.933201
Iteration 318 Loss: 0.000866 X: 6 Actual Y: 23.000000 Predicted Y: 22.970564
Iteration 319 Loss: 0.004271 X: 7 Actual Y: 26.000000 Predicted Y: 25.934645
Iteration 320 Loss: 0.016505 X: 3 Actual Y: 14.000000 Predicted Y: 13.871529
Iteration 321 Loss: 0.000818 X: 7 Actual Y: 26.000000 Predicted Y: 25.971397
Iteration 322 Loss: 0.001603 X: 0 Actual Y: 5.000000 Predicted Y: 4.959959
Iteration 323 Loss: 0.008448 X: 3 Actual Y: 14.000000 Predicted Y: 13.908085
Iteration 324 Loss: 0.202707 X: 2 Actual Y: 11.000000 Predicted Y: 11.450230
Iteration 325 Loss: 0.003145 X: 9 Actual Y: 32.000000 Predicted Y: 31.943924
Iteration 326 Loss: 0.123463 X: 2 Actual Y: 11.000000 Predicted Y: 11.351374
Iteration 327 Loss: 0.001376 X: 1 Actual Y: 8.000000 Predicted Y: 8.037100
Iteration 328 Loss: 0.000193 X: 4 Actual Y: 17.000000 Predicted Y: 16.986116
Iteration 329 Loss: 0.004008 X: 7 Actual Y: 26.000000 Predicted Y: 25.936695
Iteration 330 Loss: 0.001112 X: 1 Actual Y: 8.000000 Predicted Y: 8.033344
Iteration 331 Loss: 0.006308 X: 8 Actual Y: 29.000000 Predicted Y: 28.920578
Iteration 332 Loss: 0.000964 X: 1 Actual Y: 8.000000 Predicted Y: 8.031054
Iteration 333 Loss: 0.002369 X: 8 Actual Y: 29.000000 Predicted Y: 28.951326
Iteration 334 Loss: 0.000328 X: 7 Actual Y: 26.000000 Predicted Y: 25.981894
Iteration 335 Loss: 0.002725 X: 0 Actual Y: 5.000000 Predicted Y: 4.947795
Iteration 336 Loss: 0.082287 X: 2 Actual Y: 11.000000 Predicted Y: 11.286857
Iteration 337 Loss: 0.010372 X: 3 Actual Y: 14.000000 Predicted Y: 13.898155
Iteration 338 Loss: 0.000530 X: 7 Actual Y: 26.000000 Predicted Y: 25.976969
Iteration 339 Loss: 0.000083 X: 4 Actual Y: 17.000000 Predicted Y: 17.009136
Iteration 340 Loss: 0.005588 X: 3 Actual Y: 14.000000 Predicted Y: 13.925245
Iteration 341 Loss: 0.000318 X: 9 Actual Y: 32.000000 Predicted Y: 31.982170
Iteration 342 Loss: 0.000011 X: 6 Actual Y: 23.000000 Predicted Y: 23.003338
Iteration 343 Loss: 0.001230 X: 5 Actual Y: 20.000000 Predicted Y: 20.035076
Iteration 344 Loss: 0.053319 X: 2 Actual Y: 11.000000 Predicted Y: 11.230908
Iteration 345 Loss: 0.000091 X: 5 Actual Y: 20.000000 Predicted Y: 20.009542
Iteration 346 Loss: 0.000439 X: 6 Actual Y: 23.000000 Predicted Y: 22.979044
Iteration 347 Loss: 0.001292 X: 9 Actual Y: 32.000000 Predicted Y: 31.964058
Iteration 348 Loss: 0.000374 X: 9 Actual Y: 32.000000 Predicted Y: 31.980656
Iteration 349 Loss: 0.002550 X: 0 Actual Y: 5.000000 Predicted Y: 4.949506
Iteration 350 Loss: 0.000565 X: 1 Actual Y: 8.000000 Predicted Y: 8.023762
Iteration 351 Loss: 0.001505 X: 0 Actual Y: 5.000000 Predicted Y: 4.961202
Iteration 352 Loss: 0.034747 X: 2 Actual Y: 11.000000 Predicted Y: 11.186405
Iteration 353 Loss: 0.020554 X: 2 Actual Y: 11.000000 Predicted Y: 11.143368
Iteration 354 Loss: 0.012168 X: 2 Actual Y: 11.000000 Predicted Y: 11.110310
Iteration 355 Loss: 0.004257 X: 8 Actual Y: 29.000000 Predicted Y: 28.934752
Iteration 356 Loss: 0.000510 X: 6 Actual Y: 23.000000 Predicted Y: 22.977419
Iteration 357 Loss: 0.005510 X: 3 Actual Y: 14.000000 Predicted Y: 13.925773
Iteration 358 Loss: 0.000407 X: 9 Actual Y: 32.000000 Predicted Y: 31.979824
Iteration 359 Loss: 0.001640 X: 0 Actual Y: 5.000000 Predicted Y: 4.959502
Iteration 360 Loss: 0.000358 X: 8 Actual Y: 29.000000 Predicted Y: 28.981079
Iteration 361 Loss: 0.000195 X: 7 Actual Y: 26.000000 Predicted Y: 25.986032
Iteration 362 Loss: 0.000002 X: 9 Actual Y: 32.000000 Predicted Y: 31.998611
Iteration 363 Loss: 0.000073 X: 7 Actual Y: 26.000000 Predicted Y: 25.991428
Iteration 364 Loss: 0.000078 X: 4 Actual Y: 17.000000 Predicted Y: 17.008835
Iteration 365 Loss: 0.000842 X: 0 Actual Y: 5.000000 Predicted Y: 4.970988
Iteration 366 Loss: 0.000473 X: 0 Actual Y: 5.000000 Predicted Y: 4.978260
Iteration 367 Loss: 0.000326 X: 1 Actual Y: 8.000000 Predicted Y: 8.018054
Iteration 368 Loss: 0.000029 X: 7 Actual Y: 26.000000 Predicted Y: 25.994652
Iteration 369 Loss: 0.010943 X: 2 Actual Y: 11.000000 Predicted Y: 11.104609
Iteration 370 Loss: 0.000067 X: 9 Actual Y: 32.000000 Predicted Y: 31.991795
Iteration 371 Loss: 0.000282 X: 5 Actual Y: 20.000000 Predicted Y: 20.016804
Iteration 372 Loss: 0.000014 X: 6 Actual Y: 23.000000 Predicted Y: 22.996265
Iteration 373 Loss: 0.006470 X: 2 Actual Y: 11.000000 Predicted Y: 11.080435
Iteration 374 Loss: 0.000041 X: 1 Actual Y: 8.000000 Predicted Y: 8.006419
Iteration 375 Loss: 0.000475 X: 8 Actual Y: 29.000000 Predicted Y: 28.978205
Iteration 376 Loss: 0.000000 X: 4 Actual Y: 17.000000 Predicted Y: 16.999376
Iteration 377 Loss: 0.002902 X: 3 Actual Y: 14.000000 Predicted Y: 13.946129
Iteration 378 Loss: 0.000000 X: 6 Actual Y: 23.000000 Predicted Y: 23.000414
Iteration 379 Loss: 0.000217 X: 5 Actual Y: 20.000000 Predicted Y: 20.014732
Iteration 380 Loss: 0.000108 X: 5 Actual Y: 20.000000 Predicted Y: 20.010368
Iteration 381 Loss: 0.000009 X: 6 Actual Y: 23.000000 Predicted Y: 22.996933
Iteration 382 Loss: 0.000004 X: 6 Actual Y: 23.000000 Predicted Y: 22.997942
Iteration 383 Loss: 0.000063 X: 1 Actual Y: 8.000000 Predicted Y: 8.007967
Iteration 384 Loss: 0.000003 X: 6 Actual Y: 23.000000 Predicted Y: 22.998161
Iteration 385 Loss: 0.000144 X: 7 Actual Y: 26.000000 Predicted Y: 25.988007
Iteration 386 Loss: 0.000052 X: 8 Actual Y: 29.000000 Predicted Y: 28.992767
Iteration 387 Loss: 0.000056 X: 1 Actual Y: 8.000000 Predicted Y: 8.007481
Iteration 388 Loss: 0.001643 X: 3 Actual Y: 14.000000 Predicted Y: 13.959462
Iteration 389 Loss: 0.000000 X: 8 Actual Y: 29.000000 Predicted Y: 29.000027
Iteration 390 Loss: 0.000006 X: 7 Actual Y: 26.000000 Predicted Y: 25.997541
Iteration 391 Loss: 0.000194 X: 5 Actual Y: 20.000000 Predicted Y: 20.013924
Iteration 392 Loss: 0.000016 X: 6 Actual Y: 23.000000 Predicted Y: 23.004017
Iteration 393 Loss: 0.000007 X: 6 Actual Y: 23.000000 Predicted Y: 23.002693
Iteration 394 Loss: 0.000435 X: 0 Actual Y: 5.000000 Predicted Y: 4.979136
Iteration 395 Loss: 0.004653 X: 2 Actual Y: 11.000000 Predicted Y: 11.068216
Iteration 396 Loss: 0.000023 X: 1 Actual Y: 8.000000 Predicted Y: 8.004790
Iteration 397 Loss: 0.002734 X: 2 Actual Y: 11.000000 Predicted Y: 11.052283
Iteration 398 Loss: 0.001445 X: 3 Actual Y: 14.000000 Predicted Y: 13.961985
Iteration 399 Loss: 0.000369 X: 0 Actual Y: 5.000000 Predicted Y: 4.980781
Iteration 400 Loss: 0.000084 X: 7 Actual Y: 26.000000 Predicted Y: 25.990810
Iteration 401 Loss: 0.000034 X: 7 Actual Y: 26.000000 Predicted Y: 25.994154
Iteration 402 Loss: 0.000052 X: 5 Actual Y: 20.000000 Predicted Y: 20.007185
Iteration 403 Loss: 0.000010 X: 9 Actual Y: 32.000000 Predicted Y: 31.996838
Iteration 404 Loss: 0.001953 X: 2 Actual Y: 11.000000 Predicted Y: 11.044196
Iteration 405 Loss: 0.000076 X: 8 Actual Y: 29.000000 Predicted Y: 28.991304
Iteration 406 Loss: 0.000017 X: 5 Actual Y: 20.000000 Predicted Y: 20.004135
Iteration 407 Loss: 0.000793 X: 3 Actual Y: 14.000000 Predicted Y: 13.971838
Iteration 408 Loss: 0.000000 X: 6 Actual Y: 23.000000 Predicted Y: 22.999901
Iteration 409 Loss: 0.000204 X: 0 Actual Y: 5.000000 Predicted Y: 4.985707
Iteration 410 Loss: 0.000421 X: 3 Actual Y: 14.000000 Predicted Y: 13.979477
Iteration 411 Loss: 0.000041 X: 1 Actual Y: 8.000000 Predicted Y: 8.006400
Iteration 412 Loss: 0.000246 X: 3 Actual Y: 14.000000 Predicted Y: 13.984308
Iteration 413 Loss: 0.000091 X: 0 Actual Y: 5.000000 Predicted Y: 4.990463
Iteration 414 Loss: 0.000129 X: 3 Actual Y: 14.000000 Predicted Y: 13.988622
Iteration 415 Loss: 0.000091 X: 5 Actual Y: 20.000000 Predicted Y: 20.009542
Iteration 416 Loss: 0.000087 X: 3 Actual Y: 14.000000 Predicted Y: 13.990677
Iteration 417 Loss: 0.000043 X: 0 Actual Y: 5.000000 Predicted Y: 4.993419
Iteration 418 Loss: 0.000020 X: 6 Actual Y: 23.000000 Predicted Y: 23.004490
Iteration 419 Loss: 0.000051 X: 3 Actual Y: 14.000000 Predicted Y: 13.992849
Iteration 420 Loss: 0.001569 X: 2 Actual Y: 11.000000 Predicted Y: 11.039610
Iteration 421 Loss: 0.000002 X: 7 Actual Y: 26.000000 Predicted Y: 25.998640
Iteration 422 Loss: 0.000001 X: 7 Actual Y: 26.000000 Predicted Y: 25.999138
Iteration 423 Loss: 0.000000 X: 8 Actual Y: 29.000000 Predicted Y: 29.000443
Iteration 424 Loss: 0.000938 X: 2 Actual Y: 11.000000 Predicted Y: 11.030622
Iteration 425 Loss: 0.000001 X: 6 Actual Y: 23.000000 Predicted Y: 22.998810
Iteration 426 Loss: 0.000013 X: 5 Actual Y: 20.000000 Predicted Y: 20.003538
Iteration 427 Loss: 0.000012 X: 7 Actual Y: 26.000000 Predicted Y: 25.996519
Iteration 428 Loss: 0.000022 X: 4 Actual Y: 17.000000 Predicted Y: 17.004642
Iteration 429 Loss: 0.000012 X: 4 Actual Y: 17.000000 Predicted Y: 17.003401
Iteration 430 Loss: 0.000005 X: 5 Actual Y: 20.000000 Predicted Y: 20.002199
Iteration 431 Loss: 0.000004 X: 6 Actual Y: 23.000000 Predicted Y: 22.998108
Iteration 432 Loss: 0.000009 X: 8 Actual Y: 29.000000 Predicted Y: 28.996929
Iteration 433 Loss: 0.000084 X: 3 Actual Y: 14.000000 Predicted Y: 13.990828
Iteration 434 Loss: 0.000020 X: 1 Actual Y: 8.000000 Predicted Y: 8.004474
Iteration 435 Loss: 0.000056 X: 0 Actual Y: 5.000000 Predicted Y: 4.992502
Iteration 436 Loss: 0.000032 X: 0 Actual Y: 5.000000 Predicted Y: 4.994382
Iteration 437 Loss: 0.000000 X: 6 Actual Y: 23.000000 Predicted Y: 23.000252
Iteration 438 Loss: 0.000010 X: 5 Actual Y: 20.000000 Predicted Y: 20.003124
Iteration 439 Loss: 0.000015 X: 1 Actual Y: 8.000000 Predicted Y: 8.003808
Iteration 440 Loss: 0.000009 X: 1 Actual Y: 8.000000 Predicted Y: 8.002954
Iteration 441 Loss: 0.000021 X: 0 Actual Y: 5.000000 Predicted Y: 4.995428
Iteration 442 Loss: 0.000576 X: 2 Actual Y: 11.000000 Predicted Y: 11.024004
Iteration 443 Loss: 0.000000 X: 5 Actual Y: 20.000000 Predicted Y: 20.000391
Iteration 444 Loss: 0.000070 X: 3 Actual Y: 14.000000 Predicted Y: 13.991625
Iteration 445 Loss: 0.000003 X: 9 Actual Y: 32.000000 Predicted Y: 31.998306
Iteration 446 Loss: 0.000001 X: 9 Actual Y: 32.000000 Predicted Y: 31.999092
Iteration 447 Loss: 0.000002 X: 5 Actual Y: 20.000000 Predicted Y: 20.001431
Iteration 448 Loss: 0.000001 X: 9 Actual Y: 32.000000 Predicted Y: 31.999264
Iteration 449 Loss: 0.000004 X: 8 Actual Y: 29.000000 Predicted Y: 28.998116
Iteration 450 Loss: 0.000005 X: 1 Actual Y: 8.000000 Predicted Y: 8.002130
Iteration 451 Loss: 0.000002 X: 5 Actual Y: 20.000000 Predicted Y: 20.001318
Iteration 452 Loss: 0.000001 X: 6 Actual Y: 23.000000 Predicted Y: 22.999001
Iteration 453 Loss: 0.000016 X: 0 Actual Y: 5.000000 Predicted Y: 4.995970
Iteration 454 Loss: 0.000001 X: 5 Actual Y: 20.000000 Predicted Y: 20.001165
Iteration 455 Loss: 0.000002 X: 8 Actual Y: 29.000000 Predicted Y: 28.998682
Iteration 456 Loss: 0.000003 X: 1 Actual Y: 8.000000 Predicted Y: 8.001819
Iteration 457 Loss: 0.000000 X: 6 Actual Y: 23.000000 Predicted Y: 22.999416
Iteration 458 Loss: 0.000002 X: 1 Actual Y: 8.000000 Predicted Y: 8.001444
Iteration 459 Loss: 0.000001 X: 1 Actual Y: 8.000000 Predicted Y: 8.001121
Iteration 460 Loss: 0.000034 X: 3 Actual Y: 14.000000 Predicted Y: 13.994183
Iteration 461 Loss: 0.000009 X: 0 Actual Y: 5.000000 Predicted Y: 4.997062
Iteration 462 Loss: 0.000005 X: 0 Actual Y: 5.000000 Predicted Y: 4.997798
Iteration 463 Loss: 0.000002 X: 7 Actual Y: 26.000000 Predicted Y: 25.998552
Iteration 464 Loss: 0.000000 X: 6 Actual Y: 23.000000 Predicted Y: 23.000381
Iteration 465 Loss: 0.000000 X: 6 Actual Y: 23.000000 Predicted Y: 23.000256
Iteration 466 Loss: 0.000002 X: 1 Actual Y: 8.000000 Predicted Y: 8.001414
Iteration 467 Loss: 0.000000 X: 8 Actual Y: 29.000000 Predicted Y: 28.999996
Iteration 468 Loss: 0.000001 X: 9 Actual Y: 32.000000 Predicted Y: 32.001064
Iteration 469 Loss: 0.000000 X: 9 Actual Y: 32.000000 Predicted Y: 32.000572
Iteration 470 Loss: 0.000002 X: 7 Actual Y: 26.000000 Predicted Y: 25.998524
Iteration 471 Loss: 0.000003 X: 0 Actual Y: 5.000000 Predicted Y: 4.998349
Iteration 472 Loss: 0.000000 X: 6 Actual Y: 23.000000 Predicted Y: 23.000042
Iteration 473 Loss: 0.000000 X: 6 Actual Y: 23.000000 Predicted Y: 23.000027
Iteration 474 Loss: 0.000001 X: 7 Actual Y: 26.000000 Predicted Y: 25.999125
Iteration 475 Loss: 0.000001 X: 9 Actual Y: 32.000000 Predicted Y: 32.000885
Iteration 476 Loss: 0.000015 X: 4 Actual Y: 17.000000 Predicted Y: 17.003862
Iteration 477 Loss: 0.000008 X: 4 Actual Y: 17.000000 Predicted Y: 17.002831
Iteration 478 Loss: 0.000001 X: 1 Actual Y: 8.000000 Predicted Y: 8.000795
Iteration 479 Loss: 0.000000 X: 9 Actual Y: 32.000000 Predicted Y: 31.999426
Iteration 480 Loss: 0.000002 X: 0 Actual Y: 5.000000 Predicted Y: 4.998479
Iteration 481 Loss: 0.000001 X: 8 Actual Y: 29.000000 Predicted Y: 28.999096
Iteration 482 Loss: 0.000002 X: 7 Actual Y: 26.000000 Predicted Y: 25.998737
Iteration 483 Loss: 0.000001 X: 0 Actual Y: 5.000000 Predicted Y: 4.998949
Iteration 484 Loss: 0.000001 X: 0 Actual Y: 5.000000 Predicted Y: 4.999213
Iteration 485 Loss: 0.000000 X: 9 Actual Y: 32.000000 Predicted Y: 32.000328
Iteration 486 Loss: 0.000001 X: 5 Actual Y: 20.000000 Predicted Y: 20.001211
Iteration 487 Loss: 0.000000 X: 0 Actual Y: 5.000000 Predicted Y: 4.999366
Iteration 488 Loss: 0.000019 X: 3 Actual Y: 14.000000 Predicted Y: 13.995600
Iteration 489 Loss: 0.000011 X: 3 Actual Y: 14.000000 Predicted Y: 13.996704
Iteration 490 Loss: 0.000009 X: 4 Actual Y: 17.000000 Predicted Y: 17.003012
Iteration 491 Loss: 0.000000 X: 7 Actual Y: 26.000000 Predicted Y: 25.999477
Iteration 492 Loss: 0.000000 X: 6 Actual Y: 23.000000 Predicted Y: 23.000036
Iteration 493 Loss: 0.000000 X: 7 Actual Y: 26.000000 Predicted Y: 25.999660
Iteration 494 Loss: 0.000000 X: 6 Actual Y: 23.000000 Predicted Y: 23.000076
Iteration 495 Loss: 0.000398 X: 2 Actual Y: 11.000000 Predicted Y: 11.019955
Iteration 496 Loss: 0.000001 X: 4 Actual Y: 17.000000 Predicted Y: 17.001053
Iteration 497 Loss: 0.000015 X: 3 Actual Y: 14.000000 Predicted Y: 13.996181
Iteration 498 Loss: 0.000001 X: 9 Actual Y: 32.000000 Predicted Y: 31.999115
Iteration 499 Loss: 0.000001 X: 4 Actual Y: 17.000000 Predicted Y: 17.001205
Iteration 500 Loss: 0.000001 X: 8 Actual Y: 29.000000 Predicted Y: 28.998850
Iteration 501 Loss: 0.000244 X: 2 Actual Y: 11.000000 Predicted Y: 11.015605
Iteration 502 Loss: 0.000002 X: 0 Actual Y: 5.000000 Predicted Y: 4.998421
Iteration 503 Loss: 0.000001 X: 0 Actual Y: 5.000000 Predicted Y: 4.998816
Iteration 504 Loss: 0.000004 X: 6 Actual Y: 23.000000 Predicted Y: 22.998016
Iteration 505 Loss: 0.000011 X: 3 Actual Y: 14.000000 Predicted Y: 13.996725
Iteration 506 Loss: 0.000000 X: 1 Actual Y: 8.000000 Predicted Y: 8.000250
Iteration 507 Loss: 0.000001 X: 9 Actual Y: 32.000000 Predicted Y: 31.999067
Iteration 508 Loss: 0.000157 X: 2 Actual Y: 11.000000 Predicted Y: 11.012524
Iteration 509 Loss: 0.000000 X: 4 Actual Y: 17.000000 Predicted Y: 17.000004
Iteration 510 Loss: 0.000003 X: 6 Actual Y: 23.000000 Predicted Y: 22.998240
Iteration 511 Loss: 0.000000 X: 1 Actual Y: 8.000000 Predicted Y: 7.999835
Iteration 512 Loss: 0.000096 X: 2 Actual Y: 11.000000 Predicted Y: 11.009774
Iteration 513 Loss: 0.000000 X: 1 Actual Y: 8.000000 Predicted Y: 7.999465
Iteration 514 Loss: 0.000010 X: 7 Actual Y: 26.000000 Predicted Y: 25.996830
Iteration 515 Loss: 0.000000 X: 4 Actual Y: 17.000000 Predicted Y: 17.000019
Iteration 516 Loss: 0.000009 X: 3 Actual Y: 14.000000 Predicted Y: 13.996945
Iteration 517 Loss: 0.000002 X: 8 Actual Y: 29.000000 Predicted Y: 28.998566
Iteration 518 Loss: 0.000066 X: 2 Actual Y: 11.000000 Predicted Y: 11.008095
Iteration 519 Loss: 0.000002 X: 9 Actual Y: 32.000000 Predicted Y: 31.998499
Iteration 520 Loss: 0.000006 X: 3 Actual Y: 14.000000 Predicted Y: 13.997622
Iteration 521 Loss: 0.000000 X: 4 Actual Y: 17.000000 Predicted Y: 17.000357
Iteration 522 Loss: 0.000002 X: 7 Actual Y: 26.000000 Predicted Y: 25.998453
Iteration 523 Loss: 0.000000 X: 9 Actual Y: 32.000000 Predicted Y: 31.999788
Iteration 524 Loss: 0.000000 X: 6 Actual Y: 23.000000 Predicted Y: 22.999401
Iteration 525 Loss: 0.000000 X: 8 Actual Y: 29.000000 Predicted Y: 28.999569
Iteration 526 Loss: 0.000000 X: 8 Actual Y: 29.000000 Predicted Y: 28.999748
Iteration 527 Loss: 0.000000 X: 7 Actual Y: 26.000000 Predicted Y: 25.999300
Iteration 528 Loss: 0.000000 X: 8 Actual Y: 29.000000 Predicted Y: 28.999996
Iteration 529 Loss: 0.000001 X: 4 Actual Y: 17.000000 Predicted Y: 17.000736
Iteration 530 Loss: 0.000000 X: 6 Actual Y: 23.000000 Predicted Y: 22.999760
Iteration 531 Loss: 0.000000 X: 5 Actual Y: 20.000000 Predicted Y: 20.000397
Iteration 532 Loss: 0.000002 X: 3 Actual Y: 14.000000 Predicted Y: 13.998516
Iteration 533 Loss: 0.000000 X: 1 Actual Y: 8.000000 Predicted Y: 8.000137
Iteration 534 Loss: 0.000000 X: 1 Actual Y: 8.000000 Predicted Y: 8.000107
Iteration 535 Loss: 0.000000 X: 4 Actual Y: 17.000000 Predicted Y: 17.000633
Iteration 536 Loss: 0.000000 X: 7 Actual Y: 26.000000 Predicted Y: 25.999506
Iteration 537 Loss: 0.000000 X: 5 Actual Y: 20.000000 Predicted Y: 20.000397
Iteration 538 Loss: 0.000002 X: 0 Actual Y: 5.000000 Predicted Y: 4.998748
Iteration 539 Loss: 0.000000 X: 1 Actual Y: 8.000000 Predicted Y: 8.000109
Iteration 540 Loss: 0.000001 X: 3 Actual Y: 14.000000 Predicted Y: 13.998889
Iteration 541 Loss: 0.000047 X: 2 Actual Y: 11.000000 Predicted Y: 11.006889
Iteration 542 Loss: 0.000000 X: 5 Actual Y: 20.000000 Predicted Y: 19.999956
Iteration 543 Loss: 0.000000 X: 8 Actual Y: 29.000000 Predicted Y: 28.999580
Iteration 544 Loss: 0.000000 X: 9 Actual Y: 32.000000 Predicted Y: 31.999979
Iteration 545 Loss: 0.000000 X: 1 Actual Y: 8.000000 Predicted Y: 7.999884
Iteration 546 Loss: 0.000000 X: 1 Actual Y: 8.000000 Predicted Y: 7.999910
Iteration 547 Loss: 0.000001 X: 0 Actual Y: 5.000000 Predicted Y: 4.998846
Iteration 548 Loss: 0.000000 X: 8 Actual Y: 29.000000 Predicted Y: 28.999811
Iteration 549 Loss: 0.000000 X: 6 Actual Y: 23.000000 Predicted Y: 22.999662
Iteration 550 Loss: 0.000001 X: 3 Actual Y: 14.000000 Predicted Y: 13.998953
Iteration 551 Loss: 0.000000 X: 4 Actual Y: 17.000000 Predicted Y: 17.000463
Iteration 552 Loss: 0.000001 X: 0 Actual Y: 5.000000 Predicted Y: 4.999171
Iteration 553 Loss: 0.000001 X: 3 Actual Y: 14.000000 Predicted Y: 13.999211
Iteration 554 Loss: 0.000000 X: 1 Actual Y: 8.000000 Predicted Y: 8.000110
Iteration 555 Loss: 0.000000 X: 4 Actual Y: 17.000000 Predicted Y: 17.000431
Iteration 556 Loss: 0.000000 X: 1 Actual Y: 8.000000 Predicted Y: 8.000063
Iteration 557 Loss: 0.000000 X: 3 Actual Y: 14.000000 Predicted Y: 13.999366
Iteration 558 Loss: 0.000031 X: 2 Actual Y: 11.000000 Predicted Y: 11.005545
Iteration 559 Loss: 0.000000 X: 6 Actual Y: 23.000000 Predicted Y: 22.999512
Iteration 560 Loss: 0.000000 X: 4 Actual Y: 17.000000 Predicted Y: 17.000071
Iteration 561 Loss: 0.000000 X: 1 Actual Y: 8.000000 Predicted Y: 7.999874
Iteration 562 Loss: 0.000019 X: 2 Actual Y: 11.000000 Predicted Y: 11.004305
Iteration 563 Loss: 0.000001 X: 0 Actual Y: 5.000000 Predicted Y: 4.999022
Iteration 564 Loss: 0.000000 X: 1 Actual Y: 8.000000 Predicted Y: 7.999762
Iteration 565 Loss: 0.000011 X: 2 Actual Y: 11.000000 Predicted Y: 11.003361
Iteration 566 Loss: 0.000001 X: 3 Actual Y: 14.000000 Predicted Y: 13.998893
Iteration 567 Loss: 0.000001 X: 8 Actual Y: 29.000000 Predicted Y: 28.999287
Iteration 568 Loss: 0.000000 X: 9 Actual Y: 32.000000 Predicted Y: 31.999580
Iteration 569 Loss: 0.000001 X: 7 Actual Y: 26.000000 Predicted Y: 25.999050
Iteration 570 Loss: 0.000000 X: 6 Actual Y: 23.000000 Predicted Y: 22.999626
Iteration 571 Loss: 0.000000 X: 7 Actual Y: 26.000000 Predicted Y: 25.999454
Iteration 572 Loss: 0.000000 X: 3 Actual Y: 14.000000 Predicted Y: 13.999496
Iteration 573 Loss: 0.000000 X: 6 Actual Y: 23.000000 Predicted Y: 22.999887
Iteration 574 Loss: 0.000000 X: 3 Actual Y: 14.000000 Predicted Y: 13.999634
Iteration 575 Loss: 0.000000 X: 5 Actual Y: 20.000000 Predicted Y: 20.000172
Iteration 576 Loss: 0.000000 X: 6 Actual Y: 23.000000 Predicted Y: 22.999935
Iteration 577 Loss: 0.000000 X: 9 Actual Y: 32.000000 Predicted Y: 32.000305
Iteration 578 Loss: 0.000009 X: 2 Actual Y: 11.000000 Predicted Y: 11.002921
Iteration 579 Loss: 0.000000 X: 7 Actual Y: 26.000000 Predicted Y: 25.999435
Iteration 580 Loss: 0.000000 X: 6 Actual Y: 23.000000 Predicted Y: 22.999771
Iteration 581 Loss: 0.000000 X: 3 Actual Y: 14.000000 Predicted Y: 13.999598
Iteration 582 Loss: 0.000005 X: 2 Actual Y: 11.000000 Predicted Y: 11.002334
Iteration 583 Loss: 0.000000 X: 8 Actual Y: 29.000000 Predicted Y: 28.999855
Iteration 584 Loss: 0.000000 X: 1 Actual Y: 8.000000 Predicted Y: 7.999799
Iteration 585 Loss: 0.000000 X: 3 Actual Y: 14.000000 Predicted Y: 13.999597
Iteration 586 Loss: 0.000000 X: 6 Actual Y: 23.000000 Predicted Y: 22.999794
Iteration 587 Loss: 0.000000 X: 8 Actual Y: 29.000000 Predicted Y: 29.000013
Iteration 588 Loss: 0.000000 X: 7 Actual Y: 26.000000 Predicted Y: 25.999641
Iteration 589 Loss: 0.000000 X: 8 Actual Y: 29.000000 Predicted Y: 29.000080
Iteration 590 Loss: 0.000000 X: 5 Actual Y: 20.000000 Predicted Y: 20.000011
Iteration 591 Loss: 0.000000 X: 7 Actual Y: 26.000000 Predicted Y: 25.999752
Iteration 592 Loss: 0.000000 X: 8 Actual Y: 29.000000 Predicted Y: 29.000097
Iteration 593 Loss: 0.000000 X: 1 Actual Y: 8.000000 Predicted Y: 7.999899
Iteration 594 Loss: 0.000001 X: 0 Actual Y: 5.000000 Predicted Y: 4.999215
Iteration 595 Loss: 0.000000 X: 8 Actual Y: 29.000000 Predicted Y: 29.000092
Iteration 596 Loss: 0.000000 X: 8 Actual Y: 29.000000 Predicted Y: 29.000053
Iteration 597 Loss: 0.000000 X: 0 Actual Y: 5.000000 Predicted Y: 4.999408
Iteration 598 Loss: 0.000000 X: 5 Actual Y: 20.000000 Predicted Y: 20.000046
Iteration 599 Loss: 0.000000 X: 6 Actual Y: 23.000000 Predicted Y: 22.999931
Iteration 600 Loss: 0.000000 X: 4 Actual Y: 17.000000 Predicted Y: 17.000114
"""

import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np

# 1. Parsing the text data
iterations, losses, xs, ys, preds = [], [], [], [], []
pattern = r"Iteration (\d+) Loss: ([\d.e+inf-]+) X: ([\d.-]+) Actual Y: ([\d.-]+) Predicted Y: ([\d.e+inf-]+)"

for line in text_data.strip().split('\n'):
    match = re.search(pattern, line)
    if match:
        iterations.append(int(match.group(1)))
        losses.append(float(match.group(2)))
        xs.append(float(match.group(3)))
        ys.append(float(match.group(4)))
        preds.append(float(match.group(5)))

df = pd.DataFrame({
    'Iter': iterations,
    'Loss': losses,
    'X': xs,
    'Actual': ys,
    'Predicted': preds
})

# 2. Calculate Rolling Average Loss
# We use a window of 20 to react faster to current performance
df['Rolling_Avg'] = df['Loss'].rolling(window=20, min_periods=1).mean()

# 3. Create Subset for Table (Every 50 iterations)
df_table = df.iloc[::50].copy()

def fmt(val):
    if np.isinf(val): return "inf"
    if abs(val) > 1000 or (0 < abs(val) < 0.01): return f"{val:.2e}"
    return f"{val:.2f}"

for col in ['Loss', 'Rolling_Avg', 'X', 'Actual', 'Predicted']:
    df_table[col] = df_table[col].apply(fmt)

# 4. Visualization
fig, (ax_plot, ax_table) = plt.subplots(1, 2, figsize=(14, 6), 
                                         gridspec_kw={'width_ratios': [1.5, 1]})

# --- Left Side: Trend Plot ---
# CLIP the loss at 1e-7 so log(0) doesn't hide the pink line
plot_loss = df['Loss'].clip(lower=1e-7)
#plot_avg = df['Rolling_Avg'].clip(lower=1e-7)

ax_plot.plot(df['Iter'], plot_loss, color='crimson', alpha=0.3, label='Instant Loss')
#ax_plot.plot(df['Iter'], plot_avg, color='blue', linewidth=2, label='Rolling Avg Loss (Window=20)')

ax_plot.set_yscale('log')
ax_plot.set_title(r'Training Progress: $y = 3x+5$', fontsize=14)
ax_plot.set_xlabel('Iteration')
ax_plot.set_ylabel('Loss (Log Scale)')
ax_plot.set_ylim(1e-7, df['Loss'].max() * 1.5) # Set explicit floor for the plot
ax_plot.legend()
ax_plot.grid(True, which="both", alpha=0.3)

# --- Right Side: Filtered Data Table ---
ax_table.axis('off')
table_data = df_table[['Iter', 'Loss', 'X', 'Actual', 'Predicted']]
table = ax_table.table(cellText=table_data.values, 
                      colLabels=table_data.columns, 
                      cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.2) 

plt.tight_layout()
plt.show()