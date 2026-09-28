# Phase 6: localisation

Tampered validation images with a valid mask; post-processing tuned on one half of val and scored on the other (2-fold, grouped), so the per-method numbers are out-of-sample (the choice of main method and best single module is made on these numbers). Fusion models trained on train-split cells. The final post-processing below is tuned on all val images. Pixel AUC on every second pixel. Test set untouched.

## Protocol R

Main method: **Gradient boosting (cells)**; best single module: cm_keypoint; final post-processing (threshold, open, close, min area): (0.6, 1, 1, 0.002)

95 % CI (group bootstrap) of the main method: {'f1': [0.168, 0.2], 'iou': [0.108, 0.13], 'mcc': [0.14, 0.176]}

| protocol   | method                      |    f1 |   iou |     mcc |   pixel_auc |   oracle_f1 |   authentic_any_region |   authentic_mean_area |   n_tampered |   n_authentic |
|:-----------|:----------------------------|------:|------:|--------:|------------:|------------:|-----------------------:|----------------------:|-------------:|--------------:|
| R          | Gradient boosting (cells)   | 0.185 | 0.119 |   0.159 |       0.695 |       0.308 |                  0.888 |                 0.057 |          769 |          1124 |
| R          | Logistic regression (cells) | 0.184 | 0.121 |   0.155 |       0.686 |       0.3   |                  0.869 |                 0.058 |          769 |          1124 |
| R          | Mean of the 8 maps          | 0.159 | 0.099 |   0.1   |       0.671 |       0.23  |                  0.994 |                 0.269 |          769 |          1124 |
| R          | single: ela                 | 0.128 | 0.076 |   0.053 |       0.624 |       0.166 |                  0.982 |                 0.371 |          769 |          1124 |
| R          | single: histogram           | 0.133 | 0.084 |   0.038 |       0.56  |       0.159 |                  1     |                 0.53  |          769 |          1124 |
| R          | single: noise               | 0.132 | 0.082 |   0.011 |       0.555 |       0.168 |                  0.996 |                 0.882 |          769 |          1124 |
| R          | single: jpeg_ghost          | 0.129 | 0.081 |   0.032 |       0.549 |       0.148 |                  1     |                 0.523 |          769 |          1124 |
| R          | single: dct_dq              | 0.058 | 0.034 |   0.012 |       0.544 |       0.058 |                  0.528 |                 0.244 |          769 |          1124 |
| R          | single: edges               | 0.127 | 0.078 |   0.015 |       0.573 |       0.169 |                  1     |                 0.79  |          769 |          1124 |
| R          | single: cm_keypoint         | 0.152 | 0.102 |   0.13  |       0.62  |       0.163 |                  0.097 |                 0.009 |          769 |          1124 |
| R          | single: cm_block            | 0.06  | 0.041 |   0.029 |       0.513 |       0.061 |                  0.221 |                 0.033 |          769 |          1124 |
| R          | Whole image                 | 0.136 | 0.086 | nan     |     nan     |     nan     |                  1     |                 1     |          769 |           nan |

### By tampered area

| area_bucket   |    f1 |   iou |   mcc |   pixel_auc |
|:--------------|------:|------:|------:|------------:|
| <1%           | 0.034 | 0.02  | 0.044 |       0.617 |
| 1-5%          | 0.162 | 0.099 | 0.179 |       0.736 |
| >5%           | 0.309 | 0.206 | 0.215 |       0.706 |

### By forgery type

| forgery_type   |    f1 |   iou |   mcc |   pixel_auc |
|:---------------|------:|------:|------:|------------:|
| copy-move      | 0.204 | 0.132 | 0.189 |       0.715 |
| splicing       | 0.152 | 0.096 | 0.104 |       0.658 |

## Protocol B

Main method: **Gradient boosting (cells)**; best single module: cm_keypoint; final post-processing (threshold, open, close, min area): (0.7, 1, 17, 0.002)

95 % CI (group bootstrap) of the main method: {'f1': [0.237, 0.282], 'iou': [0.16, 0.193], 'mcc': [0.23, 0.277]}

| protocol   | method                      |    f1 |   iou |     mcc |   pixel_auc |   oracle_f1 |   authentic_any_region |   authentic_mean_area |   n_tampered |   n_authentic |
|:-----------|:----------------------------|------:|------:|--------:|------------:|------------:|-----------------------:|----------------------:|-------------:|--------------:|
| B          | Gradient boosting (cells)   | 0.26  | 0.176 |   0.254 |       0.806 |       0.436 |                  0.768 |                 0.025 |          769 |          1124 |
| B          | Logistic regression (cells) | 0.237 | 0.158 |   0.224 |       0.786 |       0.399 |                  0.917 |                 0.057 |          769 |          1124 |
| B          | Mean of the 8 maps          | 0.175 | 0.113 |   0.15  |       0.734 |       0.298 |                  0.899 |                 0.08  |          769 |          1124 |
| B          | single: ela                 | 0.124 | 0.073 |   0.044 |       0.615 |       0.158 |                  0.988 |                 0.319 |          769 |          1124 |
| B          | single: histogram           | 0.133 | 0.083 |   0.043 |       0.588 |       0.195 |                  1     |                 0.501 |          769 |          1124 |
| B          | single: noise               | 0.131 | 0.081 |   0.006 |       0.569 |       0.173 |                  0.995 |                 0.857 |          769 |          1124 |
| B          | single: jpeg_ghost          | 0.144 | 0.092 |   0.102 |       0.642 |       0.229 |                  0.997 |                 0.22  |          769 |          1124 |
| B          | single: dct_dq              | 0.157 | 0.1   |   0.132 |       0.698 |       0.238 |                  0.915 |                 0.256 |          769 |          1124 |
| B          | single: edges               | 0.124 | 0.075 |   0.009 |       0.576 |       0.167 |                  1     |                 0.767 |          769 |          1124 |
| B          | single: cm_keypoint         | 0.179 | 0.121 |   0.158 |       0.644 |       0.189 |                  0.141 |                 0.01  |          769 |          1124 |
| B          | single: cm_block            | 0.073 | 0.048 |   0.037 |       0.524 |       0.076 |                  0.336 |                 0.041 |          769 |          1124 |
| B          | Whole image                 | 0.136 | 0.086 | nan     |     nan     |     nan     |                  1     |                 1     |          769 |           nan |

### By tampered area

| area_bucket   |    f1 |   iou |   mcc |   pixel_auc |
|:--------------|------:|------:|------:|------------:|
| <1%           | 0.103 | 0.063 | 0.135 |       0.774 |
| 1-5%          | 0.269 | 0.174 | 0.298 |       0.855 |
| >5%           | 0.355 | 0.254 | 0.291 |       0.778 |

### By forgery type

| forgery_type   |    f1 |   iou |   mcc |   pixel_auc |
|:---------------|------:|------:|------:|------------:|
| copy-move      | 0.302 | 0.203 | 0.302 |       0.841 |
| splicing       | 0.184 | 0.128 | 0.167 |       0.743 |
