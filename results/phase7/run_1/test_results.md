# Phase 7: test-set results (frozen models, evaluated once)

Run #1 at 2026-09-24T19:56:29; code fingerprint 3ec0cefd38d4; test split a437cb81bb74; test images {'authentic': 1123, 'tampered': 769}. 95 % CIs from 1,000 group-bootstrap resamples of the test set.

The `forensic_all | rf` row is the deployed model's own forest (refit on all dev = identical); the deployed model adds isotonic calibration, whose output AUC is reported separately.

## Classification

| protocol   | feature_set        | model   | auc_95ci             | balanced_acc_95ci    |   precision |   recall |    f1 |
|:-----------|:-------------------|:--------|:---------------------|:---------------------|------------:|---------:|------:|
| R          | forensic_local     | rf      | 0.796 [0.769, 0.821] | 0.714 [0.688, 0.740] |       0.757 |    0.55  | 0.637 |
| R          | forensic_all       | rf      | 0.805 [0.779, 0.831] | 0.721 [0.695, 0.748] |       0.756 |    0.567 | 0.648 |
| R          | shortcuts          | rf      | 0.712 [0.682, 0.746] | 0.650 [0.621, 0.682] |       0.62  |    0.516 | 0.564 |
| R          | forensic+shortcuts | rf      | 0.815 [0.789, 0.840] | 0.722 [0.696, 0.749] |       0.725 |    0.601 | 0.657 |
| R          | forensic_all       | svm     | 0.809 [0.782, 0.834] | 0.730 [0.704, 0.754] |       0.71  |    0.64  | 0.673 |
| R          | shortcuts          | svm     | 0.753 [0.723, 0.782] | 0.695 [0.665, 0.726] |       0.633 |    0.648 | 0.64  |
| R          | forensic+shortcuts | svm     | 0.829 [0.806, 0.852] | 0.751 [0.725, 0.776] |       0.721 |    0.684 | 0.702 |
| B          | forensic_local     | rf      | 0.909 [0.892, 0.924] | 0.834 [0.814, 0.853] |       0.799 |    0.808 | 0.803 |
| B          | forensic_all       | rf      | 0.920 [0.905, 0.934] | 0.845 [0.826, 0.863] |       0.813 |    0.819 | 0.816 |
| B          | shortcuts          | rf      | 0.886 [0.870, 0.903] | 0.813 [0.792, 0.833] |       0.767 |    0.791 | 0.778 |
| B          | forensic+shortcuts | rf      | 0.936 [0.925, 0.947] | 0.857 [0.840, 0.874] |       0.831 |    0.83  | 0.83  |
| B          | forensic_all       | svm     | 0.908 [0.891, 0.925] | 0.843 [0.822, 0.863] |       0.811 |    0.817 | 0.814 |
| B          | shortcuts          | svm     | 0.883 [0.863, 0.902] | 0.820 [0.801, 0.840] |       0.765 |    0.811 | 0.787 |
| B          | forensic+shortcuts | svm     | 0.934 [0.921, 0.946] | 0.866 [0.849, 0.883] |       0.839 |    0.843 | 0.841 |
| A          | forensic_local     | rf      | 0.963 [0.954, 0.972] | 0.888 [0.868, 0.907] |       0.873 |    0.861 | 0.867 |
| A          | forensic_all       | rf      | 0.981 [0.975, 0.986] | 0.920 [0.907, 0.933] |       0.893 |    0.915 | 0.904 |
| A          | shortcuts          | rf      | 1.000 [0.999, 1.000] | 0.990 [0.984, 0.995] |       0.978 |    0.995 | 0.986 |
| A          | forensic+shortcuts | rf      | 1.000 [0.999, 1.000] | 0.992 [0.988, 0.996] |       0.98  |    0.999 | 0.989 |

## Added value over the shortcuts (paired group bootstrap)

| protocol   | model   | comparison                     |   delta_auc |   ci_low |   ci_high |
|:-----------|:--------|:-------------------------------|------------:|---------:|----------:|
| R          | rf      | forensic+shortcuts - shortcuts |      0.1026 |   0.0761 |    0.1291 |
| R          | rf      | forensic_all - shortcuts       |      0.093  |   0.0593 |    0.1251 |
| R          | rf      | forensic_local - shortcuts     |      0.0833 |   0.0476 |    0.116  |
| R          | svm     | forensic+shortcuts - shortcuts |      0.0765 |   0.054  |    0.099  |
| R          | svm     | forensic_all - shortcuts       |      0.0562 |   0.0258 |    0.085  |
| B          | rf      | forensic+shortcuts - shortcuts |      0.0496 |   0.0362 |    0.0633 |
| B          | rf      | forensic_all - shortcuts       |      0.0332 |   0.0167 |    0.0493 |
| B          | rf      | forensic_local - shortcuts     |      0.0222 |   0.0031 |    0.04   |
| B          | svm     | forensic+shortcuts - shortcuts |      0.0507 |   0.036  |    0.0664 |
| B          | svm     | forensic_all - shortcuts       |      0.025  |   0.0055 |    0.0453 |
| A          | rf      | forensic+shortcuts - shortcuts |      0.0001 |  -0      |    0.0003 |
| A          | rf      | forensic_all - shortcuts       |     -0.0189 |  -0.0241 |   -0.014  |
| A          | rf      | forensic_local - shortcuts     |     -0.0365 |  -0.0453 |   -0.0278 |

## Protocol R: deployed model, localisation

- Deployed model: {"band": 0.24, "coverage": 0.518, "accuracy_judged": 0.8582, "balanced_accuracy_judged": 0.8087, "ece": 0.0188, "brier": 0.1697, "auc_of_calibrated_output": 0.8049, "verdicts": {"tampered": 235, "authentic": 745, "uncertain": 912}}
- McNemar RF vs SVM: {'a_right_b_wrong': 119, 'a_wrong_b_right': 115, 'p_value': 0.8445721639622716}
- Localisation: {"n_tampered": 766, "n_authentic": 1123, "f1": 0.1975, "iou": 0.1277, "mcc": 0.1629, "pixel_auc": 0.7138, "f1_ci": [0.17940506333396009, 0.21643491236165485], "iou_ci": [0.11502284757305421, 0.14069136627329495], "mcc_ci": [0.14620300162515834, 0.18030648139563235], "whole_image_f1": 0.1352, "mean_f1": 0.1689, "single_cm_keypoint_f1": 0.174, "authentic_any_region_ungated": 0.8905, "authentic_any_region_gated": 0.4292, "authentic_mean_area_ungated": 0.0567, "tampered_shown": 0.8381, "tampered_f1_gated": 0.1893}

### Localisation by tampered area

|      |    f1 |   iou |   pixel_auc |
|:-----|------:|------:|------------:|
| 1-5% | 0.148 | 0.089 |       0.725 |
| <1%  | 0.04  | 0.022 |       0.682 |
| >5%  | 0.333 | 0.225 |       0.715 |

### Localisation by forgery type

|           |    f1 |   iou |   pixel_auc |
|:----------|------:|------:|------------:|
| copy-move | 0.236 | 0.154 |       0.744 |
| splicing  | 0.129 | 0.08  |       0.659 |

### Breakdown (forensic RF)

| group                         |      auc |   n_tampered |   detection_rate |   auc_vs_authentic |
|:------------------------------|---------:|-------------:|-----------------:|-------------------:|
| copy-move vs authentic        |   0.8736 |          495 |         nan      |           nan      |
| splicing vs authentic         |   0.6819 |          274 |         nan      |           nan      |
| tampered area <1%             | nan      |          138 |           0.587  |             0.8264 |
| tampered area 1-5%            | nan      |          342 |           0.5117 |             0.7759 |
| tampered area >5%             | nan      |          286 |           0.6259 |             0.8334 |
| tampered JPG source           | nan      |          309 |           0.4757 |             0.7746 |
| tampered TIF source           | nan      |          460 |           0.6283 |             0.8259 |
| authentic false-positive rate | nan      |            0 |           0.1256 |           nan      |

- Outcomes with the deployed band: {'uncertain': 912, 'TN': 619, 'TP': 222, 'FN': 126, 'FP': 13}

## Protocol B: deployed model, localisation

- Deployed model: {"band": 0.04, "coverage": 0.9572, "accuracy_judged": 0.8691, "balanced_accuracy_judged": 0.8533, "ece": 0.018, "brier": 0.1078, "auc_of_calibrated_output": 0.919, "verdicts": {"tampered": 634, "authentic": 1177, "uncertain": 81}}
- McNemar RF vs SVM: {'a_right_b_wrong': 90, 'a_wrong_b_right': 87, 'p_value': 0.8805599174040929}
- Localisation: {"n_tampered": 766, "n_authentic": 1123, "f1": 0.3081, "iou": 0.21, "mcc": 0.2961, "pixel_auc": 0.8294, "f1_ci": [0.2835015301466622, 0.3320845234990763], "iou_ci": [0.19120412239298035, 0.22820759296269807], "mcc_ci": [0.270947391030863, 0.3184952003647068], "whole_image_f1": 0.1352, "mean_f1": 0.2075, "single_cm_keypoint_f1": 0.2119, "authentic_any_region_ungated": 0.7311, "authentic_any_region_gated": 0.0855, "authentic_mean_area_ungated": 0.0252, "tampered_shown": 0.7859, "tampered_f1_gated": 0.2673}

### Localisation by tampered area

|      |    f1 |   iou |   pixel_auc |
|:-----|------:|------:|------------:|
| 1-5% | 0.295 | 0.194 |       0.857 |
| <1%  | 0.123 | 0.074 |       0.821 |
| >5%  | 0.413 | 0.295 |       0.801 |

### Localisation by forgery type

|           |    f1 |   iou |   pixel_auc |
|:----------|------:|------:|------------:|
| copy-move | 0.354 | 0.24  |       0.851 |
| splicing  | 0.226 | 0.157 |       0.791 |

### Breakdown (forensic RF)

| group                         |      auc |   n_tampered |   detection_rate |   auc_vs_authentic |
|:------------------------------|---------:|-------------:|-----------------:|-------------------:|
| copy-move vs authentic        |   0.9433 |          495 |         nan      |           nan      |
| splicing vs authentic         |   0.8768 |          274 |         nan      |           nan      |
| tampered area <1%             | nan      |          138 |           0.8333 |             0.9241 |
| tampered area 1-5%            | nan      |          342 |           0.7895 |             0.9065 |
| tampered area >5%             | nan      |          286 |           0.8462 |             0.9328 |
| tampered JPG source           | nan      |          309 |           0.8867 |             0.9437 |
| tampered TIF source           | nan      |          460 |           0.7739 |             0.9035 |
| authentic false-positive rate | nan      |            0 |           0.1291 |           nan      |
