# Phase 5: classification

Dev set (train + val, 10,722 images), grouped 5-fold CV with nested grouped 3-fold model selection; 95 % CIs from 1,000 group-bootstrap resamples. Test set untouched.

**Main protocol (pre-registered rule): R**

## Classification

| protocol   | feature_set               | model   |   n_features | auc           | pooled_auc_95ci      | balanced_acc   | f1            |
|:-----------|:--------------------------|:--------|-------------:|:--------------|:---------------------|:---------------|:--------------|
| B          | Forensic (local only)     | rf      |           70 | 0.902 ± 0.003 | 0.902 [0.895, 0.908] | 0.821 ± 0.005  | 0.787 ± 0.006 |
| B          | Forensic (local + global) | rf      |           82 | 0.910 ± 0.003 | 0.910 [0.904, 0.916] | 0.831 ± 0.005  | 0.799 ± 0.006 |
| B          | Shortcuts (Phase 2)       | rf      |           49 | 0.876 ± 0.003 | 0.876 [0.868, 0.883] | 0.797 ± 0.005  | 0.759 ± 0.006 |
| B          | Forensic + shortcuts      | rf      |          131 | 0.926 ± 0.004 | 0.926 [0.920, 0.931] | 0.849 ± 0.002  | 0.821 ± 0.003 |
| B          | Forensic (local + global) | svm     |           82 | 0.892 ± 0.004 | 0.892 [0.885, 0.900] | 0.814 ± 0.004  | 0.778 ± 0.005 |
| B          | Shortcuts (Phase 2)       | svm     |           49 | 0.876 ± 0.010 | 0.876 [0.868, 0.883] | 0.802 ± 0.007  | 0.765 ± 0.008 |
| B          | Forensic + shortcuts      | svm     |          131 | 0.922 ± 0.004 | 0.922 [0.915, 0.928] | 0.854 ± 0.004  | 0.827 ± 0.005 |
| R          | Forensic (local only)     | rf      |           70 | 0.781 ± 0.012 | 0.781 [0.769, 0.792] | 0.693 ± 0.007  | 0.607 ± 0.012 |
| R          | Forensic (local + global) | rf      |           82 | 0.795 ± 0.008 | 0.795 [0.783, 0.806] | 0.705 ± 0.011  | 0.628 ± 0.018 |
| R          | Shortcuts (Phase 2)       | rf      |           49 | 0.721 ± 0.013 | 0.721 [0.708, 0.734] | 0.655 ± 0.010  | 0.569 ± 0.015 |
| R          | Forensic + shortcuts      | rf      |          131 | 0.810 ± 0.006 | 0.809 [0.799, 0.820] | 0.714 ± 0.013  | 0.644 ± 0.021 |
| R          | Forensic (local + global) | svm     |           82 | 0.793 ± 0.006 | 0.793 [0.782, 0.804] | 0.719 ± 0.011  | 0.658 ± 0.015 |
| R          | Shortcuts (Phase 2)       | svm     |           49 | 0.771 ± 0.010 | 0.771 [0.760, 0.782] | 0.713 ± 0.016  | 0.662 ± 0.022 |
| R          | Forensic + shortcuts      | svm     |          131 | 0.826 ± 0.008 | 0.826 [0.816, 0.835] | 0.744 ± 0.007  | 0.692 ± 0.011 |
| A          | Forensic (local only)     | rf      |           70 | 0.962 ± 0.005 | 0.962 [0.959, 0.966] | 0.885 ± 0.010  | 0.864 ± 0.012 |
| A          | Forensic (local + global) | rf      |           82 | 0.978 ± 0.003 | 0.978 [0.976, 0.981] | 0.920 ± 0.006  | 0.905 ± 0.007 |
| A          | Shortcuts (Phase 2)       | rf      |           49 | 1.000 ± 0.000 | 1.000 [0.999, 1.000] | 0.991 ± 0.002  | 0.988 ± 0.003 |
| A          | Forensic + shortcuts      | rf      |          131 | 0.999 ± 0.000 | 0.999 [0.999, 1.000] | 0.992 ± 0.002  | 0.989 ± 0.003 |

## Added value over the shortcut features (paired group bootstrap)

| protocol   | model   | comparison                     |   delta_auc |   ci_low |   ci_high |
|:-----------|:--------|:-------------------------------|------------:|---------:|----------:|
| B          | rf      | forensic+shortcuts - shortcuts |      0.0498 |   0.0442 |    0.0553 |
| B          | rf      | forensic_all - shortcuts       |      0.0344 |   0.0272 |    0.0412 |
| B          | rf      | forensic_local - shortcuts     |      0.0259 |   0.0184 |    0.0332 |
| B          | svm     | forensic+shortcuts - shortcuts |      0.0463 |   0.0401 |    0.0526 |
| B          | svm     | forensic_all - shortcuts       |      0.0167 |   0.0085 |    0.0249 |
| R          | rf      | forensic+shortcuts - shortcuts |      0.0879 |   0.0782 |    0.0975 |
| R          | rf      | forensic_all - shortcuts       |      0.0736 |   0.062  |    0.0851 |
| R          | rf      | forensic_local - shortcuts     |      0.0594 |   0.0468 |    0.0721 |
| R          | svm     | forensic+shortcuts - shortcuts |      0.0545 |   0.0453 |    0.0631 |
| R          | svm     | forensic_all - shortcuts       |      0.0216 |   0.0101 |    0.0333 |
| A          | rf      | forensic+shortcuts - shortcuts |     -0.0001 |  -0.0002 |   -0      |
| A          | rf      | forensic_all - shortcuts       |     -0.0212 |  -0.0234 |   -0.019  |
| A          | rf      | forensic_local - shortcuts     |     -0.0371 |  -0.0408 |   -0.0337 |

## Statistics on protocol R

- McNemar, RF vs SVM (forensic features): {'a_right_b_wrong': 603, 'a_wrong_b_right': 644, 'p_value': 0.2573182866044448}
- Calibration: ECE 0.034 -> 0.009 after isotonic calibration; Brier 0.176 -> 0.175
- Uncertain band per protocol (chosen on out-of-fold calibrated predictions; accuracy and balanced accuracy on the judged images): {'B': {'coverage': 0.9633, 'accuracy': 0.8508, 'balanced_accuracy': 0.8385}, 'R': {'coverage': 0.5124, 'accuracy': 0.8555, 'balanced_accuracy': 0.8105}}

### Breakdown (forensic features, RF, out-of-fold)

| group                         |     auc |   n_tampered |   detection_rate |   auc_vs_authentic |
|:------------------------------|--------:|-------------:|-----------------:|-------------------:|
| copy-move vs authentic        |   0.851 |         2800 |          nan     |            nan     |
| splicing vs authentic         |   0.694 |         1554 |          nan     |            nan     |
| tampered area <1%             | nan     |          976 |            0.525 |              0.768 |
| tampered area 1-5%            | nan     |         1681 |            0.53  |              0.787 |
| tampered area >5%             | nan     |         1681 |            0.596 |              0.818 |
| tampered JPG source           | nan     |         1755 |            0.477 |              0.769 |
| tampered TIF source           | nan     |         2599 |            0.606 |              0.812 |
| authentic false-positive rate | nan     |            0 |            0.144 |            nan     |

## Ablations (RF, fixed parameters)

| module      |   ('B', 'all modules') |   ('B', 'leave one out') |   ('B', 'module alone') |   ('B', 'module alone (local only)') |   ('R', 'all modules') |   ('R', 'leave one out') |   ('R', 'module alone') |   ('R', 'module alone (local only)') |
|:------------|-----------------------:|-------------------------:|------------------------:|-------------------------------------:|-----------------------:|-------------------------:|------------------------:|-------------------------------------:|
| -           |                   0.91 |                  nan     |                 nan     |                              nan     |                  0.795 |                  nan     |                 nan     |                              nan     |
| cm_block    |                 nan    |                    0.91  |                   0.6   |                                0.594 |                nan     |                    0.792 |                   0.579 |                                0.569 |
| cm_keypoint |                 nan    |                    0.871 |                   0.747 |                                0.738 |                nan     |                    0.715 |                   0.72  |                                0.714 |
| dct_dq      |                 nan    |                    0.864 |                   0.821 |                                0.8   |                nan     |                    0.794 |                   0.519 |                                0.517 |
| edges       |                 nan    |                    0.911 |                   0.626 |                                0.607 |                nan     |                    0.793 |                   0.605 |                                0.598 |
| ela         |                 nan    |                    0.91  |                   0.723 |                                0.703 |                nan     |                    0.782 |                   0.664 |                                0.638 |
| histogram   |                 nan    |                    0.911 |                   0.569 |                                0.564 |                nan     |                    0.796 |                   0.566 |                                0.555 |
| jpeg_ghost  |                 nan    |                    0.9   |                   0.758 |                                0.751 |                nan     |                    0.794 |                   0.624 |                                0.618 |
| noise       |                 nan    |                    0.91  |                   0.645 |                                0.639 |                nan     |                    0.791 |                   0.617 |                                0.612 |

## Copy-move: keypoint vs block matching

| protocol   | features                                      |   auc_copymove_vs_authentic |   pixel_auc_copymove |   share_pixel_auc_gt_0.6 |
|:-----------|:----------------------------------------------|----------------------------:|---------------------:|-------------------------:|
| B          | keypoint                                      |                       0.865 |              nan     |                  nan     |
| B          | block                                         |                       0.633 |              nan     |                  nan     |
| B          | both                                          |                       0.878 |              nan     |                  nan     |
| B          | all modules                                   |                       0.925 |              nan     |                  nan     |
| B          | cm_keypoint localisation (Phase 4 val sample) |                     nan     |                0.719 |                    0.562 |
| B          | cm_block localisation (Phase 4 val sample)    |                     nan     |                0.527 |                    0.141 |
| R          | keypoint                                      |                       0.821 |              nan     |                  nan     |
| R          | block                                         |                       0.603 |              nan     |                  nan     |
| R          | both                                          |                       0.839 |              nan     |                  nan     |
| R          | all modules                                   |                       0.858 |              nan     |                  nan     |
| R          | cm_keypoint localisation (Phase 4 val sample) |                     nan     |                0.683 |                    0.479 |
| R          | cm_block localisation (Phase 4 val sample)    |                     nan     |                0.517 |                    0.099 |

## SHAP: summed mean |SHAP| per module

|             |      0 |
|:------------|-------:|
| cm_keypoint | 0.1864 |
| ela         | 0.0975 |
| edges       | 0.0478 |
| noise       | 0.0443 |
| jpeg_ghost  | 0.039  |
| cm_block    | 0.0236 |
| histogram   | 0.0154 |
| dct_dq      | 0.0062 |

## Type classifier (copy-move vs splicing, tampered only): AUC {'B': 0.8744295826438684, 'R': 0.830478258871116}
