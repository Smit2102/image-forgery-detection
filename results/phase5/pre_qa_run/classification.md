# Phase 5: classification

Dev set (train + val, 10,722 images), grouped 5-fold CV with nested grouped 3-fold model selection; 95 % CIs from 1,000 group-bootstrap resamples. Test set untouched.

**Main protocol (pre-registered rule): R**

## Classification

| protocol   | feature_set               | model   |   n_features | auc           | pooled_auc_95ci      | balanced_acc   | f1            |
|:-----------|:--------------------------|:--------|-------------:|:--------------|:---------------------|:---------------|:--------------|
| B          | Forensic (local only)     | rf      |           70 | 0.899 ± 0.005 | 0.898 [0.891, 0.905] | 0.818 ± 0.007  | 0.784 ± 0.009 |
| B          | Forensic (local + global) | rf      |           82 | 0.907 ± 0.004 | 0.907 [0.900, 0.913] | 0.827 ± 0.005  | 0.794 ± 0.006 |
| B          | Shortcuts (Phase 2)       | rf      |           49 | 0.876 ± 0.003 | 0.876 [0.868, 0.883] | 0.797 ± 0.005  | 0.759 ± 0.006 |
| B          | Forensic + shortcuts      | rf      |          131 | 0.928 ± 0.003 | 0.928 [0.922, 0.933] | 0.852 ± 0.004  | 0.824 ± 0.005 |
| B          | Forensic (local + global) | svm     |           82 | 0.896 ± 0.006 | 0.895 [0.888, 0.902] | 0.819 ± 0.005  | 0.784 ± 0.006 |
| B          | Shortcuts (Phase 2)       | svm     |           49 | 0.876 ± 0.010 | 0.876 [0.868, 0.883] | 0.802 ± 0.007  | 0.765 ± 0.008 |
| B          | Forensic + shortcuts      | svm     |          131 | 0.924 ± 0.006 | 0.923 [0.917, 0.929] | 0.854 ± 0.005  | 0.827 ± 0.006 |
| R          | Forensic (local only)     | rf      |           70 | 0.801 ± 0.011 | 0.801 [0.791, 0.812] | 0.719 ± 0.013  | 0.653 ± 0.020 |
| R          | Forensic (local + global) | rf      |           82 | 0.812 ± 0.012 | 0.812 [0.802, 0.822] | 0.725 ± 0.015  | 0.660 ± 0.022 |
| R          | Shortcuts (Phase 2)       | rf      |           49 | 0.721 ± 0.013 | 0.721 [0.708, 0.734] | 0.655 ± 0.010  | 0.569 ± 0.015 |
| R          | Forensic + shortcuts      | rf      |          131 | 0.821 ± 0.011 | 0.820 [0.811, 0.830] | 0.727 ± 0.013  | 0.664 ± 0.018 |
| R          | Forensic (local + global) | svm     |           82 | 0.811 ± 0.013 | 0.811 [0.801, 0.821] | 0.730 ± 0.012  | 0.674 ± 0.017 |
| R          | Shortcuts (Phase 2)       | svm     |           49 | 0.771 ± 0.010 | 0.771 [0.760, 0.782] | 0.713 ± 0.016  | 0.662 ± 0.022 |
| R          | Forensic + shortcuts      | svm     |          131 | 0.841 ± 0.008 | 0.841 [0.832, 0.849] | 0.760 ± 0.008  | 0.713 ± 0.011 |
| A          | Forensic (local only)     | rf      |           70 | 0.958 ± 0.003 | 0.958 [0.954, 0.962] | 0.882 ± 0.007  | 0.860 ± 0.007 |
| A          | Forensic (local + global) | rf      |           82 | 0.980 ± 0.003 | 0.980 [0.977, 0.982] | 0.926 ± 0.009  | 0.910 ± 0.011 |
| A          | Shortcuts (Phase 2)       | rf      |           49 | 1.000 ± 0.000 | 1.000 [0.999, 1.000] | 0.991 ± 0.002  | 0.988 ± 0.003 |
| A          | Forensic + shortcuts      | rf      |          131 | 0.999 ± 0.000 | 0.999 [0.999, 1.000] | 0.991 ± 0.002  | 0.988 ± 0.003 |

## Added value over the shortcut features (paired group bootstrap)

| protocol   | model   | comparison                     |   delta_auc |   ci_low |   ci_high |
|:-----------|:--------|:-------------------------------|------------:|---------:|----------:|
| B          | rf      | forensic+shortcuts - shortcuts |      0.0519 |   0.0459 |    0.0576 |
| B          | rf      | forensic_all - shortcuts       |      0.0309 |   0.0239 |    0.0383 |
| B          | rf      | forensic_local - shortcuts     |      0.0225 |   0.0146 |    0.0305 |
| B          | svm     | forensic+shortcuts - shortcuts |      0.0476 |   0.0413 |    0.0535 |
| B          | svm     | forensic_all - shortcuts       |      0.0199 |   0.0118 |    0.0276 |
| R          | rf      | forensic+shortcuts - shortcuts |      0.0993 |   0.0888 |    0.1091 |
| R          | rf      | forensic_all - shortcuts       |      0.0904 |   0.0786 |    0.1023 |
| R          | rf      | forensic_local - shortcuts     |      0.08   |   0.0671 |    0.0927 |
| R          | svm     | forensic+shortcuts - shortcuts |      0.0697 |   0.0604 |    0.0783 |
| R          | svm     | forensic_all - shortcuts       |      0.0402 |   0.0294 |    0.0511 |
| A          | rf      | forensic+shortcuts - shortcuts |     -0.0001 |  -0.0002 |    0      |
| A          | rf      | forensic_all - shortcuts       |     -0.0198 |  -0.0221 |   -0.0177 |
| A          | rf      | forensic_local - shortcuts     |     -0.0416 |  -0.0451 |   -0.0381 |

## Statistics on protocol R

- McNemar, RF vs SVM (forensic features): {'a_right_b_wrong': 622, 'a_wrong_b_right': 592, 'p_value': 0.40524010052501813}
- Calibration: ECE 0.030 -> 0.005 after isotonic calibration; Brier 0.170 -> 0.169
- Uncertain band: 0.5 ± 0.22

### Breakdown (forensic features, RF, out-of-fold)

| group                         |     auc |   n_tampered |   detection_rate |   auc_vs_authentic |
|:------------------------------|--------:|-------------:|-----------------:|-------------------:|
| copy-move vs authentic        |   0.853 |         2800 |          nan     |            nan     |
| splicing vs authentic         |   0.736 |         1554 |          nan     |            nan     |
| tampered area <1%             | nan     |          976 |            0.556 |              0.777 |
| tampered area 1-5%            | nan     |         1681 |            0.578 |              0.797 |
| tampered area >5%             | nan     |         1681 |            0.663 |              0.846 |
| tampered JPG source           | nan     |         1755 |            0.567 |              0.798 |
| tampered TIF source           | nan     |         2599 |            0.633 |              0.821 |
| authentic false-positive rate | nan     |            0 |            0.157 |            nan     |

## Ablations (RF, fixed parameters)

| module      |   ('B', 'all modules') |   ('B', 'leave one out') |   ('B', 'module alone') |   ('B', 'module alone (local only)') |   ('R', 'all modules') |   ('R', 'leave one out') |   ('R', 'module alone') |   ('R', 'module alone (local only)') |
|:------------|-----------------------:|-------------------------:|------------------------:|-------------------------------------:|-----------------------:|-------------------------:|------------------------:|-------------------------------------:|
| -           |                  0.906 |                  nan     |                 nan     |                              nan     |                  0.812 |                  nan     |                 nan     |                              nan     |
| cm_block    |                nan     |                    0.906 |                   0.603 |                                0.609 |                nan     |                    0.811 |                   0.605 |                                0.625 |
| cm_keypoint |                nan     |                    0.868 |                   0.721 |                                0.729 |                nan     |                    0.746 |                   0.72  |                                0.713 |
| dct_dq      |                nan     |                    0.869 |                   0.809 |                                0.769 |                nan     |                    0.81  |                   0.598 |                                0.58  |
| edges       |                nan     |                    0.907 |                   0.621 |                                0.602 |                nan     |                    0.809 |                   0.608 |                                0.6   |
| ela         |                nan     |                    0.905 |                   0.711 |                                0.69  |                nan     |                    0.783 |                   0.692 |                                0.671 |
| histogram   |                nan     |                    0.907 |                   0.572 |                                0.561 |                nan     |                    0.812 |                   0.577 |                                0.566 |
| jpeg_ghost  |                nan     |                    0.896 |                   0.757 |                                0.747 |                nan     |                    0.812 |                   0.631 |                                0.627 |
| noise       |                nan     |                    0.906 |                   0.592 |                                0.586 |                nan     |                    0.812 |                   0.577 |                                0.574 |

## Copy-move: keypoint vs block matching

| protocol   | features                                      |   auc_copymove_vs_authentic |   pixel_auc_copymove |   share_pixel_auc_gt_0.6 |
|:-----------|:----------------------------------------------|----------------------------:|---------------------:|-------------------------:|
| B          | keypoint                                      |                       0.844 |              nan     |                  nan     |
| B          | block                                         |                       0.645 |              nan     |                  nan     |
| B          | both                                          |                       0.869 |              nan     |                  nan     |
| B          | all modules                                   |                       0.925 |              nan     |                  nan     |
| B          | cm_keypoint localisation (Phase 4 val sample) |                     nan     |                0.716 |                    0.559 |
| B          | cm_block localisation (Phase 4 val sample)    |                     nan     |                0.546 |                    0.201 |
| R          | keypoint                                      |                       0.82  |              nan     |                  nan     |
| R          | block                                         |                       0.651 |              nan     |                  nan     |
| R          | both                                          |                       0.838 |              nan     |                  nan     |
| R          | all modules                                   |                       0.863 |              nan     |                  nan     |
| R          | cm_keypoint localisation (Phase 4 val sample) |                     nan     |                0.682 |                    0.479 |
| R          | cm_block localisation (Phase 4 val sample)    |                     nan     |                0.529 |                    0.121 |

## SHAP: summed mean |SHAP| per module

|             |      0 |
|:------------|-------:|
| cm_keypoint | 0.1947 |
| ela         | 0.1526 |
| edges       | 0.0429 |
| jpeg_ghost  | 0.0301 |
| cm_block    | 0.0274 |
| noise       | 0.0264 |
| dct_dq      | 0.0226 |
| histogram   | 0.0119 |

## Type classifier (copy-move vs splicing, tampered only): AUC {'B': 0.872520454127597, 'R': 0.8335608567751425}
