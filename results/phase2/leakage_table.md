# Phase 2: shortcut-baseline (leakage) study

Dev set only (train + val, 10,722 images), grouped 5-fold CV, random forest (300 trees, balanced class weights). Test set untouched.

Protocols: **A** as released; **C** JPEG files only (both classes); **B<q>** every image re-encoded once to JPEG at quality q (main: B85); **R85** downscaled x0.75 then re-encoded; the last two rows restrict the main protocol to one source format.

## Balanced accuracy overview (mean of 5 folds)

| feature_set              |     A |     C |   B75 |   B85 |   B95 |   R85 |   B85-jpeg-sources |   B85-au-vs-tiff |
|:-------------------------|------:|------:|------:|------:|------:|------:|-------------------:|-----------------:|
| File metadata            | 0.99  | 0.991 | 0.503 | 0.508 | 0.509 | 0.503 |              0.516 |            0.511 |
| Image dimensions         | 0.603 | 0.597 | 0.603 | 0.603 | 0.603 | 0.604 |              0.597 |            0.613 |
| Naive global ELA         | 0.696 | 0.678 | 0.541 | 0.564 | 0.665 | 0.539 |              0.579 |            0.535 |
| Global DCT double-quant. | 0.844 | 0.822 | 0.779 | 0.788 | 0.808 | 0.557 |              0.915 |            0.726 |
| Global ghost curve       | 0.911 | 0.944 | 0.676 | 0.665 | 0.84  | 0.624 |              0.669 |            0.656 |
| Compression history      | 0.928 | 0.955 | 0.778 | 0.789 | 0.821 | 0.608 |              0.919 |            0.725 |
| All shortcuts            | 0.989 | 0.986 | 0.791 | 0.797 | 0.855 | 0.649 |              0.925 |            0.735 |

## Full table

| protocol         | feature_set              |   n_images | balanced_accuracy   | roc_auc       |   majority_accuracy |
|:-----------------|:-------------------------|-----------:|:--------------------|:--------------|--------------------:|
| A                | File metadata            |      10722 | 0.990 ± 0.002       | 0.999 ± 0.000 |               0.594 |
| A                | Image dimensions         |      10722 | 0.603 ± 0.007       | 0.642 ± 0.006 |               0.594 |
| A                | Naive global ELA         |      10722 | 0.696 ± 0.007       | 0.778 ± 0.007 |               0.594 |
| A                | Global DCT double-quant. |      10722 | 0.844 ± 0.013       | 0.941 ± 0.009 |               0.594 |
| A                | Global ghost curve       |      10722 | 0.911 ± 0.005       | 0.968 ± 0.003 |               0.594 |
| A                | Compression history      |      10722 | 0.928 ± 0.006       | 0.981 ± 0.003 |               0.594 |
| A                | All shortcuts            |      10722 | 0.989 ± 0.003       | 0.999 ± 0.000 |               0.594 |
| C                | File metadata            |       8077 | 0.991 ± 0.002       | 0.996 ± 0.001 |               0.783 |
| C                | Image dimensions         |       8077 | 0.597 ± 0.015       | 0.645 ± 0.012 |               0.783 |
| C                | Naive global ELA         |       8077 | 0.678 ± 0.015       | 0.782 ± 0.015 |               0.783 |
| C                | Global DCT double-quant. |       8077 | 0.822 ± 0.019       | 0.935 ± 0.017 |               0.783 |
| C                | Global ghost curve       |       8077 | 0.944 ± 0.006       | 0.988 ± 0.002 |               0.783 |
| C                | Compression history      |       8077 | 0.955 ± 0.011       | 0.990 ± 0.003 |               0.783 |
| C                | All shortcuts            |       8077 | 0.986 ± 0.003       | 0.998 ± 0.001 |               0.783 |
| B75              | File metadata            |      10722 | 0.503 ± 0.004       | 0.507 ± 0.008 |               0.594 |
| B75              | Image dimensions         |      10722 | 0.603 ± 0.007       | 0.642 ± 0.006 |               0.594 |
| B75              | Naive global ELA         |      10722 | 0.541 ± 0.010       | 0.562 ± 0.009 |               0.594 |
| B75              | Global DCT double-quant. |      10722 | 0.779 ± 0.007       | 0.864 ± 0.008 |               0.594 |
| B75              | Global ghost curve       |      10722 | 0.676 ± 0.011       | 0.739 ± 0.013 |               0.594 |
| B75              | Compression history      |      10722 | 0.778 ± 0.008       | 0.869 ± 0.007 |               0.594 |
| B75              | All shortcuts            |      10722 | 0.791 ± 0.007       | 0.872 ± 0.008 |               0.594 |
| B85              | File metadata            |      10722 | 0.508 ± 0.016       | 0.510 ± 0.015 |               0.594 |
| B85              | Image dimensions         |      10722 | 0.603 ± 0.007       | 0.642 ± 0.006 |               0.594 |
| B85              | Naive global ELA         |      10722 | 0.564 ± 0.010       | 0.597 ± 0.010 |               0.594 |
| B85              | Global DCT double-quant. |      10722 | 0.788 ± 0.006       | 0.861 ± 0.009 |               0.594 |
| B85              | Global ghost curve       |      10722 | 0.665 ± 0.009       | 0.718 ± 0.007 |               0.594 |
| B85              | Compression history      |      10722 | 0.789 ± 0.010       | 0.870 ± 0.006 |               0.594 |
| B85              | All shortcuts            |      10722 | 0.797 ± 0.007       | 0.876 ± 0.004 |               0.594 |
| B95              | File metadata            |      10722 | 0.509 ± 0.004       | 0.517 ± 0.005 |               0.594 |
| B95              | Image dimensions         |      10722 | 0.603 ± 0.007       | 0.642 ± 0.006 |               0.594 |
| B95              | Naive global ELA         |      10722 | 0.665 ± 0.009       | 0.736 ± 0.011 |               0.594 |
| B95              | Global DCT double-quant. |      10722 | 0.808 ± 0.004       | 0.882 ± 0.005 |               0.594 |
| B95              | Global ghost curve       |      10722 | 0.840 ± 0.011       | 0.921 ± 0.006 |               0.594 |
| B95              | Compression history      |      10722 | 0.821 ± 0.005       | 0.902 ± 0.003 |               0.594 |
| B95              | All shortcuts            |      10722 | 0.855 ± 0.007       | 0.936 ± 0.005 |               0.594 |
| R85              | File metadata            |      10722 | 0.503 ± 0.006       | 0.509 ± 0.012 |               0.594 |
| R85              | Image dimensions         |      10722 | 0.604 ± 0.006       | 0.643 ± 0.006 |               0.594 |
| R85              | Naive global ELA         |      10722 | 0.539 ± 0.011       | 0.556 ± 0.009 |               0.594 |
| R85              | Global DCT double-quant. |      10722 | 0.557 ± 0.011       | 0.611 ± 0.016 |               0.594 |
| R85              | Global ghost curve       |      10722 | 0.624 ± 0.013       | 0.672 ± 0.014 |               0.594 |
| R85              | Compression history      |      10722 | 0.608 ± 0.004       | 0.683 ± 0.007 |               0.594 |
| R85              | All shortcuts            |      10722 | 0.649 ± 0.011       | 0.720 ± 0.012 |               0.594 |
| B85-jpeg-sources | File metadata            |       8077 | 0.516 ± 0.018       | 0.514 ± 0.022 |               0.783 |
| B85-jpeg-sources | Image dimensions         |       8077 | 0.597 ± 0.015       | 0.645 ± 0.012 |               0.783 |
| B85-jpeg-sources | Naive global ELA         |       8077 | 0.579 ± 0.013       | 0.643 ± 0.013 |               0.783 |
| B85-jpeg-sources | Global DCT double-quant. |       8077 | 0.915 ± 0.014       | 0.976 ± 0.005 |               0.783 |
| B85-jpeg-sources | Global ghost curve       |       8077 | 0.669 ± 0.020       | 0.767 ± 0.021 |               0.783 |
| B85-jpeg-sources | Compression history      |       8077 | 0.919 ± 0.008       | 0.978 ± 0.004 |               0.783 |
| B85-jpeg-sources | All shortcuts            |       8077 | 0.925 ± 0.006       | 0.983 ± 0.003 |               0.783 |
| B85-au-vs-tiff   | File metadata            |       8967 | 0.511 ± 0.014       | 0.512 ± 0.020 |               0.71  |
| B85-au-vs-tiff   | Image dimensions         |       8967 | 0.613 ± 0.004       | 0.650 ± 0.011 |               0.71  |
| B85-au-vs-tiff   | Naive global ELA         |       8967 | 0.535 ± 0.014       | 0.563 ± 0.010 |               0.71  |
| B85-au-vs-tiff   | Global DCT double-quant. |       8967 | 0.726 ± 0.008       | 0.798 ± 0.013 |               0.71  |
| B85-au-vs-tiff   | Global ghost curve       |       8967 | 0.656 ± 0.014       | 0.720 ± 0.015 |               0.71  |
| B85-au-vs-tiff   | Compression history      |       8967 | 0.725 ± 0.010       | 0.811 ± 0.010 |               0.71  |
| B85-au-vs-tiff   | All shortcuts            |       8967 | 0.735 ± 0.009       | 0.818 ± 0.007 |               0.71  |
