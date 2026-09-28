# Phase 8: ELA-CNN vs classical (test split)

Created 2026-09-27T18:23:24

| Protocol | CNN | classical | CNN - classical | rank average | combination - classical | classical + shortcuts | CNN - (classical + shortcuts) | shortcuts |
|---|---|---|---|---|---|---|---|---|
| R | 0.770 [0.744, 0.795] | 0.805 [0.779, 0.831] | -0.036 [-0.067, -0.005] | 0.824 [0.803, 0.846] | +0.019 [+0.003, +0.035] | 0.815 [0.789, 0.840] | -0.045 [-0.074, -0.017] | 0.712 [0.682, 0.746] |
| B | 0.811 [0.788, 0.834] | 0.920 [0.905, 0.934] | -0.109 [-0.128, -0.088] | 0.903 [0.886, 0.919] | -0.017 [-0.025, -0.008] | 0.936 [0.925, 0.947] | -0.125 [-0.145, -0.104] | 0.886 [0.870, 0.903] |
| A | 0.994 [0.991, 0.997] | 0.981 [0.975, 0.986] | +0.014 [+0.008, +0.019] | 0.996 [0.994, 0.998] | +0.015 [+0.011, +0.019] | 1.000 [0.999, 1.000] | -0.005 [-0.009, -0.003] | 1.000 [0.999, 1.000] |

## Subsets (AUC; CNN - classical)

- R jpeg_only (n=1424): CNN 0.765 [0.718, 0.805], classical 0.776 [0.728, 0.819], delta -0.011 [-0.062, +0.039]
- R authentic_vs_copy-move (n=1618): CNN 0.778 [0.751, 0.802], classical 0.874 [0.850, 0.895], delta -0.096 [-0.125, -0.069]
- R authentic_vs_splicing (n=1397): CNN 0.755 [0.709, 0.796], classical 0.682 [0.628, 0.735], delta +0.073 [+0.019, +0.128]
- B jpeg_only (n=1424): CNN 0.856 [0.826, 0.883], classical 0.944 [0.921, 0.962], delta -0.088 [-0.116, -0.061]
- B authentic_vs_copy-move (n=1618): CNN 0.814 [0.790, 0.840], classical 0.943 [0.929, 0.956], delta -0.129 [-0.152, -0.106]
- B authentic_vs_splicing (n=1397): CNN 0.805 [0.760, 0.845], classical 0.877 [0.840, 0.906], delta -0.071 [-0.101, -0.042]
- A jpeg_only (n=1424): CNN 0.995 [0.989, 0.999], classical 0.994 [0.990, 0.997], delta +0.001 [-0.005, +0.006]
- A authentic_vs_copy-move (n=1618): CNN 0.994 [0.991, 0.997], classical 0.980 [0.974, 0.986], delta +0.014 [+0.008, +0.020]
- A authentic_vs_splicing (n=1397): CNN 0.995 [0.990, 0.998], classical 0.982 [0.974, 0.987], delta +0.013 [+0.006, +0.020]

## Image size (tile count)

- R: tile count alone AUC 0.603, pixel count alone 0.602; within-area-tercile mean AUC CNN 0.640, classical 0.748; Spearman(CNN, classical) 0.57
- B: tile count alone AUC 0.602, pixel count alone 0.602; within-area-tercile mean AUC CNN 0.737, classical 0.897; Spearman(CNN, classical) 0.62
- A: tile count alone AUC 0.602, pixel count alone 0.602; within-area-tercile mean AUC CNN 0.984, classical 0.958; Spearman(CNN, classical) 0.78

## Degradations and MICC-F220

- R clean: CNN 0.770 [0.744, 0.795] (unpadded 0.771; 1% of images smaller than a tile), classical 0.805, delta -0.036 [-0.067, -0.005], rank average 0.824
- R JPEG Q75: CNN 0.762 [0.734, 0.787] (unpadded 0.763; 1% of images smaller than a tile), classical 0.799, delta -0.037 [-0.070, -0.008], rank average 0.817
- R resize x0.5: CNN 0.685 [0.654, 0.713] (unpadded 0.670; 95% of images smaller than a tile), classical 0.693, delta -0.008 [-0.043, +0.027], rank average 0.720
- R social (x0.8 + JPEG Q70): CNN 0.733 [0.702, 0.760] (unpadded 0.729; 82% of images smaller than a tile), classical 0.762, delta -0.029 [-0.065, +0.004], rank average 0.788
- R MICC-F220: CNN 0.529 [0.360, 0.700] (unpadded 0.529; 0% of images smaller than a tile), classical 0.972, delta -0.443 [-0.615, -0.282], rank average 0.811
- B clean: CNN 0.811 [0.788, 0.834] (unpadded 0.811; 1% of images smaller than a tile), classical 0.920, delta -0.109 [-0.128, -0.088], rank average 0.903
- B JPEG Q75: CNN 0.746 [0.715, 0.778] (unpadded 0.746; 1% of images smaller than a tile), classical 0.739, delta +0.007 [-0.034, +0.047], rank average 0.803
- B resize x0.5: CNN 0.672 [0.638, 0.704] (unpadded 0.690; 87% of images smaller than a tile), classical 0.698, delta -0.026 [-0.064, +0.011], rank average 0.723
- B social (x0.8 + JPEG Q70): CNN 0.674 [0.642, 0.702] (unpadded 0.677; 1% of images smaller than a tile), classical 0.721, delta -0.047 [-0.088, -0.005], rank average 0.755
- B MICC-F220: CNN 0.454 [0.226, 0.660] (unpadded 0.454; 0% of images smaller than a tile), classical 0.902, delta -0.449 [-0.660, -0.262], rank average 0.724
