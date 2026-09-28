# Phase 2: splits

- Split groups: 7,262 from 7,491 host ids (196 merges from exact duplicates, 17 from 694 verified near-duplicate pairs, 16 from donor-dominant images); largest group 69 images
- Every split group, host group and exact-duplicate set lies in one split and one CV fold (asserted before anything is written).
- Largest deviation from the 70/15/15 target: 0.23 percentage points (54-image BMP stratum: 1.67).
- Tampered images whose donor sits in a different split: 830 ({'splicing': 828, 'copy-move': 2}); their pasted region covers a median 5.5% of the image (90th percentile 34.5%). Reported, not prevented: merging every donor would create one group of 4,027 images (31.9%).
- Frozen hashes: test `a437cb81bb74d6f8cb9030adb6ffbba5bb41b0201fad49b94d981c9ffca0b9b4`, full split `6991cc0c03323bd7e3fd9900d188ddb706760cb5ddaa42fb9adb0d7d74a04468`

## Composition (forgery type x format)

| stratum       | train        | val          | test         |
|:--------------|:-------------|:-------------|:-------------|
| copy-move_jpg | 676 (69.9%)  | 146 (15.1%)  | 145 (15.0%)  |
| copy-move_tif | 1629 (70.0%) | 349 (15.0%)  | 350 (15.0%)  |
| none_bmp      | 37 (68.5%)   | 9 (16.7%)    | 8 (14.8%)    |
| none_jpg      | 5207 (70.0%) | 1115 (15.0%) | 1115 (15.0%) |
| splicing_jpg  | 769 (70.1%)  | 164 (14.9%)  | 164 (14.9%)  |
| splicing_tif  | 510 (69.8%)  | 111 (15.2%)  | 110 (15.0%)  |
| TOTAL         | 8828 (70.0%) | 1894 (15.0%) | 1892 (15.0%) |

## Dev-set CV folds (train + val)

| stratum       |    0 |    1 |    2 |    3 |    4 |
|:--------------|-----:|-----:|-----:|-----:|-----:|
| copy-move_jpg |  164 |  165 |  164 |  165 |  164 |
| copy-move_tif |  396 |  396 |  394 |  396 |  396 |
| none_bmp      |    9 |   10 |    9 |    9 |    9 |
| none_jpg      | 1265 | 1264 | 1265 | 1264 | 1264 |
| splicing_jpg  |  187 |  187 |  186 |  186 |  187 |
| splicing_tif  |  124 |  124 |  124 |  125 |  124 |
