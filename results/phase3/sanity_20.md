# Phase 3: module sanity check (20 validation images)

8 copy-move + 8 splicing images (mask 1-50% of the frame) and 4 authentic images, fixed seed.

## Median pixel ROC-AUC of the evidence map against the mask (0.5 = unrelated)

| module      |     B |     R |
|:------------|------:|------:|
| ela         | 0.643 | 0.625 |
| histogram   | 0.544 | 0.545 |
| noise       | 0.442 | 0.515 |
| jpeg_ghost  | 0.643 | 0.534 |
| dct_dq      | 0.647 | 0.552 |
| edges       | 0.595 | 0.64  |
| cm_keypoint | 0.589 | 0.5   |
| cm_block    | 0.5   | 0.5   |

## Share of tampered images with pixel AUC > 0.6

| module      |    B |    R |
|:------------|-----:|-----:|
| ela         | 0.69 | 0.62 |
| histogram   | 0.31 | 0.44 |
| noise       | 0.25 | 0.38 |
| jpeg_ghost  | 0.56 | 0.38 |
| dct_dq      | 0.62 | 0.38 |
| edges       | 0.5  | 0.56 |
| cm_keypoint | 0.5  | 0.25 |
| cm_block    | 0.06 | 0.06 |

## Median pixel AUC by forgery type

| module      |   ('B', 'copy-move') |   ('R', 'copy-move') |   ('B', 'splicing') |   ('R', 'splicing') |
|:------------|---------------------:|---------------------:|--------------------:|--------------------:|
| ela         |                0.616 |                0.514 |               0.747 |               0.665 |
| histogram   |                0.488 |                0.512 |               0.618 |               0.651 |
| noise       |                0.477 |                0.576 |               0.442 |               0.484 |
| jpeg_ghost  |                0.696 |                0.528 |               0.598 |               0.551 |
| dct_dq      |                0.703 |                0.648 |               0.6   |               0.525 |
| edges       |                0.64  |                0.641 |               0.572 |               0.594 |
| cm_keypoint |                0.797 |                0.611 |               0.5   |               0.5   |
| cm_block    |                0.497 |                0.494 |               0.5   |               0.5   |

## Mean evidence on the 4 authentic images (lower = fewer false alarms)

| module      |     B |     R |
|:------------|------:|------:|
| ela         | 0.071 | 0.073 |
| histogram   | 0.102 | 0.109 |
| noise       | 0.087 | 0.088 |
| jpeg_ghost  | 0.073 | 0.076 |
| dct_dq      | 0.053 | 0.047 |
| edges       | 0.114 | 0.121 |
| cm_keypoint | 0     | 0     |
| cm_block    | 0.001 | 0.006 |
