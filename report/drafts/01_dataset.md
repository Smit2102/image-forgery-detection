# III. Dataset

*Source numbers: `results/phase1/dataset_stats.{md,json}`. Figures: `results/phase1/fig_*.png`.*

## A. Composition

We use CASIA v2.0 [ref], which has 12,614 images: 7,491 authentic and 5,123 tampered. The tampered images were made with editing software as either **copy-move** (a region duplicated within one image) or **splicing** (a region pasted from a different image). There are nine scene categories (animal, architecture, art, character, indoor, nature, plant, scene, texture). Most images are small: 85.8% of authentic and 64.7% of tampered images are 384×256 or 256×384 pixels.

**The forgery type is corrected by majority vote.** It is normally read from the file-name letter (`Tp_S` = copy-move, `Tp_D` = splicing), which gives 3,274 copy-move and 1,849 splicing images. For 115 images this letter conflicts with other evidence: the letter in the mask's file name, and whether the host and donor ids are equal (equal ids mean copy-move). We take the majority of these three votes. This changes 99 labels (60 splicing → copy-move, 39 copy-move → splicing). The corrected type is used for stratification and all per-type results; the original letter is kept in the manifest.

**Table I: Class and file format (corrected type)** (`fig_class_format.png`)

| Class | Images | JPEG | TIFF | BMP |
|---|---:|---:|---:|---:|
| Authentic | 7,491 | 7,437 | 0 | 54 |
| Copy-move | 3,295 | 967 | 2,328 | 0 |
| Splicing | 1,828 | 1,097 | 731 | 0 |

## B. Ground-truth masks

Each tampered image has a binary mask marking the content taken from the donor. Masks could not be matched to images by file name alone: 142 mask names differ from their image name. Of these:
- 97 differ only in the S/D letter
- 44 differ in the host or donor id (10 of them use the placeholder id `xxx00000`)
- 1 is named `…_gt3.png`

We match masks on the trailing tamper id, which is unique across the dataset. Every tampered image then has exactly one mask.

Of the 5,123 masks, 19 are unusable for pixel-level scoring:
- 5 are transposed (width and height swapped: four 256×384 masks for 384×256 images, and one 638×336 mask for a 336×638 image)
- 12 have other sizes: 9 differ from the image by 1 to 11 pixels, and 3 by 125 to 200 pixels
- 2 cover more than 95% of the image

These 19 images remain in the classification experiments but are excluded from localisation scoring. That leaves **5,104 valid masks**.

We checked mask correctness against an independent signal: where the tampered image differs from its background image (blurred absolute grey-level difference > 20). The background image is the named authentic image that best matches the area *outside* the mask. It is the host for 5,029 masks and the second-named image for 5, where the ids are swapped (§C). This could be checked for 5,034 masks. The median share of mask pixels that differ from the background (recall) is 0.885, and the median share of changed pixels inside the mask (precision) is 0.991. So the masks are generally accurate. We flag 52 masks with recall < 0.2 for inspection but do not remove them.

**Tampered regions are small** (`fig_tampered_area.png`). The median tampered area is 3.1% of the image, the 10th percentile is 0.5%, and the 90th percentile is 28%. We report localisation in three area buckets: <1% (1,114 images), 1–5% (2,023) and >5% (1,967).

## C. Which named image is the background?

A tampered file name `Tp_<S|D>_…_<id1>_<id2>_<n>` names two authentic images. We measured, for every tampered image, the share of pixels within 10 grey levels of each named image (sizes must match).

For the 1,778 splicing-named images whose ids differ:
- `id1` supplies most pixels in 1,689
- `id2` in 20
- neither in 69 (27 because the host has a different size, 42 because neither image matches more than half of the pixels)

For the 44 copy-move-named images with differing ids, the counts are 37, 3 and 4.

We therefore treat `id1` as the **host**. In 23 images the second-named image supplies most pixels:
- **18 large pastes:** the mask, which marks the pasted region, covers 52–96% of the frame.
- **5 swapped ids:** the pasted region is small (0.8–6.8% of the image), and the unmasked background matches `id2`, so the two ids appear to be swapped in the file name.

All 23 are additionally grouped with the second-named image (Section IV).

## D. Pitfalls that affect evaluation

1. **File format tracks the label.** 60% of tampered images are uncompressed TIFF, but no authentic image is.
2. **JPEG settings track the label.** Among JPEG files, 89.1% of authentic images use a standard libjpeg (IJG) quantisation table, compared with 0% of tampered images. Their tables look like those written by editing software. The closest IJG quality factor is mostly 84 or 90 for authentic images and 93 for tampered ones (`fig_jpeg_quality.png`).
3. **Image size tracks the label.** Tampered images are more often 640×480 or 800×600.
4. **Duplicates.** 196 authentic images appear twice under different ids (e.g. `cha10xxx` = `cha20xxx`, identical pixels). About 480 tampered images are near-identical to an authentic image other than their named host.
5. **Label noise in names.** See the type correction in §A.

Points 1–3 mean that a classifier can separate the classes without looking at any forensic evidence. Section IV measures this directly.
