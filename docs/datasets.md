# Datasets

Datasets are not stored in this repo (size, licences). Run `data_prep/download_datasets.sh`, then the conversion scripts.

| Dataset | Source | Licence | Used for | Processing |
|---|---|---|---|---|
| VisDrone | https://github.com/VisDrone/VisDrone-Dataset | Non-commercial | Stage 1, general drone-view pre-training | Keep pedestrian and people only, map to class 0, drop ignored regions |
| HERIDAL | https://zenodo.org/records/5662351 | _check_ | Stage 2, small camouflaged people | Upscale 1.5-2x, tile at 640 px, 20% overlap, keep boxes with 40%+ visible |
| SARD | Kaggle: nikolasgegenava/sard-search-and-rescue | _check_ | Stage 3, SAR poses | All pose classes to class 0, tile or train at larger imgsz |
| Own footage | Collected at 25 m with dummies | Ours | Stage 4 and test set | Label as class 0 |

## Rules

- One class: `0 = person`.
- Split by group (source image, video or flight), never at random per image.
- The test set is fixed before training and never trained on.
- Keep about 10% empty tiles as negatives.

## Shared converted data

_Add a Kaggle or Drive link here once the converted version exists._
