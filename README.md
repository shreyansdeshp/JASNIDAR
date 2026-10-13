# NIDAR RescueSwarm (PS1)

Autonomous multi-drone system that searches a 10-hectare flood-affected area, detects and geotags survivors, and drops 200 g kit parcels. Built for NIDAR 2026-27 (Drone Federation India / MeitY SwaYaan).

**Team:** Shreyans, Jaanya
**Mentor:** _to be added_

## Status

Work in progress. See `docs/results.md` for the latest numbers.

## System overview

1. **Detection:** YOLO11s fine-tuned in stages on VisDrone, HERIDAL and SARD, then on our own 25 m footage.
2. **Confirmation:** multi-frame tracking, so no manual tagging is needed.
3. **Geotagging:** pixel to GPS from altitude, heading and camera field of view.
4. **Drop:** centre over the survivor and release autonomously.

Details are in `docs/architecture.md`.

## Repository layout

| Folder | Contents |
|---|---|
| `configs/` | Dataset yaml files and model definitions (`configs/models/`) |
| `data_prep/` | Download, conversion, tiling and split scripts |
| `training/` | Staged training and evaluation |
| `inference/` | Detection, tracking and geolocation |
| `drone/` | Autopilot, mission planning, area split |
| `docs/` | Architecture, datasets, results |
| `notebooks/` | Analysis and visualisation |
| `samples/` | A few example images with labels |

## Setup

```bash
git clone <this-repo-url>
cd nidar-rescueswarm
pip install -r requirements.txt
```

Datasets are **not** stored in this repo. See `docs/datasets.md` and run `data_prep/download_datasets.sh`.

## Training

See `training/train_stages.sh`. Stages: VisDrone, then HERIDAL, then SARD, then own footage. Every stage is scored on one fixed test set that is never trained on.

## Credits

- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) (AGPL-3.0)
- [VisDrone](https://github.com/VisDrone/VisDrone-Dataset), Tianjin University
- HERIDAL, University of Split (we use the [Roboflow re-export](https://universe.roboflow.com/new-workspace-qsuag/heridal-t428z); original: https://zenodo.org/records/5662351)
- SARD (Search and Rescue dataset, Kaggle)
- [AlienSight](https://github.com/7amzaGH/UAV-SAR-Human-Detection-and-Geolocation) (MIT), reference for the geolocation approach

## Licence

To be decided with the mentor. Note that Ultralytics is AGPL-3.0.
