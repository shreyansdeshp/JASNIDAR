# Architecture

## Model

YOLO11s (Ultralytics). Model definition lives in `configs/models/`.

To add it: copy `ultralytics/cfg/models/11/yolo11.yaml` from the Ultralytics clone to `configs/models/yolo11s.yaml`, with a comment at the top crediting Ultralytics (AGPL-3.0).

A P2 variant (`yolo11s-p2.yaml`) is added only if evaluation shows misses on people under about 16 px.

## Training pipeline

VisDrone, then HERIDAL, then SARD, then own 25 m footage. Each stage starts from the previous `best.pt` at a lower learning rate, and is scored on the fixed test set.

## Onboard pipeline

Detect, confirm across frames with a tracker, geotag, descend and centre, drop. No manual intervention (each costs 50 points).

## Design decisions

_Add diagrams and the reasoning behind each decision here._
