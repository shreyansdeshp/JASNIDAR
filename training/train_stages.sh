#!/usr/bin/env bash
# Staged fine-tuning. Keep imgsz the same in every stage.
# Check the weights path printed at the end of each run; adjust below if yours differs.
set -e

IMGSZ=640

# Smoke test first: 3 epochs, to check paths and labels
# yolo train model=yolo11s.pt data=configs/data_visdrone.yaml imgsz=$IMGSZ epochs=3 batch=8 name=smoke

# Stage 1: VisDrone, from COCO-pretrained weights
yolo train model=yolo11s.pt data=configs/data_visdrone.yaml imgsz=$IMGSZ epochs=50 batch=16 name=s1_visdrone
yolo val model=runs/detect/s1_visdrone/weights/best.pt data=configs/data_test.yaml imgsz=$IMGSZ name=s1_visdrone_test

# Stage 2: HERIDAL tiles, from stage 1
yolo train model=runs/detect/s1_visdrone/weights/best.pt data=configs/data_heridal.yaml imgsz=$IMGSZ epochs=50 lr0=0.002 name=s2_heridal
yolo val model=runs/detect/s2_heridal/weights/best.pt data=configs/data_test.yaml imgsz=$IMGSZ name=s2_heridal_test

# Stage 3 (set aside for now): SARD, from stage 2
# yolo train model=runs/detect/s2_heridal/weights/best.pt data=configs/data_sard.yaml imgsz=$IMGSZ epochs=50 lr0=0.002 name=s3_sard

# Stage 4 (when footage exists): own 25 m dummy footage, lower learning rate
# yolo train model=runs/detect/s3_sard/weights/best.pt data=configs/data_own.yaml imgsz=$IMGSZ epochs=50 lr0=0.001 name=s4_own

# Record each stage's test-set mAP in docs/results.md after it runs.
