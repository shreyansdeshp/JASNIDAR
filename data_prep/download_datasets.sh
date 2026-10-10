#!/usr/bin/env bash
# Downloads the raw datasets into datasets/raw/. Nothing here is committed to git.
# Needs a Kaggle account and API token at ~/.kaggle/kaggle.json (never commit it).
set -e

mkdir -p datasets/raw/{sard,heridal,visdrone}

# SARD (about 4.4 GB)
kaggle datasets download -d nikolasgegenava/sard-search-and-rescue -p datasets/raw/sard --unzip

# HERIDAL
kaggle datasets download -d imadeddinelassakeur/heridal -p datasets/raw/heridal --unzip

# VisDrone: download the DET train and val zips from the official repo
# https://github.com/VisDrone/VisDrone-Dataset and unzip into datasets/raw/visdrone
echo "Download VisDrone manually (see link above), then unzip into datasets/raw/visdrone"
