#!/bin/bash

# This script downloads the Booster dataset (testing - 1.8 GB)
# Booster MONO: https://amsacta.unibo.it/id/eprint/7161/
# Download link: https://amsacta.unibo.it/id/eprint/7161/2/booster_mono.zip
# The file would be downloaded and extracted under ./datasets/booster_mono/
# [OPTIONAL] full dataset: https://1drv.ms/f/s!AgV49D1Z6rmGgY8k0W0bijzTx40BIg?e=HtHnDU

# Create the datasets directory if it doesn't exist
mkdir -p datasets
# Download the dataset
wget -O datasets/booster_mono.zip https://amsacta.unibo.it/id/eprint/7161/2/booster_mono.zip
# Unzip the dataset
unzip datasets/booster_mono.zip -d datasets/booster_mono
# Remove the zip file
rm datasets/booster_mono.zip
# Verbose
echo "Booster MONO dataset downloaded successfully and extracted to datasets/booster_mono/"

# NOTE: file structure
# ./booster_mono/
#     test_mono/
#         (It contains 21 testing sequences, 
#         each contained in a separate folder:)
#         Microwave1/ - folder 
#         Window3/ - folder
#         .../
#             (Each sequence folder contains (a subset of) 
#             the following folders and files:)
#             camera_00/ - folder (has .png files)
#                 (It contains left camera images, 
#                 under different illumination.)
#                 0000.png
#                 ...
#             calib_00-02.xml
#                 (Calibration file containing intrinsic/extrinsic 
#                 parameters.)
