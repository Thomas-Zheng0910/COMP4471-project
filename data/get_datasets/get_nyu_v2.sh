#!/bin/bash

# This script downloads the NYU Depth V2 dataset
# The file would be placed under ./datasets/nyu_depth_v2_labeled.mat

# Create the datasets directory if it doesn't exist
mkdir -p datasets
# Download the dataset
wget -O datasets/nyu_depth_v2_labeled.mat http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
# Verbose
echo "NYU Depth V2 dataset downloaded successfully and saved to datasets/nyu_depth_v2_labeled.mat"