#!/bin/bash
# Starting script for the demo Flask app

set -e

# Cache directory
# torch.hub download location
export TORCH_HOME="./cache"
# Set hugging face and transformers cache location
export TRANSFORMERS_CACHE="./cache"
export HF_HOME="./cache"
export HF_HUB_CACHE="./cache"

# Set DEVICE and CHECKPOINT_PATH environment variables for the app
export DEVICE="cuda:0"
export CHECKPOINT_PATH="..."

# Run the Flask app
python -m demo.app
