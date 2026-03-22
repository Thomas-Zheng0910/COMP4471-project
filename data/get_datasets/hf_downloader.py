# Script to download dataset from hugging face
# requires huggingface_hub library, install with pip install huggingface_hub

import os
import time
import logging
from huggingface_hub import snapshot_download
from requests.exceptions import RequestException
import argparse

REPO_ID = "..."
TARGET_PATH = "..."

# Recommended environment settings
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0" 
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"  # Increased timeout for large files

logging.basicConfig(level=logging.WARNING)

def safe_download():
    max_retries = 10
    base_delay = 5
    
    print(f"Starting download for {REPO_ID} to {TARGET_PATH}...")
    
    for attempt in range(max_retries):
        try:
            # snapshot_download is the preferred way to download entire datasets
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                local_dir=TARGET_PATH,
                # Ignores git metadata and hidden files automatically
                ignore_patterns=[".*", ".git*", "README.md"], 
                max_workers=8  # Parallel downloads for speed
            )
            print("Download completed successfully!")
            break
        except (RequestException, Exception) as e:
            # Exponential backoff: 5s, 10s, 20s...
            delay = base_delay * (2 ** attempt)
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Please check your network.")

if __name__ == "__main__":

    # Optional: parse command line arguments for repo_id and target_path
    parser = argparse.ArgumentParser(description = "Download dataset from Hugging Face Hub")
    parser.add_argument("--repo_id", help = "ID of the repository to download")
    parser.add_argument("--target_path", help = "Path where the dataset should be downloaded")
    # Parse
    args = parser.parse_args()
    # Assign if provided
    if args.repo_id:
        REPO_ID = args.repo_id
    if args.target_path:
        TARGET_PATH = args.target_path

    start_time = time.time()
    try:
        safe_download()
    except KeyboardInterrupt:
        print("\nDownload interrupted by user.")
    
    total_min = (time.time() - start_time) / 60
    print(f"Total time elapsed: {total_min:.2f} minutes")