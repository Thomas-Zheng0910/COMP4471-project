#!/usr/bin/env python3
"""
Download TODD (Toronto Transparent Object Depth Dataset)
https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/ZJJAJ3

TODD dataset contains:
- train.7z: Training set with 4 types of backgrounds
- test.7z: Test set
- val.7z: Validation set

Usage:
    python get_todd.py
"""

import os
import sys
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
import ssl


# Borealis Dataverse file IDs for TODD dataset
FILE_IDS = {
    "train.7z": 200799,
    "test.7z": 200798,
    "val.7z": 200797,
}

BASE_URL = "https://borealisdata.ca/api/access/datafile/"


def get_project_root() -> Path:
    """Get project root directory."""
    script_path = Path(__file__).resolve()
    return script_path.parent.parent.parent


def download_file(file_id: int, output_path: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file from Borealis Dataverse.
    
    Args:
        file_id: Dataverse file ID
        output_path: Where to save the file
        chunk_size: Download chunk size
    
    Returns:
        True if successful, False otherwise
    """
    url = f"{BASE_URL}{file_id}"
    
    print(f"  URL: {url}")
    print(f"  Output: {output_path}")
    
    # Create SSL context that allows us to connect
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    # Set up request with headers
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.0"
    }
    
    try:
        req = Request(url, headers=headers)
        
        with urlopen(req, context=ssl_context, timeout=60) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            
            with open(output_path, 'wb') as f:
                downloaded = 0
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Progress bar
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        bar_length = 40
                        filled = int(bar_length * downloaded / total_size)
                        bar = '█' * filled + '░' * (bar_length - filled)
                        print(f"\r  [{bar}] {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')
                
                print()  # New line after progress bar
        
        # Verify download
        if output_path.exists() and output_path.stat().st_size > 0:
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Downloaded: {size_mb:.2f} MB")
            return True
        else:
            print(f"  ✗ Download failed: file is empty")
            return False
            
    except HTTPError as e:
        print(f"  ✗ HTTP Error {e.code}: {e.reason}")
        if e.code == 403:
            print(f"    Access forbidden. The dataset may require authentication.")
        elif e.code == 404:
            print(f"    File not found. Check if the file ID is correct.")
        return False
    except URLError as e:
        print(f"  ✗ URL Error: {e.reason}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def extract_7z(archive_path: Path) -> bool:
    """Extract 7z archive if tools are available."""
    import subprocess
    
    # Try 7z or 7za
    for cmd in ['7z', '7za']:
        try:
            result = subprocess.run(
                [cmd, 'x', '-y', str(archive_path)],
                capture_output=True,
                text=True,
                cwd=str(archive_path.parent)
            )
            if result.returncode == 0:
                print(f"  ✓ Extracted with {cmd}")
                return True
        except FileNotFoundError:
            continue
    
    print(f"  ! 7z not found. Install p7zip to extract: sudo apt-get install p7zip-full")
    return False


def main():
    """Main download function."""
    print("=" * 60)
    print("TODD Dataset Downloader")
    print("Toronto Transparent Object Depth Dataset")
    print("=" * 60)
    
    # Setup paths
    project_root = get_project_root()
    todd_dir = project_root / "datasets" / "todd"
    todd_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownload directory: {todd_dir}")
    print("")
    
    # Download each file
    success_count = 0
    for filename, file_id in FILE_IDS.items():
        print(f"[{list(FILE_IDS.keys()).index(filename) + 1}/{len(FILE_IDS)}] Downloading {filename}...")
        
        output_path = todd_dir / filename
        
        # Skip if already exists
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  ! File already exists ({size_mb:.2f} MB), skipping download")
            success_count += 1
        else:
            if download_file(file_id, output_path):
                success_count += 1
        
        print()
    
    # Summary
    print("=" * 60)
    print("Download Summary")
    print("=" * 60)
    
    if success_count == len(FILE_IDS):
        print(f"✓ All {len(FILE_IDS)} files downloaded successfully!")
    else:
        print(f"! Downloaded {success_count}/{len(FILE_IDS)} files")
    
    print(f"\nLocation: {todd_dir}")
    print("\nContents:")
    for item in sorted(todd_dir.iterdir()):
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  {item.name:20s} {size_mb:8.2f} MB")
    
    # Try to extract
    print("\n" + "=" * 60)
    print("Extracting Archives")
    print("=" * 60)
    
    for filename in FILE_IDS.keys():
        archive_path = todd_dir / filename
        if archive_path.exists():
            print(f"\nExtracting {filename}...")
            extract_7z(archive_path)
    
    print("\n" + "=" * 60)
    print("TODD Dataset Setup Complete!")
    print("=" * 60)
    
    # List final directory structure
    print("\nDataset structure:")
    for item in sorted(todd_dir.iterdir()):
        if item.is_dir():
            num_files = len(list(item.rglob("*")))
            print(f"  📁 {item.name}/ ({num_files} items)")
        else:
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  📄 {item.name} ({size_mb:.2f} MB)")
    
    return 0 if success_count == len(FILE_IDS) else 1


if __name__ == "__main__":
    sys.exit(main())
