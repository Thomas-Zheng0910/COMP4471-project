"""
Scan ToM depth maps for obviously broken files (unreadable, near-constant,
mostly-zero, low range, very dark, tiny) and move them — along with their
corresponding RGB images and challenging/ counterparts — to a _rejected/
folder so they are excluded from training.

Uses the same fast statistical checks as tmp_scan_depth.py.

Usage
-----
    # Dry run — report only:
    python -m data.preprocess.detect_and_remove_bad_depth --dry_run

    # Move bad depth maps (+ RGB + challenging copies) to _rejected/:
    python -m data.preprocess.detect_and_remove_bad_depth
"""

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFile

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOM_ROOT = "datasets/Diffusion4RobustDepth/ToM"
DEPTH_SUFFIX = "_depth_anything.png"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
UINT16_MAX = 65534

# Thresholds for flagging (same as tmp_scan_depth.py)
MIN_UNIQUE = 100           # depth map should have rich gradation
MAX_ZERO_FRAC = 0.5        # more than 50 % zeros is suspicious
MIN_RANGE = 1000           # uint16 range should use at least this much of 0-65534
MAX_CONST_FRAC = 0.95      # >95 % pixels same value = likely broken
MIN_MAX_VAL = 100          # max pixel value too low = very dark / empty
MIN_DIM = 10               # width or height below this = tiny


# ---------------------------------------------------------------------------
# Scan a single depth map
# ---------------------------------------------------------------------------

def scan_depth(dpath: Path) -> Tuple[Dict, List[str]]:
    """
    Read a depth map and return (stats_dict, list_of_issue_tags).
    If the file is unreadable the stats dict will be mostly None and
    the issue list will contain 'unreadable'.
    """
    issues: List[str] = []
    stats: Dict = {"path": str(dpath)}

    try:
        raw = np.array(Image.open(dpath), dtype=np.float32)
    except Exception as e:
        stats["error"] = str(e)
        issues.append("unreadable")
        return stats, issues

    h, w = raw.shape[:2]
    n_pixels = h * w
    unique_vals = np.unique(raw)
    n_unique = len(unique_vals)
    vmin, vmax = float(raw.min()), float(raw.max())
    val_range = vmax - vmin
    zero_frac = float(np.sum(raw == 0) / n_pixels)
    mode_val = int(np.bincount(raw.astype(np.int64).ravel()).argmax())
    mode_frac = float(np.sum(raw == mode_val) / n_pixels)

    stats.update({
        "shape": (h, w), "min": vmin, "max": vmax, "range": val_range,
        "unique": n_unique, "zero_frac": zero_frac,
        "mode_val": mode_val, "mode_frac": mode_frac,
    })

    if n_unique < MIN_UNIQUE:
        issues.append("low_unique")
    if zero_frac > MAX_ZERO_FRAC:
        issues.append("high_zero_frac")
    if val_range < MIN_RANGE:
        issues.append("low_range")
    if mode_frac > MAX_CONST_FRAC:
        issues.append("near_constant")
    if vmax < MIN_MAX_VAL:
        issues.append("very_dark")
    if h < MIN_DIM or w < MIN_DIM:
        issues.append("tiny_image")

    return stats, issues


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rgb_for_depth(depth_path: Path) -> Optional[Path]:
    """Return the source RGB image that corresponds to a depth map."""
    stem = depth_path.stem.replace("_depth_anything", "")
    for ext in IMAGE_EXTS:
        candidate = depth_path.parent / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def _challenging_counterpart(depth_path: Path, root: Path) -> Optional[Path]:
    """Return the matching depth file under challenging/ (if it exists)."""
    parent = depth_path.parent
    if parent.name == "easy":
        chal = parent.parent / "challenging" / depth_path.name
        return chal if chal.exists() else None
    elif parent.name == "challenging":
        easy = parent.parent / "easy" / depth_path.name
        return easy if easy.exists() else None
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Scan ToM depth maps for broken files and move them to _rejected/."
    )
    p.add_argument("--root", type=str, default=TOM_ROOT)
    p.add_argument("--min_unique", type=int, default=MIN_UNIQUE)
    p.add_argument("--max_zero_frac", type=float, default=MAX_ZERO_FRAC)
    p.add_argument("--min_range", type=float, default=MIN_RANGE)
    p.add_argument("--max_const_frac", type=float, default=MAX_CONST_FRAC)
    p.add_argument("--min_max_val", type=float, default=MIN_MAX_VAL)
    p.add_argument("--min_dim", type=int, default=MIN_DIM)
    p.add_argument("--dry_run", action="store_true",
                   help="Report only; do not move any files.")
    p.add_argument("--reject_dir", type=str, default="_rejected",
                   help="Sub-directory name under each scene dir for rejected files.")
    p.add_argument("--exclusion_list_path", type=str,
                   default="datasets/Diffusion4RobustDepth/ToM/bad_depth_exclusion.json",
                   help="Path to save/read the JSON exclusion list.")
    p.add_argument("--from_exclusion_list", action="store_true",
                   help="Skip scanning; read the existing exclusion list JSON and "
                        "move the listed files directly.")
    return p.parse_args()


def _move_from_exclusion_list(args):
    """Read the existing exclusion list JSON and move listed files to _rejected/."""
    root = Path(args.root).expanduser()
    excl_path = Path(args.exclusion_list_path)
    if not excl_path.exists():
        raise FileNotFoundError(
            f"Exclusion list not found: {excl_path}\n"
            "Run a --dry_run scan first to generate it."
        )

    with open(excl_path) as f:
        data = json.load(f)
    depth_rels = data.get("excluded_depths", [])
    print(f"Loaded {len(depth_rels)} entries from {excl_path}")

    if args.dry_run:
        print(f"[DRY RUN] Would move {len(depth_rels)} depth maps "
              f"(+ RGB) to {args.reject_dir}/.")
        return

    moved = 0
    for rel in depth_rels:
        dpath = root / rel
        if not dpath.exists():
            continue
        reject_dir = dpath.parent / args.reject_dir
        reject_dir.mkdir(parents=True, exist_ok=True)

        shutil.move(str(dpath), str(reject_dir / dpath.name))
        moved += 1

        rgb = _rgb_for_depth(dpath)
        if rgb is not None and rgb.exists():
            shutil.move(str(rgb), str(reject_dir / rgb.name))
            moved += 1

    print(f"Moved {moved} files to '{args.reject_dir}/' sub-directories.")
    print("To undo: move files from _rejected/ back to their parent directories.")


def main():
    args = parse_args()

    if args.from_exclusion_list:
        _move_from_exclusion_list(args)
        return

    # Override module-level thresholds with CLI values
    global MIN_UNIQUE, MAX_ZERO_FRAC, MIN_RANGE, MAX_CONST_FRAC, MIN_MAX_VAL, MIN_DIM
    MIN_UNIQUE = args.min_unique
    MAX_ZERO_FRAC = args.max_zero_frac
    MIN_RANGE = args.min_range
    MAX_CONST_FRAC = args.max_const_frac
    MIN_MAX_VAL = args.min_max_val
    MIN_DIM = args.min_dim

    root = Path(args.root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"ToM root not found: {root}")

    # Collect all depth maps
    depth_files = sorted(root.rglob(f"*{DEPTH_SUFFIX}"))
    print(f"Found {len(depth_files)} depth maps to scan\n")

    issues_all: Dict[str, List] = defaultdict(list)
    stats_all: List[Dict] = []
    bad_depths: List[Path] = []

    for i, dpath in enumerate(depth_files):
        if i % 500 == 0:
            print(f"  Scanning {i}/{len(depth_files)} ...", flush=True)

        stats, issues = scan_depth(dpath)

        rel = str(dpath.relative_to(root))
        stats["path"] = rel
        stats_all.append(stats)

        if issues:
            bad_depths.append(dpath)
            for tag in issues:
                issues_all[tag].append(rel)

    # ---- Also check for missing depth maps ----
    print(f"\n  Checking for missing depth maps ...")
    missing: List[str] = []
    for img_path in sorted(root.rglob("*")):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        if DEPTH_SUFFIX.replace(".png", "") in img_path.stem:
            continue
        expected = img_path.with_name(img_path.stem + DEPTH_SUFFIX)
        if not expected.exists():
            missing.append(str(img_path.relative_to(root)))

    # ---- Report ----
    print(f"\n{'=' * 70}")
    print(f"SCAN COMPLETE: {len(depth_files)} depth maps, "
          f"{len(bad_depths)} bad ({100 * len(bad_depths) / max(len(depth_files), 1):.1f}%)")
    print(f"{'=' * 70}\n")

    if not any(issues_all.values()) and not missing:
        print("ALL CLEAR — no issues found!")
    else:
        for issue_type, entries in sorted(issues_all.items()):
            print(f"\n--- {issue_type.upper()} ({len(entries)} files) ---")
            for entry in entries[:20]:
                print(f"  {entry}")
            if len(entries) > 20:
                print(f"  ... and {len(entries) - 20} more")

        if missing:
            print(f"\n--- MISSING DEPTH MAPS ({len(missing)} images) ---")
            for m in missing[:20]:
                print(f"  {m}")
            if len(missing) > 20:
                print(f"  ... and {len(missing) - 20} more")

    # Global stats summary
    if stats_all:
        good = [s for s in stats_all if "error" not in s]
        if good:
            ranges = [s["range"] for s in good]
            uniques = [s["unique"] for s in good]
            zero_fracs = [s["zero_frac"] for s in good]
            print(f"\n--- GLOBAL STATS ---")
            print(f"  Range:     min={min(ranges):.0f}  median={np.median(ranges):.0f}  max={max(ranges):.0f}")
            print(f"  Unique:    min={min(uniques)}  median={np.median(uniques):.0f}  max={max(uniques)}")
            print(f"  Zero frac: min={min(zero_fracs):.3f}  median={np.median(zero_fracs):.3f}  max={max(zero_fracs):.3f}")

    # ---- Save exclusion list (JSON) ----
    exclusion_entries: List[str] = []
    for dpath in bad_depths:
        rel = str(dpath.relative_to(root))
        exclusion_entries.append(rel)
        # Also include the challenging / easy counterpart
        chal = _challenging_counterpart(dpath, root)
        if chal is not None:
            exclusion_entries.append(str(chal.relative_to(root)))

    exclusion_data = {
        "description": "Auto-generated list of broken depth maps to exclude from training.",
        "thresholds": {
            "min_unique": MIN_UNIQUE,
            "max_zero_frac": MAX_ZERO_FRAC,
            "min_range": MIN_RANGE,
            "max_const_frac": MAX_CONST_FRAC,
            "min_max_val": MIN_MAX_VAL,
            "min_dim": MIN_DIM,
        },
        "n_excluded": len(set(exclusion_entries)),
        "excluded_depths": sorted(set(exclusion_entries)),
    }
    excl_path = Path(args.exclusion_list_path)
    excl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(excl_path, "w") as f:
        json.dump(exclusion_data, f, indent=2)
    print(f"\nExclusion list saved to {excl_path} "
          f"({exclusion_data['n_excluded']} entries)")

    # ---- Move bad files ----
    if not bad_depths:
        print("\nNothing to move.")
        return

    if args.dry_run:
        print(f"\n[DRY RUN] Would move {len(bad_depths)} bad depth maps "
              f"(+ RGB + challenging copies) to {args.reject_dir}/. "
              f"Re-run without --dry_run to apply.")
        return

    moved = 0
    for dpath in bad_depths:
        reject_dir = dpath.parent / args.reject_dir
        reject_dir.mkdir(parents=True, exist_ok=True)

        # Move the depth map
        shutil.move(str(dpath), str(reject_dir / dpath.name))
        moved += 1

        # Move the corresponding RGB image
        rgb = _rgb_for_depth(dpath)
        if rgb is not None and rgb.exists():
            shutil.move(str(rgb), str(reject_dir / rgb.name))
            moved += 1

        # Move the challenging (or easy) counterpart depth + RGB
        chal_depth = _challenging_counterpart(dpath, root)
        if chal_depth is not None and chal_depth.exists():
            chal_reject = chal_depth.parent / args.reject_dir
            chal_reject.mkdir(parents=True, exist_ok=True)
            shutil.move(str(chal_depth), str(chal_reject / chal_depth.name))
            moved += 1
            chal_rgb = _rgb_for_depth(chal_depth)
            if chal_rgb is not None and chal_rgb.exists():
                shutil.move(str(chal_rgb), str(chal_reject / chal_rgb.name))
                moved += 1

    print(f"\nMoved {moved} files to '{args.reject_dir}/' sub-directories.")
    print("To undo: move files from _rejected/ back to their parent directories.")


if __name__ == "__main__":
    main()
