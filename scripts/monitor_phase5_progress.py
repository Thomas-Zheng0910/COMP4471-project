#!/usr/bin/env python3
import argparse
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


METHOD_NAME = {
    "rgb_only": "RGB-only",
    "supervision_only": "Supervision-only",
    "late_fusion": "Late fusion",
    "token_fusion": "Token fusion",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase5 monitor: report epoch and ETA every minute.")
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--stop_file", type=str, required=True)
    parser.add_argument("--output_log", type=str, required=True)
    return parser.parse_args()


def to_seconds(hhmmss: str) -> Optional[int]:
    if not hhmmss:
        return None
    parts = hhmmss.split(":")
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + int(s)
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + int(s)
    return None


def tail_lines(path: Path, max_lines: int = 120) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(0, 2)
        end = f.tell()
        block = 4096
        data = b""
        while end > 0 and data.count(b"\n") <= max_lines:
            read_size = min(block, end)
            end -= read_size
            f.seek(end)
            data = f.read(read_size) + data
        return data.decode("utf-8", errors="ignore")


def parse_progress(text: str, total_epochs: int) -> Dict[str, Optional[float]]:
    # tqdm line example:
    # Epoch 2/100:  84%|...| 167/198 [02:13<00:24,  1.25batch/s]
    tqdm_pat = re.compile(
        r"Epoch\s+(\d+)/(\d+):.*?(\d+)/(\d+)\s*\[[^<]*<([^,\]]+),\s*([0-9.]+)batch/s\]",
        re.S,
    )
    avg_pat = re.compile(r"Epoch\s*\[(\d+)/(\d+)\]\s*avg\s*loss")

    info = {
        "epoch": None,
        "epoch_total": float(total_epochs),
        "batch": None,
        "batch_total": None,
        "speed": None,
        "eta_sec": None,
        "ratio": 0.0,
    }

    matches = list(tqdm_pat.finditer(text))
    if matches:
        m = matches[-1]
        epoch = int(m.group(1))
        epoch_total = int(m.group(2))
        batch = int(m.group(3))
        batch_total = int(m.group(4))
        speed = float(m.group(6)) if m.group(6) else 0.0

        info["epoch"] = float(epoch)
        info["epoch_total"] = float(epoch_total)
        info["batch"] = float(batch)
        info["batch_total"] = float(batch_total)
        info["speed"] = float(speed)

        ratio = ((epoch - 1) + (batch / max(batch_total, 1))) / max(epoch_total, 1)
        info["ratio"] = max(0.0, min(1.0, ratio))

        if speed > 0:
            remaining_batches = max(0, (epoch_total - epoch) * batch_total + (batch_total - batch))
            info["eta_sec"] = float(remaining_batches / speed)
        return info

    avg_matches = list(avg_pat.finditer(text))
    if avg_matches:
        m = avg_matches[-1]
        epoch_done = int(m.group(1))
        epoch_total = int(m.group(2))
        info["epoch"] = float(epoch_done)
        info["epoch_total"] = float(epoch_total)
        info["ratio"] = max(0.0, min(1.0, epoch_done / max(epoch_total, 1)))
    return info


def format_eta(sec: Optional[float]) -> str:
    if sec is None:
        return "N/A"
    s = int(max(0, sec))
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60
    return f"{h:02d}:{m:02d}:{ss:02d}"


def monitor_once(timestamp: str, total_epochs: int) -> Dict[str, Dict[str, Optional[float]]]:
    report = {}
    for slug, method in METHOD_NAME.items():
        log_path = Path(f"runs/phase5_{timestamp}_{slug}_gpu")
        candidates = sorted(Path("runs").glob(f"phase5_{timestamp}_{slug}_gpu*.log"))
        if not candidates:
            report[method] = {
                "status": "pending",
                "ratio": 0.0,
                "epoch": None,
                "batch": None,
                "batch_total": None,
                "eta_sec": None,
            }
            continue

        text = tail_lines(candidates[-1], max_lines=160)
        info = parse_progress(text, total_epochs)

        if "Phase 5 Ablation Complete" in text:
            status = "done"
            ratio = 1.0
        elif "Traceback (most recent call last):" in text:
            status = "error"
            ratio = info["ratio"]
        else:
            status = "running"
            ratio = info["ratio"]

        report[method] = {
            "status": status,
            "ratio": ratio,
            "epoch": info["epoch"],
            "batch": info["batch"],
            "batch_total": info["batch_total"],
            "eta_sec": info["eta_sec"],
        }

    return report


def main() -> None:
    args = parse_args()
    stop_file = Path(args.stop_file)
    output_log = Path(args.output_log)
    output_log.parent.mkdir(parents=True, exist_ok=True)

    while not stop_file.exists():
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = monitor_once(args.timestamp, args.epochs)

        ratios = [v["ratio"] for v in report.values()]
        overall_ratio = sum(ratios) / max(len(ratios), 1)

        lines = [f"[{now}] Phase5 Progress | Overall: {overall_ratio * 100:.1f}%"]
        for method, s in report.items():
            epoch_txt = "N/A"
            if s["epoch"] is not None:
                if s["batch"] is not None and s["batch_total"] is not None:
                    epoch_txt = f"{int(s['epoch'])}/{args.epochs} (batch {int(s['batch'])}/{int(s['batch_total'])})"
                else:
                    epoch_txt = f"{int(s['epoch'])}/{args.epochs}"

            lines.append(
                f"  - {method}: status={s['status']}, epoch={epoch_txt}, ETA={format_eta(s['eta_sec'])}"
            )

        text = "\n".join(lines)
        print(text, flush=True)
        with output_log.open("a", encoding="utf-8") as f:
            f.write(text + "\n")

        time.sleep(max(1, args.interval))


if __name__ == "__main__":
    main()
