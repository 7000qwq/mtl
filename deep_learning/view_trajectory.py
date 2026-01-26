import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

# python view_trajectory.py --scan_all --root flight_data_random

def load_positions(file_path: Path, start: int = 0, end: int = None, stride: int = 1) -> Tuple[np.ndarray, str, str]:
    """Load positions from a trajectory json file."""
    with open(file_path, "r") as f:
        data = json.load(f)

    traj = data.get("trajectory", [])
    if end is None or end > len(traj):
        end = len(traj)

    points = []
    for step in traj[start:end:stride]:
        pos = step.get("position", {})
        points.append((pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0)))

    intent = data.get("intent_type", "unknown")
    run_number = data.get("run_number", "?")
    return np.array(points, dtype=float), intent, run_number


def get_intent_color(intent: str) -> str:
    # Assign a color deterministically; fallback to gray
    static_map = {
        "takeoff": "tab:green",
        "hover": "tab:blue",
        "straight_line": "tab:orange",
        "turn": "tab:red",
        "landing": "tab:purple",
        "z_scan": "tab:cyan",
    }
    return static_map.get(intent, "tab:gray")


def plot_trajectory(points: np.ndarray, intent: str, run_number: str, title: str, save_path: Path = None):
    """Plot 3D trajectory with XY projection."""
    color = get_intent_color(intent)
    fig = plt.figure(figsize=(12, 5))

    ax3d = fig.add_subplot(121, projection="3d")
    ax3d.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=1.5, marker="o", markersize=2)
    ax3d.scatter(points[0, 0], points[0, 1], points[0, 2], color="tab:green", label="Start", s=40)
    ax3d.scatter(points[-1, 0], points[-1, 1], points[-1, 2], color="tab:red", label="End", s=40)
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_title(f"3D trajectory ({intent}, run {run_number})")
    ax3d.legend()
    ax3d.view_init(elev=25, azim=-60)

    ax2d = fig.add_subplot(122)
    ax2d.plot(points[:, 0], points[:, 1], color=color, linewidth=1.5)
    ax2d.scatter(points[0, 0], points[0, 1], color="tab:green", label="Start", s=30)
    ax2d.scatter(points[-1, 0], points[-1, 1], color="tab:red", label="End", s=30)
    ax2d.set_xlabel("X")
    ax2d.set_ylabel("Y")
    ax2d.set_title("XY projection")
    ax2d.legend()
    ax2d.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize 3D trajectory json files.")
    parser.add_argument("--file", help="Path to a single trajectory json file (e.g., flight_data_random/hover/hover_*.json)")
    parser.add_argument("--root", default="flight_data_random", help="Root directory to scan all trajectory json files")
    parser.add_argument("--scan_all", action="store_true", help="Scan all json files under root and plot each")
    parser.add_argument("--start", type=int, default=0, help="Start index in the trajectory")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive) in the trajectory")
    parser.add_argument("--stride", type=int, default=1, help="Subsample stride for the trajectory")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save a single plot as PNG")
    parser.add_argument("--save_dir", type=str, default=None, help="Optional directory to save all plots when scan_all is used")

    args = parser.parse_args()

    if args.scan_all:
        root = Path(args.root)
        if not root.exists():
            raise FileNotFoundError(f"Root directory not found: {root}")
        files = sorted(root.rglob("*.json"))
        if not files:
            raise FileNotFoundError(f"No json files found under {root}")
        print(f"Found {len(files)} json files under {root}")
        save_dir = Path(args.save_dir) if args.save_dir else None
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

        for fp in files:
            points, intent, run_number = load_positions(fp, start=args.start, end=args.end, stride=args.stride)
            if points.size == 0:
                print(f"Skip empty trajectory: {fp}")
                continue
            title = f"{fp.name} | intent: {intent}"
            save_path = (save_dir / f"{fp.stem}.png") if save_dir else None
            print(f"Plotting {fp} (intent={intent}, run={run_number})")
            plot_trajectory(points, intent, run_number, title, save_path)
        return

    if not args.file:
        raise ValueError("Please provide --file for single plot or --scan_all to process all files.")

    file_path = Path(args.file)
    if not file_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {file_path}")

    points, intent, run_number = load_positions(file_path, start=args.start, end=args.end, stride=args.stride)
    if points.size == 0:
        raise ValueError("No trajectory points found in the file.")

    title = f"{file_path.name} | {intent}"
    save_path = Path(args.save) if args.save else None
    plot_trajectory(points, intent, run_number, title, save_path)


if __name__ == "__main__":
    main()
