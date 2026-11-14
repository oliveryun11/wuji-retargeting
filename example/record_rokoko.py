"""Record Rokoko glove hands to a .pkl that `teleop_sim.py --play` can replay.

Needs, all on this machine:
  - rokoko-sdk running, with both gloves on Wi-Fi
  - rkk-hand-solver --no-driver --emit-openxr running

Usage (from example/):
    python record_rokoko.py --seconds 10 --out data/rokoko_sample.pkl
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from input_devices.rokoko import RokokoHandStream


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=10.0, help="How long to record")
    parser.add_argument("--rate", type=float, default=30.0, help="Frames saved per second")
    parser.add_argument("--out", default="data/rokoko_sample.pkl",
                        help="Output .pkl (relative paths are relative to example/)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12277)
    args = parser.parse_args()

    stream = RokokoHandStream(args.host, args.port)
    zeros = np.zeros((21, 3), dtype=np.float32)
    frames = []
    try:
        # Wait for the first hand to arrive.
        deadline = time.monotonic() + 5.0
        while stream.get("left") is None and stream.get("right") is None:
            if time.monotonic() > deadline:
                raise SystemExit("No hand data after 5 s. Is rkk-hand-solver running with "
                                 "--emit-openxr, and do both gloves show UDP in `rokoko device list`?")
            time.sleep(0.05)

        print(f"Recording {args.seconds:.0f} s: open and close your hands, move each finger...")
        start = time.monotonic()
        while (t := time.monotonic() - start) < args.seconds:
            left, right = stream.get("left"), stream.get("right")
            frames.append({
                "t": t,
                "left_fingers": zeros if left is None else left,
                "right_fingers": zeros if right is None else right,
            })
            time.sleep(1.0 / args.rate)
    finally:
        stream.close()

    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parent / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(frames, f)

    print(f"Saved {len(frames)} frames to {out}")
    for side in ("left", "right"):
        hands = [fr[f"{side}_fingers"] for fr in frames if fr[f"{side}_fingers"].any()]
        if not hands:
            print(f"  {side}: no data")
            continue
        # Sanity check: wrist (0) to middle fingertip (12) is ~15-20 cm on an open adult hand.
        reach_cm = 100 * np.linalg.norm(hands[-1][12] - hands[-1][0])
        print(f"  {side}: data in {len(hands)}/{len(frames)} frames, "
              f"wrist->middle tip {reach_cm:.1f} cm in the last frame")


if __name__ == "__main__":
    main()
