"""Direct joint control of a real Wuji Hand with GUI sliders — no input device.

Bypasses retargeting entirely: one slider per joint, sent straight to the hardware
backend as a qpos vector. Useful for bring-up, checking joint direction/limits, and
posing the hand without a glove or camera.

Usage:
    # Preview only — prints qpos, never touches the hand
    python slider_control.py --dry-run

    # Drive the real hand (USB Wuji Hand)
    python slider_control.py --hand right

    # Networked Wuji Hand 2
    python slider_control.py --hand right --hand-model wuji_hand_2

Sliders are in degrees and clamped to the URDF joint limits. Targets are sent at
--rate Hz with a slew limit (--max-step) so yanking a slider ramps the joint
instead of stepping it.
"""

import argparse
import sys
import tkinter as tk
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DESCRIPTION_DIR = PROJECT_ROOT / "wuji_retargeting" / "wuji-description"


def load_joint_limits(hand_model: str, hand_side: str):
    """Return [(joint_name, lower_rad, upper_rad), ...] in URDF declaration order.

    For the Wuji Hand that order is finger1..5 x joint1..4, which is exactly the
    (5, 4) layout wujihandpy expects, so no remap is needed.
    """
    asset_dir = "hand2" if hand_model == "wuji_hand_2" else "hand"
    urdf = DESCRIPTION_DIR / asset_dir / "body" / "urdf" / f"{hand_side}.urdf"
    if not urdf.exists():
        raise FileNotFoundError(
            f"{urdf} not found — run: git submodule update --init "
            f"wuji_retargeting/wuji-description"
        )
    joints = []
    for j in ET.parse(urdf).getroot().iter("joint"):
        if j.get("type") == "fixed":
            continue
        limit = j.find("limit")
        joints.append((j.get("name"), float(limit.get("lower")), float(limit.get("upper"))))
    return joints


class SliderApp:
    def __init__(self, root, joints, backend, rate_hz, max_step_rad):
        self._joints = joints
        self._backend = backend
        self._period_ms = max(1, int(1000.0 / rate_hz))
        self._max_step = max_step_rad

        # Rest pose: 0 rad where the joint allows it, otherwise the nearest limit
        # (the thumb's joint1 lower bound is above zero).
        self._rest = np.array([np.clip(0.0, lo, hi) for _, lo, hi in joints])
        # Commanded state, slew-limited toward the slider targets.
        self._current = self._rest.copy()

        self._vars = []
        self._value_labels = []

        header = tk.Frame(root)
        header.pack(fill="x", padx=8, pady=(8, 0))
        self._status = tk.Label(header, text="", anchor="w", font=("TkDefaultFont", 10))
        self._status.pack(side="left")
        tk.Button(header, text="Rest pose", command=self.reset).pack(side="right")

        grid = tk.Frame(root)
        grid.pack(fill="both", expand=True, padx=8, pady=8)

        # One column per finger (4 joints each for both hand models).
        for i, (name, lo, hi) in enumerate(joints):
            col, row = divmod(i, 4)
            if row == 0:
                tk.Label(
                    grid, text=name.rsplit("_", 1)[0], font=("TkDefaultFont", 10, "bold")
                ).grid(row=0, column=col, pady=(0, 4))

            cell = tk.Frame(grid)
            cell.grid(row=row + 1, column=col, padx=6, pady=2, sticky="n")

            var = tk.DoubleVar(value=np.degrees(self._rest[i]))
            self._vars.append(var)
            tk.Label(cell, text=name.rsplit("_", 1)[1], width=7).pack()
            tk.Scale(
                cell,
                variable=var,
                from_=np.degrees(hi),
                to=np.degrees(lo),
                resolution=0.5,
                length=180,
                showvalue=False,
                orient="vertical",
            ).pack()
            lbl = tk.Label(cell, textvariable=var, width=7)
            lbl.pack()
            self._value_labels.append(lbl)

        self._tick()

    def reset(self):
        for var, target in zip(self._vars, self._rest):
            var.set(round(np.degrees(target), 1))

    def _tick(self):
        target = np.array([np.radians(v.get()) for v in self._vars])
        # Clamp to limits defensively: a slider can only be dragged in range, but
        # this also guards the rounding at the endpoints.
        lows = np.array([lo for _, lo, _ in self._joints])
        highs = np.array([hi for _, _, hi in self._joints])
        target = np.clip(target, lows, highs)

        # Slew limit toward the target so a fast drag ramps rather than steps.
        delta = np.clip(target - self._current, -self._max_step, self._max_step)
        self._current = self._current + delta

        self._backend.send(self._current)
        moving = np.abs(target - self._current).max()
        self._status.config(
            text=f"max |target-cmd| = {np.degrees(moving):5.1f}deg"
            + ("   (ramping)" if moving > 1e-4 else "   (settled)")
        )
        self._root_after()

    def _root_after(self):
        self._status.after(self._period_ms, self._tick)


class DryRunBackend:
    """Prints instead of sending. Same interface as the hardware backends."""

    def __init__(self):
        self._n = 0

    def send(self, qpos):
        self._n += 1
        if self._n % 25 == 0:  # ~2 Hz at the default 50 Hz loop
            print("qpos(deg): " + " ".join(f"{v:6.1f}" for v in np.degrees(qpos)))

    def close(self):
        print("dry-run: nothing to close")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hand", default="right", choices=["left", "right"])
    parser.add_argument("--hand-model", default="wuji_hand",
                        choices=["wuji_hand", "wuji_hand_2"],
                        help="wuji_hand: USB via wujihandpy. wuji_hand_2: network via wuji_sdk.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print qpos instead of driving hardware")
    parser.add_argument("--hand-serial", default="", help="Wuji Hand serial number")
    parser.add_argument("--wuji-hand-2-ip", default="", help="e.g. 192.168.1.111:50001")
    parser.add_argument("--kp", type=float, default=3.0)
    parser.add_argument("--kd", type=float, default=0.1)
    parser.add_argument("--current-limit", type=float, default=1.0,
                        help="Wuji Hand 2 per-joint current limit, amps (default 1.0 — "
                             "lower than teleop's 1.5 since sliders are hand-driven)")
    parser.add_argument("--rate", type=float, default=50.0, help="Send rate, Hz")
    parser.add_argument("--max-step", type=float, default=2.0,
                        help="Slew limit in degrees per send (default 2.0)")
    args = parser.parse_args()

    joints = load_joint_limits(args.hand_model, args.hand)
    print(f"{len(joints)} joints from the {args.hand_model} {args.hand} URDF")

    if args.dry_run:
        backend = DryRunBackend()
    else:
        # Reuse the tested bring-up/teardown from the teleop script rather than
        # opening the hardware a second, divergent way.
        from teleop_real import WujiHandBackend, WujiHand2Backend

        if args.hand_model == "wuji_hand_2":
            backend = WujiHand2Backend(
                ip=args.wuji_hand_2_ip,
                kp=args.kp,
                kd=args.kd,
                current_limit=args.current_limit,
                handedness=args.hand,
            )
        else:
            backend = WujiHandBackend(hand_serial=args.hand_serial)

    root = tk.Tk()
    root.title(f"Wuji {args.hand} joint sliders"
               + (" [DRY RUN]" if args.dry_run else " [LIVE]"))
    try:
        SliderApp(root, joints, backend, args.rate, np.radians(args.max_step))
        root.mainloop()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        try:
            backend.close()
        except Exception as e:
            print(f"backend.close() raised: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
