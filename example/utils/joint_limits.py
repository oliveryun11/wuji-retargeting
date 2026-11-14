"""Joint limits for the Wuji hands, read from the URDF in wuji-description.

Shared by slider_control.py (slider ranges) and teleop_real.py (command clipping),
so both read the same limits the same way.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
