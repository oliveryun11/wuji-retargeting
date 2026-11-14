"""Rokoko Smartgloves input, via the Rokoko Device SDK hand solver.

Data path:
    gloves --Wi-Fi--> rokoko-sdk (driver, RGMP on :12276)
           --> rkk-hand-solver --no-driver --emit-openxr (RGMP on :12277)
           --> this module

For each glove, the solver publishes a `joints_openxr` group: 26 OpenXR hand
joints, each a TRANSFORM (x, y, z, qx, qy, qz, qw) relative to `xr_base`. That
frame is fixed to the glove, with the wrist at the origin. Positions are in
metres. We keep only the positions, map the 26 joints to the 21 MediaPipe
points that wuji-retargeting expects, and hold the latest frame per hand.
"""

import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .base import InputDeviceBase

# Rokoko's Python RGMP client ships inside the SDK, not on PyPI.
_SDK_PY = Path(os.environ.get("ROKOKO_SDK_HOME", "/opt/rokoko-device-sdk")) / "examples" / "python"
if str(_SDK_PY) not in sys.path:
    sys.path.append(str(_SDK_PY))
from rgmp_core.client import RgmpClient  # noqa: E402
from rgmp_core.layout import read_slot  # noqa: E402

SOLVER_PORT = 12277
GROUP_NAME = "joints_openxr"

# MediaPipe index -> OpenXR joint. An OpenXR joint's position is at the base of
# its bone, so "proximal" sits at the knuckle (MCP), "intermediate" at the PIP,
# and so on. xr_palm and the four finger metacarpals are not used.
MEDIAPIPE_FROM_OPENXR = [
    "xr_wrist",                                                                              # 0
    "xr_thumb_metacarpal", "xr_thumb_proximal", "xr_thumb_distal", "xr_thumb_tip",           # 1-4
    "xr_index_proximal", "xr_index_intermediate", "xr_index_distal", "xr_index_tip",         # 5-8
    "xr_middle_proximal", "xr_middle_intermediate", "xr_middle_distal", "xr_middle_tip",     # 9-12
    "xr_ring_proximal", "xr_ring_intermediate", "xr_ring_distal", "xr_ring_tip",             # 13-16
    "xr_little_proximal", "xr_little_intermediate", "xr_little_distal", "xr_little_tip",     # 17-20
]

# Used when the stream definition doesn't say which hand a glove is.
# Device IDs come from `rokoko device list`.
KNOWN_GLOVES = {3872341061: "right", 1948277338: "left"}


def _side_from_definition(definition: dict) -> Optional[str]:
    """Look for 'left'/'right' in device_info under a key mentioning side or hand."""
    info = definition.get("device_info") or {}
    for key, value in info.items():
        if (isinstance(value, str) and value.lower() in ("left", "right")
                and any(word in key.lower() for word in ("side", "hand"))):
            return value.lower()
    return KNOWN_GLOVES.get(definition["device_id"])


class RokokoHandStream:
    """Connects to the hand solver and keeps the latest (21, 3) hand per side.

    Reading happens on a background thread. `get()` returns a copy of the
    latest hand, or None if that hand has gone quiet for longer than `max_age`.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = SOLVER_PORT):
        self._lock = threading.Lock()
        self._latest: Dict[str, tuple] = {}  # side -> (monotonic time, (21, 3) array)
        self._plans: Dict[int, tuple] = {}   # device_id -> (side, group_id, slots)

        self._client = RgmpClient(host, port)
        self._client.on_definition = self._on_definition
        self._client.on_data = self._on_data
        self._client.on_disconnect = self._on_disconnect
        self._client.on_error = lambda e: print(f"[rokoko] stream error: {e}")
        self._client.__enter__()  # connect now, so a missing solver fails here, loudly
        self._thread = threading.Thread(target=self._client.run, daemon=True)
        self._thread.start()

    # Callbacks, run on the background thread

    def _on_definition(self, definition, layout):
        device_id = definition["device_id"]
        group_id = next((g for g, n in layout.group_names.items() if n == GROUP_NAME), None)
        if group_id is None:
            return  # not a solved hand, or the solver was started without --emit-openxr
        by_name = {slot.entry.get("target_frame"): slot for slot in layout.groups[group_id]}
        missing = [n for n in MEDIAPIPE_FROM_OPENXR if n not in by_name]
        if missing:
            print(f"[rokoko] device {device_id}: missing joints {missing}, ignoring it")
            return
        side = _side_from_definition(definition)
        if side is None:
            print(f"[rokoko] device {device_id}: unknown hand side, ignoring it "
                  f"(device_info={definition.get('device_info')})")
            return
        self._plans[device_id] = (side, group_id, [by_name[n] for n in MEDIAPIPE_FROM_OPENXR])
        print(f"[rokoko] device {device_id}: {side} hand "
              f"(device_info={definition.get('device_info')})")

    def _on_data(self, device_id, group_id, ts_us, payload, layout):
        plan = self._plans.get(device_id)
        if plan is None or group_id != plan[1]:
            return
        side, _, slots = plan
        points = np.array([read_slot(slot, payload)[:3] for slot in slots], dtype=np.float32)
        with self._lock:
            self._latest[side] = (time.monotonic(), points)

    def _on_disconnect(self, device_id):
        plan = self._plans.pop(device_id, None)
        if plan is not None:
            with self._lock:
                self._latest.pop(plan[0], None)
            print(f"[rokoko] device {device_id} ({plan[0]}) disconnected")

    # Public API

    def get(self, side: str, max_age: float = 0.2) -> Optional[np.ndarray]:
        with self._lock:
            entry = self._latest.get(side)
        if entry is None or time.monotonic() - entry[0] > max_age:
            return None
        return entry[1].copy()

    def close(self):
        self._client.close()


class RokokoDevice(InputDeviceBase):
    """wuji-retargeting input device. A hand with no fresh data comes back as zeros."""

    def __init__(self, host: str = "127.0.0.1", port: int = SOLVER_PORT, max_age: float = 0.2):
        self.stream = RokokoHandStream(host, port)
        self.max_age = max_age

    def get_fingers_data(self) -> Dict[str, np.ndarray]:
        data = {}
        for side in ("left", "right"):
            points = self.stream.get(side, self.max_age)
            data[f"{side}_fingers"] = points if points is not None else np.zeros((21, 3), np.float32)
        return data
