# Controlling a Wuji Hand — agent handoff

How to drive real Wuji Hand hardware from Python. Written for an agent starting in a
different repo. Everything here was verified on 2026-08-06 against a right Wuji Hand,
firmware 1.2.1, `wujihandpy` 1.8.0.

Read the whole doc before running anything. There are two traps that cost hours if you
hit them blind, and both are in the first two sections.

---

## 1. Clear ROS from the environment, or nothing works

If the machine has ROS (Jazzy) sourced, it shadows the project's Python **twice**:

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH <your-python> <script>
```

- `PYTHONPATH` puts ROS's site-packages first, so `import pinocchio` gets ROS's build.
- `LD_LIBRARY_PATH` puts ROS's `libeigenpy.so` first. The cmeel `pin` wheels link with
  `DT_RUNPATH`, which **loses to** `LD_LIBRARY_PATH` (unlike `DT_RPATH`), so the wrong
  library wins.

The symptom is `undefined symbol: EIGENPY_ARRAY_APIPyArray_RUNTIME_VERSION`. That is
library shadowing. **Do not** try to fix it by changing `pin` / `eigenpy` versions.

A *different* error — `module compiled using NumPy 1.x cannot be run in NumPy 2.x` — is a
genuine version problem (eigenpy wheels are built against NumPy 1.x). Both can appear in
one session. Read the message before acting.

Pure hardware control (`wujihandpy` only, no retargeting) does not import pinocchio, so it
survives a dirty environment — but prefix the command anyway; it costs nothing.

## 2. Install `wujihandpy`

It is on PyPI. The catch: this project's `.venv` has **no `pip`**, so use `uv`:

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH uv pip install --python .venv/bin/python wujihandpy
```

Confirm the device is attached before anything else:

```bash
lsusb | grep -i wuji     # expect: ID 0483:2000 STMicroelectronics WUJIHAND
```

`wuji_sdk` is a **different package** — needed only for the Wuji Hand 2 (networked) and the
Wuji Glove, not for the USB Wuji Hand. It is not on PyPI; it ships with Wuji Studio.

---

## 3. Files to copy

### Minimum for direct joint control (recommended — start here)

Copy **two things**:

| Copy | From | Why |
|---|---|---|
| `example/slider_control.py` | this repo | GUI sliders → joint targets. Self-contained apart from the backend import. |
| `wuji_retargeting/wuji-description/hand/body/urdf/{right,left}.urdf` | this repo's submodule | Joint names and limits. Parsed at startup; do not hardcode limits. |

`slider_control.py` imports its hardware backend from `teleop_real.py`. If you don't want
that repo's whole import chain (see below), replace the import with this — extracted
verbatim from `WujiHandBackend` in `example/teleop_real.py:130`, which is the code that
actually ran:

```python
import time
import numpy as np
import wujihandpy

class WujiHandBackend:
    def __init__(self, hand_serial: str = ""):
        self._hand = (
            wujihandpy.Hand(serial_number=hand_serial) if hand_serial else wujihandpy.Hand()
        )
        self._hand.write_joint_enabled(True)
        self._controller = self._hand.realtime_controller(
            enable_upstream=False,
            filter=wujihandpy.filter.LowPass(cutoff_freq=5.0),
        )
        time.sleep(0.5)

    def send(self, qpos: np.ndarray) -> None:
        self._controller.set_joint_target_position(qpos.reshape(5, 4))

    def close(self) -> None:
        if getattr(self, "_hand", None) is not None:
            self._hand.write_joint_enabled(False)
```

That is the entire hardware interface. Twenty floats in, hand moves.

### If you also want retargeting (human hand pose → joint angles)

Copying piecemeal will not work — the chain is wide. Take the whole repo and the
submodule:

```bash
git clone <this-repo>
git submodule update --init wuji_retargeting/wuji-description   # empty by default
```

`example/teleop_real.py` imports, unconditionally: the `wuji_retargeting` package
(pinocchio + nlopt), `example/utils/config_paths.py`, `example/input_devices/visionpro.py`
(which needs `avp_stream` **even if you never use a VisionPro**), `mediapipe_replay.py`,
and `mano_device.py`. Camera devices are wrapped in `try/except ImportError` and are
optional. To exercise its argparse offline without hardware, stub `wujihandpy` with an
empty module.

Also note: `example/data/*.pkl` are **Git-LFS pointers**. A 132-byte "pickle" that fails
with `invalid load key, 'v'` is a pointer file, not corruption.

---

## 4. The contract

`qpos` is **20 floats, radians**, in URDF declaration order:

```
finger1_joint1..4, finger2_joint1..4, ... finger5_joint1..4
```

`finger1` is the **thumb**. That order reshapes directly to the `(5, 4)` array
`set_joint_target_position` expects — no remap for the Wuji Hand.

Limits, read from `hand/body/urdf/right.urdf` (radians):

| joint | lower | upper | note |
|---|---|---|---|
| `finger1_joint1` (thumb) | **0.0475** | 1.6033 | lower bound is **above zero** — 0.0 is out of range |
| `finger1_joint2` | -0.1387 | 0.9324 | |
| `fingerN_joint2` (N=2..5) | -0.3700 | 0.3700 | abduction/spread — narrow, easy to over-drive |
| `fingerN_joint3`, `joint4` | ~-0.47 | ~1.56 | flexion |
| `fingerN_joint1` (N=2..5) | ~-0.16 | ~1.56 | flexion |

**Parse these from the URDF at runtime.** They vary by hand and firmware; the table is
orientation, not a source of truth. A neutral rest pose is 0.0 per joint clipped into
limits.

The Wuji Hand **2** is a different hand: networked over Ethernet via `wuji_sdk`, MIT
control mode, and its URDF declares fingers in a different order than its MJCF — so it
needs a name-based joint remap (`qpos_reorder_perm` in `example/utils/config_paths.py`).
Do not assume identity there. Its assets are under `wuji-description/hand2/`.

## 5. Bring-up sequence and safety

Order matters. `slider_control.py` follows it; keep it if you write your own.

1. **Check handedness before energizing.** `Hand.read_handedness()` returns an int;
   `wujihandpy.Hand.Side.Right == 0`, `Left == 1`. Driving a left hand with right-hand
   limits is silently wrong.
2. **Do all fallible setup first** — parse the URDF, build the GUI — and open the hardware
   **last**. Energizing early means a later failure leaves a live hand.
3. **Open the hardware inside the `try`**, so the `finally` always reaches `close()`.
   Ctrl+C must de-energize.
4. **Clip every command to the joint limits** even when the UI already constrains them.
5. **Slew-limit the target.** `slider_control.py` sends at 50 Hz with a 2°/send cap, so a
   yanked slider ramps instead of stepping. The 5 Hz low-pass in the controller helps but
   is not a substitute — it shapes the command, it does not bound the step.
6. **Never `SIGKILL` a running controller.** The joints stay enabled holding their last
   target. Close the window or Ctrl+C.

Two things the SDK does **not** give you, so don't plan around them:

- **No read-back of the enable state.** `write_joint_enabled` exists; there is no
  `read_joint_enabled`. A clean exit is your only evidence the hand de-energized.
- **Per-joint sign/offset calibration is deliberately not exposed** in this repo
  (`example/teleop_real.py:96`). An unvalidated sign flip goes straight to the controller
  and cannot be previewed in simulation. Do not add it as a user-facing knob.

## 6. Verify without touching hardware

`slider_control.py --dry-run` swaps in a printing backend with the same
`{send(qpos), close()}` interface and never opens the device. Use it to confirm joint
count, ordering, and rate before energizing. Confirmed working output:

```
20 joints from the wuji_hand right URDF
qpos(deg):    2.9   -0.1    0.0    0.2   -0.1    0.2  ...
```

`2.9°` on the first joint is the thumb sitting at its non-zero lower limit — that is
correct, not drift.

## 7. Known-good invocation

```bash
cd example
env -u PYTHONPATH -u LD_LIBRARY_PATH ../.venv/bin/python slider_control.py --hand right
```

Flags: `--dry-run`, `--hand {left,right}`, `--hand-model {wuji_hand,wuji_hand_2}`,
`--hand-serial`, `--rate` (Hz, default 50), `--max-step` (deg/send, default 2.0).

Expected startup log — if you don't see the firmware line, the device is not connected:

```
20 joints from the wuji_hand right URDF
[wuji] [info] SDK v1.8.0 initialized
[wuji] [info] Using firmware version: 1.2.1, SN: LQSQJR.260504.005
```

## 8. First thing to check on real hardware

Move one joint at a time and confirm direction and range. A joint that moves the wrong way
or hits a hard stop early means the URDF and the physical hand disagree — stop and
recalibrate rather than compensating in software.
