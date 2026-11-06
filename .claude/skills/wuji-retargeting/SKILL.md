---
name: wuji-retargeting
description: Work on the wuji-retargeting hand-pose pipeline - running the sim or teleop scripts, adding an input device (MANO, glove, camera, mocap), converting a skeleton to the 21-point MediaPipe layout, or tuning retargeting parameters. Covers the ROS/pinocchio environment trap that blocks every run, the assets that are not actually in the repo, MANO's joint-order pitfalls, and how to verify a new input path instead of eyeballing it.
---

# wuji-retargeting

Maps human hand keypoints to Wuji Hand joint angles. Everything funnels through one
contract, so most work is conversion, not algorithms.

## Run nothing before fixing the environment

ROS Jazzy shadows the project's Python **twice**, and both must be cleared:

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH ./.venv/bin/python <script>
```

- `PYTHONPATH` puts `/opt/ros/jazzy/.../python3.12/site-packages` first, so `import
  pinocchio` gets ROS's build instead of the venv's.
- `LD_LIBRARY_PATH` puts ROS's `libeigenpy.so` first. cmeel wheels (`pin`) link with
  `DT_RUNPATH`, which **loses to** `LD_LIBRARY_PATH` (unlike `DT_RPATH`), so the wrong
  library wins and you get `undefined symbol: EIGENPY_ARRAY_APIPyArray_RUNTIME_VERSION`.

That symbol error means library shadowing, not a version mismatch — do not "fix" it by
churning `pin`/`eigenpy` versions. A *different* error, `module compiled using NumPy 1.x
cannot be run in NumPy 2.x`, is the real version problem: eigenpy wheels are built
against NumPy 1.x. Both can appear in one session; read the message before acting.

## Assets that are not in the repo

A fresh clone cannot run. Check these first when something is "missing":

- **`wuji-description` submodule** holds every URDF and MJCF. Empty by default:
  `git submodule update --init wuji_retargeting/wuji-description`.
- **`example/data/*.pkl` are Git-LFS pointers** (`*.pkl filter=lfs` in `.gitattributes`).
  A 132-byte "pickle" that fails with `invalid load key, 'v'` is a pointer file. Without
  `git-lfs`, fetch through the LFS batch API and verify the SHA-256 against the pointer.
- **`teleop_sim.py` imports `avp_stream` unconditionally** (the VisionPro device is a
  top-level import), so even replay mode needs it. `teleop_real.py` likewise needs
  `wujihandpy` at import — stub it with an empty module to test its argparse offline.
- MANO models are licence-restricted. Keep them gitignored; never commit them.

## The data contract

Every input device returns `{"left_fingers", "right_fingers"}`, each `(21, 3)`,
**metres, wrist-relative, MediaPipe landmark order**, zeros when a hand is absent. Match
that and the whole retarget/sim/hardware stack works unchanged. `docs/new-device-integration.md`
is the authority; follow it rather than inventing a path.

Register a new device in **both** `teleop_sim.py` and `teleop_real.py` — `device_map`,
the `--input` choices, a `--<name>` shortcut, and the auto-config switch. They are
parallel by design and drift is a real bug.

## MANO specifics

Already implemented in `wuji_retargeting/mano.py`; this is why it looks the way it does.

- **Joint order is wrist, index, middle, pinky, ring, thumb.** Pinky before ring. This
  is the single most common mismapping bug.
- **Fingertips are not skeleton joints.** MANO has 16 joints; tips are mesh vertices
  (smplx: thumb 744, index 320, middle 443, ring 554, pinky 671), conventionally
  appended as joints 16–20 in thumb, index, middle, ring, pinky order.
- **MANO v1.2 pkls are chumpy-pickled.** chumpy will not install on modern NumPy.
  Unpickle with stub classes, then pull arrays out: `Ch` holds data in `.x`; `Select` is
  a lazy gather, `source.ravel()[obj.idxs].reshape(obj.preferred_shape)` (that is how
  `shapedirs` is stored). `J_regressor` is scipy sparse — call `.toarray()`.
- The forward pass is ~40 lines of numpy LBS and runs in 0.2 ms. **Do not add torch** for
  this; the package is numpy/scipy/nlopt and needs no autograd.
- Global orient is irrelevant to retargeting: `apply_mediapipe_transformations`
  re-derives the wrist frame from keypoints 0/5/9 and discards world rotation. Support it
  for correct vertices, but never debug orientation by tweaking it.

## Derive constants from the model, never from memory

Both MANO facts above were confirmed against the model data, and one remembered constant
was wrong when checked. Use the data:

- **Chain identity from rest-pose geometry** — the longest chain is middle, the shortest
  and most lateral is pinky, the chain whose root sits ~4 cm from the wrist is thumb.
- **Tip vertices from LBS skinning weights** — the vertex farthest from each chain's
  distal joint. This yields middle 444 / pinky 672, ~3 mm from the smplx constants: same
  fingertip, different vertex. Prefer the ecosystem constants for interoperability, but
  know they are a convention, not ground truth.

## Verify a new input path with two tests, not by looking at it

1. **Round trip** — permute known-good MediaPipe keypoints into your device's order,
   push them back through your converter, assert exact recovery *and* bit-identical qpos
   versus the native path. Catches an inconsistent permutation.
2. **Anatomical semantics** — round trip passes even if the whole mapping is rotated, so
   also assert on real device output: tip distances rank middle > index > ring > pinky,
   thumb tip nearest the wrist, each chain monotonically outward. Catches a *wrong*
   permutation.

Then assert on the retargeted qpos: zero joint-limit violation (allow float32 epsilon —
`_run_optimization` casts to float32, so ~5e-9 rad of slack is rounding, not a breach),
every joint moving, and bounded frame-to-frame steps.

## Tune from measurements

`segment_scaling` is `[PIP, DIP, TIP]` multipliers on wrist→joint target vectors. Derive
a starting point by measuring per-segment lengths of your source against a config that
is already tuned, and multiply — do not guess.

Do not assume an existing config's oddities are sensor compensation. The AVP config's
boosted pinky looks like tracker bias, but substituting a pure size ratio made mean IK
cost **worse** (2.09 → 3.82): those values encode robot fit. Test before "correcting".

**Watch for flat-basin solver flips.** On a smooth MANO sweep the pinky DIP jumped 58°
between frames while the input moved 0.18 cm and the cost did not improve — SLSQP hopping
between equivalent solutions, not tracking. `norm_delta` is the lever; sweep it and read
the trade directly:

| norm_delta | worst step | mean cost |
|-----------|-----------|-----------|
| 0.04 | 58.2° | 2.091 |
| 0.20 | 32.6° | 2.093 |
| 0.40 | 22.3° | 2.096 |
| 0.80 | 15.6° | 2.104 |

0.4 removed most of the jump for 0.24% cost. Report a jump like this rather than hiding
it behind `lp_alpha`; filtering is the second line of defence, not the diagnosis.

## Facts worth not re-deriving

- `retarget()` does **not** mutate the caller's array. `apply_mediapipe_transformations`
  allocates fresh, so `_apply_offset`'s in-place writes never escape. (Verified — it
  reads like an aliasing bug and is not one.)
- `mjpython` in the README is macOS-only; on Linux use plain `python`.
- MuJoCo passive-viewer scripts exit code 0 when the window closes — that is a clean
  exit, not a crash. To test a teleop loop without a window, stub
  `mujoco.viewer.launch_passive` with a fake whose `is_running()` counts down.
- End-to-end MANO→qpos runs ~220 Hz; the IK, not the hand model, is the cost.
