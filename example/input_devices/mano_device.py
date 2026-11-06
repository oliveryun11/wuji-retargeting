"""MANO input device for teleoperation.

Feeds MANO hand model output into the standard 21-point MediaPipe interface, so
the existing retargeting / simulation / hardware pipeline is reused unchanged
(see docs/new-device-integration.md, "Category B" -- custom skeleton order).

Three source kinds, auto-detected from the file contents:

1. MANO *parameters* -- ``.npz``/``.pkl`` holding ``pose`` (+ optional ``betas``,
   ``global_orient``). Run through ``ManoLayer`` on the fly. This is what
   HaMeR/FrankMocap/dataset exports typically give you.
2. MANO *joints* -- ``joints`` of shape (N, 21, 3), already in MANO order with
   fingertips appended. Reordered only; no model file needed.
3. A live callable, via ``ManoDevice(source=fn)``, where ``fn()`` returns a pose
   vector each frame. For streaming MANO estimators.

MANO is metres and wrist-relative after conversion, which is exactly what the
retargeter wants, so there is no unit or axis fixup here.
"""

import pickle
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import numpy as np

from wuji_retargeting.mano import ManoLayer, mano_joints_to_mediapipe

from .base import InputDeviceBase

_EMPTY = np.zeros((21, 3), dtype=np.float32)


def _example_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_model_path(hand_side: str) -> Path:
    return _example_dir() / "data" / "mano" / f"MANO_{hand_side.upper()}.pkl"


class ManoDevice(InputDeviceBase):
    """Drive the Wuji Hand from MANO hand poses.

    Args:
        source: Path to a MANO sequence (``.npz`` or ``.pkl``), or a callable
            returning a pose vector per frame for live use.
        hand_side: Which hand this source describes ('left' or 'right'). The
            other hand is reported as zeros, which the teleop loop skips.
        model_path: MANO_{LEFT,RIGHT}.pkl. Defaults to
            ``example/data/mano/MANO_<SIDE>.pkl``. Only needed for parameter
            sources; joint sources skip the model entirely.
        use_pca: Pose vectors are PCA coefficients rather than 45 axis-angle
            values. Inferred from the data when a file supplies ``use_pca``.
        num_pca_comps: PCA component count when ``use_pca``.
        flat_hand_mean: See ``ManoLayer``. False (MANO default) means zero pose
            is the relaxed hand.
        betas: (10,) shape coefficients applied to every frame. A file-supplied
            ``betas`` wins over this.
        fps: Playback rate for file sources.
        loop: Restart the sequence when it ends.
        playback_speed: Multiplier on ``fps``.
    """

    def __init__(
        self,
        source: Union[str, Path, Callable[[], np.ndarray]],
        hand_side: str = "right",
        model_path: Optional[Union[str, Path]] = None,
        use_pca: bool = False,
        num_pca_comps: int = 6,
        flat_hand_mean: bool = False,
        betas: Optional[np.ndarray] = None,
        fps: float = 30.0,
        loop: bool = True,
        playback_speed: float = 1.0,
    ):
        self.hand_side = hand_side.lower()
        if self.hand_side not in ("left", "right"):
            raise ValueError(f"hand_side must be 'left' or 'right', got {hand_side}")

        self.fps = fps
        self.loop = loop
        self.playback_speed = playback_speed
        self.betas = None if betas is None else np.asarray(betas, dtype=np.float64)

        self._live_fn: Optional[Callable[[], np.ndarray]] = None
        self._joints = None          # (N, 21, 3) pre-converted MediaPipe keypoints
        self._poses = None           # (N, D) MANO pose vectors
        self._global_orient = None   # (N, 3) or None
        self._start_time = None
        self._frame = 0

        # Defaults from the constructor; a file source may override them from its
        # own metadata in _load_sequence.
        self._use_pca = use_pca
        self._num_pca_comps = num_pca_comps

        if callable(source):
            self._live_fn = source
            self.total_frames = 0
        else:
            self._load_sequence(Path(source), use_pca, num_pca_comps)

        # A model is only needed when we still have parameters to evaluate.
        self.mano = None
        if self._joints is None:
            path = Path(model_path) if model_path else _default_model_path(self.hand_side)
            self.mano = ManoLayer(
                path,
                use_pca=self._use_pca,
                num_pca_comps=self._num_pca_comps,
                flat_hand_mean=flat_hand_mean,
            )
            print(f"Loaded MANO model: {path}")

        print(f"MANO device ready: hand={self.hand_side}, "
              f"{'live callable' if self._live_fn else f'{self.total_frames} frames @ {fps} fps'}, "
              f"{'joints' if self._joints is not None else 'parameters'} source")

    # ------------------------------------------------------------------
    # InputDeviceBase
    # ------------------------------------------------------------------
    def get_fingers_data(self) -> Dict[str, np.ndarray]:
        """Return {'left_fingers', 'right_fingers'} as (21, 3) metre arrays."""
        keypoints = self._next_keypoints()
        if keypoints is None:
            return {"left_fingers": _EMPTY.copy(), "right_fingers": _EMPTY.copy()}

        keypoints = keypoints.astype(np.float32)
        out = {"left_fingers": _EMPTY.copy(), "right_fingers": _EMPTY.copy()}
        out[f"{self.hand_side}_fingers"] = keypoints
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _next_keypoints(self) -> Optional[np.ndarray]:
        if self._live_fn is not None:
            pose = self._live_fn()
            if pose is None:
                return None
            return self.mano.forward(np.asarray(pose, dtype=np.float64), betas=self.betas)

        frame = self._current_frame_index()
        if frame is None:
            return None
        self._frame = frame

        if self._joints is not None:
            return self._joints[frame]

        global_orient = None if self._global_orient is None else self._global_orient[frame]
        return self.mano.forward(
            self._poses[frame], betas=self.betas, global_orient=global_orient
        )

    def _current_frame_index(self) -> Optional[int]:
        """Index for the current wall-clock time, or None once a non-looping
        sequence has run out."""
        if self._start_time is None:
            self._start_time = time.time()
        elapsed = (time.time() - self._start_time) * self.playback_speed
        frame = int(elapsed * self.fps)
        if frame >= self.total_frames:
            if not self.loop:
                return None
            self._start_time = time.time()
            frame = 0
        return frame

    def _load_sequence(self, path: Path, use_pca: bool, num_pca_comps: int):
        if not path.is_absolute():
            path = _example_dir() / path
        if not path.exists():
            raise FileNotFoundError(f"MANO sequence not found: {path}")

        if path.suffix == ".npz":
            data = dict(np.load(path, allow_pickle=True))
        else:
            with open(path, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if not isinstance(data, dict):
                raise ValueError(
                    f"{path} must contain a dict with 'pose' or 'joints', got {type(data).__name__}"
                )

        self._use_pca = bool(data.get("use_pca", use_pca))
        self._num_pca_comps = int(data.get("num_pca_comps", num_pca_comps))

        if "joints" in data:
            joints = np.asarray(data["joints"], dtype=np.float64)
            if joints.ndim != 3 or joints.shape[1:] != (21, 3):
                raise ValueError(f"'joints' must be (N, 21, 3), got {joints.shape}")
            # Convert once up front -- it is a pure gather, so per-frame work is nil.
            self._joints = mano_joints_to_mediapipe(joints)
            self.total_frames = len(self._joints)
        else:
            key = next((k for k in ("pose", "hand_pose", "poses") if k in data), None)
            if key is None:
                raise ValueError(
                    f"{path} has none of 'joints', 'pose', 'hand_pose', 'poses'. "
                    f"Found keys: {sorted(data)}"
                )
            poses = np.asarray(data[key], dtype=np.float64)
            if poses.ndim == 1:
                poses = poses[None]
            if poses.ndim != 2:
                raise ValueError(f"'{key}' must be (N, D), got {poses.shape}")
            self._poses = poses
            self.total_frames = len(poses)
            self._use_pca = bool(data.get("use_pca", poses.shape[1] != 45))

            if "global_orient" in data:
                go = np.asarray(data["global_orient"], dtype=np.float64).reshape(-1, 3)
                if len(go) != self.total_frames:
                    raise ValueError(
                        f"'global_orient' has {len(go)} frames but pose has {self.total_frames}"
                    )
                self._global_orient = go

        if "betas" in data and self.betas is None:
            self.betas = np.asarray(data["betas"], dtype=np.float64).reshape(-1)

        if self.total_frames == 0:
            raise ValueError(f"{path} contains no frames")

    def get_progress(self) -> tuple:
        """(current_frame, total_frames), for parity with MediaPipeReplay."""
        return self._frame, self.total_frames

    def reset(self):
        """Restart playback."""
        self._start_time = None
        self._frame = 0


__all__ = ["ManoDevice"]
