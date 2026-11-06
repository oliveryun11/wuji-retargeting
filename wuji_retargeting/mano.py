"""MANO hand model support for Wuji Hand retargeting.

MANO (Romero et al., SIGGRAPH Asia 2017) is the standard parametric hand model.
This module converts MANO output into the 21-point MediaPipe layout that the rest
of this package speaks, so a MANO source plugs into the existing pipeline:

    MANO (betas, pose) -> 21 joints -> MediaPipe order -> Retargeter -> Wuji Hand qpos

Two entry points, depending on what you already have:

- ``mano_joints_to_mediapipe(joints)`` -- you already have MANO joint positions
  (from smplx, manopth, HaMeR, a dataset, ...). Pure reordering, no model file.
- ``ManoLayer`` -- you have MANO *parameters*. Runs the model forward in numpy
  (no torch, no chumpy) and returns MediaPipe-ordered keypoints.

Joint ordering
--------------
MANO's 16 skeleton joints run wrist, index, middle, **pinky**, ring, thumb --
pinky comes before ring, which is the usual source of mismapped-finger bugs.
The 5 fingertips are not skeleton joints; they are read off mesh vertices and
conventionally appended at indices 16..20 in thumb, index, middle, ring, pinky
order (the smplx convention). ``MANO_TO_MEDIAPIPE`` encodes both facts.

Both orderings were verified against the MANO v1.2 model data rather than taken
on trust: chain identity from rest-pose geometry (chain 7-9 is the shortest and
most lateral => pinky; chain 4-6 is the longest => middle), and the fingertip
vertices from the LBS skinning weights (the vertex farthest from each chain's
distal joint).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Union

import numpy as np

# MANO's 16 skeleton joints, in model order.
MANO_JOINT_NAMES = (
    "wrist",
    "index1", "index2", "index3",
    "middle1", "middle2", "middle3",
    "pinky1", "pinky2", "pinky3",
    "ring1", "ring2", "ring3",
    "thumb1", "thumb2", "thumb3",
)

# Fingertip mesh vertices, appended as joints 16..20 in this order.
# Values are the smplx constants (smplx/vertex_ids.py), which is what the MANO
# ecosystem overwhelmingly emits -- keeping them makes our joints directly
# comparable to smplx output. Deriving tips from the skinning weights instead
# gives middle=444 and pinky=672; those sit 3.4mm and 3.0mm from these, i.e. the
# same fingertip, different vertex on it. Override via ManoLayer(tip_vertices=...).
MANO_TIP_VERTICES = {
    "thumb": 744,
    "index": 320,
    "middle": 443,
    "ring": 554,
    "pinky": 671,
}
_TIP_ORDER = ("thumb", "index", "middle", "ring", "pinky")

# MediaPipe index -> index into MANO's 21 (16 skeleton + 5 appended tips).
# MediaPipe order is wrist, then thumb/index/middle/ring/pinky, 4 points each.
MANO_TO_MEDIAPIPE = np.array([
    0,                  # 0  wrist
    13, 14, 15, 16,     # 1-4   thumb  CMC, MCP, IP, TIP
    1, 2, 3, 17,        # 5-8   index  MCP, PIP, DIP, TIP
    4, 5, 6, 18,        # 9-12  middle
    10, 11, 12, 19,     # 13-16 ring
    7, 8, 9, 20,        # 17-20 pinky
], dtype=np.int64)

NUM_MANO_JOINTS = 16
NUM_MEDIAPIPE_KEYPOINTS = 21


def mano_joints_to_mediapipe(joints: np.ndarray) -> np.ndarray:
    """Reorder MANO joints into the MediaPipe 21-point layout.

    Args:
        joints: (..., 21, 3) MANO joints with fingertips already appended at
            16..20 in thumb, index, middle, ring, pinky order.

    Returns:
        (..., 21, 3) keypoints in MediaPipe order, wrist-relative, same units in
        as out (MANO is metres, which is what the retargeter expects).

    Raises:
        ValueError: if the trailing dims are not (21, 3). A (16, 3) input is
            rejected explicitly -- fingertips cannot be recovered from the
            skeleton alone, they come from mesh vertices.
    """
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim < 2 or joints.shape[-1] != 3:
        raise ValueError(f"expected (..., 21, 3), got {joints.shape}")
    if joints.shape[-2] == NUM_MANO_JOINTS:
        raise ValueError(
            "got 16 MANO skeleton joints; fingertips (5) must be appended first. "
            "Fingertips are mesh vertices, not skeleton joints -- use ManoLayer, "
            "or index MANO_TIP_VERTICES into your posed vertices."
        )
    if joints.shape[-2] != NUM_MEDIAPIPE_KEYPOINTS:
        raise ValueError(f"expected (..., 21, 3), got {joints.shape}")

    reordered = joints[..., MANO_TO_MEDIAPIPE, :]
    # Wrist-relative, matching the convention of every other input device here.
    return reordered - reordered[..., 0:1, :]


def _resolve_chumpy(obj):
    """Return a plain ndarray from a value in a MANO pkl.

    MANO v1.2 pkls are pickled with chumpy objects, which cannot be unpickled
    without chumpy installed (and chumpy does not work on modern numpy). We
    unpickle with stub classes and then pull the arrays back out here:
    ``Ch`` holds its data in ``.x``, and ``Select`` is a lazy gather of ``.a``
    through ``.idxs`` (this is how shapedirs is stored -- the MANO hand
    shapedirs are a slice of the SMPL+H ones).
    """
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "idxs") and hasattr(obj, "a"):  # chumpy Select
        source = np.asarray(_resolve_chumpy(obj.a))
        return source.ravel()[np.asarray(obj.idxs)].reshape(obj.preferred_shape)
    if hasattr(obj, "x"):  # chumpy Ch
        return np.asarray(obj.x)
    if hasattr(obj, "toarray"):  # scipy sparse (J_regressor)
        return obj.toarray()
    return np.asarray(obj)


class _ManoUnpickler(pickle.Unpickler):
    """Unpickler that tolerates chumpy's absence by stubbing its classes."""

    def find_class(self, module: str, name: str):
        if module.startswith("chumpy"):
            return type(name, (object,), {})
        return super().find_class(module, name)


def load_mano_model(model_path: Union[str, Path]) -> dict:
    """Load a MANO_LEFT.pkl / MANO_RIGHT.pkl into plain numpy arrays.

    Args:
        model_path: Path to the official MANO v1.2 pkl.

    Returns:
        dict with v_template (778,3), shapedirs (778,3,10), posedirs (778,3,135),
        J_regressor (16,778), weights (778,16), parents (16,), hands_mean (45,),
        hands_components (45,45), faces (1538,3).
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"MANO model not found: {model_path}. Download mano_v1_2.zip from "
            "https://mano.is.tue.mpg.de (registration required) and extract "
            "models/MANO_LEFT.pkl and models/MANO_RIGHT.pkl."
        )
    with open(model_path, "rb") as f:
        raw = _ManoUnpickler(f, encoding="latin1").load()

    parents = np.asarray(raw["kintree_table"][0], dtype=np.int64).copy()
    parents[0] = -1  # stored as uint32 max

    return {
        "v_template": _resolve_chumpy(raw["v_template"]).astype(np.float64),
        "shapedirs": _resolve_chumpy(raw["shapedirs"]).astype(np.float64),
        "posedirs": _resolve_chumpy(raw["posedirs"]).astype(np.float64),
        "J_regressor": _resolve_chumpy(raw["J_regressor"]).astype(np.float64),
        "weights": _resolve_chumpy(raw["weights"]).astype(np.float64),
        "parents": parents,
        "hands_mean": _resolve_chumpy(raw["hands_mean"]).astype(np.float64),
        "hands_components": _resolve_chumpy(raw["hands_components"]).astype(np.float64),
        "faces": np.asarray(raw["f"], dtype=np.int64),
    }


def _rodrigues(axis_angle: np.ndarray) -> np.ndarray:
    """Axis-angle (K, 3) -> rotation matrices (K, 3, 3)."""
    theta = np.linalg.norm(axis_angle, axis=1, keepdims=True)
    # At theta=0 the axis is undefined; the 1e-12 floor makes the formula fall
    # through to identity (sin/cos terms vanish) instead of producing NaN.
    axis = axis_angle / np.maximum(theta, 1e-12)
    cos, sin = np.cos(theta)[..., None], np.sin(theta)[..., None]
    K = np.zeros((axis.shape[0], 3, 3), dtype=np.float64)
    K[:, 0, 1], K[:, 0, 2] = -axis[:, 2], axis[:, 1]
    K[:, 1, 0], K[:, 1, 2] = axis[:, 2], -axis[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -axis[:, 1], axis[:, 0]
    eye = np.broadcast_to(np.eye(3), K.shape)
    return eye + sin * K + (1.0 - cos) * (K @ K)


class ManoLayer:
    """Numpy MANO forward pass producing MediaPipe-ordered keypoints.

    Deliberately torch-free: this package's dependency set is numpy/scipy/nlopt,
    and a retargeting pipeline does not need autograd. One forward pass is a few
    hundred microseconds, well under the optimizer's own budget.

    Args:
        model_path: Path to MANO_LEFT.pkl or MANO_RIGHT.pkl.
        use_pca: If True, ``pose`` is PCA coefficients; if False, 45 axis-angle
            values (15 joints x 3). Matches smplx's flag of the same name.
        num_pca_comps: Number of PCA components when ``use_pca``.
        flat_hand_mean: If False (the MANO/smplx default), ``hands_mean`` is
            added to the pose, so zero pose is the natural relaxed hand. If True,
            zero pose is a flat hand.
        tip_vertices: Override the fingertip vertex indices (see MANO_TIP_VERTICES).
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        use_pca: bool = True,
        num_pca_comps: int = 6,
        flat_hand_mean: bool = False,
        tip_vertices: Optional[dict] = None,
    ):
        self.model = load_mano_model(model_path)
        self.use_pca = use_pca
        self.num_pca_comps = num_pca_comps
        self.flat_hand_mean = flat_hand_mean

        tips = dict(MANO_TIP_VERTICES if tip_vertices is None else tip_vertices)
        missing = set(_TIP_ORDER) - set(tips)
        if missing:
            raise ValueError(f"tip_vertices missing fingers: {sorted(missing)}")
        self.tip_vertex_ids = np.array([tips[f] for f in _TIP_ORDER], dtype=np.int64)

        self.num_betas = self.model["shapedirs"].shape[-1]

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.forward(*args, **kwargs)

    def forward(
        self,
        pose: Optional[np.ndarray] = None,
        betas: Optional[np.ndarray] = None,
        global_orient: Optional[np.ndarray] = None,
        return_verts: bool = False,
    ):
        """Run MANO forward.

        Args:
            pose: (num_pca_comps,) PCA coefficients, or (45,) axis-angle.
            betas: (10,) shape coefficients. Defaults to the mean shape.
            global_orient: (3,) axis-angle wrist rotation. Irrelevant to
                retargeting -- ``apply_mediapipe_transformations`` re-derives the
                wrist frame from the keypoints and discards global rotation --
                but supported so posed vertices come out right.
            return_verts: Also return the (778, 3) posed mesh.

        Returns:
            (21, 3) keypoints in MediaPipe order, wrist-relative, metres.
            If ``return_verts``, a (keypoints, vertices) tuple.
        """
        m = self.model

        betas = np.zeros(self.num_betas) if betas is None else np.asarray(betas, dtype=np.float64)
        if betas.shape != (self.num_betas,):
            raise ValueError(f"betas must be ({self.num_betas},), got {betas.shape}")

        full_pose = self._resolve_hand_pose(pose)
        global_orient = (
            np.zeros(3) if global_orient is None
            else np.asarray(global_orient, dtype=np.float64).reshape(3)
        )
        pose_aa = np.concatenate([global_orient, full_pose]).reshape(NUM_MANO_JOINTS, 3)

        # Shape blend shapes, then the rest-pose skeleton for this shape.
        v_shaped = m["v_template"] + m["shapedirs"] @ betas
        J_rest = m["J_regressor"] @ v_shaped

        # Pose blend shapes, driven by the non-root rotations flattened as (R - I).
        rot_mats = _rodrigues(pose_aa)
        pose_feature = (rot_mats[1:] - np.eye(3)).reshape(-1)
        v_posed = v_shaped + m["posedirs"] @ pose_feature

        joints, skinning_mats = self._rigid_transform(rot_mats, J_rest, m["parents"])

        # Linear blend skinning.
        T = np.einsum("vk,kij->vij", m["weights"], skinning_mats)
        v_homo = np.concatenate([v_posed, np.ones((v_posed.shape[0], 1))], axis=1)
        vertices = np.einsum("vij,vj->vi", T, v_homo)[:, :3]

        # Fingertips come from the mesh; append as joints 16..20.
        tips = vertices[self.tip_vertex_ids]
        joints_21 = np.concatenate([joints, tips], axis=0)

        keypoints = mano_joints_to_mediapipe(joints_21)
        if return_verts:
            return keypoints, vertices
        return keypoints

    def _resolve_hand_pose(self, pose: Optional[np.ndarray]) -> np.ndarray:
        """Expand PCA coefficients / validate axis-angle, and add the hand mean."""
        m = self.model
        if pose is None:
            pose = np.zeros(self.num_pca_comps if self.use_pca else 45)
        pose = np.asarray(pose, dtype=np.float64).reshape(-1)

        if self.use_pca:
            if pose.shape[0] != self.num_pca_comps:
                raise ValueError(
                    f"use_pca=True expects {self.num_pca_comps} coefficients, "
                    f"got {pose.shape[0]}. Pass use_pca=False for 45 axis-angle values."
                )
            full_pose = pose @ m["hands_components"][: self.num_pca_comps]
        else:
            if pose.shape[0] != 45:
                raise ValueError(
                    f"use_pca=False expects 45 axis-angle values, got {pose.shape[0]}"
                )
            full_pose = pose

        if not self.flat_hand_mean:
            full_pose = full_pose + m["hands_mean"]
        return full_pose

    @staticmethod
    def _rigid_transform(rot_mats: np.ndarray, joints: np.ndarray, parents: np.ndarray):
        """Walk the kinematic chain.

        Returns:
            (posed joint positions (16,3), skinning matrices (16,4,4)). The
            skinning matrices have the rest-pose offset removed, which is what
            LBS consumes; the posed joints are read off before that subtraction.
        """
        num_joints = joints.shape[0]
        rel_joints = joints.copy()
        rel_joints[1:] -= joints[parents[1:]]

        local = np.zeros((num_joints, 4, 4), dtype=np.float64)
        local[:, :3, :3] = rot_mats
        local[:, :3, 3] = rel_joints
        local[:, 3, 3] = 1.0

        chain = [local[0]]
        for i in range(1, num_joints):
            chain.append(chain[parents[i]] @ local[i])
        globals_ = np.stack(chain, axis=0)

        posed_joints = globals_[:, :3, 3].copy()

        # Subtract each joint's rest position carried through its own rotation,
        # so the matrices map rest-pose vertices rather than re-applying offsets.
        skinning = globals_.copy()
        skinning[:, :3, 3] -= np.einsum("kij,kj->ki", globals_[:, :3, :3], joints)
        return posed_joints, skinning


__all__ = [
    "MANO_JOINT_NAMES",
    "MANO_TIP_VERTICES",
    "MANO_TO_MEDIAPIPE",
    "ManoLayer",
    "load_mano_model",
    "mano_joints_to_mediapipe",
]
