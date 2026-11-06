"""Generate a sample MANO parameter sequence for testing the MANO input path.

Produces a synthetic open/close/pinch cycle in MANO's PCA pose space, saved in
the .npz layout ``ManoDevice`` reads. Useful for exercising the pipeline without
a dataset or a MANO estimator in the loop.

    python make_mano_sample.py --hand right --out data/mano_sample.npz
    python teleop_sim.py --mano data/mano_sample.npz --hand right
"""

import argparse
from pathlib import Path

import numpy as np


def build_sequence(num_frames: int = 300, num_pca: int = 6) -> np.ndarray:
    """Build (num_frames, num_pca) PCA coefficients for a grasp cycle.

    MANO's first PCA component is dominated by overall finger curl, so sweeping
    it produces a natural open->closed hand. The later components add smaller
    out-of-phase motion so the fingers do not move in perfect lockstep, which is
    what makes the pinch-detection blend in the optimizer actually engage.
    """
    t = np.linspace(0.0, 2.0 * np.pi, num_frames, endpoint=False)
    poses = np.zeros((num_frames, num_pca), dtype=np.float64)

    # Component 0: main curl, 0 (open) -> ~2.5 (closed) -> 0.
    poses[:, 0] = 1.25 * (1.0 - np.cos(t))
    if num_pca > 1:
        poses[:, 1] = 0.8 * np.sin(t)           # thumb opposition, quarter-phase
    if num_pca > 2:
        poses[:, 2] = 0.4 * np.sin(2.0 * t)     # spread
    for k in range(3, num_pca):
        poses[:, k] = 0.15 * np.sin((k + 1) * t)
    return poses


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hand", default="right", choices=["left", "right"])
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--pca", type=int, default=6, help="Number of PCA components")
    ap.add_argument("--out", default=None, help="Output .npz (default: data/mano_sample.npz)")
    args = ap.parse_args()

    out = Path(args.out) if args.out else Path(__file__).parent / "data" / "mano_sample.npz"
    out.parent.mkdir(parents=True, exist_ok=True)

    poses = build_sequence(args.frames, args.pca)
    np.savez(
        out,
        pose=poses,
        betas=np.zeros(10),        # mean shape
        use_pca=True,
        num_pca_comps=args.pca,
        hand=args.hand,
    )
    print(f"Wrote {out}: {poses.shape[0]} frames, {poses.shape[1]} PCA components, {args.hand} hand")


if __name__ == "__main__":
    main()
