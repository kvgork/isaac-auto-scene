"""Robust point-cloud registration backends (Phase 7+).

Public API
----------
register_with_fpfh_ransac  — FPFH features + mutual matching + custom
                              RANSAC Procrustes (Kabsch SVD).  Drop-in
                              replacement for ``register_global_local``
                              when classic ICP can't bridge a partial-
                              overlap, cluttered target (real D435 arm
                              cloud with cables vs URDF CAD).

Implementation note
-------------------
Open3D's bundled
``registration_ransac_based_on_feature_matching`` segfaults on this
build (see project CLAUDE.md).  This module rolls a tiny RANSAC over
mutual-NN FPFH correspondences, estimating each candidate SE(3) by
SVD-based Procrustes (Kabsch) and scoring by inlier count + RMSE.  The
best candidate then gets refined with ``o3d.t.pipelines.registration.icp``
(tensor API — the only stable ICP variant on this build).

The module name reflects the original intent to host learned matchers
(GeoTransformer, PREDATOR).  Wiring those needs PyTorch + a checkpoint
and is deferred until the geometric approach is exhausted.  Today only
the FPFH+RANSAC backend is wired.

No extra dependencies beyond numpy + Open3D 0.18.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import open3d as o3d

from isaac_auto_scene.register import RegistrationResult


def register_with_fpfh_ransac(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    *,
    voxel_size: float = 0.005,
    feature_radius_multiplier: float = 5.0,
    fpfh_radius_multiplier: float = 10.0,
    ransac_iterations: int = 4000,
    ransac_sample_size: int = 4,
    ransac_inlier_distance_multiplier: float = 2.5,
    refine_distance_multiplier: float = 2.0,
    seed: int = 0,
) -> RegistrationResult:
    """Register ``source`` onto ``target`` via FPFH + RANSAC + ICP refine.

    Pipeline
    --------
    1. Voxel-downsample both clouds at ``voxel_size``.
    2. Estimate normals at ``voxel_size * feature_radius_multiplier``.
    3. Compute FPFH features at ``voxel_size * fpfh_radius_multiplier``.
    4. Mutual nearest-neighbour matching in FPFH feature space.
    5. RANSAC: repeatedly sample ``ransac_sample_size`` correspondences,
       solve SE(3) via Kabsch SVD, count inliers within
       ``voxel_size * ransac_inlier_distance_multiplier``.
    6. Best candidate refined with point-to-plane tensor-API ICP at
       ``voxel_size * refine_distance_multiplier``.

    Returns a :class:`RegistrationResult` whose ``fitness`` and
    ``inlier_rmse_m`` come from the refinement step so the result is
    directly comparable to ``register_global_local`` output.
    """
    src_down, src_fpfh = _downsample_and_feature(
        source, voxel_size, feature_radius_multiplier, fpfh_radius_multiplier
    )
    tgt_down, tgt_fpfh = _downsample_and_feature(
        target, voxel_size, feature_radius_multiplier, fpfh_radius_multiplier
    )

    src_pts = np.asarray(src_down.points)  # (N_src, 3)
    tgt_pts = np.asarray(tgt_down.points)  # (N_tgt, 3)
    src_feat = np.asarray(src_fpfh.data)   # (33, N_src)
    tgt_feat = np.asarray(tgt_fpfh.data)   # (33, N_tgt)

    if src_pts.shape[0] < ransac_sample_size or tgt_pts.shape[0] < ransac_sample_size:
        raise RuntimeError(
            f"too few downsampled points for RANSAC: "
            f"src={src_pts.shape[0]} tgt={tgt_pts.shape[0]} "
            f"need >= {ransac_sample_size}"
        )

    src_idx, tgt_idx = _mutual_feature_match(src_feat, tgt_feat)
    if len(src_idx) < ransac_sample_size:
        raise RuntimeError(
            f"only {len(src_idx)} mutual FPFH matches; RANSAC needs >= "
            f"{ransac_sample_size}"
        )

    src_corr = src_pts[src_idx]  # (M, 3)
    tgt_corr = tgt_pts[tgt_idx]  # (M, 3)

    inlier_dist = voxel_size * ransac_inlier_distance_multiplier
    T_best, best_inliers = _ransac_se3(
        src_corr,
        tgt_corr,
        sample_size=ransac_sample_size,
        n_iters=ransac_iterations,
        inlier_distance=inlier_dist,
        seed=seed,
    )
    if T_best is None:
        raise RuntimeError(
            f"RANSAC found no SE(3) consensus over {ransac_iterations} iterations"
        )

    # Refine with tensor-API point-to-plane ICP (the only ICP variant that
    # doesn't segfault on this Open3D build — see CLAUDE.md).
    src_t = o3d.t.geometry.PointCloud.from_legacy(src_down)
    tgt_t = o3d.t.geometry.PointCloud.from_legacy(tgt_down)
    refined = o3d.t.pipelines.registration.icp(
        src_t,
        tgt_t,
        max_correspondence_distance=voxel_size * refine_distance_multiplier,
        init_source_to_target=o3d.core.Tensor(T_best, dtype=o3d.core.Dtype.Float64),
        estimation_method=o3d.t.pipelines.registration.TransformationEstimationPointToPlane(),
    )

    return RegistrationResult(
        T=np.asarray(refined.transformation.numpy(), dtype=np.float64),
        fitness=float(refined.fitness),
        inlier_rmse_m=float(refined.inlier_rmse),
        used_fallback=True,
        n_restarts=0,
    )


# Backwards-compatible alias for the original "teaser" CLI value.  The
# Teaser++ source build is heavier than this build wants to take on, so
# the implementation is the FPFH+RANSAC pipeline above.
register_with_teaser = register_with_fpfh_ransac


def _downsample_and_feature(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    feature_radius_multiplier: float,
    fpfh_radius_multiplier: float,
) -> tuple[o3d.geometry.PointCloud, Any]:
    """Voxel-downsample then compute FPFH features."""
    down = pcd.voxel_down_sample(voxel_size)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * feature_radius_multiplier, max_nn=30
        )
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * fpfh_radius_multiplier, max_nn=100
        ),
    )
    return down, fpfh


def _mutual_feature_match(
    src_feat: np.ndarray, tgt_feat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mutual nearest-neighbour matching of (33, N) FPFH descriptors."""
    src = src_feat.T  # (N_src, 33)
    tgt = tgt_feat.T  # (N_tgt, 33)
    d2 = (
        np.sum(src * src, axis=1, keepdims=True)
        + np.sum(tgt * tgt, axis=1)
        - 2.0 * (src @ tgt.T)
    )
    nn_src_to_tgt = np.argmin(d2, axis=1)
    nn_tgt_to_src = np.argmin(d2, axis=0)

    src_idx_list: list[int] = []
    tgt_idx_list: list[int] = []
    for i, j in enumerate(nn_src_to_tgt):
        if nn_tgt_to_src[j] == i:
            src_idx_list.append(int(i))
            tgt_idx_list.append(int(j))
    return np.asarray(src_idx_list, dtype=np.int64), np.asarray(
        tgt_idx_list, dtype=np.int64
    )


def _kabsch(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """SVD-based rigid transform (Kabsch) from N>=3 paired 3D points.

    Returns a 4x4 SE(3) that maps source -> target.
    Both inputs are (N, 3).
    """
    src_c = src - src.mean(axis=0)
    tgt_c = tgt - tgt.mean(axis=0)
    H = src_c.T @ tgt_c
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = tgt.mean(axis=0) - R @ src.mean(axis=0)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _ransac_se3(
    src: np.ndarray,
    tgt: np.ndarray,
    *,
    sample_size: int,
    n_iters: int,
    inlier_distance: float,
    seed: int,
) -> tuple[np.ndarray | None, int]:
    """Naive RANSAC over paired correspondences for a robust SE(3) fit.

    Returns ``(best_T, best_inlier_count)`` — ``best_T`` is None when
    RANSAC fails to find any inliers across ``n_iters``.
    """
    rng = np.random.default_rng(seed)
    n_pairs = src.shape[0]
    best_T: np.ndarray | None = None
    best_score = 0
    best_inlier_mask: np.ndarray | None = None
    dist_thresh_sq = float(inlier_distance) ** 2

    for _ in range(n_iters):
        idx = rng.choice(n_pairs, size=sample_size, replace=False)
        try:
            T = _kabsch(src[idx], tgt[idx])
        except np.linalg.LinAlgError:
            continue
        # Transform all source correspondences and count inliers.
        ones = np.ones((n_pairs, 1))
        src_h = np.hstack([src, ones])
        proj = src_h @ T.T
        diff = proj[:, :3] - tgt
        d2 = np.sum(diff * diff, axis=1)
        inlier_mask = d2 < dist_thresh_sq
        score = int(inlier_mask.sum())
        if score > best_score:
            best_score = score
            best_T = T
            best_inlier_mask = inlier_mask

    # Refit on all inliers from the best hypothesis.
    if best_T is not None and best_inlier_mask is not None and best_score >= sample_size:
        best_T = _kabsch(src[best_inlier_mask], tgt[best_inlier_mask])

    return best_T, best_score


__all__ = ["register_with_fpfh_ransac", "register_with_teaser"]
