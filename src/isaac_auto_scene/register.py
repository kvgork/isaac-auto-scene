"""Point-cloud registration with random-restart ICP (Phase 5).

Public API
----------
RegistrationResult       — frozen dataclass: T (4x4), fitness, rmse, used_fallback
register_global_local()  — N random-rotation restart ICP + point-to-plane refine
register_multi_pose()    — aggregate per-pose ICP results into one transform
MultiPoseResult          — aggregated transform + per-pose diagnostics
QUALITY_GATE             — (fitness_min, rmse_max_m) from research §6

Design note
-----------
Research doc recommends FPFH+FGR for global init, but Open3D 0.18 on this
platform segfaults in ``registration_fgr_*``, ``registration_ransac_*``
and the legacy ``registration_icp``.  The tensor-API ``o3d.t.pipelines
.registration.icp`` is stable, so we use it instead.  Global init is
approximated by ``n_restarts`` random small-rotation seeds around the
centroid offset; this matches the use-case (fixed D435 mount + known
SO-101 pose) where the misalignment is bounded to a few cm / tens of
degrees.  The ``fallback`` hook is the production escape route
(GeoTransformer / TEASER++).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import open3d as o3d


QUALITY_GATE: tuple[float, float] = (0.65, 0.005)  # fitness > 0.65, RMSE < 5 mm


@dataclass(frozen=True)
class RegistrationResult:
    """Output of one registration run."""

    T: np.ndarray
    fitness: float
    inlier_rmse_m: float
    used_fallback: bool
    n_restarts: int


def _to_tpcd(pcd: o3d.geometry.PointCloud) -> o3d.t.geometry.PointCloud:
    """Convert a legacy PointCloud to a tensor PointCloud (float32)."""
    pts = np.asarray(pcd.points, dtype=np.float32)
    t = o3d.core.Tensor(np.ascontiguousarray(pts), o3d.core.float32)
    out = o3d.t.geometry.PointCloud(t)
    if pcd.has_normals():
        ns = np.asarray(pcd.normals, dtype=np.float32)
        out.point.normals = o3d.core.Tensor(np.ascontiguousarray(ns), o3d.core.float32)
    return out


def _voxel_down_with_normals(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    normal_radius: float,
) -> o3d.t.geometry.PointCloud:
    down = pcd.voxel_down_sample(voxel_size)
    if not down.has_normals():
        down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
        )
    return _to_tpcd(down)


def _icp_tensor(
    src_t: o3d.t.geometry.PointCloud,
    tgt_t: o3d.t.geometry.PointCloud,
    init: np.ndarray,
    max_correspondence_distance: float,
    use_point_to_plane: bool,
    max_iter: int,
) -> o3d.t.pipelines.registration.RegistrationResult:
    init_t = o3d.core.Tensor(
        np.ascontiguousarray(init, dtype=np.float32), o3d.core.float32
    )
    estimator = (
        o3d.t.pipelines.registration.TransformationEstimationPointToPlane()
        if use_point_to_plane
        else o3d.t.pipelines.registration.TransformationEstimationPointToPoint()
    )
    crit = o3d.t.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=max_iter
    )
    return o3d.t.pipelines.registration.icp(
        src_t, tgt_t, max_correspondence_distance, init_t, estimator, crit
    )


def _random_so3(rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation matrix via Shoemake's quaternion method.

    Returns a 3x3 rotation matrix drawn uniformly from SO(3) — every
    orientation is equally likely, unlike axis-angle with bounded angle
    which biases toward identity.
    """
    u1, u2, u3 = rng.uniform(size=3)
    sqrt1mu1 = np.sqrt(1.0 - u1)
    sqrtu1 = np.sqrt(u1)
    q = np.array(
        [
            sqrt1mu1 * np.sin(2.0 * np.pi * u2),
            sqrt1mu1 * np.cos(2.0 * np.pi * u2),
            sqrtu1 * np.sin(2.0 * np.pi * u3),
            sqrtu1 * np.cos(2.0 * np.pi * u3),
        ]
    )
    x, y, z, w = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return R


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array(
        [
            [0.0, -a[2], a[1]],
            [a[2], 0.0, -a[0]],
            [-a[1], a[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def register_global_local(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    *,
    voxel_size: float = 0.005,
    n_restarts: int = 5,
    coarse_distance: float = 0.05,
    fine_distance: float = 0.01,
    fallback: Callable[
        [o3d.geometry.PointCloud, o3d.geometry.PointCloud],
        RegistrationResult,
    ] | None = None,
    fallback_fitness: float = 0.40,
    coarse_max_iter: int = 50,
    fine_max_iter: int = 150,
    full_so3_init: bool = True,
) -> RegistrationResult:
    """Random-restart tensor-API ICP global init + point-to-plane refine.

    Parameters
    ----------
    n_restarts:
        Number of random initial rotations attempted.  Higher = more
        chance of escaping local minima on a partial-view target.
    coarse_max_iter / fine_max_iter:
        Per-restart ICP iteration budgets.  Doubled vs the pre-Phase-7
        defaults (30 / 80) — the prior values truncated convergence on
        real D435 captures with cluttered targets.
    full_so3_init:
        When True (default), random restarts sample full SO(3) via
        unit quaternions (any orientation in 3D).  When False, fall
        back to the legacy ±0.6 rad wedge — only useful when the
        caller has already pre-aligned the source and just wants a
        small wiggle search.  Partial views with no prior knowledge
        of arm orientation need True; the legacy ±34° wedge silently
        traps ICP in different per-pose basins (observed: 72° transform
        dispersion across 5 real captures).
    """
    radius_normal = voxel_size * 2.0
    src_down_t = _voxel_down_with_normals(source, voxel_size, radius_normal)
    tgt_down_t = _voxel_down_with_normals(target, voxel_size, radius_normal)

    src_pts = src_down_t.point.positions.numpy()
    tgt_pts = tgt_down_t.point.positions.numpy()
    src_centroid = src_pts.mean(axis=0)
    tgt_centroid = tgt_pts.mean(axis=0)
    t_init = (tgt_centroid - src_centroid).astype(np.float64)

    rng = np.random.default_rng(0)
    best_fitness = -1.0
    best_T: np.ndarray = np.eye(4)
    best_T[:3, 3] = t_init

    for i in range(n_restarts):
        T_try = np.eye(4)
        T_try[:3, 3] = t_init
        if i > 0:
            if full_so3_init:
                T_try[:3, :3] = _random_so3(rng)
            else:
                axis = rng.normal(size=3)
                axis /= np.linalg.norm(axis)
                angle = float(rng.uniform(-0.6, 0.6))
                T_try[:3, :3] = _rodrigues(axis, angle)

        coarse = _icp_tensor(
            src_down_t, tgt_down_t, T_try, coarse_distance,
            use_point_to_plane=False, max_iter=coarse_max_iter,
        )
        fit = float(coarse.fitness)
        if fit > best_fitness:
            best_fitness = fit
            best_T = coarse.transformation.numpy().astype(np.float64)

    if best_fitness < fallback_fitness and fallback is not None:
        return fallback(source, target)

    fine = _icp_tensor(
        src_down_t, tgt_down_t, best_T, fine_distance,
        use_point_to_plane=True, max_iter=fine_max_iter,
    )

    return RegistrationResult(
        T=fine.transformation.numpy().astype(np.float64),
        fitness=float(fine.fitness),
        inlier_rmse_m=float(fine.inlier_rmse),
        used_fallback=False,
        n_restarts=n_restarts,
    )


def passes_quality_gate(
    result: RegistrationResult,
    gate: tuple[float, float] | None = None,
) -> bool:
    """Check a result against the quality gate.

    Parameters
    ----------
    result:
        RegistrationResult to evaluate.
    gate:
        Optional override of ``(fitness_min, rmse_max_m)``.  Falls back to
        :data:`QUALITY_GATE` (0.65 / 5 mm) when None.  Hardware bring-up
        with cluttered captures often needs a looser gate (e.g.
        ``(0.30, 0.012)``) until the CAD model is refined or the
        registration backend is upgraded.
    """
    f_min, rmse_max = gate if gate is not None else QUALITY_GATE
    return result.fitness >= f_min and result.inlier_rmse_m <= rmse_max


# ---------------------------------------------------------------------------
# Multi-pose aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerPoseRegistration:
    """One pose's contribution to a multi-pose registration."""

    pose_name: str
    accepted: bool
    fitness: float
    inlier_rmse_m: float
    T: np.ndarray
    reason: str = ""


@dataclass(frozen=True)
class MultiPoseResult:
    """Aggregated multi-pose registration output.

    Attributes
    ----------
    T:
        4x4 mean transform (Markley quaternion average + weighted translation).
    quat_xyzw:
        Mean rotation as unit XYZW quaternion.
    translation_m:
        Mean translation in metres.
    dispersion_rad:
        Weighted-mean rotation angle between each accepted pose's transform
        and the mean.  Treat large values (> ~0.05 rad ≈ 3°) as a hint that
        the pose set is inconsistent.
    n_accepted:
        Number of poses that passed the per-pose quality gate.
    n_total:
        Total poses considered.
    per_pose:
        Per-pose diagnostics (accepted + reason for rejection if any).
    """

    T: np.ndarray
    quat_xyzw: np.ndarray
    translation_m: np.ndarray
    dispersion_rad: float
    n_accepted: int
    n_total: int
    per_pose: tuple[PerPoseRegistration, ...]


def _registration_weight(result: RegistrationResult) -> float:
    """Combine fitness + (1/rmse) into a positive scalar weight.

    A small epsilon stops a tiny RMSE from dominating; using fitness as a
    multiplier penalises low-overlap fits even when their RMSE is small.
    """
    eps = 1e-4  # 0.1 mm
    rmse = max(float(result.inlier_rmse_m), eps)
    return max(float(result.fitness), 0.0) / rmse


def register_multi_pose(
    pairs: list[
        tuple[
            str,
            o3d.geometry.PointCloud,
            o3d.geometry.PointCloud,
        ]
    ],
    *,
    voxel_size: float = 0.005,
    n_restarts: int = 5,
    coarse_distance: float = 0.05,
    fine_distance: float = 0.01,
    min_accepted: int = 2,
    quality_gate: tuple[float, float] | None = None,
    fallback: Callable[
        [o3d.geometry.PointCloud, o3d.geometry.PointCloud],
        RegistrationResult,
    ]
    | None = None,
) -> MultiPoseResult:
    """Run per-pose ICP, reject outliers, aggregate into one transform.

    Parameters
    ----------
    pairs:
        List of ``(pose_name, source_cad_pcd, target_arm_pcd)`` tuples.
        Each entry's ``T_cam_arm`` is estimated independently.
    voxel_size, n_restarts, coarse_distance, fine_distance:
        Passed through to :func:`register_global_local`.
    min_accepted:
        Minimum number of poses that must pass :data:`QUALITY_GATE` for the
        aggregation to succeed.  Raises ``RuntimeError`` if fewer accepted.

    Returns
    -------
    MultiPoseResult
    """
    from isaac_auto_scene.utils.transforms import (
        mean_se3,
        rotation_matrix_to_quat_xyzw,
    )

    if not pairs:
        raise ValueError("register_multi_pose requires at least one pose pair")

    per_pose: list[PerPoseRegistration] = []
    accepted_Ts: list[np.ndarray] = []
    accepted_weights: list[float] = []

    effective_gate = quality_gate if quality_gate is not None else QUALITY_GATE

    for pose_name, src, tgt in pairs:
        try:
            reg = register_global_local(
                src,
                tgt,
                voxel_size=voxel_size,
                n_restarts=n_restarts,
                coarse_distance=coarse_distance,
                fine_distance=fine_distance,
                fallback=fallback,
            )
        except Exception as exc:
            per_pose.append(
                PerPoseRegistration(
                    pose_name=pose_name,
                    accepted=False,
                    fitness=0.0,
                    inlier_rmse_m=float("inf"),
                    T=np.eye(4),
                    reason=f"icp_error: {type(exc).__name__}: {exc}",
                )
            )
            continue

        if passes_quality_gate(reg, gate=effective_gate):
            per_pose.append(
                PerPoseRegistration(
                    pose_name=pose_name,
                    accepted=True,
                    fitness=reg.fitness,
                    inlier_rmse_m=reg.inlier_rmse_m,
                    T=np.asarray(reg.T, dtype=np.float64),
                )
            )
            accepted_Ts.append(np.asarray(reg.T, dtype=np.float64))
            accepted_weights.append(_registration_weight(reg))
        else:
            per_pose.append(
                PerPoseRegistration(
                    pose_name=pose_name,
                    accepted=False,
                    fitness=reg.fitness,
                    inlier_rmse_m=reg.inlier_rmse_m,
                    T=np.asarray(reg.T, dtype=np.float64),
                    reason=(
                        f"quality_gate(fitness>={effective_gate[0]:.2f},"
                        f"rmse<={effective_gate[1]*1000:.0f}mm)  "
                        f"actual fitness={reg.fitness:.3f} "
                        f"rmse={reg.inlier_rmse_m*1000:.2f}mm"
                    ),
                )
            )

    n_accepted = len(accepted_Ts)
    if n_accepted < min_accepted:
        raise RuntimeError(
            f"multi-pose registration failed: only {n_accepted}/{len(pairs)} "
            f"poses passed quality gate (min={min_accepted}). "
            f"Per-pose reasons: "
            + "; ".join(f"{p.pose_name}={p.reason or 'ok'}" for p in per_pose)
        )

    stacked = np.stack(accepted_Ts, axis=0)
    weights = np.asarray(accepted_weights, dtype=np.float64)
    T_mean, q_mean, dispersion = mean_se3(stacked, weights)

    # mean_se3 already gives quat_xyzw; recompute is a no-op but keeps
    # MultiPoseResult.quat_xyzw consistent if mean_se3 internals ever change.
    _ = rotation_matrix_to_quat_xyzw  # mark imported

    return MultiPoseResult(
        T=T_mean,
        quat_xyzw=q_mean,
        translation_m=T_mean[:3, 3].copy(),
        dispersion_rad=dispersion,
        n_accepted=n_accepted,
        n_total=len(pairs),
        per_pose=tuple(per_pose),
    )
