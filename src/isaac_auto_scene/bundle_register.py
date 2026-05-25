"""Joint multi-pose bundle adjustment for camera<->arm calibration (Phase 7+).

Public API
----------
register_bundle()    — solve ONE SE(3) ``T_cam_arm`` jointly across N
                       arm poses with known joint angles.
BundleResult         — frozen dataclass: T, quat_xyzw, translation_m,
                       per_pose_fitness, per_pose_rmse_m.

Motivation
----------
Per-pose ICP (``register_multi_pose``) fits N independent SE(3) matrices
and weighted-averages them.  On a partial view of a cylindrical robot
arm the per-pose fit is **degenerate**: rotation about the arm's long
axis is unobservable from a single hemisphere.  Different poses land in
different per-pose minima; the average is meaningless (observed: 72°
transform dispersion across 5 real captures).

This module instead optimises one shared ``T_cam_arm`` that explains
*all* N observations simultaneously.  Joint configurations differ across
poses, so points that are symmetry-ambiguous in one pose are constrained
by another.  The result has principled, non-arbitrary geometry.

Math
----
Variables: ξ ∈ ℝ⁶ — se(3) Lie-algebra parametrisation of T_cam_arm.
For each pose p:
    X_p^arm = assemble_pcd(urdf, joint_angles_p)   (CAD in arm root frame)
    X_p^cam = T(ξ) · X_p^arm                       (transformed to cam frame)
    r_p,i   = X_p^cam_i - nn(arm_cloud_p, X_p^cam_i)
Cost: Σ_p Σ_i ||r_p,i||²

Solved by Levenberg-Marquardt on ξ using
``scipy.optimize.least_squares``.  Per-iteration nearest-neighbour
lookups use ``scipy.spatial.cKDTree`` against each pose's arm_cloud (cheap
since the trees are built once and queried at every iteration).

The se(3) exponential map maps ξ → SE(3) via the standard formula (see
Sola, "A micro Lie theory for state estimation in robotics", §C).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import open3d as o3d


@dataclass(frozen=True)
class BundleResult:
    """Output of a joint multi-pose bundle registration."""

    T: np.ndarray
    quat_xyzw: np.ndarray
    translation_m: np.ndarray
    cost: float
    per_pose_fitness: tuple[float, ...]
    per_pose_rmse_m: tuple[float, ...]
    n_iterations: int


def register_bundle(
    cad_clouds: Sequence[np.ndarray],
    arm_clouds: Sequence[o3d.geometry.PointCloud],
    *,
    T_init: np.ndarray | None = None,
    inlier_distance_m: float = 0.02,
    max_nfev: int = 200,
    ftol: float = 1e-6,
    xtol: float = 1e-8,
    cad_subsample: int | None = 2000,
) -> BundleResult:
    """Bundle-optimise one SE(3) over N pose observations.

    Parameters
    ----------
    cad_clouds:
        Per-pose CAD point clouds in arm-root frame.  Each entry is an
        ``(M_p, 3)`` numpy array — call ``assemble_pcd(urdf, joints_p)``
        before invoking this function.
    arm_clouds:
        Per-pose Open3D point clouds in camera frame (output of
        ``segment_table_arm(...).arm_cloud``).  Must align 1:1 with
        ``cad_clouds`` by index.
    T_init:
        4x4 initial guess for ``T_cam_arm``.  Default: identity.  Pass
        the best per-pose ICP result here for fast convergence.
    inlier_distance_m:
        Residuals from CAD points whose nearest neighbour lies farther
        than this are clamped to this distance.  Robustifies against the
        partial-view "missing back of arm" failure mode where 50% of
        CAD points cannot find a match no matter what T is chosen.
    max_nfev:
        scipy.optimize.least_squares function-evaluation budget.
    ftol, xtol:
        Convergence thresholds (relative cost / parameter change).
    cad_subsample:
        Per-pose CAD downsample to keep residual evaluation tractable.
        ``None`` keeps all points.  Default 2000 ≈ 5 ms per evaluation
        on a 5-pose problem.

    Returns
    -------
    BundleResult
    """
    from scipy.optimize import least_squares
    from scipy.spatial import cKDTree

    if len(cad_clouds) != len(arm_clouds):
        raise ValueError(
            f"cad_clouds and arm_clouds must have equal length "
            f"(got {len(cad_clouds)} and {len(arm_clouds)})"
        )
    if len(cad_clouds) == 0:
        raise ValueError("register_bundle requires at least one pose")

    rng = np.random.default_rng(0)
    cad_arr: list[np.ndarray] = []
    arm_trees: list[cKDTree] = []
    arm_arrs: list[np.ndarray] = []
    for cad_pts, arm_pcd in zip(cad_clouds, arm_clouds):
        pts = np.asarray(cad_pts, dtype=np.float64)
        if cad_subsample is not None and len(pts) > cad_subsample:
            idx = rng.choice(len(pts), size=cad_subsample, replace=False)
            pts = pts[idx]
        cad_arr.append(pts)
        arm_pts = np.asarray(arm_pcd.points, dtype=np.float64)
        arm_arrs.append(arm_pts)
        arm_trees.append(cKDTree(arm_pts))

    T0 = np.eye(4) if T_init is None else np.asarray(T_init, dtype=np.float64)
    xi0 = se3_log(T0)

    inlier_d = float(inlier_distance_m)

    def residuals(xi: np.ndarray) -> np.ndarray:
        T = se3_exp(xi)
        R = T[:3, :3]
        t = T[:3, 3]
        parts = []
        for cad_pts, tree in zip(cad_arr, arm_trees):
            cam_pts = cad_pts @ R.T + t
            dist, _ = tree.query(cam_pts, k=1)
            clamped = np.minimum(dist, inlier_d)
            parts.append(clamped)
        return np.concatenate(parts)

    result = least_squares(
        residuals,
        xi0,
        method="lm",
        max_nfev=max_nfev,
        ftol=ftol,
        xtol=xtol,
    )

    T_opt = se3_exp(result.x)
    R_opt = T_opt[:3, :3]
    t_opt = T_opt[:3, 3]
    q = _rotation_to_quat_xyzw(R_opt)

    per_pose_fit: list[float] = []
    per_pose_rmse: list[float] = []
    for cad_pts, arm_pts, tree in zip(cad_arr, arm_arrs, arm_trees):
        cam_pts = cad_pts @ R_opt.T + t_opt
        dist, _ = tree.query(cam_pts, k=1)
        mask = dist < inlier_d
        n_inliers = int(mask.sum())
        per_pose_fit.append(n_inliers / max(1, len(cad_pts)))
        if n_inliers > 0:
            per_pose_rmse.append(float(np.sqrt(np.mean(dist[mask] ** 2))))
        else:
            per_pose_rmse.append(float("inf"))

    return BundleResult(
        T=T_opt,
        quat_xyzw=q,
        translation_m=t_opt.copy(),
        cost=float(result.cost),
        per_pose_fitness=tuple(per_pose_fit),
        per_pose_rmse_m=tuple(per_pose_rmse),
        n_iterations=int(result.nfev),
    )


# ---------------------------------------------------------------------------
# se(3) exp / log helpers
# ---------------------------------------------------------------------------


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Exponential map se(3) -> SE(3).

    ``xi = (rho, phi)`` where ``rho`` is the translation part (R³) and
    ``phi`` is the rotation part (so(3) axis-angle vector, R³).

    Returns 4x4 homogeneous SE(3) matrix.
    Reference: Sola, "A micro Lie theory" §C, eqn 137.
    """
    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    rho = xi[:3]
    phi = xi[3:]
    theta = float(np.linalg.norm(phi))

    if theta < 1e-9:
        R = np.eye(3) + _skew(phi)
        V = np.eye(3) + 0.5 * _skew(phi)
    else:
        a = phi / theta
        K = _skew(a)
        s = np.sin(theta)
        c = np.cos(theta)
        R = np.eye(3) + s * K + (1 - c) * (K @ K)
        V = (
            np.eye(3)
            + ((1 - c) / theta) * K
            + ((theta - s) / theta) * (K @ K)
        )

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ rho
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    """Logarithm map SE(3) -> se(3).

    Inverse of :func:`se3_exp`.  Used to convert an initial 4x4 guess to
    a ξ ∈ ℝ⁶ vector for the solver.
    """
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    # so(3) log
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = float(np.arccos(cos_theta))

    if theta < 1e-9:
        phi = np.zeros(3)
        V_inv = np.eye(3)
    else:
        ln_R = (theta / (2.0 * np.sin(theta))) * (R - R.T)
        phi = np.array([ln_R[2, 1], ln_R[0, 2], ln_R[1, 0]])
        a = phi / theta
        K = _skew(a)
        s = np.sin(theta)
        c = np.cos(theta)
        V_inv = (
            np.eye(3)
            - 0.5 * (theta * K)
            + (1.0 / theta**2)
            * (1.0 - (theta * s) / (2.0 * (1.0 - c)))
            * ((theta * K) @ (theta * K))
        )

    rho = V_inv @ t
    return np.concatenate([rho, phi])


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def _rotation_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> (x, y, z, w) unit quaternion (Shepperd, stable)."""
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    n = np.linalg.norm(q)
    return q / n if n > 0 else q


@dataclass(frozen=True)
class JointOffsetBundleResult:
    """Output of a bundle solve that also calibrates per-joint offsets."""

    T: np.ndarray
    quat_xyzw: np.ndarray
    translation_m: np.ndarray
    joint_offsets: dict[str, float]
    cost: float
    per_pose_fitness: tuple[float, ...]
    per_pose_rmse_m: tuple[float, ...]
    n_iterations: int


def register_bundle_with_joints(
    urdf,
    joint_angles_per_pose: Sequence[dict[str, float]],
    arm_clouds: Sequence[o3d.geometry.PointCloud],
    *,
    optimize_joints: Sequence[str] | None = None,
    cad_target_n_points: int = 2000,
    T_init: np.ndarray | None = None,
    delta_init: np.ndarray | None = None,
    inlier_distance_m: float = 0.02,
    delta_bound_rad: float = 0.35,
    max_nfev: int = 400,
    ftol: float = 1e-6,
    xtol: float = 1e-8,
) -> JointOffsetBundleResult:
    """Bundle solve T_cam_arm jointly with a constant per-joint offset Δθ.

    The per-pose CAD is reassembled at every LM evaluation using
    ``joint_angles_per_pose[p] + Δθ`` so the optimiser can absorb a
    systematic kinematic offset (LeRobot servo zero vs URDF zero, link-
    length tolerances, backlash bias) that otherwise compounds along the
    chain and makes the bundle T fit only near-home poses.

    Parameters
    ----------
    urdf:
        Loaded ``yourdfpy.URDF`` for forward kinematics.
    joint_angles_per_pose:
        Per-pose dict of joint_name -> radians (the readback).  All names
        in ``optimize_joints`` must appear in every dict.
    arm_clouds:
        Per-pose segmented arm clouds in camera frame.
    optimize_joints:
        Names of joints whose offset is freed.  ``None`` defaults to all
        actuated joints.
    cad_target_n_points:
        Target Poisson-disk samples per pose CAD.  Lower = faster LM.
    T_init / delta_init:
        Optional starting points.  Defaults: identity / zeros.
    inlier_distance_m:
        Residual clamp (m).  Caps the back-half-of-arm "no NN match" mode.
    delta_bound_rad:
        ``±delta_bound_rad`` box constraint per joint offset.  Default
        0.35 rad (~20°) keeps the optimisation in the linear FK regime.
    """
    from scipy.optimize import least_squares
    from scipy.spatial import cKDTree

    if len(joint_angles_per_pose) != len(arm_clouds):
        raise ValueError(
            f"joint_angles_per_pose and arm_clouds must align (got "
            f"{len(joint_angles_per_pose)} and {len(arm_clouds)})"
        )
    if len(arm_clouds) == 0:
        raise ValueError("register_bundle_with_joints requires >=1 pose")

    if optimize_joints is None:
        optimize_joints = tuple(urdf.actuated_joint_names)
    else:
        optimize_joints = tuple(optimize_joints)

    # Pre-compute link-local CAD samples ONCE (independent of joint state).
    link_local_pts = _sample_link_local_pcds(urdf, cad_target_n_points)

    # KDTrees for arm clouds (once).
    arm_trees: list[cKDTree] = []
    arm_arrs: list[np.ndarray] = []
    for arm_pcd in arm_clouds:
        arm_pts = np.asarray(arm_pcd.points, dtype=np.float64)
        arm_arrs.append(arm_pts)
        arm_trees.append(cKDTree(arm_pts))

    n_joints = len(optimize_joints)
    T0 = np.eye(4) if T_init is None else np.asarray(T_init, dtype=np.float64)
    xi0 = se3_log(T0)
    delta0 = (
        np.zeros(n_joints, dtype=np.float64)
        if delta_init is None
        else np.asarray(delta_init, dtype=np.float64)
    )
    x0 = np.concatenate([xi0, delta0])
    lb = np.concatenate([-np.full(6, np.inf), -delta_bound_rad * np.ones(n_joints)])
    ub = np.concatenate([np.full(6, np.inf), delta_bound_rad * np.ones(n_joints)])

    inlier_d = float(inlier_distance_m)

    def residuals(x: np.ndarray) -> np.ndarray:
        T = se3_exp(x[:6])
        R = T[:3, :3]
        t = T[:3, 3]
        delta = {n: float(x[6 + i]) for i, n in enumerate(optimize_joints)}

        parts = []
        for p_idx, joints in enumerate(joint_angles_per_pose):
            applied = {k: float(v) + delta.get(k, 0.0) for k, v in joints.items()}
            urdf.update_cfg(applied)
            cad_arm = _assemble_from_link_local(urdf, link_local_pts)
            cam_pts = cad_arm @ R.T + t
            dist, _ = arm_trees[p_idx].query(cam_pts, k=1)
            parts.append(np.minimum(dist, inlier_d))
        return np.concatenate(parts)

    result = least_squares(
        residuals,
        x0,
        method="trf",  # supports bounds; method='lm' does not
        bounds=(lb, ub),
        max_nfev=max_nfev,
        ftol=ftol,
        xtol=xtol,
    )

    T_opt = se3_exp(result.x[:6])
    R_opt = T_opt[:3, :3]
    t_opt = T_opt[:3, 3]
    delta_opt = {n: float(result.x[6 + i]) for i, n in enumerate(optimize_joints)}
    q = _rotation_to_quat_xyzw(R_opt)

    per_pose_fit: list[float] = []
    per_pose_rmse: list[float] = []
    for p_idx, joints in enumerate(joint_angles_per_pose):
        applied = {k: float(v) + delta_opt.get(k, 0.0) for k, v in joints.items()}
        urdf.update_cfg(applied)
        cad_arm = _assemble_from_link_local(urdf, link_local_pts)
        cam_pts = cad_arm @ R_opt.T + t_opt
        dist, _ = arm_trees[p_idx].query(cam_pts, k=1)
        mask = dist < inlier_d
        n_in = int(mask.sum())
        per_pose_fit.append(n_in / max(1, len(cam_pts)))
        per_pose_rmse.append(
            float(np.sqrt(np.mean(dist[mask] ** 2))) if n_in else float("inf")
        )

    return JointOffsetBundleResult(
        T=T_opt,
        quat_xyzw=q,
        translation_m=t_opt.copy(),
        joint_offsets=delta_opt,
        cost=float(result.cost),
        per_pose_fitness=tuple(per_pose_fit),
        per_pose_rmse_m=tuple(per_pose_rmse),
        n_iterations=int(result.nfev),
    )


def _sample_link_local_pcds(urdf, target_n_points: int) -> dict[str, np.ndarray]:
    """Sample each visual link's mesh ONCE in its own local frame.

    Returns ``{link_name: (N_link, 3) numpy array}``.  N_link is allocated
    proportional to the link's surface area so all links are represented.
    """
    import trimesh
    from isaac_auto_scene.cad import _link_visual_mesh

    meshes: dict[str, trimesh.Trimesh] = {}
    for link_name in urdf.link_map.keys():
        m = _link_visual_mesh(urdf, link_name)
        if m is not None and isinstance(m, trimesh.Trimesh) and len(m.faces) > 0:
            meshes[link_name] = m

    if not meshes:
        raise ValueError("URDF has no visual meshes to sample")

    total_area = sum(m.area for m in meshes.values())
    out: dict[str, np.ndarray] = {}
    for ln, m in meshes.items():
        share = max(50, int(round(target_n_points * m.area / total_area)))
        pts, _ = trimesh.sample.sample_surface(m, count=share)
        out[ln] = np.asarray(pts, dtype=np.float64)
    return out


def _assemble_from_link_local(
    urdf,
    link_local_pts: dict[str, np.ndarray],
) -> np.ndarray:
    """Apply current URDF FK to pre-sampled link-local CAD points."""
    parts: list[np.ndarray] = []
    for link_name, local_pts in link_local_pts.items():
        T = np.asarray(urdf.get_transform(link_name), dtype=np.float64)
        R = T[:3, :3]
        t = T[:3, 3]
        parts.append(local_pts @ R.T + t)
    return np.vstack(parts)


__all__ = [
    "BundleResult",
    "JointOffsetBundleResult",
    "register_bundle",
    "register_bundle_with_joints",
    "se3_exp",
    "se3_log",
]
