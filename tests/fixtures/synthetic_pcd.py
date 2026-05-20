"""Synthetic point cloud fixture reused by Phase 3 segmentation tests.

Creates a deterministic table + arm-proxy PCD that can be used without any
hardware or Isaac Sim dependencies.

Phase 3 imports ``make_table_plus_arm_pcd`` from here.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d


def make_table_plus_arm_pcd(
    *,
    n_table: int = 8000,
    n_arm: int = 2000,
    table_z: float = 0.0,
    table_xlim: tuple[float, float] = (-0.4, 0.4),
    table_ylim: tuple[float, float] = (-0.3, 0.3),
    arm_center: tuple[float, float, float] = (0.0, 0.0, 0.15),
    arm_radius: float = 0.08,
    noise_std: float = 0.002,
    seed: int = 42,
) -> o3d.geometry.PointCloud:
    """Return a synthetic Open3D PointCloud with a table plane + arm proxy.

    The table lies in the XY plane at ``table_z``.  The arm proxy is a
    hemisphere centred at ``arm_center`` with the given ``arm_radius``.

    Parameters
    ----------
    n_table:
        Number of table surface points.
    n_arm:
        Number of arm-proxy hemisphere points.
    table_z:
        Z coordinate of the table plane.
    table_xlim, table_ylim:
        X and Y extent of the table in metres.
    arm_center:
        Centre of the arm hemisphere (should be above table_z).
    arm_radius:
        Radius of the arm hemisphere in metres.
    noise_std:
        Gaussian noise added to all point positions (metres).
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    o3d.geometry.PointCloud
        Combined table + arm points with estimated normals.
    """
    rng = np.random.default_rng(seed)

    # ---- Table points (uniform XY + fixed Z + noise) --------------------
    table_pts_xy = rng.uniform(
        low=[table_xlim[0], table_ylim[0]],
        high=[table_xlim[1], table_ylim[1]],
        size=(n_table, 2),
    )
    table_pts = np.column_stack(
        [table_pts_xy, np.full(n_table, table_z, dtype=np.float64)]
    )

    # ---- Arm hemisphere points ------------------------------------------
    # Sample points on the upper hemisphere (z >= arm_center_z)
    theta = rng.uniform(0.0, np.pi / 2.0, size=n_arm)  # polar angle 0..90°
    phi = rng.uniform(0.0, 2 * np.pi, size=n_arm)
    x = arm_radius * np.sin(theta) * np.cos(phi) + arm_center[0]
    y = arm_radius * np.sin(theta) * np.sin(phi) + arm_center[1]
    z = arm_radius * np.cos(theta) + arm_center[2]
    arm_pts = np.column_stack([x, y, z])

    # ---- Combine + add noise --------------------------------------------
    all_pts = np.vstack([table_pts, arm_pts])
    all_pts += rng.normal(0.0, noise_std, all_pts.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )

    return pcd
