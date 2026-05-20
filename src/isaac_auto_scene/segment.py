"""Plane + arm segmentation pipeline (Phase 3).

Public API
----------
SegmentResult        — frozen dataclass: T_world_table, arm_cloud, table_cloud
segment_table_arm()  — Open3D RANSAC plane + DBSCAN cluster largest

Strategy
--------
1. Detect dominant plane with ``segment_plane`` (5 mm threshold).
2. Build ``T_world_table`` so the plane normal becomes +Z and a point on
   the plane becomes the origin (right-handed frame, X axis arbitrary).
3. Split the remaining points (non-plane) into clusters with
   ``cluster_dbscan`` and keep the largest as the SO-101 arm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass(frozen=True)
class SegmentResult:
    """Output of one segmentation run.

    Attributes
    ----------
    T_world_table:
        4x4 homogeneous transform from camera/world frame to the table
        frame (+Z aligned with plane normal, origin on the plane).
    plane_model:
        (a, b, c, d) coefficients of the fitted plane ax+by+cz+d=0
        in the input frame.
    table_cloud:
        Open3D PointCloud of inlier (table) points.
    arm_cloud:
        Open3D PointCloud of the largest DBSCAN cluster after plane removal.
    inlier_rmse_m:
        RMS distance of inliers to the fitted plane, in metres.
    """

    T_world_table: np.ndarray
    plane_model: tuple[float, float, float, float]
    table_cloud: o3d.geometry.PointCloud
    arm_cloud: o3d.geometry.PointCloud
    inlier_rmse_m: float


def segment_table_arm(
    pcd: o3d.geometry.PointCloud,
    *,
    plane_threshold_m: float = 0.005,
    plane_ransac_n: int = 3,
    plane_num_iterations: int = 2000,
    dbscan_eps_m: float = 0.015,
    dbscan_min_points: int = 30,
) -> SegmentResult:
    """Segment the dominant plane and the largest off-plane cluster.

    Parameters
    ----------
    pcd:
        Coloured or uncoloured Open3D point cloud (camera/world frame).
    plane_threshold_m:
        RANSAC inlier distance threshold (5 mm matches D435 noise floor).
    plane_ransac_n:
        Number of samples per RANSAC iteration (3 for plane).
    plane_num_iterations:
        RANSAC iterations.
    dbscan_eps_m:
        DBSCAN neighbourhood radius (15 mm; larger than D435 noise).
    dbscan_min_points:
        Minimum cluster size to keep.

    Returns
    -------
    SegmentResult

    Raises
    ------
    ValueError
        If ``pcd`` has fewer than 100 points or no off-plane cluster
        satisfies ``dbscan_min_points``.
    """
    if len(pcd.points) < 100:
        raise ValueError(f"input pcd too small: {len(pcd.points)} points")

    plane_model_tuple, inlier_indices = pcd.segment_plane(
        distance_threshold=plane_threshold_m,
        ransac_n=plane_ransac_n,
        num_iterations=plane_num_iterations,
    )
    a, b, c, d = plane_model_tuple

    table_cloud = pcd.select_by_index(inlier_indices)
    offplane_cloud = pcd.select_by_index(inlier_indices, invert=True)

    plane_normal = np.array([a, b, c], dtype=np.float64)
    norm = float(np.linalg.norm(plane_normal))
    plane_normal /= norm
    inlier_pts = np.asarray(table_cloud.points)
    signed = inlier_pts @ plane_normal + d / norm
    inlier_rmse_m = float(np.sqrt(np.mean(signed**2)))

    centroid = inlier_pts.mean(axis=0)

    if plane_normal[2] < 0:
        plane_normal = -plane_normal

    helper = np.array([1.0, 0.0, 0.0]) if abs(plane_normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(helper, plane_normal)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(plane_normal, x_axis)

    R = np.column_stack([x_axis, y_axis, plane_normal])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = centroid

    if len(offplane_cloud.points) < dbscan_min_points:
        raise ValueError(
            f"too few off-plane points for clustering: {len(offplane_cloud.points)}"
        )

    labels = np.asarray(
        offplane_cloud.cluster_dbscan(
            eps=dbscan_eps_m,
            min_points=dbscan_min_points,
            print_progress=False,
        )
    )
    valid = labels >= 0
    if not valid.any():
        raise ValueError("DBSCAN found no clusters meeting min_points")

    unique, counts = np.unique(labels[valid], return_counts=True)
    largest_label = int(unique[counts.argmax()])
    arm_indices = np.where(labels == largest_label)[0]
    arm_cloud = offplane_cloud.select_by_index(arm_indices.tolist())

    return SegmentResult(
        T_world_table=T,
        plane_model=(float(a), float(b), float(c), float(d)),
        table_cloud=table_cloud,
        arm_cloud=arm_cloud,
        inlier_rmse_m=inlier_rmse_m,
    )
