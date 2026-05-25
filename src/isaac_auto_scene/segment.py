"""Plane + arm segmentation pipeline (Phase 3).

Public API
----------
SegmentResult        — frozen dataclass: T_world_table, arm_cloud, table_cloud
segment_table_arm()  — Open3D RANSAC plane + DBSCAN cluster largest

Strategy
--------
1. Optional Z-range crop (rejects far-background points that otherwise
   dominate the RANSAC inlier count).
2. Detect dominant plane with ``segment_plane`` (5 mm threshold).
3. Optionally constrain the plane normal to be within ``up_tolerance_deg``
   of ``expected_up`` — if the first RANSAC plane is sloped (e.g. a wall),
   peel it off and re-fit until the up-aligned plane is found.
4. Build ``T_world_table`` so the plane normal becomes +Z and a point on
   the plane becomes the origin (right-handed frame, X axis arbitrary).
5. Split the remaining points (non-plane) into clusters with
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
    workspace_z_max_m: float | None = None,
    workspace_z_min_m: float | None = None,
    expected_up: tuple[float, float, float] | None = None,
    up_tolerance_deg: float = 30.0,
    max_plane_attempts: int = 5,
    arm_merge_radius_m: float = 0.0,
    outlier_nb_neighbors: int = 0,
    outlier_std_ratio: float = 2.0,
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
    workspace_z_max_m:
        Optional Z-axis crop (metres in input frame).  Points with z above
        this are discarded before plane fit.  Use to suppress background
        walls / monitors that otherwise dominate the largest-plane search.
    workspace_z_min_m:
        Optional lower Z-axis bound (e.g. 0.1 m drops sensor noise just
        in front of the lens).
    expected_up:
        Optional 3-vector giving the expected plane normal direction in
        the input frame.  When supplied, planes whose normal differs from
        this direction by more than ``up_tolerance_deg`` are rejected and
        the search continues by peeling those inliers off and re-fitting.
        Useful when a wall/desk side is bigger than the table.
    up_tolerance_deg:
        Angular tolerance (degrees) used with ``expected_up``.  Default 30°
        catches table-mount slop without admitting walls.
    max_plane_attempts:
        Safety cap on the peel-and-refit loop when ``expected_up`` is set.
    arm_merge_radius_m:
        When > 0, also include any DBSCAN cluster whose centroid is within
        this distance of the largest cluster's centroid (after taking the
        largest first).  Useful when an articulated arm appears as two or
        three disconnected clusters in the depth cloud (e.g. base + forearm
        separated by a thin servo joint that DBSCAN does not bridge).
        SO-101 reach ≈ 0.30 m, so a default of 0.0 (off) is conservative;
        callers exposing this via the CLI typically pass 0.30.
    outlier_nb_neighbors:
        When > 0, run Open3D ``remove_statistical_outlier`` on the final
        arm cloud with this neighbour count.  Drops sparse outliers — most
        commonly the cable/mount stragglers that pad the bounding box and
        push ICP RMSE above the quality gate.  20 is a reasonable starting
        value for D435 clouds at 640×480.
    outlier_std_ratio:
        Standard-deviation multiplier for the outlier filter (lower = more
        aggressive).  Ignored when ``outlier_nb_neighbors`` is 0.

    Returns
    -------
    SegmentResult

    Raises
    ------
    ValueError
        If ``pcd`` has fewer than 100 points after cropping, no off-plane
        cluster satisfies ``dbscan_min_points``, or no plane within
        ``up_tolerance_deg`` of ``expected_up`` is found in
        ``max_plane_attempts`` RANSAC passes.
    """
    if len(pcd.points) < 100:
        raise ValueError(f"input pcd too small: {len(pcd.points)} points")

    work_pcd = _apply_workspace_crop(pcd, workspace_z_min_m, workspace_z_max_m)
    if len(work_pcd.points) < 100:
        raise ValueError(
            f"after workspace crop only {len(work_pcd.points)} points remain "
            "(check workspace_z_min_m / workspace_z_max_m)"
        )

    plane_model_tuple, inlier_indices, peeled_indices = _find_up_aligned_plane(
        work_pcd,
        plane_threshold_m=plane_threshold_m,
        plane_ransac_n=plane_ransac_n,
        plane_num_iterations=plane_num_iterations,
        expected_up=expected_up,
        up_tolerance_deg=up_tolerance_deg,
        max_plane_attempts=max_plane_attempts,
    )
    a, b, c, d = plane_model_tuple

    table_cloud = work_pcd.select_by_index(inlier_indices)
    # Exclude both the table inliers AND any sloped planes peeled along
    # the way — those are walls / desk-sides whose points would otherwise
    # form a giant DBSCAN cluster that swamps the arm.
    excluded = sorted(set(inlier_indices) | set(peeled_indices))
    offplane_cloud = work_pcd.select_by_index(excluded, invert=True)

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

    offplane_pts = np.asarray(offplane_cloud.points)
    if arm_merge_radius_m > 0.0:
        largest_centroid = offplane_pts[arm_indices].mean(axis=0)
        for lbl in unique:
            if int(lbl) == largest_label:
                continue
            idx = np.where(labels == lbl)[0]
            centroid = offplane_pts[idx].mean(axis=0)
            if np.linalg.norm(centroid - largest_centroid) <= arm_merge_radius_m:
                arm_indices = np.concatenate([arm_indices, idx])

    arm_cloud = offplane_cloud.select_by_index(arm_indices.tolist())

    if outlier_nb_neighbors > 0 and len(arm_cloud.points) > outlier_nb_neighbors:
        arm_cloud, _ = arm_cloud.remove_statistical_outlier(
            nb_neighbors=int(outlier_nb_neighbors),
            std_ratio=float(outlier_std_ratio),
        )

    return SegmentResult(
        T_world_table=T,
        plane_model=(float(a), float(b), float(c), float(d)),
        table_cloud=table_cloud,
        arm_cloud=arm_cloud,
        inlier_rmse_m=inlier_rmse_m,
    )


def _apply_workspace_crop(
    pcd: o3d.geometry.PointCloud,
    z_min: float | None,
    z_max: float | None,
) -> o3d.geometry.PointCloud:
    """Return a Z-axis cropped copy of ``pcd`` (no-op when both bounds are None)."""
    if z_min is None and z_max is None:
        return pcd
    pts = np.asarray(pcd.points)
    mask = np.ones(len(pts), dtype=bool)
    if z_min is not None:
        mask &= pts[:, 2] >= float(z_min)
    if z_max is not None:
        mask &= pts[:, 2] <= float(z_max)
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts[mask])
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        out.colors = o3d.utility.Vector3dVector(colors[mask])
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        out.normals = o3d.utility.Vector3dVector(normals[mask])
    return out


def _find_up_aligned_plane(
    pcd: o3d.geometry.PointCloud,
    *,
    plane_threshold_m: float,
    plane_ransac_n: int,
    plane_num_iterations: int,
    expected_up: tuple[float, float, float] | None,
    up_tolerance_deg: float,
    max_plane_attempts: int,
) -> tuple[tuple[float, float, float, float], list[int], list[int]]:
    """Peel off mis-oriented planes until one fits ``expected_up``.

    When ``expected_up`` is None, behaves identically to a single
    ``segment_plane`` call (largest-plane semantics preserved).

    Returns
    -------
    plane_model:
        (a, b, c, d) for the chosen up-aligned plane.
    inlier_indices:
        Indices (into the input pcd) of the chosen plane's inliers.
    peeled_indices:
        Indices (into the input pcd) of all mis-aligned planes that were
        peeled off during the search.  Caller should exclude these from
        the off-plane cloud — otherwise a peeled wall ends up in DBSCAN
        and dominates the largest cluster.
    """
    if expected_up is None:
        plane, inliers = pcd.segment_plane(
            distance_threshold=plane_threshold_m,
            ransac_n=plane_ransac_n,
            num_iterations=plane_num_iterations,
        )
        return plane, list(inliers), []

    up = np.asarray(expected_up, dtype=np.float64)
    up /= np.linalg.norm(up)
    tol_cos = float(np.cos(np.deg2rad(up_tolerance_deg)))

    remaining_idx = np.arange(len(pcd.points), dtype=np.int64)
    peeled: list[int] = []
    work = pcd

    for _ in range(max_plane_attempts):
        plane, local_inliers = work.segment_plane(
            distance_threshold=plane_threshold_m,
            ransac_n=plane_ransac_n,
            num_iterations=plane_num_iterations,
        )
        normal = np.asarray(plane[:3], dtype=np.float64)
        normal /= np.linalg.norm(normal)
        if abs(float(normal @ up)) >= tol_cos:
            inliers_global = remaining_idx[list(local_inliers)]
            return plane, [int(i) for i in inliers_global], peeled

        # Mis-aligned plane: record its global indices, drop from work.
        peeled_global = remaining_idx[list(local_inliers)]
        peeled.extend(int(i) for i in peeled_global)
        all_local = np.arange(len(work.points), dtype=np.int64)
        keep_local = np.setdiff1d(all_local, np.asarray(local_inliers), assume_unique=False)
        if len(keep_local) < 100:
            break
        remaining_idx = remaining_idx[keep_local]
        work = pcd.select_by_index(remaining_idx.tolist())

    raise ValueError(
        f"no plane within {up_tolerance_deg:.1f}° of expected_up={tuple(up)} "
        f"found in {max_plane_attempts} RANSAC attempts"
    )
