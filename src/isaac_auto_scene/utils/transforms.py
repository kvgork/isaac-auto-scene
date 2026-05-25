"""Quaternion and rigid-body transform helpers (XYZW convention throughout).

Public API
----------
rotation_matrix_to_quat_xyzw  — 3x3 rotation -> unit (x, y, z, w)
quat_xyzw_to_rotation_matrix  — (x, y, z, w) -> 3x3 rotation
average_quaternions_xyzw      — Markley weighted quaternion mean
mean_se3                      — weighted mean of N SE(3) transforms

Convention: XYZW order matches Isaac Lab + SciPy default.
"""

from __future__ import annotations

import numpy as np


def rotation_matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a unit XYZW quaternion.

    Implementation matches `calibrate._rot_to_quat_xyzw` and uses
    Shepperd's method (numerically stable across sign of trace).
    """
    m = np.asarray(R, dtype=np.float64)
    tr = m.trace()
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_xyzw_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a unit XYZW quaternion to a 3x3 rotation matrix."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
            [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
            [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return R


def average_quaternions_xyzw(
    quats: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """Markley's weighted mean quaternion (XYZW).

    Reference: F. Landis Markley et al., "Averaging Quaternions",
    J. Guidance, Control, and Dynamics, Vol. 30, No. 4, 2007.

    The mean is the eigenvector with the largest eigenvalue of
    ``M = sum_i w_i * q_i q_i^T``.  This is rotation-sign invariant — no
    pre-flipping of antipodal quaternions is needed.

    Parameters
    ----------
    quats:
        (N, 4) array of XYZW quaternions.  Each row should be unit-norm.
    weights:
        Optional (N,) weights.  Non-negative.  Defaults to uniform.

    Returns
    -------
    np.ndarray
        Unit XYZW quaternion of shape (4,).
    """
    q = np.asarray(quats, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"quats must have shape (N, 4); got {q.shape}")
    if q.shape[0] == 0:
        raise ValueError("average_quaternions_xyzw needs >= 1 quaternion")

    norms = np.linalg.norm(q, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("zero-norm quaternion in input")
    q = q / norms

    if weights is None:
        w = np.ones(q.shape[0], dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (q.shape[0],):
            raise ValueError(
                f"weights shape {w.shape} does not match {q.shape[0]} quats"
            )
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        if w.sum() <= 0:
            raise ValueError("weights must have positive sum")

    M = (w[:, None, None] * (q[:, :, None] @ q[:, None, :])).sum(axis=0)
    eigvals, eigvecs = np.linalg.eigh(M)
    q_mean = eigvecs[:, np.argmax(eigvals)]
    if q_mean[3] < 0.0:  # canonicalise sign: prefer w >= 0
        q_mean = -q_mean
    return q_mean / np.linalg.norm(q_mean)


def mean_se3(
    transforms: np.ndarray, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """Weighted mean of N SE(3) transforms.

    Translation is a weighted arithmetic mean; rotation is Markley quaternion
    averaging.  Returns the mean transform plus a scalar dispersion metric
    (mean rotation angle to mean, in radians) — useful for detecting an
    inconsistent set of pose estimates.

    Parameters
    ----------
    transforms:
        (N, 4, 4) stack of homogeneous transforms.
    weights:
        Optional (N,) non-negative weights; defaults to uniform.

    Returns
    -------
    T_mean:
        (4, 4) mean transform.
    quat_mean_xyzw:
        (4,) mean rotation as XYZW quaternion.
    dispersion_rad:
        Weighted-mean rotation angle from each input to the mean (radians).
    """
    Ts = np.asarray(transforms, dtype=np.float64)
    if Ts.ndim != 3 or Ts.shape[1:] != (4, 4):
        raise ValueError(f"transforms must have shape (N, 4, 4); got {Ts.shape}")
    n = Ts.shape[0]
    if n == 0:
        raise ValueError("mean_se3 needs >= 1 transform")

    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (n,):
            raise ValueError(f"weights shape {w.shape} does not match {n} transforms")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        if w.sum() <= 0:
            raise ValueError("weights must have positive sum")

    w_norm = w / w.sum()
    t_mean = (w_norm[:, None] * Ts[:, :3, 3]).sum(axis=0)

    quats = np.stack(
        [rotation_matrix_to_quat_xyzw(Ts[i, :3, :3]) for i in range(n)], axis=0
    )
    q_mean = average_quaternions_xyzw(quats, w)
    R_mean = quat_xyzw_to_rotation_matrix(q_mean)

    # Dispersion: weighted-mean angle between each rotation and the mean.
    dots = np.abs(np.einsum("ij,j->i", quats, q_mean))
    dots = np.clip(dots, -1.0, 1.0)
    angles = 2.0 * np.arccos(dots)
    dispersion = float((w_norm * angles).sum())

    T_mean = np.eye(4, dtype=np.float64)
    T_mean[:3, :3] = R_mean
    T_mean[:3, 3] = t_mean
    return T_mean, q_mean, dispersion
