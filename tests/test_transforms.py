"""Unit tests for XYZW quaternion + SE(3) helpers."""

from __future__ import annotations

import numpy as np
import pytest

from isaac_auto_scene.utils.transforms import (
    average_quaternions_xyzw,
    mean_se3,
    quat_xyzw_to_rotation_matrix,
    rotation_matrix_to_quat_xyzw,
)


def _R_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def test_quat_roundtrip_identity():
    q = rotation_matrix_to_quat_xyzw(np.eye(3))
    np.testing.assert_allclose(np.abs(q), [0, 0, 0, 1], atol=1e-9)
    R = quat_xyzw_to_rotation_matrix(q)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-12)


def test_quat_roundtrip_random():
    rng = np.random.default_rng(1)
    for _ in range(20):
        angle = rng.uniform(-np.pi, np.pi)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        K = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        q = rotation_matrix_to_quat_xyzw(R)
        R2 = quat_xyzw_to_rotation_matrix(q)
        np.testing.assert_allclose(R, R2, atol=1e-9)


def test_average_quaternions_single_returns_input():
    q = rotation_matrix_to_quat_xyzw(_R_z(0.4))
    out = average_quaternions_xyzw(np.stack([q]))
    np.testing.assert_allclose(out, q, atol=1e-10)


def test_average_quaternions_symmetric_pair():
    q1 = rotation_matrix_to_quat_xyzw(_R_z(0.2))
    q2 = rotation_matrix_to_quat_xyzw(_R_z(-0.2))
    mean = average_quaternions_xyzw(np.stack([q1, q2]))
    # By symmetry, mean rotation = identity.
    R = quat_xyzw_to_rotation_matrix(mean)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-6)


def test_average_quaternions_antipodal_robust():
    """Markley's method must handle sign-flipped inputs."""
    q = rotation_matrix_to_quat_xyzw(_R_z(0.3))
    mean = average_quaternions_xyzw(np.stack([q, -q]))
    # Either +q or -q is a valid mean; both encode the same rotation.
    R_mean = quat_xyzw_to_rotation_matrix(mean)
    R_target = quat_xyzw_to_rotation_matrix(q)
    np.testing.assert_allclose(R_mean, R_target, atol=1e-6)


def test_average_quaternions_weights_bias_result():
    q1 = rotation_matrix_to_quat_xyzw(_R_z(0.4))
    q2 = rotation_matrix_to_quat_xyzw(_R_z(-0.4))
    # Heavy weight on q1: mean should rotate toward +0.4 rad.
    mean = average_quaternions_xyzw(np.stack([q1, q2]), np.array([10.0, 1.0]))
    R = quat_xyzw_to_rotation_matrix(mean)
    # Angle of R about z must be positive and close to 0.4 * 10/11 ≈ 0.36 rad.
    angle = np.arctan2(R[1, 0], R[0, 0])
    assert angle > 0.2


def test_average_quaternions_rejects_bad_shape():
    with pytest.raises(ValueError):
        average_quaternions_xyzw(np.zeros((3,)))
    with pytest.raises(ValueError):
        average_quaternions_xyzw(np.zeros((0, 4)))


def test_average_quaternions_rejects_zero_norm():
    q = rotation_matrix_to_quat_xyzw(np.eye(3))
    with pytest.raises(ValueError):
        average_quaternions_xyzw(np.stack([q, np.zeros(4)]))


def test_mean_se3_identical_inputs():
    T = _T(_R_z(0.3), np.array([0.1, -0.2, 0.05]))
    stack = np.stack([T, T, T])
    T_mean, q_mean, dispersion = mean_se3(stack)
    np.testing.assert_allclose(T_mean, T, atol=1e-9)
    assert dispersion < 1e-9


def test_mean_se3_translation_average():
    Ts = np.stack(
        [_T(np.eye(3), np.array([0.0, 0.0, 0.0])),
         _T(np.eye(3), np.array([1.0, 0.0, 0.0]))]
    )
    T_mean, _, dispersion = mean_se3(Ts)
    np.testing.assert_allclose(T_mean[:3, 3], [0.5, 0.0, 0.0], atol=1e-9)
    assert dispersion < 1e-9


def test_mean_se3_weighted_translation():
    Ts = np.stack(
        [_T(np.eye(3), np.array([0.0, 0.0, 0.0])),
         _T(np.eye(3), np.array([1.0, 0.0, 0.0]))]
    )
    T_mean, _, _ = mean_se3(Ts, np.array([3.0, 1.0]))
    np.testing.assert_allclose(T_mean[:3, 3], [0.25, 0.0, 0.0], atol=1e-9)


def test_mean_se3_rejects_empty_stack():
    with pytest.raises(ValueError, match="needs >= 1"):
        mean_se3(np.zeros((0, 4, 4)))


def test_mean_se3_rejects_negative_weights():
    Ts = np.stack([_T(np.eye(3), np.zeros(3)), _T(np.eye(3), np.ones(3))])
    with pytest.raises(ValueError, match="non-negative"):
        mean_se3(Ts, np.array([1.0, -1.0]))


def test_mean_se3_rejects_zero_weight_sum():
    Ts = np.stack([_T(np.eye(3), np.zeros(3)), _T(np.eye(3), np.ones(3))])
    with pytest.raises(ValueError, match="positive sum"):
        mean_se3(Ts, np.array([0.0, 0.0]))


def test_mean_se3_dispersion_for_spread():
    Ts = np.stack([_T(_R_z(0.2), np.zeros(3)), _T(_R_z(-0.2), np.zeros(3))])
    _, _, dispersion = mean_se3(Ts)
    # Each transform sits ~0.2 rad from the identity mean.
    assert 0.15 < dispersion < 0.25
