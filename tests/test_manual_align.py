"""Tests for isaac_auto_scene.manual_align — module-load + helper functions.

The full GUI is exercised manually with a display attached.  These tests
cover the pure-Python pieces (axis rotations, translation helper) that
don't require an OpenGL context.
"""

from __future__ import annotations

import numpy as np
import pytest

from isaac_auto_scene.manual_align import _rot_axis, _translate, ManualAlignState


def test_module_imports_without_display() -> None:
    """Importing should not require a display server."""
    from isaac_auto_scene import manual_align

    assert hasattr(manual_align, "run_manual_align")


def test_translate_matrix() -> None:
    T = _translate(0.1, -0.2, 0.3)
    np.testing.assert_allclose(T[:3, 3], [0.1, -0.2, 0.3])
    np.testing.assert_allclose(T[:3, :3], np.eye(3))


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_rot_axis_orthonormal(axis: str) -> None:
    T = _rot_axis(axis, 0.4)
    R = T[:3, :3]
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert abs(np.linalg.det(R) - 1.0) < 1e-12


def test_rot_axis_quarter_turn_z() -> None:
    """Z quarter turn rotates +X into +Y."""
    T = _rot_axis("z", np.pi / 2)
    R = T[:3, :3]
    np.testing.assert_allclose(R @ np.array([1, 0, 0]), [0, 1, 0], atol=1e-12)


def test_rot_axis_invalid() -> None:
    with pytest.raises(ValueError, match="axis must be"):
        _rot_axis("w", 0.5)


def test_manual_align_state_dataclass() -> None:
    state = ManualAlignState(
        T=np.eye(4), T_init=np.eye(4), step_m=0.01, rot_step_rad=0.087
    )
    assert state.step_m == 0.01
    np.testing.assert_allclose(state.T, np.eye(4))


def test_manual_align_subparser_offline() -> None:
    """Offline mode (--captures + --pose) parses cleanly."""
    from isaac_auto_scene.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "manual-align",
            "--captures", "/tmp/x",
            "--urdf", "/tmp/x.urdf",
            "--pose", "home",
            "--out", "/tmp/calib.json",
        ]
    )
    assert args.cmd == "manual-align"
    assert args.pose == "home"
    assert args.live is False


def test_manual_align_subparser_live_default() -> None:
    """Live mode (only --urdf) works — --captures + --pose are optional."""
    from isaac_auto_scene.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "manual-align",
            "--urdf", "/tmp/x.urdf",
            "--out", "/tmp/calib.json",
        ]
    )
    assert args.cmd == "manual-align"
    assert args.captures is None
    assert args.pose is None
    assert args.arm_port == "/dev/ttyACM0"
    assert args.frames == 15
