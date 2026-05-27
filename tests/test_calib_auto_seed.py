"""Tests for the auto-seed behaviour of manual-align and manual-align-all.

The GUI viewer (run_manual_align) and live-capture machinery are patched out so
no display or hardware is required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IDENTITY = [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]]

# Non-identity transform: translate 0.1 m along X.
_T_SEED = [[1.0, 0.0, 0.0, 0.1],
           [0.0, 1.0, 0.0, 0.0],
           [0.0, 0.0, 1.0, 0.0],
           [0.0, 0.0, 0.0, 1.0]]

_QUAT_IDENTITY = [0.0, 0.0, 0.0, 1.0]  # XYZW
_TRANSLATION_SEED = [0.1, 0.0, 0.0]


def _write_seed_calib(path: Path) -> None:
    """Write a minimal calib.json with a known non-identity transform."""
    data = {
        "T_cam_arm": _T_SEED,
        "quat_xyzw": _QUAT_IDENTITY,
        "translation_m": _TRANSLATION_SEED,
        "intrinsics": {
            "fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0,
            "width": 640, "height": 480, "depth_unit": 0.001,
        },
        "icp_fitness": 0.9,
        "inlier_rmse_m": 0.002,
        "joint_angles_at_capture": {"shoulder": 0.0},
        "T_cam_table": None,
    }
    path.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# Unit tests for the _resolve_init_calib helper
# ---------------------------------------------------------------------------

class TestResolveInitCalib:
    def test_explicit_wins_over_default(self, tmp_path: Path) -> None:
        from isaac_auto_scene.cli import _resolve_init_calib

        default = tmp_path / "default_calib.json"
        default.write_text("{}")  # exists
        explicit = str(tmp_path / "explicit.json")

        with patch("isaac_auto_scene.cli._default_calib_path", return_value=default):
            result = _resolve_init_calib(explicit, no_auto_seed=False)
        assert result == explicit

    def test_no_auto_seed_returns_none(self, tmp_path: Path) -> None:
        from isaac_auto_scene.cli import _resolve_init_calib

        default = tmp_path / "default_calib.json"
        default.write_text("{}")  # exists but should be ignored

        with patch("isaac_auto_scene.cli._default_calib_path", return_value=default):
            result = _resolve_init_calib(None, no_auto_seed=True)
        assert result is None

    def test_auto_seeds_from_default_when_exists(self, tmp_path: Path) -> None:
        from isaac_auto_scene.cli import _resolve_init_calib

        default = tmp_path / "calib.json"
        default.write_text("{}")

        with patch("isaac_auto_scene.cli._default_calib_path", return_value=default):
            result = _resolve_init_calib(None, no_auto_seed=False)
        assert result == str(default)

    def test_returns_none_when_default_absent(self, tmp_path: Path) -> None:
        from isaac_auto_scene.cli import _resolve_init_calib

        missing = tmp_path / "nonexistent.json"

        with patch("isaac_auto_scene.cli._default_calib_path", return_value=missing):
            result = _resolve_init_calib(None, no_auto_seed=False)
        assert result is None


# ---------------------------------------------------------------------------
# Integration-level tests for cmd_manual_align
# ---------------------------------------------------------------------------

def _make_manual_align_args(
    tmp_path: Path,
    *,
    init_from: str | None = None,
    no_auto_seed: bool = False,
) -> Any:
    """Build a minimal args namespace for cmd_manual_align."""
    import argparse
    from tests.fixtures.minimal_urdf import write_minimal_urdf

    urdf = write_minimal_urdf(tmp_path)
    ns = argparse.Namespace(
        urdf=str(urdf),
        live=False,
        captures=None,
        pose=None,
        arm_port="/dev/ttyACM0",
        frames=5,
        home_offset=None,
        target_n_points=500,
        workspace_z_max=None,
        workspace_z_min=None,
        expected_up=None,
        up_tol_deg=30.0,
        arm_merge_radius=0.0,
        outlier_neighbors=0,
        outlier_std=2.0,
        step=0.01,
        rot_step=5.0,
        icp_threshold=0.02,
        no_icp_refine=False,
        out=str(tmp_path / "out_calib.json"),
        init_from=init_from,
        no_auto_seed=no_auto_seed,
    )
    return ns


def _fake_capture_result(tmp_path: Path) -> MagicMock:
    """Fake CaptureResult with minimal fields."""
    import open3d as o3d

    cap = MagicMock()
    pts = np.random.default_rng(0).uniform(0, 1, (50, 3)).astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    cap.pcd = pcd
    cap.intrinsics = MagicMock()
    cap.depth_image = np.zeros((480, 640), dtype=np.float32)
    cap.color_image = np.zeros((480, 640, 3), dtype=np.uint8)
    return cap


def _fake_segment_result() -> MagicMock:
    import open3d as o3d

    seg = MagicMock()
    pts = np.random.default_rng(1).uniform(0, 0.5, (30, 3)).astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    seg.arm_cloud = pcd
    seg.T_world_table = np.eye(4)
    return seg


def _run_manual_align_with_mocks(
    args: Any,
    tmp_path: Path,
    seed_calib_path: Path | None,
) -> "tuple[np.ndarray | None, MagicMock]":
    """Run cmd_manual_align with all heavy machinery patched.

    Returns (T_init_captured, run_manual_align_mock).
    T_init_captured is the T_init passed to run_manual_align.
    """
    cap = _fake_capture_result(tmp_path)
    seg = _fake_segment_result()

    # T_init is captured from the call to run_manual_align
    captured: dict[str, Any] = {}

    def fake_run_manual_align(cad_pts, scene_pts, T_init, **kwargs):  # noqa: ANN001
        captured["T_init"] = T_init.copy()
        return T_init  # return non-None so save path executes

    mock_run = MagicMock(side_effect=fake_run_manual_align)

    with (
        patch(
            "isaac_auto_scene.cli._live_capture_for_manual_align",
            return_value=(cap, {"shoulder": 0.0}),
        ),
        patch("isaac_auto_scene.segment.segment_table_arm", return_value=seg),
        patch("isaac_auto_scene.manual_align.run_manual_align", mock_run),
        patch(
            "isaac_auto_scene.cli._default_calib_path",
            return_value=seed_calib_path if seed_calib_path else (tmp_path / "no_calib.json"),
        ),
        patch("isaac_auto_scene.calibrate.save_calibration"),
        patch("isaac_auto_scene.calibrate.build_calibration", return_value=MagicMock()),
    ):
        from isaac_auto_scene.cli import cmd_manual_align
        cmd_manual_align(args)

    return captured.get("T_init"), mock_run


class TestManualAlignAutoSeed:
    def test_auto_seeds_from_default_calib(self, tmp_path: Path) -> None:
        """With default calib present and no --init-from, T_init is seeded from it."""
        default_calib = tmp_path / "calib.json"
        _write_seed_calib(default_calib)

        args = _make_manual_align_args(tmp_path, init_from=None, no_auto_seed=False)
        T_init, _ = _run_manual_align_with_mocks(args, tmp_path, seed_calib_path=default_calib)

        assert T_init is not None
        # The seed translation is [0.1, 0, 0]
        np.testing.assert_allclose(T_init[:3, 3], [0.1, 0.0, 0.0], atol=1e-9)

    def test_explicit_init_from_wins_over_default(self, tmp_path: Path) -> None:
        """Explicit --init-from is used even when a default calib exists."""
        default_calib = tmp_path / "calib.json"
        _write_seed_calib(default_calib)

        explicit_calib = tmp_path / "explicit.json"
        # Explicit has a different translation [0.5, 0, 0]
        explicit_data = json.loads(default_calib.read_text())
        explicit_data["translation_m"] = [0.5, 0.0, 0.0]
        explicit_data["T_cam_arm"] = (
            [[1.0, 0.0, 0.0, 0.5],
             [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]]
        )
        explicit_calib.write_text(json.dumps(explicit_data))

        args = _make_manual_align_args(tmp_path, init_from=str(explicit_calib), no_auto_seed=False)
        T_init, _ = _run_manual_align_with_mocks(args, tmp_path, seed_calib_path=default_calib)

        assert T_init is not None
        np.testing.assert_allclose(T_init[:3, 3], [0.5, 0.0, 0.0], atol=1e-9)

    def test_no_auto_seed_uses_identity(self, tmp_path: Path) -> None:
        """--no-auto-seed with a default calib present results in identity T_init."""
        default_calib = tmp_path / "calib.json"
        _write_seed_calib(default_calib)

        args = _make_manual_align_args(tmp_path, init_from=None, no_auto_seed=True)
        T_init, _ = _run_manual_align_with_mocks(args, tmp_path, seed_calib_path=default_calib)

        assert T_init is not None
        np.testing.assert_allclose(T_init, np.eye(4), atol=1e-9)

    def test_identity_when_no_calib_and_no_init_from(self, tmp_path: Path) -> None:
        """When default calib does not exist and no --init-from, T_init is identity."""
        args = _make_manual_align_args(tmp_path, init_from=None, no_auto_seed=False)
        # seed_calib_path=None -> default path points at nonexistent file
        T_init, _ = _run_manual_align_with_mocks(args, tmp_path, seed_calib_path=None)

        assert T_init is not None
        np.testing.assert_allclose(T_init, np.eye(4), atol=1e-9)


# ---------------------------------------------------------------------------
# Argparse: verify new flags parse correctly
# ---------------------------------------------------------------------------

class TestManualAlignArgparse:
    def test_no_auto_seed_flag_manual_align(self) -> None:
        from isaac_auto_scene.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "manual-align",
            "--urdf", "/tmp/x.urdf",
            "--out", "/tmp/c.json",
            "--no-auto-seed",
        ])
        assert args.no_auto_seed is True

    def test_no_auto_seed_default_false_manual_align(self) -> None:
        from isaac_auto_scene.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "manual-align",
            "--urdf", "/tmp/x.urdf",
            "--out", "/tmp/c.json",
        ])
        assert args.no_auto_seed is False

    def test_no_auto_seed_flag_manual_align_all(self) -> None:
        from isaac_auto_scene.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "manual-align-all",
            "--captures", "/tmp/caps",
            "--urdf", "/tmp/x.urdf",
            "--no-auto-seed",
        ])
        assert args.no_auto_seed is True

    def test_init_from_help_no_longer_claims_default_search(self) -> None:
        """The old false claim 'default search' must not appear in the help."""
        from isaac_auto_scene.cli import build_parser
        import io

        parser = build_parser()
        buf = io.StringIO()
        try:
            parser.parse_args(["manual-align", "--help"])
        except SystemExit:
            pass
        # Build sub-parser and inspect its help directly
        sub_parser = None
        for action in parser._actions:
            if hasattr(action, '_name_parser_map'):
                sub_parser = action._name_parser_map.get("manual-align")
                break
        assert sub_parser is not None
        help_text = sub_parser.format_help()
        assert "default search" not in help_text


# ---------------------------------------------------------------------------
# Integration-level tests for cmd_manual_align_all auto-seed behaviour
# ---------------------------------------------------------------------------

def _make_manual_align_all_args(
    tmp_path: Path,
    *,
    captures_dir: Path,
    init_from: str | None = None,
    no_auto_seed: bool = False,
) -> Any:
    """Build a minimal args namespace for cmd_manual_align_all."""
    import argparse
    from tests.fixtures.minimal_urdf import write_minimal_urdf

    urdf = write_minimal_urdf(tmp_path)
    ns = argparse.Namespace(
        urdf=str(urdf),
        captures=str(captures_dir),
        out_dir=str(tmp_path / "align_all_out"),
        home_offset=None,
        target_n_points=500,
        workspace_z_max=None,
        workspace_z_min=None,
        expected_up=None,
        up_tol_deg=30.0,
        arm_merge_radius=0.0,
        outlier_neighbors=0,
        outlier_std=2.0,
        step=0.01,
        rot_step=5.0,
        icp_threshold=0.02,
        no_icp_refine=False,
        init_from=init_from,
        no_auto_seed=no_auto_seed,
    )
    return ns


def _make_fake_manifest(tmp_path: Path) -> MagicMock:
    """Return a minimal manifest mock with one ok pose record."""
    from isaac_auto_scene.capture_multi import PoseCaptureRecord

    rec = PoseCaptureRecord(
        name="pose0",
        status="ok",
        commanded_joints={"shoulder": 0.0},
        readback_joints={"shoulder": 0.0},
        pose_dir="pose0",
    )
    manifest = MagicMock()
    manifest.poses = [rec]
    return manifest


def _run_manual_align_all_with_mocks(
    args: Any,
    tmp_path: Path,
    seed_calib_path: Path | None,
) -> "tuple[np.ndarray | None, MagicMock]":
    """Run cmd_manual_align_all with all heavy machinery patched.

    Returns (T_init_captured, run_manual_align_mock).
    T_init_captured is the T_init passed to run_manual_align for the first pose.
    """
    import open3d as o3d

    # Fake point cloud for capture and segment results
    pts = np.random.default_rng(42).uniform(0, 1, (50, 3)).astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    cap = MagicMock()
    cap.pcd = pcd
    cap.intrinsics = MagicMock()

    seg = MagicMock()
    seg.arm_cloud = pcd
    seg.T_world_table = np.eye(4)

    # cad mock with .points
    cad = MagicMock()
    cad.points = pcd

    # Capture T_init from run_manual_align call
    captured: dict[str, Any] = {}

    def fake_run_manual_align(cad_pts, scene_pts, T_init, **kwargs):  # noqa: ANN001
        captured["T_init"] = T_init.copy()
        return T_init

    mock_run = MagicMock(side_effect=fake_run_manual_align)

    # bundle_register result mock
    bundle_result = MagicMock()
    bundle_result.T = np.eye(4)
    bundle_result.per_pose_fitness = [0.9]
    bundle_result.per_pose_rmse_m = [0.001]

    with (
        patch(
            "isaac_auto_scene.capture_multi.load_manifest",
            return_value=_make_fake_manifest(tmp_path),
        ),
        patch("isaac_auto_scene.capture_multi.load_pose_capture", return_value=cap),
        patch("isaac_auto_scene.cad.load_urdf", return_value=MagicMock()),
        patch("isaac_auto_scene.cad.assemble_pcd", return_value=cad),
        patch("isaac_auto_scene.segment.segment_table_arm", return_value=seg),
        patch("isaac_auto_scene.manual_align.run_manual_align", mock_run),
        patch("isaac_auto_scene.bundle_register.register_bundle", return_value=bundle_result),
        patch("isaac_auto_scene.calibrate.build_calibration", return_value=MagicMock()),
        patch("isaac_auto_scene.calibrate.save_calibration"),
        patch(
            "isaac_auto_scene.cli._default_calib_path",
            return_value=seed_calib_path if seed_calib_path else (tmp_path / "no_calib.json"),
        ),
    ):
        from isaac_auto_scene.cli import cmd_manual_align_all
        cmd_manual_align_all(args)

    return captured.get("T_init"), mock_run


class TestManualAlignAllAutoSeed:
    def test_no_auto_seed_uses_identity(self, tmp_path: Path) -> None:
        """--no-auto-seed with a default calib present results in identity T_init.

        Mirrors test_no_auto_seed_uses_identity from TestManualAlignAutoSeed but
        for cmd_manual_align_all, which reads T_cam_arm directly (not rebuilt from
        quat/translation like cmd_manual_align does).
        """
        default_calib = tmp_path / "calib.json"
        _write_seed_calib(default_calib)

        args = _make_manual_align_all_args(
            tmp_path,
            captures_dir=tmp_path / "caps",
            init_from=None,
            no_auto_seed=True,
        )
        T_init, _ = _run_manual_align_all_with_mocks(args, tmp_path, seed_calib_path=default_calib)

        assert T_init is not None
        np.testing.assert_allclose(T_init, np.eye(4), atol=1e-9)

    def test_auto_seeds_from_default_calib_via_T_cam_arm(self, tmp_path: Path) -> None:
        """Without --no-auto-seed, cmd_manual_align_all seeds T_init from T_cam_arm directly.

        Unlike cmd_manual_align which rebuilds T from quat/translation fields,
        cmd_manual_align_all uses prev.T_cam_arm directly. Both paths go through
        _resolve_init_calib → the same auto-seed logic.
        """
        default_calib = tmp_path / "calib.json"
        _write_seed_calib(default_calib)  # T_cam_arm has [0.1, 0, 0] translation

        args = _make_manual_align_all_args(
            tmp_path,
            captures_dir=tmp_path / "caps",
            init_from=None,
            no_auto_seed=False,
        )
        T_init, _ = _run_manual_align_all_with_mocks(args, tmp_path, seed_calib_path=default_calib)

        assert T_init is not None
        # _T_SEED has translation [0.1, 0, 0] stored in T_cam_arm
        np.testing.assert_allclose(T_init[:3, 3], [0.1, 0.0, 0.0], atol=1e-9)
