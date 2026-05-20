"""Tests for isaac_auto_scene.cli (Phase 6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from isaac_auto_scene.cli import build_parser, main
from tests.fixtures.minimal_urdf import write_minimal_urdf


def test_help_does_not_error() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_calibrate_subcommand_runs_mock(tmp_path: Path) -> None:
    urdf = write_minimal_urdf(tmp_path)
    calib_path = tmp_path / "calib.json"

    rc = main(
        [
            "calibrate",
            "--urdf", str(urdf),
            "--mock",
            "--frames", "5",
            "--target-n-points", "3000",
            "--restarts", "2",
            "--out", str(calib_path),
        ]
    )
    # Mock data is geometrically very different from the URDF box, so the
    # quality gate is allowed to fail (rc=2) — we just need the file to be
    # produced with a valid schema.
    assert rc in (0, 2)
    assert calib_path.exists()
    data = json.loads(calib_path.read_text())
    for key in (
        "T_cam_arm",
        "quat_xyzw",
        "translation_m",
        "intrinsics",
        "icp_fitness",
        "inlier_rmse_m",
        "joint_angles_at_capture",
    ):
        assert key in data, f"missing calib key: {key}"


def test_generate_subcommand_writes_usd(tmp_path: Path) -> None:
    urdf = write_minimal_urdf(tmp_path)
    calib_path = tmp_path / "calib.json"
    scene_path = tmp_path / "scene.usda"

    rc_cal = main(
        [
            "calibrate",
            "--urdf", str(urdf),
            "--mock",
            "--frames", "5",
            "--target-n-points", "3000",
            "--restarts", "2",
            "--out", str(calib_path),
        ]
    )
    assert rc_cal in (0, 2)

    rc_gen = main(
        ["generate", "--calib", str(calib_path), "--out", str(scene_path)]
    )
    assert rc_gen == 0
    assert scene_path.exists() and scene_path.stat().st_size > 200


def test_validate_subcommand_emits_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    urdf = write_minimal_urdf(tmp_path)
    calib_path = tmp_path / "calib.json"
    scene_path = tmp_path / "scene.usda"

    main(
        [
            "calibrate",
            "--urdf", str(urdf),
            "--mock",
            "--frames", "5",
            "--target-n-points", "3000",
            "--restarts", "2",
            "--out", str(calib_path),
        ]
    )
    main(["generate", "--calib", str(calib_path), "--out", str(scene_path)])
    capsys.readouterr()  # drop stdout from above

    rc = main(["validate", "--calib", str(calib_path), "--scene", str(scene_path)])
    out = capsys.readouterr().out
    assert rc in (0, 2)
    parsed = json.loads(out)
    assert "icp_fitness" in parsed
    assert "quality_gate_pass" in parsed
