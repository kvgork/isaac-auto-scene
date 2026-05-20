"""Smoke test for the Isaac Sim render pipeline (Phase 6 +).

Skipped automatically when the Isaac Sim Python interpreter is missing.
Marked ``hardware`` so it runs only with ``pytest --run-hardware``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from isaac_auto_scene.cli import build_parser, main
from tests.fixtures.minimal_urdf import write_minimal_urdf


ISAAC_PY = Path.home() / "workspaces/lerobot-isaac-training/.pixi/envs/sim/bin/python"


@pytest.mark.hardware
@pytest.mark.skipif(
    not ISAAC_PY.exists(),
    reason=f"Isaac Sim Python not found at {ISAAC_PY}",
)
def test_render_with_ros2_bridge(tmp_path: Path) -> None:
    """Smoke: --ros2 flag attaches the publisher graph and render still succeeds."""
    urdf = write_minimal_urdf(tmp_path)
    calib_path = tmp_path / "calib.json"
    frame_path = tmp_path / "frame.png"

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
    assert rc in (0, 2)
    rc = main(
        [
            "render",
            "--calib", str(calib_path),
            "--out", str(frame_path),
            "--ros2",
            "--ros2-frames", "10",
        ]
    )
    assert rc == 0
    assert frame_path.exists() and frame_path.stat().st_size > 1000


@pytest.mark.hardware
@pytest.mark.skipif(
    not ISAAC_PY.exists(),
    reason=f"Isaac Sim Python not found at {ISAAC_PY}",
)
def test_render_produces_png(tmp_path: Path) -> None:
    urdf = write_minimal_urdf(tmp_path)
    calib_path = tmp_path / "calib.json"
    frame_path = tmp_path / "frame.png"

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
    assert rc in (0, 2)
    assert calib_path.exists()

    rc = main(["render", "--calib", str(calib_path), "--out", str(frame_path)])
    assert rc == 0
    assert frame_path.exists() and frame_path.stat().st_size > 1000


def test_render_subparser_present() -> None:
    parser = build_parser()
    # Just check it parses without error
    ns = parser.parse_args([
        "render", "--calib", "x.json", "--out", "y.png", "--isaac-python", "/bin/false",
    ])
    assert ns.cmd == "render"
    assert ns.calib == "x.json"


def test_render_missing_isaac_python_returns_1(tmp_path: Path) -> None:
    """If --isaac-python points to a non-existent binary, cli exits 1."""
    calib = tmp_path / "calib.json"
    calib.write_text("{}")  # invalid but render never reads it
    rc = main(
        [
            "render",
            "--calib", str(calib),
            "--out", str(tmp_path / "out.png"),
            "--isaac-python", "/nonexistent/python",
        ]
    )
    assert rc == 1
