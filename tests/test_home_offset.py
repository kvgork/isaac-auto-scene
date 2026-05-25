"""Tests for the home-offset flow: set-home + apply_home_offset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from isaac_auto_scene.cli import _apply_home_offset, _load_home_offset, build_parser


def test_load_home_offset_none() -> None:
    # auto_discover=False to ignore any real ~/.config file on the dev host.
    assert _load_home_offset(None, auto_discover=False) == {}


def test_load_home_offset_file(tmp_path: Path) -> None:
    p = tmp_path / "home.json"
    p.write_text(
        json.dumps(
            {
                "home_offset_rad": {"shoulder_pan": 0.1, "elbow_flex": -0.2},
                "captured_at": "2026-01-01T00:00:00Z",
            }
        )
    )
    out = _load_home_offset(str(p))
    assert out == {"shoulder_pan": 0.1, "elbow_flex": -0.2}


def test_apply_home_offset_empty() -> None:
    """No home offset = identity."""
    joints = {"shoulder_pan": 0.5, "elbow_flex": 0.3}
    assert _apply_home_offset(joints, {}) == joints


def test_apply_home_offset_subtracts_per_joint() -> None:
    joints = {"shoulder_pan": 0.5, "elbow_flex": 0.3, "wrist_flex": 0.1}
    home = {"shoulder_pan": 0.1, "elbow_flex": -0.2}  # wrist_flex missing -> 0
    out = _apply_home_offset(joints, home)
    assert out["shoulder_pan"] == pytest.approx(0.4)
    assert out["elbow_flex"] == pytest.approx(0.5)
    assert out["wrist_flex"] == pytest.approx(0.1)


def test_apply_home_offset_does_not_mutate_input() -> None:
    joints = {"shoulder_pan": 0.5}
    home = {"shoulder_pan": 0.1}
    _apply_home_offset(joints, home)
    assert joints["shoulder_pan"] == 0.5


def test_calibrate_arm_subparser() -> None:
    args = build_parser().parse_args(["calibrate-arm", "--arm-port", "/dev/ttyACM1"])
    assert args.cmd == "calibrate-arm"
    assert args.arm_port == "/dev/ttyACM1"


def test_set_home_subparser() -> None:
    args = build_parser().parse_args(
        ["set-home", "--urdf", "/tmp/x.urdf", "--out", "/tmp/home.json"]
    )
    assert args.cmd == "set-home"
    assert args.out == "/tmp/home.json"
    assert args.arm_port == "/dev/ttyACM0"


def test_manual_align_home_offset_arg() -> None:
    args = build_parser().parse_args(
        [
            "manual-align",
            "--urdf", "/tmp/x.urdf",
            "--out", "/tmp/c.json",
            "--home-offset", "/tmp/home.json",
        ]
    )
    assert args.home_offset == "/tmp/home.json"
