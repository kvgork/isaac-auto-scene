"""Tests for isaac_auto_scene.lerobot_arm (Phase 7).

The driver is structurally tested in the default env (no lerobot installed) —
we cover the radians<->degrees boundary, joint-name remapping, and the
connect() failure path.  The actual hardware round-trip lives in the
hardware-gated smoke test.
"""

from __future__ import annotations

import math

import pytest

from isaac_auto_scene.lerobot_arm import LeRobotSO101Config, LeRobotSO101Driver


def test_default_port() -> None:
    cfg = LeRobotSO101Config()
    assert cfg.port == "/dev/ttyACM0"
    assert cfg.calibrate is False
    assert cfg.joint_name_map is None


def test_satisfies_arm_driver_protocol() -> None:
    """Structural duck-typing check against the ArmDriver protocol."""
    from isaac_auto_scene.poses import ArmDriver

    driver = LeRobotSO101Driver()
    assert isinstance(driver, ArmDriver)


def test_connect_fails_clean_without_lerobot(monkeypatch: pytest.MonkeyPatch) -> None:
    """When lerobot isn't installed, connect() raises RuntimeError, not import error."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("lerobot"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    driver = LeRobotSO101Driver()
    with pytest.raises(RuntimeError, match="lerobot is not installed"):
        driver.connect()


def test_command_without_connect_raises() -> None:
    driver = LeRobotSO101Driver()
    with pytest.raises(RuntimeError, match="call connect"):
        driver.command_joints({"shoulder_pan": 0.0})


def test_read_without_connect_raises() -> None:
    driver = LeRobotSO101Driver()
    with pytest.raises(RuntimeError, match="call connect"):
        driver.read_joints()


def test_command_uses_degrees_on_wire(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify rad->deg conversion at the lerobot boundary."""
    sent: dict[str, float] = {}

    class FakeRobot:
        is_connected = True

        def send_action(self, action: dict[str, float]) -> dict[str, float]:
            sent.update(action)
            return action

        def disconnect(self) -> None:
            pass

    driver = LeRobotSO101Driver()
    driver._robot = FakeRobot()
    driver.command_joints({"shoulder_pan": math.pi / 2, "elbow_flex": -math.pi / 4})

    assert sent["shoulder_pan.pos"] == pytest.approx(90.0)
    assert sent["elbow_flex.pos"] == pytest.approx(-45.0)


def test_read_converts_degrees_to_radians() -> None:
    """Verify deg->rad conversion + .pos suffix stripping on observation."""

    class FakeRobot:
        is_connected = True

        def get_observation(self) -> dict[str, object]:
            return {
                "shoulder_pan.pos": 90.0,
                "elbow_flex.pos": -45.0,
                "main_cam": b"\x00",  # non-joint channel — must be ignored
            }

        def disconnect(self) -> None:
            pass

    driver = LeRobotSO101Driver()
    driver._robot = FakeRobot()
    rad = driver.read_joints()
    assert set(rad.keys()) == {"shoulder_pan", "elbow_flex"}
    assert rad["shoulder_pan"] == pytest.approx(math.pi / 2)
    assert rad["elbow_flex"] == pytest.approx(-math.pi / 4)


def test_joint_name_map_remaps_both_directions() -> None:
    """When a custom URDF<->motor mapping is supplied, both directions remap."""
    sent: dict[str, float] = {}

    class FakeRobot:
        is_connected = True

        def send_action(self, action: dict[str, float]) -> dict[str, float]:
            sent.update(action)
            return action

        def get_observation(self) -> dict[str, object]:
            return {"motor_a.pos": 30.0, "motor_b.pos": 60.0}

        def disconnect(self) -> None:
            pass

    cfg = LeRobotSO101Config(
        joint_name_map={"motor_a": "joint_one", "motor_b": "joint_two"},
    )
    driver = LeRobotSO101Driver(config=cfg)
    driver._robot = FakeRobot()

    driver.command_joints({"joint_one": math.radians(30.0)})
    assert "motor_a.pos" in sent

    rad = driver.read_joints()
    assert set(rad.keys()) == {"joint_one", "joint_two"}


def test_context_manager_calls_connect_disconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    """__enter__/__exit__ wire through connect/disconnect."""
    called: list[str] = []
    driver = LeRobotSO101Driver()

    monkeypatch.setattr(driver, "connect", lambda: called.append("connect"))
    monkeypatch.setattr(driver, "disconnect", lambda: called.append("disconnect"))

    with driver as d:
        assert d is driver
    assert called == ["connect", "disconnect"]
