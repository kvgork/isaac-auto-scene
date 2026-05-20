"""Tests for isaac_auto_scene.ros2_bridge (D10).

Only the dataclass / topic-naming surface is exercised in CI — the actual
OmniGraph build requires Isaac Sim and is covered by the hardware-gated
test in :mod:`test_render_isaac`.
"""

from __future__ import annotations

from isaac_auto_scene.ros2_bridge import ROS2BridgeCfg


def test_default_topic_names_match_realsense_driver() -> None:
    """Default topics must mirror the real ``realsense2_camera`` ROS2 driver."""
    cfg = ROS2BridgeCfg()
    assert cfg.rgb_topic == "/camera/color/image_raw"
    assert cfg.depth_topic == "/camera/depth/image_rect_raw"
    assert cfg.pcl_topic == "/camera/depth/color/points"
    assert cfg.camera_info_topic == "/camera/color/camera_info"


def test_default_frame_id() -> None:
    assert ROS2BridgeCfg().frame_id == "d435_link"


def test_camera_prim_path_default() -> None:
    cfg = ROS2BridgeCfg()
    assert cfg.camera_prim_path == "/World/D435"
    assert cfg.graph_path == "/World/ROS2Bridge"


def test_override_namespace_propagates() -> None:
    cfg = ROS2BridgeCfg(namespace="robot1", frame_id="custom_link", queue_size=5)
    assert cfg.namespace == "robot1"
    assert cfg.frame_id == "custom_link"
    assert cfg.queue_size == 5


def test_frozen_dataclass() -> None:
    """Cfg should be hashable + immutable (frozen)."""
    cfg = ROS2BridgeCfg()
    try:
        cfg.namespace = "x"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("ROS2BridgeCfg must be frozen")
