"""ROS2 OmniGraph publisher for the Isaac Sim D435 camera (D10).

Public API
----------
ROS2BridgeCfg            — frozen dataclass: topic names, frame_id, queue, namespace
attach_ros2_camera_publisher(...)   — builds the OmniGraph attached to a render
                                       product; publishes /camera/color/image_raw,
                                       /camera/depth/image_rect_raw,
                                       /camera/depth/color/points,
                                       /camera/color/camera_info

All ``omni.*`` and ``isaacsim.*`` imports are deferred to the function body so
this module is safe to import from environments without Isaac Sim (e.g. the
default pixi env used for unit tests).

Topic-name convention mirrors the real RealSense ROS2 driver
(``realsense2_camera``) so a downstream node cannot tell the rendered scene
from a real D435.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ROS2BridgeCfg:
    """Configuration for the ROS2 publisher graph.

    Defaults mirror the topic layout produced by ``realsense2_camera`` for a
    D435 with depth-aligned colour.
    """

    graph_path: str = "/World/ROS2Bridge"
    camera_prim_path: str = "/World/D435"
    namespace: str = ""
    frame_id: str = "d435_link"
    rgb_topic: str = "/camera/color/image_raw"
    depth_topic: str = "/camera/depth/image_rect_raw"
    pcl_topic: str = "/camera/depth/color/points"
    camera_info_topic: str = "/camera/color/camera_info"
    queue_size: int = 10


def attach_ros2_camera_publisher(cfg: ROS2BridgeCfg) -> str:  # pragma: no cover - Isaac Sim only
    """Build and attach the OmniGraph publisher for a camera prim.

    Returns the graph path that was created.  Must be called **after**
    ``AppLauncher`` has booted and the camera prim exists.

    Nodes
    -----
    OnPlaybackTick                                  — frame trigger
    IsaacCreateRenderProduct                        — attaches to camera prim
    ROS2CameraHelper(type="rgb")                    — RGB publisher
    ROS2CameraHelper(type="depth")                  — depth publisher
    ROS2CameraHelper(type="depth_pcl")              — depth -> PointCloud2
    ROS2CameraInfoHelper                            — camera_info publisher
    """
    # Required extensions must be enabled before the node types resolve
    import omni.kit.app
    ext_mgr = omni.kit.app.get_app().get_extension_manager()
    for ext_name in ("isaacsim.core.nodes", "isaacsim.ros2.bridge"):
        ext_mgr.set_extension_enabled_immediate(ext_name, True)

    import omni.graph.core as og

    keys = og.Controller.Keys

    create_render_product = "create_render_product"
    rgb_helper = "ros2_rgb_helper"
    depth_helper = "ros2_depth_helper"
    pcl_helper = "ros2_pcl_helper"
    info_helper = "ros2_camera_info_helper"

    (graph_handle, _, _, _) = og.Controller.edit(
        {"graph_path": cfg.graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnPlaybackTick"),
                (create_render_product, "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                (rgb_helper, "isaacsim.ros2.bridge.ROS2CameraHelper"),
                (depth_helper, "isaacsim.ros2.bridge.ROS2CameraHelper"),
                (pcl_helper, "isaacsim.ros2.bridge.ROS2CameraHelper"),
                (info_helper, "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
            ],
            keys.SET_VALUES: [
                (f"{create_render_product}.inputs:cameraPrim", [cfg.camera_prim_path]),
                (f"{create_render_product}.inputs:enabled", True),
                (f"{rgb_helper}.inputs:type", "rgb"),
                (f"{rgb_helper}.inputs:topicName", cfg.rgb_topic),
                (f"{rgb_helper}.inputs:frameId", cfg.frame_id),
                (f"{rgb_helper}.inputs:nodeNamespace", cfg.namespace),
                (f"{rgb_helper}.inputs:queueSize", cfg.queue_size),
                (f"{depth_helper}.inputs:type", "depth"),
                (f"{depth_helper}.inputs:topicName", cfg.depth_topic),
                (f"{depth_helper}.inputs:frameId", cfg.frame_id),
                (f"{depth_helper}.inputs:nodeNamespace", cfg.namespace),
                (f"{depth_helper}.inputs:queueSize", cfg.queue_size),
                (f"{pcl_helper}.inputs:type", "depth_pcl"),
                (f"{pcl_helper}.inputs:topicName", cfg.pcl_topic),
                (f"{pcl_helper}.inputs:frameId", cfg.frame_id),
                (f"{pcl_helper}.inputs:nodeNamespace", cfg.namespace),
                (f"{pcl_helper}.inputs:queueSize", cfg.queue_size),
                (f"{info_helper}.inputs:topicName", cfg.camera_info_topic),
                (f"{info_helper}.inputs:frameId", cfg.frame_id),
                (f"{info_helper}.inputs:nodeNamespace", cfg.namespace),
                (f"{info_helper}.inputs:queueSize", cfg.queue_size),
            ],
            keys.CONNECT: [
                ("OnTick.outputs:tick", f"{create_render_product}.inputs:execIn"),
                (
                    f"{create_render_product}.outputs:execOut",
                    f"{rgb_helper}.inputs:execIn",
                ),
                (
                    f"{create_render_product}.outputs:execOut",
                    f"{depth_helper}.inputs:execIn",
                ),
                (
                    f"{create_render_product}.outputs:execOut",
                    f"{pcl_helper}.inputs:execIn",
                ),
                (
                    f"{create_render_product}.outputs:execOut",
                    f"{info_helper}.inputs:execIn",
                ),
                (
                    f"{create_render_product}.outputs:renderProductPath",
                    f"{rgb_helper}.inputs:renderProductPath",
                ),
                (
                    f"{create_render_product}.outputs:renderProductPath",
                    f"{depth_helper}.inputs:renderProductPath",
                ),
                (
                    f"{create_render_product}.outputs:renderProductPath",
                    f"{pcl_helper}.inputs:renderProductPath",
                ),
                (
                    f"{create_render_product}.outputs:renderProductPath",
                    f"{info_helper}.inputs:renderProductPath",
                ),
            ],
        },
    )

    return cfg.graph_path
