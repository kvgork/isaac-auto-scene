"""Headless Isaac Sim renderer for an isaac-auto-scene calib.json.

Must be invoked with the Isaac Sim environment Python — e.g.::

    ~/workspaces/lerobot-isaac-training/.pixi/envs/sim/bin/python \
        scripts/render_isaac_scene.py \
        --calib calib.json --out frame_000.png

Boots AppLauncher(headless=True, enable_cameras=True), spawns a pinhole
camera at the calibrated pose, a CuboidCfg table at the origin, and a
DomeLightCfg.  Runs the mandatory 30-frame warm-up (IsaacLab#3250) before
reading the RGB buffer.

This script must come AFTER the AppLauncher boot for any isaacsim / omni
imports to work — that's why all heavy imports are inline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _query_free_vram_mb() -> int | None:
    """Return free VRAM in MiB on GPU 0, or None if nvidia-smi is unavailable."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return int(out.stdout.strip().splitlines()[0])
    except (subprocess.SubprocessError, FileNotFoundError, ValueError, IndexError):
        return None


# Minimum free VRAM (MiB) required for the headless render product to
# allocate its colour + depth buffers without ERROR_OUT_OF_DEVICE_MEMORY.
# With --ros2 the OmniGraph adds RGB + depth + PCL + camera_info publisher
# render-targets, so the budget roughly doubles.
MIN_FREE_VRAM_MB = 1500
MIN_FREE_VRAM_MB_ROS2 = 2500


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calib", required=True, help="path to calib.json")
    parser.add_argument("--out", required=True, help="output PNG path")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--ros2",
        action="store_true",
        help="attach the ROS2 OmniGraph camera publisher (D10)",
    )
    parser.add_argument(
        "--ros2-frames",
        type=int,
        default=0,
        help="extra simulation steps after warm-up so the ROS2 publisher has "
        "time to push frames out (default 0 = single-shot render only)",
    )
    parser.add_argument(
        "--camera-convention",
        choices=["ros", "world", "opengl"],
        default="ros",
        help="frame convention for the camera orientation passed to "
        "Camera.set_world_poses. 'ros' = optical (+Z fwd, -Y up), matches the "
        "RealSense calibration frame. Swap to diagnose a rotated render.",
    )

    # Inject Isaac Lab AppLauncher CLI args
    from isaaclab.app import AppLauncher  # type: ignore
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()
    # Force headless + cameras on regardless of CLI defaults
    args.headless = True
    args.enable_cameras = True

    # Pre-flight VRAM check. ERROR_OUT_OF_DEVICE_MEMORY inside Vulkan during
    # render-product allocation produces a silent retry loop (no rgb buffer
    # ever populates, the warmup loop completes without an exception).
    # Catch it before AppLauncher boots so we exit cleanly instead of hanging.
    free_mb = _query_free_vram_mb()
    if free_mb is not None:
        budget = MIN_FREE_VRAM_MB_ROS2 if args.ros2 else MIN_FREE_VRAM_MB
        if free_mb < budget:
            print(
                f"ERROR: insufficient GPU memory: {free_mb} MiB free, need >= "
                f"{budget} MiB ({'with --ros2' if args.ros2 else 'baseline'}). "
                "Stop competing GPU workloads (check `nvidia-smi --query-compute-apps`).",
                file=sys.stderr,
            )
            return 1

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app  # noqa: F841

    # ---- Heavy imports must happen AFTER AppLauncher boots ----
    import torch  # noqa: E402
    from PIL import Image  # noqa: E402

    import isaaclab.sim as sim_utils  # noqa: E402

    if args.ros2:
        # Load ROS2 bridge extensions BEFORE any further graph activity so
        # node type IDs resolve when ros2_bridge.attach_ros2_camera_publisher
        # is called.  set_extension_enabled_immediate is synchronous.
        import omni.kit.app  # noqa: E402
        ext_mgr = omni.kit.app.get_app().get_extension_manager()
        for ext_name in ("isaacsim.core.nodes", "isaacsim.ros2.bridge"):
            ext_mgr.set_extension_enabled_immediate(ext_name, True)

    from isaac_auto_scene.calibrate import load_calibration  # noqa: E402
    from isaac_auto_scene.scene_gen import build_isaac_scene, build_scene_spec  # noqa: E402

    calib = load_calibration(args.calib)
    spec = build_scene_spec(calib)

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 60.0, device="cpu")
    )
    entities = build_isaac_scene(spec)
    camera = entities["camera"]
    arm = entities.get("arm")

    sim.reset()

    # Set SO-101 joint angles to match the captured pose.  Must happen
    # AFTER sim.reset() so the Articulation indices are initialised.
    if arm is not None and spec.arm_joint_angles_rad:
        import torch as _torch
        joint_names = list(arm.joint_names)
        joint_pos = _torch.tensor(
            [[float(spec.arm_joint_angles_rad.get(n, 0.0)) for n in joint_names]],
            dtype=_torch.float32,
        )
        joint_vel = _torch.zeros_like(joint_pos)
        arm.write_joint_state_to_sim(joint_pos, joint_vel)
        arm.set_joint_position_target(joint_pos)
        print(
            f"[render] applied joint state: "
            f"{ {n: round(float(spec.arm_joint_angles_rad.get(n, 0.0)), 3) for n in joint_names} }",
            flush=True,
        )

    # Set camera world pose from the calibrated spec (post-reset so internal
    # indices are populated).
    #
    # World frame == SO-101 arm base at origin.  Camera pose is inv(T_cam_arm),
    # which was computed in build_scene_spec from the calibration.
    #
    # spec.camera_quat_xyzw is in XYZW order (our internal convention).
    # Camera.set_world_poses expects WXYZ order and a convention kwarg that
    # describes the frame that the quaternion represents.
    # convention="ros" (default) means the camera's +Z is the optical forward
    # axis and -Y is the optical up axis — matching the RealSense optical frame
    # that the calibration uses.  Isaac Lab internally converts from this
    # convention to its OpenGL stage convention before setting the prim pose.
    # Override via --camera-convention to diagnose a rotated render.
    x, y, z, w = spec.camera_quat_xyzw
    cam_pos = torch.tensor([list(spec.camera_position_m)], dtype=torch.float32)
    cam_ori = torch.tensor([[w, x, y, z]], dtype=torch.float32)  # wxyz for isaaclab
    camera.set_world_poses(cam_pos, cam_ori, convention=args.camera_convention)
    print(f"[render] camera convention: {args.camera_convention}", flush=True)

    if args.ros2:
        from isaac_auto_scene.ros2_bridge import (  # noqa: E402
            ROS2BridgeCfg,
            attach_ros2_camera_publisher,
        )
        bridge_cfg = ROS2BridgeCfg(camera_prim_path="/World/D435")
        attach_ros2_camera_publisher(bridge_cfg)
        print(f"ROS2 bridge attached at {bridge_cfg.graph_path}", flush=True)

    for _ in range(args.warmup):
        sim.step()
        camera.update(dt=sim.get_physics_dt())

    sim.step()
    camera.update(dt=sim.get_physics_dt())

    for _ in range(args.ros2_frames):
        sim.step()
        camera.update(dt=sim.get_physics_dt())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rgb = camera.data.output.get("rgb")
    if rgb is None:
        print("ERROR: camera produced no rgb buffer", file=sys.stderr)
        simulation_app.close()
        return 1
    rgb_np = rgb[0].cpu().numpy().astype("uint8")
    if rgb_np.shape[-1] == 4:
        rgb_np = rgb_np[..., :3]
    Image.fromarray(rgb_np).save(str(out))
    print(f"wrote {out} ({out.stat().st_size} bytes)", flush=True)

    # simulation_app.close() can deadlock in headless rendering kits on this
    # build; force-exit after a successful write.
    import os as _os
    _os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
