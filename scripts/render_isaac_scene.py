"""Headless Isaac Sim renderer for an isaac-auto-scene calib.json.

Must be invoked with the Isaac Sim environment Python — e.g.::

    /home/koen/workspaces/lerobot-isaac-training/.pixi/envs/sim/bin/python \
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
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calib", required=True, help="path to calib.json")
    parser.add_argument("--out", required=True, help="output PNG path")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)

    # Inject Isaac Lab AppLauncher CLI args
    from isaaclab.app import AppLauncher  # type: ignore
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()
    # Force headless + cameras on regardless of CLI defaults
    args.headless = True
    args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app  # noqa: F841

    # ---- Heavy imports must happen AFTER AppLauncher boots ----
    import numpy as np  # noqa: E402
    import torch  # noqa: E402
    from PIL import Image  # noqa: E402

    import isaaclab.sim as sim_utils  # noqa: E402

    from isaac_auto_scene.calibrate import load_calibration  # noqa: E402
    from isaac_auto_scene.scene_gen import build_isaac_scene, build_scene_spec  # noqa: E402

    calib = load_calibration(args.calib)
    spec = build_scene_spec(calib)

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 60.0, device="cpu")
    )
    entities = build_isaac_scene(spec)
    camera = entities["camera"]

    sim.reset()

    # Pose camera (post-reset so internal indices are populated)
    cam_t = np.array(spec.arm_position_m, dtype=np.float32)
    cam_view_pos = torch.tensor(
        [[float(cam_t[0]), float(cam_t[1]), float(0.3)]], dtype=torch.float32
    )
    target = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    camera.set_world_poses_from_view(cam_view_pos, target)

    for _ in range(args.warmup):
        sim.step()
        camera.update(dt=sim.get_physics_dt())

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
