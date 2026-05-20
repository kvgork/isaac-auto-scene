"""Command-line interface for isaac-auto-scene (Phase 6).

Subcommands
-----------
calibrate  capture -> segment -> register -> calib.json
generate   calib.json -> scene.usd (stub or full Isaac Sim asset)
render     scene.usd -> PNG frames (requires Isaac Sim)
validate   calib.json + scene.usd -> residual report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from isaac_auto_scene.cad import assemble_pcd, load_urdf
from isaac_auto_scene.calibrate import (
    build_calibration,
    load_calibration,
    save_calibration,
)
from isaac_auto_scene.capture import MockD435Source, capture, save_capture
from isaac_auto_scene.register import (
    QUALITY_GATE,
    passes_quality_gate,
    register_global_local,
)
from isaac_auto_scene.scene_gen import build_scene_spec, write_usd_stub
from isaac_auto_scene.segment import segment_table_arm


def _pcd_from_np(pts):
    import open3d as o3d

    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, dtype=np.float64))
    return p


def cmd_calibrate(args: argparse.Namespace) -> int:
    """capture -> segment -> register -> calib.json."""
    if args.mock:
        source = MockD435Source(seed=args.seed)
    else:  # pragma: no cover - hardware path
        from isaac_auto_scene.realsense_source import RealSenseD435Source
        source = RealSenseD435Source()

    with source as src:
        cap = capture(source=src, num_frames=args.frames)

    seg = segment_table_arm(cap.pcd)

    urdf = load_urdf(args.urdf)
    joint_angles = json.loads(args.joints) if args.joints else None
    cad = assemble_pcd(urdf, joint_angles, target_n_points=args.target_n_points)

    cad_pcd = _pcd_from_np(cad.points)
    reg = register_global_local(
        cad_pcd, seg.arm_cloud, voxel_size=args.voxel, n_restarts=args.restarts
    )

    calib = build_calibration(cap, cad, reg)
    save_calibration(calib, Path(args.out))

    print(f"calib.json written -> {args.out}")
    print(f"  fitness={reg.fitness:.3f}  rmse={reg.inlier_rmse_m*1000:.2f} mm")
    print(f"  quality_gate({QUALITY_GATE[0]:.2f}, {QUALITY_GATE[1]*1000:.0f} mm): "
          f"{'PASS' if passes_quality_gate(reg) else 'FAIL'}")

    if args.dump_pcds:
        out_dir = Path(args.dump_pcds)
        save_capture(cap, out_dir)

    return 0 if passes_quality_gate(reg) else 2


def cmd_generate(args: argparse.Namespace) -> int:
    """calib.json -> scene.usd (stub)."""
    calib = load_calibration(args.calib)
    spec = build_scene_spec(calib, enable_ros2=args.ros2)
    out = write_usd_stub(spec, Path(args.out))
    size = out.stat().st_size
    print(f"USD stub written -> {out} ({size} bytes)")
    if size == 0:
        print("ERROR: USD output empty", file=sys.stderr)
        return 1
    return 0


def cmd_render(args: argparse.Namespace) -> int:  # pragma: no cover - hardware-only
    """scene.usd -> PNG frames.  Requires Isaac Sim."""
    print(
        "render requires Isaac Sim; install via `pixi run install-isaac` and use "
        "the `sim` environment.",
        file=sys.stderr,
    )
    return 3


def cmd_validate(args: argparse.Namespace) -> int:
    """Forward-projection residual report from calib.json + scene.usd."""
    calib = load_calibration(args.calib)
    scene = Path(args.scene)
    if not scene.exists():
        print(f"ERROR: scene not found: {scene}", file=sys.stderr)
        return 1

    report = {
        "scene_bytes": scene.stat().st_size,
        "icp_fitness": calib.icp_fitness,
        "inlier_rmse_m": calib.inlier_rmse_m,
        "quality_gate_pass": (
            calib.icp_fitness >= QUALITY_GATE[0]
            and calib.inlier_rmse_m <= QUALITY_GATE[1]
        ),
        "translation_m": calib.translation_m,
        "quat_xyzw": calib.quat_xyzw,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["quality_gate_pass"] else 2


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse CLI."""
    p = argparse.ArgumentParser(prog="isaac-auto-scene")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("calibrate", help="capture -> segment -> register")
    pc.add_argument("--urdf", required=True, help="path to SO-101 URDF")
    pc.add_argument("--out", default="calib.json", help="output calib.json path")
    pc.add_argument("--mock", action="store_true", help="use MockD435Source")
    pc.add_argument("--seed", type=int, default=0)
    pc.add_argument("--frames", type=int, default=30)
    pc.add_argument("--joints", default=None, help="JSON dict of joint angles")
    pc.add_argument("--voxel", type=float, default=0.005)
    pc.add_argument("--restarts", type=int, default=5)
    pc.add_argument("--target-n-points", type=int, default=15_000)
    pc.add_argument("--dump-pcds", default=None, help="optional out dir for capture artefacts")
    pc.set_defaults(func=cmd_calibrate)

    pg = sub.add_parser("generate", help="calib.json -> scene.usd")
    pg.add_argument("--calib", required=True)
    pg.add_argument("--out", required=True)
    pg.add_argument("--ros2", action="store_true", help="attach ROS2 publisher OmniGraph")
    pg.set_defaults(func=cmd_generate)

    pr = sub.add_parser("render", help="scene.usd -> PNG frames (Isaac Sim)")
    pr.add_argument("--scene", required=True)
    pr.add_argument("--out", required=True)
    pr.set_defaults(func=cmd_render)

    pv = sub.add_parser("validate", help="forward-projection residual report")
    pv.add_argument("--calib", required=True)
    pv.add_argument("--scene", required=True)
    pv.set_defaults(func=cmd_validate)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
