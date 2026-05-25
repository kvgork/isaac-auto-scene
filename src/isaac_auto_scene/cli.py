"""Command-line interface for isaac-auto-scene (Phase 6).

Subcommands
-----------
calibrate      capture -> segment -> register -> calib.json
capture-poses  drive arm through pose set -> per-pose RGB-D + manifest
register-multi capture manifest -> aggregated calib.json
generate       calib.json -> scene.usd (stub or full Isaac Sim asset)
render         scene.usd -> PNG frames (requires Isaac Sim)
validate       calib.json + scene.usd -> residual report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

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


def _resolve_fallback(args: argparse.Namespace):
    """Return the fallback registration callable specified by --fallback, or None."""
    name = getattr(args, "fallback", None) or "none"
    name = str(name).lower()
    if name in ("none", ""):
        return None
    if name in ("fpfh_ransac", "fpfh-ransac"):
        from isaac_auto_scene.learned_register import register_with_fpfh_ransac

        voxel = float(getattr(args, "voxel", 0.005))

        def _wrapped(source, target):
            return register_with_fpfh_ransac(source, target, voxel_size=voxel)

        return _wrapped
    raise ValueError(
        f"unknown --fallback backend: {name!r} (supported: none, fpfh_ransac)"
    )


def _parse_expected_up(value: str | None) -> tuple[float, float, float] | None:
    """Parse '--expected-up x,y,z' (or None) into a 3-tuple."""
    if value is None or value == "":
        return None
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise ValueError(
            f"--expected-up must be 'x,y,z' (got {value!r})"
        )
    return (float(parts[0]), float(parts[1]), float(parts[2]))


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


def cmd_capture_poses(args: argparse.Namespace) -> int:
    """Drive arm through pose set; write per-pose RGB-D + manifest.yaml."""
    from isaac_auto_scene.capture_multi import capture_pose_set
    from isaac_auto_scene.poses import MockArmDriver, load_poses

    urdf_path = Path(args.urdf)
    urdf = load_urdf(urdf_path)
    poses = load_poses(args.poses)

    if args.mock_arm:
        driver = MockArmDriver(
            joint_names=tuple(urdf.actuated_joint_names),
            readback_noise_rad=args.servo_noise,
        )
    else:  # pragma: no cover - hardware path
        from isaac_auto_scene.lerobot_arm import (
            LeRobotSO101Config,
            LeRobotSO101Driver,
        )

        driver = LeRobotSO101Driver(
            config=LeRobotSO101Config(
                port=args.arm_port,
                calibrate=args.arm_calibrate,
            )
        )

    if args.mock_cam:
        source = MockD435Source(seed=args.seed)
    else:  # pragma: no cover - hardware path
        from isaac_auto_scene.realsense_source import RealSenseD435Source

        source = RealSenseD435Source()

    with driver as drv, source as src:
        manifest = capture_pose_set(
            poses,
            drv,
            src,
            urdf,
            urdf_path,
            out_dir=Path(args.out),
            frames_per_pose=args.frames,
        )

    print(
        f"capture manifest -> {args.out}/manifest.yaml  "
        f"({manifest.num_ok}/{manifest.num_poses} poses ok)"
    )
    return 0 if manifest.num_ok == manifest.num_poses else 2


def cmd_register_multi(args: argparse.Namespace) -> int:
    """Aggregate per-pose ICP runs into one calib.json."""
    from isaac_auto_scene.capture_multi import load_manifest, load_pose_capture
    from isaac_auto_scene.register import register_multi_pose

    manifest_dir = Path(args.captures)
    manifest = load_manifest(manifest_dir)
    urdf = load_urdf(args.urdf)

    pairs = []
    last_cap = None
    last_cad = None
    for record in manifest.poses:
        if record.status != "ok":
            print(
                f"skipping pose {record.name!r} (status={record.status})",
                file=sys.stderr,
            )
            continue
        cap = load_pose_capture(manifest_dir, record)
        from isaac_auto_scene.segment import segment_table_arm

        seg = segment_table_arm(
            cap.pcd,
            workspace_z_max_m=args.workspace_z_max,
            workspace_z_min_m=args.workspace_z_min,
            expected_up=_parse_expected_up(args.expected_up),
            up_tolerance_deg=args.up_tol_deg,
            arm_merge_radius_m=args.arm_merge_radius,
            outlier_nb_neighbors=args.outlier_neighbors,
            outlier_std_ratio=args.outlier_std,
        )
        cad = assemble_pcd(
            urdf, record.readback_joints, target_n_points=args.target_n_points
        )
        pairs.append((record.name, _pcd_from_np(cad.points), seg.arm_cloud))
        last_cap = cap
        last_cad = cad

    if not pairs:
        print("ERROR: no usable poses in manifest", file=sys.stderr)
        return 1

    gate_override: tuple[float, float] | None = None
    if args.gate_fitness is not None or args.gate_rmse is not None:
        f_min, rmse_max = QUALITY_GATE
        if args.gate_fitness is not None:
            f_min = float(args.gate_fitness)
        if args.gate_rmse is not None:
            rmse_max = float(args.gate_rmse)
        gate_override = (f_min, rmse_max)
        print(
            f"[register-multi] quality gate override: fitness>={f_min:.2f} "
            f"rmse<={rmse_max*1000:.1f}mm (default {QUALITY_GATE[0]:.2f}/"
            f"{QUALITY_GATE[1]*1000:.0f}mm)",
            file=sys.stderr,
        )

    fallback_fn = _resolve_fallback(args)

    multi = register_multi_pose(
        pairs,
        voxel_size=args.voxel,
        n_restarts=args.restarts,
        min_accepted=args.min_accepted,
        quality_gate=gate_override,
        fallback=fallback_fn,
    )

    if getattr(args, "backend", "per_pose") == "bundle_joints":
        from isaac_auto_scene.bundle_register import register_bundle_with_joints

        if multi.per_pose:
            best_pp = max(multi.per_pose, key=lambda p: p.fitness)
            T_init = np.asarray(best_pp.T, dtype=np.float64)
            print(
                f"[register-multi] bundle_joints init: pose={best_pp.pose_name!r} "
                f"fitness={best_pp.fitness:.3f}",
                file=sys.stderr,
            )
        else:
            T_init = multi.T

        joints_per_pose = [
            dict(rec.readback_joints) for rec in manifest.poses if rec.status == "ok"
        ]
        arm_clouds_only = [p[2] for p in pairs]
        opt_joints = (
            tuple(args.optimize_joints.split(",")) if args.optimize_joints else None
        )
        bj = register_bundle_with_joints(
            urdf,
            joints_per_pose,
            arm_clouds_only,
            optimize_joints=opt_joints,
            T_init=T_init,
            cad_target_n_points=args.target_n_points // 5,
            inlier_distance_m=float(args.bundle_inlier_distance),
            delta_bound_rad=float(args.joint_offset_bound),
            max_nfev=int(args.bundle_max_nfev),
        )
        print(
            f"[register-multi] bundle_joints: cost={bj.cost:.4f} nfev={bj.n_iterations}",
            file=sys.stderr,
        )
        print(f"[register-multi] joint offsets (rad):", file=sys.stderr)
        for k, v in bj.joint_offsets.items():
            print(f"    {k:<16} {v:+.4f} ({np.degrees(v):+.2f} deg)", file=sys.stderr)

        from isaac_auto_scene.register import MultiPoseResult, PerPoseRegistration

        per_pose_records = tuple(
            PerPoseRegistration(
                pose_name=p[0],
                accepted=True,
                fitness=bj.per_pose_fitness[i],
                inlier_rmse_m=bj.per_pose_rmse_m[i],
                T=bj.T.copy(),
            )
            for i, p in enumerate(pairs)
        )
        multi = MultiPoseResult(
            T=bj.T,
            quat_xyzw=bj.quat_xyzw,
            translation_m=bj.translation_m,
            dispersion_rad=0.0,
            n_accepted=len(pairs),
            n_total=len(pairs),
            per_pose=per_pose_records,
        )

    elif getattr(args, "backend", "per_pose") == "bundle":
        # Bundle-refine using the *best per-pose* ICP result as init.
        # The averaged T from register_multi_pose is biased by cylindrical-
        # symmetry ambiguity (each per-pose solve lands in a different basin
        # and the average is meaningless).  The single highest-fitness pose
        # is closer to the true T.
        from isaac_auto_scene.bundle_register import register_bundle

        if multi.per_pose:
            best_pp = max(multi.per_pose, key=lambda p: p.fitness)
            T_init = np.asarray(best_pp.T, dtype=np.float64)
            print(
                f"[register-multi] bundle init: pose={best_pp.pose_name!r} "
                f"fitness={best_pp.fitness:.3f}",
                file=sys.stderr,
            )
        else:
            T_init = multi.T

        cad_arrays = [np.asarray(p[1].points, dtype=np.float64) for p in pairs]
        arm_clouds = [p[2] for p in pairs]
        bundle = register_bundle(
            cad_arrays,
            arm_clouds,
            T_init=T_init,
            inlier_distance_m=float(args.bundle_inlier_distance),
            max_nfev=int(args.bundle_max_nfev),
        )
        # Overwrite multi with bundle output, keep per_pose stats for the
        # final report.
        from isaac_auto_scene.register import (
            MultiPoseResult,
            PerPoseRegistration,
        )

        per_pose_records = tuple(
            PerPoseRegistration(
                pose_name=p[0],
                accepted=True,
                fitness=bundle.per_pose_fitness[i],
                inlier_rmse_m=bundle.per_pose_rmse_m[i],
                T=bundle.T.copy(),
            )
            for i, p in enumerate(pairs)
        )
        multi = MultiPoseResult(
            T=bundle.T,
            quat_xyzw=bundle.quat_xyzw,
            translation_m=bundle.translation_m,
            dispersion_rad=0.0,
            n_accepted=len(pairs),
            n_total=len(pairs),
            per_pose=per_pose_records,
        )
        print(
            f"[register-multi] bundle solver: cost={bundle.cost:.4f} "
            f"nfev={bundle.n_iterations}",
            file=sys.stderr,
        )

    # Reuse build_calibration's quat conversion / intrinsics packaging via
    # a synthetic single-pose RegistrationResult-like object.
    from isaac_auto_scene.register import RegistrationResult

    synthetic_reg = RegistrationResult(
        T=multi.T,
        fitness=float(np.mean([p.fitness for p in multi.per_pose if p.accepted])),
        inlier_rmse_m=float(
            np.mean([p.inlier_rmse_m for p in multi.per_pose if p.accepted])
        ),
        used_fallback=False,
        n_restarts=args.restarts,
    )
    calib = build_calibration(last_cap, last_cad, synthetic_reg)
    save_calibration(calib, Path(args.out))

    print(
        f"multi-pose calib -> {args.out}  "
        f"({multi.n_accepted}/{multi.n_total} accepted, "
        f"dispersion={multi.dispersion_rad*180/np.pi:.2f}°)"
    )
    for p in multi.per_pose:
        tag = "OK " if p.accepted else "REJ"
        rmse_mm = (
            f"{p.inlier_rmse_m*1000:.2f} mm"
            if np.isfinite(p.inlier_rmse_m)
            else "inf"
        )
        print(
            f"  [{tag}] {p.pose_name:<16} fitness={p.fitness:.3f} "
            f"rmse={rmse_mm} {p.reason}"
        )
    return 0 if multi.n_accepted >= args.min_accepted else 2


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


def _detect_ros_distro() -> str:
    """Pick the bundled Isaac Sim ROS2 distro by Ubuntu major version.

    Ubuntu 22.x -> humble, Ubuntu 24.x -> jazzy.  Defaults to jazzy.
    """
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("VERSION_ID="):
                    ver = line.split("=", 1)[1].strip().strip('"')
                    major = ver.split(".")[0]
                    if major == "22":
                        return "humble"
                    if major == "24":
                        return "jazzy"
    except OSError:
        pass
    return "jazzy"


def cmd_render(args: argparse.Namespace) -> int:  # pragma: no cover - external Isaac Sim env
    """calib.json -> PNG frame via the lerobot-isaac-training Isaac Sim env.

    Delegates to ``scripts/render_isaac_scene.py`` using the Isaac Sim Python
    interpreter from ``~/workspaces/lerobot-isaac-training/.pixi/envs/sim``.
    Set ``--isaac-python`` to override.
    """
    import os
    import shutil
    import subprocess

    isaac_python = (
        args.isaac_python
        or os.environ.get("ISAAC_PYTHON")
        or str(
            Path.home()
            / "workspaces/lerobot-isaac-training/.pixi/envs/sim/bin/python"
        )
    )
    if not Path(isaac_python).exists():
        print(
            f"ERROR: Isaac Sim Python not found at {isaac_python}.  "
            "Override with --isaac-python or set ISAAC_PYTHON.",
            file=sys.stderr,
        )
        return 1

    repo_root = Path(__file__).resolve().parents[2]
    render_script = repo_root / "scripts" / "render_isaac_scene.py"

    env = os.environ.copy()
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = (
        src_path + os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else src_path
    )

    if args.ros2:
        # Isaac Sim's ROS2 bridge needs LD_LIBRARY_PATH + ROS_DISTRO +
        # RMW_IMPLEMENTATION pointing at the bundled jazzy/humble libs.
        # Detect distro by Ubuntu major version.
        ros_distro = _detect_ros_distro()
        isaacsim_root = Path(isaac_python).parent.parent / (
            "lib/python3.12/site-packages/isaacsim/exts/isaacsim.ros2.core"
        )
        ros_lib_dir = isaacsim_root / ros_distro / "lib"
        if not ros_lib_dir.exists():
            print(
                f"ERROR: bundled ROS2 libs not found at {ros_lib_dir}",
                file=sys.stderr,
            )
            return 1
        env["ROS_DISTRO"] = ros_distro
        env["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
        env["LD_LIBRARY_PATH"] = str(ros_lib_dir) + os.pathsep + env.get(
            "LD_LIBRARY_PATH", ""
        )

    cmd = [
        isaac_python,
        str(render_script),
        "--calib", str(args.calib),
        "--out", str(args.out),
        "--headless",
        "--enable_cameras",
    ]
    if args.ros2:
        cmd += ["--ros2", "--ros2-frames", str(args.ros2_frames)]
    print("$", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, env=env, check=False)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if proc.returncode != 0:
        return proc.returncode
    if not Path(args.out).exists() or Path(args.out).stat().st_size == 0:
        print(f"ERROR: render produced empty output {args.out}", file=sys.stderr)
        return 1
    print(f"render ok -> {args.out}")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """End-to-end smoke: capture-poses -> register-multi -> render.

    Hardware path by default (real D435 + SO-101 follower).  ``--mock`` swaps
    both for the synthetic mock sources so the command is also useful for
    local pipeline verification without hardware connected.

    The smoke writes a small artifact tree to ``--out``::

        out/
        ├── captures/                    (one subdir per pose)
        │   └── manifest.yaml
        ├── calib.json
        └── frame.png
    """
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    captures_dir = out_root / "captures"
    calib_path = out_root / "calib.json"
    frame_path = out_root / "frame.png"

    # ---- Stage 1: capture-poses (reuse cmd_capture_poses for parity) ----
    capture_args = argparse.Namespace(
        urdf=args.urdf,
        poses=args.poses,
        out=str(captures_dir),
        mock_arm=args.mock,
        mock_cam=args.mock,
        seed=args.seed,
        frames=args.frames,
        servo_noise=0.0,
        arm_port=args.arm_port,
        arm_calibrate=False,
    )
    rc = cmd_capture_poses(capture_args)
    if rc != 0:
        print(f"[smoke] capture-poses failed (rc={rc})", file=sys.stderr)
        return rc

    # ---- Stage 2: register-multi (reuse cmd_register_multi) ----
    reg_args = argparse.Namespace(
        captures=str(captures_dir),
        urdf=args.urdf,
        out=str(calib_path),
        voxel=0.005,
        restarts=5,
        target_n_points=15_000,
        min_accepted=max(1, len(yaml.safe_load(Path(args.poses).read_text())["poses"]) // 2),
        workspace_z_max=args.workspace_z_max,
        workspace_z_min=args.workspace_z_min,
        expected_up=args.expected_up,
        up_tol_deg=args.up_tol_deg,
        arm_merge_radius=args.arm_merge_radius,
        outlier_neighbors=args.outlier_neighbors,
        outlier_std=args.outlier_std,
        gate_fitness=args.gate_fitness,
        gate_rmse=args.gate_rmse,
        fallback=args.fallback,
        backend=args.backend,
        bundle_inlier_distance=args.bundle_inlier_distance,
        bundle_max_nfev=args.bundle_max_nfev,
        optimize_joints=args.optimize_joints,
        joint_offset_bound=args.joint_offset_bound,
    )
    try:
        rc = cmd_register_multi(reg_args)
    except RuntimeError as exc:
        # register_multi_pose raises when zero poses clear the quality gate.
        # On the synthetic mock path (pose-invariant MockD435Source) every
        # pose reuses the same depth, so the gate cannot be satisfied.  The
        # rest of the pipeline plumbing was verified by stage 1; surface
        # the gate failure and skip render rather than crashing.
        print(f"[smoke] register-multi quality gate fail: {exc}", file=sys.stderr)
        print(
            "[smoke] no calib.json produced; expected on --mock (synthetic "
            "mock is pose-invariant). Real hardware: investigate ICP "
            "convergence.",
            file=sys.stderr,
        )
        return 2
    if rc not in (0, 2):  # 2 = quality-gate fail; calib.json still written
        print(f"[smoke] register-multi failed (rc={rc})", file=sys.stderr)
        return rc
    if not calib_path.exists():
        print("[smoke] register-multi returned but no calib.json written; halting",
              file=sys.stderr)
        return 2

    # ---- Stage 3: render ----
    render_args = argparse.Namespace(
        calib=str(calib_path),
        out=str(frame_path),
        isaac_python=None,
        ros2=False,
        ros2_frames=0,
    )
    rc = cmd_render(render_args)
    if rc != 0:
        print(f"[smoke] render failed (rc={rc})", file=sys.stderr)
        return rc

    print(f"[smoke] OK -> {frame_path}")
    return 0


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

    pcp = sub.add_parser(
        "capture-poses", help="drive arm through pose set -> per-pose RGB-D"
    )
    pcp.add_argument("--urdf", required=True)
    pcp.add_argument("--poses", required=True, help="path to poses.yaml")
    pcp.add_argument("--out", required=True, help="output capture dir")
    pcp.add_argument("--mock-arm", action="store_true")
    pcp.add_argument("--mock-cam", action="store_true")
    pcp.add_argument("--seed", type=int, default=0)
    pcp.add_argument("--frames", type=int, default=30)
    pcp.add_argument(
        "--servo-noise",
        type=float,
        default=0.0,
        help="Gaussian readback noise (rad) for MockArmDriver",
    )
    pcp.add_argument(
        "--arm-port",
        default="/dev/ttyACM0",
        help="serial port for the SO-101 follower (ignored when --mock-arm)",
    )
    pcp.add_argument(
        "--arm-calibrate",
        action="store_true",
        help="run LeRobot's interactive calibration prompt on connect",
    )
    pcp.set_defaults(func=cmd_capture_poses)

    prm = sub.add_parser(
        "register-multi", help="capture manifest -> aggregated calib.json"
    )
    prm.add_argument("--captures", required=True, help="capture run directory")
    prm.add_argument("--urdf", required=True)
    prm.add_argument("--out", default="calib.json")
    prm.add_argument("--voxel", type=float, default=0.005)
    prm.add_argument("--restarts", type=int, default=5)
    prm.add_argument("--target-n-points", type=int, default=15_000)
    prm.add_argument("--min-accepted", type=int, default=2)
    prm.add_argument(
        "--workspace-z-max",
        type=float,
        default=None,
        help="Drop points beyond this Z (metres, camera frame) before "
        "plane fit. Use to suppress background walls.",
    )
    prm.add_argument(
        "--workspace-z-min",
        type=float,
        default=None,
        help="Drop points closer than this Z (metres). E.g. 0.15 to ignore "
        "noise just in front of the lens.",
    )
    prm.add_argument(
        "--expected-up",
        default=None,
        help="Expected table normal in camera frame as 'x,y,z' (e.g. "
        "'0,-1,0' for a level D435 looking forward at a table). When set, "
        "RANSAC planes within --up-tol-deg of this direction are preferred "
        "over the largest plane.",
    )
    prm.add_argument(
        "--up-tol-deg",
        type=float,
        default=30.0,
        help="Angular tolerance (deg) for --expected-up (default 30).",
    )
    prm.add_argument(
        "--arm-merge-radius",
        type=float,
        default=0.0,
        help="Merge DBSCAN clusters within this radius (m) of the largest "
        "into the arm cloud. Use ~0.30 for SO-101 to capture base + "
        "forearm when joints split them across clusters. Default 0 = off.",
    )
    prm.add_argument(
        "--outlier-neighbors",
        type=int,
        default=0,
        help="When > 0, run Open3D statistical outlier removal on the arm "
        "cloud with this neighbour count (drops cable / mount stragglers). "
        "Typical: 20.  Default 0 = off.",
    )
    prm.add_argument(
        "--outlier-std",
        type=float,
        default=2.0,
        help="Std-dev multiplier for outlier filter (lower = more aggressive).",
    )
    prm.add_argument(
        "--gate-fitness",
        type=float,
        default=None,
        help="Override the quality-gate minimum fitness (default 0.65). "
        "Hardware bring-up with cluttered captures often needs ~0.30.",
    )
    prm.add_argument(
        "--gate-rmse",
        type=float,
        default=None,
        help="Override the quality-gate maximum RMSE in metres (default "
        "0.005). Try ~0.012 for noisy real captures.",
    )
    prm.add_argument(
        "--fallback",
        default="none",
        choices=["none", "fpfh_ransac"],
        help="Robust registration backend invoked when classic ICP scores "
        "below the per-restart fallback threshold. 'fpfh_ransac' uses "
        "Open3D FPFH features + custom RANSAC Procrustes (Kabsch SVD) — "
        "no extra deps, robust to partial overlap + clutter.",
    )
    prm.add_argument(
        "--backend",
        default="per_pose",
        choices=["per_pose", "bundle", "bundle_joints"],
        help="Aggregation backend. 'per_pose' = independent ICP + weighted "
        "average (default, fast). 'bundle' = single SE(3) jointly optimised "
        "via se(3) LM. 'bundle_joints' = single SE(3) + per-joint offset "
        "Δθ jointly optimised; absorbs systematic FK error (LeRobot servo "
        "zero ≠ URDF zero) that otherwise compounds for large-excursion "
        "poses.",
    )
    prm.add_argument(
        "--optimize-joints",
        default=None,
        help="Comma-separated joint names to free in bundle_joints (default "
        "= all actuated).",
    )
    prm.add_argument(
        "--joint-offset-bound",
        type=float,
        default=0.35,
        help="±rad bound on each joint offset (default 0.35 ≈ ±20°).",
    )
    prm.add_argument(
        "--bundle-inlier-distance",
        type=float,
        default=0.02,
        help="Bundle residual clamp distance (m).  CAD points whose nearest "
        "real-arm neighbour exceeds this are still counted with the clamp "
        "value, robustifying against the missing-back-half-of-arm failure "
        "mode.",
    )
    prm.add_argument(
        "--bundle-max-nfev",
        type=int,
        default=200,
        help="scipy.optimize.least_squares max function-eval budget for "
        "the bundle solver.",
    )
    prm.set_defaults(func=cmd_register_multi)

    pg = sub.add_parser("generate", help="calib.json -> scene.usd")
    pg.add_argument("--calib", required=True)
    pg.add_argument("--out", required=True)
    pg.add_argument("--ros2", action="store_true", help="attach ROS2 publisher OmniGraph")
    pg.set_defaults(func=cmd_generate)

    pr = sub.add_parser("render", help="calib.json -> PNG via Isaac Sim env")
    pr.add_argument("--calib", required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument(
        "--isaac-python",
        default=None,
        help="Path to the Isaac Sim Python interpreter (default: "
        "~/workspaces/lerobot-isaac-training/.pixi/envs/sim/bin/python)",
    )
    pr.add_argument(
        "--ros2",
        action="store_true",
        help="attach the ROS2 OmniGraph publisher (D10)",
    )
    pr.add_argument(
        "--ros2-frames",
        type=int,
        default=60,
        help="extra sim steps after warm-up so the ROS2 publisher can push "
        "frames out (default 60 = ~1 s @ 60 Hz)",
    )
    pr.set_defaults(func=cmd_render)

    pv = sub.add_parser("validate", help="forward-projection residual report")
    pv.add_argument("--calib", required=True)
    pv.add_argument("--scene", required=True)
    pv.set_defaults(func=cmd_validate)

    ps = sub.add_parser(
        "smoke",
        help="end-to-end: capture-poses -> register-multi -> render",
    )
    ps.add_argument("--urdf", required=True, help="path to SO-101 URDF")
    ps.add_argument("--poses", required=True, help="path to poses.yaml")
    ps.add_argument("--out", required=True, help="output artifact dir")
    ps.add_argument(
        "--mock",
        action="store_true",
        help="use MockArmDriver + MockD435Source (no hardware required)",
    )
    ps.add_argument("--frames", type=int, default=30)
    ps.add_argument("--seed", type=int, default=0)
    ps.add_argument(
        "--arm-port",
        default="/dev/ttyACM0",
        help="serial port for the SO-101 follower (ignored when --mock)",
    )
    ps.add_argument(
        "--workspace-z-max",
        type=float,
        default=None,
        help="Drop points beyond this Z (metres) before plane fit.",
    )
    ps.add_argument(
        "--workspace-z-min",
        type=float,
        default=None,
        help="Drop points closer than this Z before plane fit.",
    )
    ps.add_argument(
        "--expected-up",
        default=None,
        help="Expected table normal in camera frame as 'x,y,z'.",
    )
    ps.add_argument(
        "--up-tol-deg",
        type=float,
        default=30.0,
        help="Angular tolerance (deg) for --expected-up.",
    )
    ps.add_argument(
        "--arm-merge-radius",
        type=float,
        default=0.0,
        help="Merge DBSCAN clusters within this radius (m) of the largest "
        "into the arm cloud (0 = off).",
    )
    ps.add_argument(
        "--outlier-neighbors",
        type=int,
        default=0,
        help="Statistical outlier removal neighbour count (0 = off).",
    )
    ps.add_argument(
        "--outlier-std",
        type=float,
        default=2.0,
        help="Std-dev multiplier for outlier filter.",
    )
    ps.add_argument(
        "--gate-fitness",
        type=float,
        default=None,
        help="Override quality-gate minimum fitness.",
    )
    ps.add_argument(
        "--gate-rmse",
        type=float,
        default=None,
        help="Override quality-gate maximum RMSE (m).",
    )
    ps.add_argument(
        "--fallback",
        default="none",
        choices=["none", "fpfh_ransac"],
        help="Robust registration backend (default none).",
    )
    ps.add_argument(
        "--backend",
        default="per_pose",
        choices=["per_pose", "bundle", "bundle_joints"],
        help="Multi-pose aggregation backend.",
    )
    ps.add_argument("--bundle-inlier-distance", type=float, default=0.02)
    ps.add_argument("--bundle-max-nfev", type=int, default=200)
    ps.add_argument("--optimize-joints", default=None)
    ps.add_argument("--joint-offset-bound", type=float, default=0.35)
    ps.set_defaults(func=cmd_smoke)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
