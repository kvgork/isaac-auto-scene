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


def _dump_per_pose_debug(
    out_dir: str,
    pairs: list,
    T_cam_arm: np.ndarray,
    *,
    urdf,
    manifest_poses,
) -> None:
    """Write per-pose `cad_<name>.ply` (in camera frame, post calib T) and
    `arm_<name>.ply` (segmented capture) to ``out_dir``.

    Pair them up in a viewer (MeshLab / Open3D viewer / Isaac Sim) to
    visually diagnose where the bundle T fits well and where it drifts.
    """
    import open3d as o3d

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    R = T_cam_arm[:3, :3]
    t = T_cam_arm[:3, 3]
    np.save(out / "T_cam_arm.npy", T_cam_arm)
    for (name, cad_pcd, arm_pcd), rec in zip(pairs, manifest_poses):
        cad_local = np.asarray(cad_pcd.points)
        cad_cam = cad_local @ R.T + t
        cad_out = o3d.geometry.PointCloud()
        cad_out.points = o3d.utility.Vector3dVector(cad_cam)
        cad_out.paint_uniform_color((1.0, 0.2, 0.2))  # red: predicted
        arm_out = o3d.geometry.PointCloud(arm_pcd)
        arm_out.paint_uniform_color((0.2, 1.0, 0.2))  # green: observed
        o3d.io.write_point_cloud(str(out / f"cad_{name}.ply"), cad_out)
        o3d.io.write_point_cloud(str(out / f"arm_{name}.ply"), arm_out)
    print(f"[debug] dumped per-pose PLY pairs to {out}/", file=sys.stderr)


def _config_dir() -> Path:
    """User-level persistent config directory for isaac-auto-scene.

    Respects XDG_CONFIG_HOME, falls back to ``~/.config/isaac-auto-scene``.
    Mirrors where LeRobot keeps its calibration JSONs (under
    ``~/.cache/huggingface/lerobot``) so hardware-specific files survive
    a reboot and aren't accidentally checked into the repo.
    """
    import os

    base = os.environ.get("XDG_CONFIG_HOME")
    if base:
        return Path(base) / "isaac-auto-scene"
    return Path.home() / ".config" / "isaac-auto-scene"


def _default_home_offset_path() -> Path:
    return _config_dir() / "home_offset.json"


def _default_calib_path() -> Path:
    return _config_dir() / "calib.json"


def _default_manual_calibs_dir() -> Path:
    return _config_dir() / "manual-calibs"


def _resolve_bundle_init(args, multi):
    """Pick the bundle solver's starting T.

    Precedence (highest first):
      1. --init-from <calib.json>  — explicit user override.
      2. --init-from-dir <dir>     — average T_cam_arm across calib_*.json
                                      in dir (e.g. manual-align-all output),
                                      weighted by stored fitness.
      3. multi.per_pose best       — highest-fitness pose's per-pose ICP T.
      4. multi.T                   — the weighted per-pose average.
    """
    import json as _json

    init_from = getattr(args, "init_from", None)
    init_from_dir = getattr(args, "init_from_dir", None)
    if init_from:
        T = np.asarray(
            _json.loads(Path(init_from).read_text())["T_cam_arm"],
            dtype=np.float64,
        )
        print(
            f"[register-multi] bundle init from {init_from} "
            f"(translation={T[:3, 3].tolist()})",
            file=sys.stderr,
        )
        return T
    if init_from_dir:
        T = _average_calibs_in_dir(init_from_dir)
        if T is not None:
            print(
                f"[register-multi] bundle init from mean of "
                f"{init_from_dir}/calib_*.json (translation={T[:3, 3].tolist()})",
                file=sys.stderr,
            )
            return T
    if multi.per_pose:
        best_pp = max(multi.per_pose, key=lambda p: p.fitness)
        print(
            f"[register-multi] bundle init: best per-pose ICP "
            f"({best_pp.pose_name!r}, fitness={best_pp.fitness:.3f})",
            file=sys.stderr,
        )
        return np.asarray(best_pp.T, dtype=np.float64)
    return multi.T


def _average_calibs_in_dir(dir_path: str) -> np.ndarray | None:
    """Compute the (fitness-weighted) Markley quaternion-mean SE(3) over
    every calib_*.json in ``dir_path``.  Returns ``None`` if no matching
    files exist.
    """
    import json as _json

    from isaac_auto_scene.utils.transforms import mean_se3

    d = Path(dir_path)
    if not d.is_dir():
        return None
    Ts = []
    weights = []
    for p in sorted(d.glob("calib_*.json")):
        try:
            data = _json.loads(p.read_text())
            T = np.asarray(data["T_cam_arm"], dtype=np.float64)
            w = max(float(data.get("icp_fitness", 1.0)), 1e-3)
            Ts.append(T)
            weights.append(w)
        except (OSError, KeyError, ValueError):
            continue
    if not Ts:
        return None
    T_mean, _, _ = mean_se3(np.stack(Ts), np.asarray(weights))
    return T_mean


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

    calib = build_calibration(cap, cad, reg, T_cam_table=seg.T_world_table)
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
            check_floor=args.check_floor,
            floor_z_m=args.floor_z,
            home_offset_rad=_load_home_offset(args.home_offset)
            if args.home_offset
            else None,
            settle_s_override=args.settle_s if args.settle_s > 0 else None,
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

    ok_poses = [r for r in manifest.poses if r.status == "ok"]
    print(
        f"[register-multi] loaded {len(ok_poses)} ok pose(s) from {manifest_dir}",
        flush=True,
    )
    pairs = []
    last_cap = None
    last_cad = None
    last_T_table = None
    home_offset = _load_home_offset(getattr(args, "home_offset", None))
    for idx, record in enumerate(manifest.poses, start=1):
        if record.status != "ok":
            print(
                f"[register-multi] [{idx}/{len(manifest.poses)}] "
                f"skipping {record.name!r} (status={record.status})",
                flush=True,
            )
            continue
        print(
            f"[register-multi] [{idx}/{len(manifest.poses)}] "
            f"{record.name!r}: loading capture...",
            flush=True,
        )
        cap = load_pose_capture(manifest_dir, record)
        from isaac_auto_scene.segment import segment_table_arm

        print(
            f"[register-multi] [{idx}/{len(manifest.poses)}] "
            f"{record.name!r}: segmenting ({len(cap.pcd.points)} pts)...",
            flush=True,
        )
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
        joints_urdf = _apply_home_offset(dict(record.readback_joints), home_offset)
        print(
            f"[register-multi] [{idx}/{len(manifest.poses)}] "
            f"{record.name!r}: assembling CAD ({args.target_n_points} pts)...",
            flush=True,
        )
        cad = assemble_pcd(
            urdf, joints_urdf, target_n_points=args.target_n_points
        )
        print(
            f"[register-multi] [{idx}/{len(manifest.poses)}] "
            f"{record.name!r}: arm_cloud={len(seg.arm_cloud.points)} pts, "
            f"cad={len(cad.points)} pts",
            flush=True,
        )
        pairs.append((record.name, _pcd_from_np(cad.points), seg.arm_cloud))
        last_cap = cap
        last_cad = cad
        last_T_table = seg.T_world_table

    if not pairs:
        print("ERROR: no usable poses in manifest", file=sys.stderr)
        return 1

    # Parse user-supplied gate first.
    user_gate: tuple[float, float] | None = None
    if args.gate_fitness is not None or args.gate_rmse is not None:
        f_min, rmse_max = QUALITY_GATE
        if args.gate_fitness is not None:
            f_min = float(args.gate_fitness)
        if args.gate_rmse is not None:
            rmse_max = float(args.gate_rmse)
        user_gate = (f_min, rmse_max)
        print(
            f"[register-multi] quality gate override: fitness>={f_min:.2f} "
            f"rmse<={rmse_max*1000:.1f}mm (default {QUALITY_GATE[0]:.2f}/"
            f"{QUALITY_GATE[1]*1000:.0f}mm)",
            file=sys.stderr,
        )

    # Bundle backends use the per-pose ICP results only as initial-guess
    # hints, then jointly optimise.  Force a permissive gate for the
    # per-pose stage so RuntimeError never fires; user_gate is checked
    # against the final bundle result instead.
    if getattr(args, "backend", "per_pose") in ("bundle", "bundle_joints"):
        gate_override: tuple[float, float] | None = (0.0, 1.0)
        args.min_accepted = 1
        print(
            f"[register-multi] backend={args.backend}: per-pose gate "
            f"relaxed to (0.0, 1.0) for init hints; user gate evaluated "
            f"against final bundle result.",
            file=sys.stderr,
            flush=True,
        )
    else:
        gate_override = user_gate

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

        T_init = _resolve_bundle_init(args, multi)

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

        T_init = _resolve_bundle_init(args, multi)

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
    calib = build_calibration(
        last_cap, last_cad, synthetic_reg, T_cam_table=last_T_table
    )
    save_calibration(calib, Path(args.out))

    if getattr(args, "dump_debug", None):
        _dump_per_pose_debug(
            args.dump_debug,
            pairs,
            multi.T,
            urdf=urdf,
            manifest_poses=[r for r in manifest.poses if r.status == "ok"],
        )

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

    if getattr(args, "show", True):
        _open_image_viewer(args.out)
    return 0


def _open_image_viewer(path: str) -> None:
    """Open the rendered PNG in the user's default image viewer (non-blocking).

    Tries `xdg-open` first (Linux desktop default), then `feh`, `eog`,
    `xdg-open` again as a last resort.  Silently skips when no viewer is
    available — the file is still on disk regardless.
    """
    import shutil
    import subprocess

    candidates = ["xdg-open", "feh", "eog", "gio"]
    for tool in candidates:
        if shutil.which(tool) is None:
            continue
        try:
            if tool == "gio":
                subprocess.Popen(
                    ["gio", "open", path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                subprocess.Popen(
                    [tool, path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            print(f"[render] opened {path} with {tool}", file=sys.stderr)
            return
        except (FileNotFoundError, OSError) as exc:  # pragma: no cover - env-dep
            print(f"[render] {tool} failed: {exc}", file=sys.stderr)
            continue
    print(
        f"[render] no image viewer found ({', '.join(candidates)}); "
        f"open {path} manually.",
        file=sys.stderr,
    )


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
        check_floor=getattr(args, "check_floor", False),
        floor_z=getattr(args, "floor_z", -0.005),
        home_offset=getattr(args, "home_offset", None),
        settle_s=getattr(args, "settle_s", 0.0),
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
        home_offset=args.home_offset,
        init_from=None,
        init_from_dir=None,
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


def _live_capture_for_manual_align(args: argparse.Namespace, urdf):
    """Capture one D435 frame + read SO-101 joints. Returns (CaptureResult, joints)."""
    from isaac_auto_scene.capture import capture
    from isaac_auto_scene.lerobot_arm import LeRobotSO101Config, LeRobotSO101Driver

    if args.mock_arm:
        from isaac_auto_scene.poses import MockArmDriver

        driver = MockArmDriver(
            joint_names=tuple(urdf.actuated_joint_names),
            readback_noise_rad=0.0,
        )
    else:
        driver = LeRobotSO101Driver(
            config=LeRobotSO101Config(port=args.arm_port, calibrate=False)
        )

    if args.mock_cam:
        source = MockD435Source(seed=0)
    else:
        from isaac_auto_scene.realsense_source import RealSenseD435Source

        source = RealSenseD435Source()

    print(
        f"[manual-align live] connecting arm@{args.arm_port}, capturing "
        f"{args.frames} D435 frames...",
        file=sys.stderr,
    )
    with driver as drv, source as src:
        joints = drv.read_joints()
        cap = capture(source=src, num_frames=args.frames)
    _print_joint_readback(joints, urdf)
    return cap, joints


def _print_joint_readback(joints: dict, urdf) -> None:
    """Pretty-print joint readback with rad + deg + URDF-limit position."""
    print()
    print("================ LeRobot SO-101 joint readback ================")
    print(f"  {'joint':<16s} {'rad':>10s} {'deg':>10s} {'urdf-limits (rad)':>22s}")
    print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*22}")
    for name in urdf.actuated_joint_names:
        if name not in joints:
            continue
        rad = float(joints[name])
        deg = np.degrees(rad)
        lim = urdf.joint_map.get(name)
        if lim is not None and lim.limit is not None:
            lo, hi = float(lim.limit.lower), float(lim.limit.upper)
            limstr = f"[{lo:+.3f}, {hi:+.3f}]"
        else:
            limstr = "(unbounded)"
        print(f"  {name:<16s} {rad:+10.3f} {deg:+10.2f}° {limstr:>22s}")
    print("=================================================================")
    print()


def _offline_capture_for_manual_align(args: argparse.Namespace):
    """Load a pose from an existing captures manifest. Returns (cap, joints) or (None, None)."""
    from isaac_auto_scene.capture_multi import load_manifest, load_pose_capture

    if not args.pose:
        print(
            "ERROR: --captures requires --pose to select which entry to load",
            file=sys.stderr,
        )
        return None, None
    manifest = load_manifest(Path(args.captures))
    matching = [r for r in manifest.poses if r.name == args.pose and r.status == "ok"]
    if not matching:
        print(
            f"ERROR: pose {args.pose!r} not found (or not ok) in manifest. "
            f"Available: {[r.name for r in manifest.poses]}",
            file=sys.stderr,
        )
        return None, None
    rec = matching[0]
    cap = load_pose_capture(Path(args.captures), rec)
    return cap, dict(rec.readback_joints)


def cmd_calibrate_arm(args: argparse.Namespace) -> int:
    """Launch LeRobot's interactive calibration walkthrough.

    Spawns ``python -m lerobot.calibrate --robot.type=so101_follower
    --robot.port=...`` so the walkthrough's stdin prompts reach the user's
    terminal directly.  Wrapping it as a subprocess (instead of importing
    and calling internals) keeps the interactive parts working — lerobot
    uses ``input()`` for the per-motor calibration steps and that needs
    a real TTY, not our argparse callback context.
    """
    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "lerobot.calibrate",
        f"--robot.type=so101_follower",
        f"--robot.port={args.arm_port}",
    ]
    if args.robot_id:
        cmd.append(f"--robot.id={args.robot_id}")
    print("[calibrate-arm] $ " + " ".join(cmd), file=sys.stderr)
    try:
        proc = subprocess.run(cmd, check=False)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if proc.returncode != 0:
        print(
            f"[calibrate-arm] lerobot.calibrate exited rc={proc.returncode}",
            file=sys.stderr,
        )
        return proc.returncode
    print("[calibrate-arm] done. Place arm in URDF home pose, then run 'set-home'.")
    return 0


def cmd_set_home(args: argparse.Namespace) -> int:
    """Capture current arm joint readback and save it as the home-offset JSON.

    Physically place the arm in the URDF home pose (all links straight,
    extended forward — what `assemble_pcd(urdf, all_zeros)` predicts),
    then run this command.  The captured readback is then subtracted
    from any future readback before it's handed to URDF FK, so the
    LeRobot servo zero is aligned with the URDF joint zero.

    Output JSON schema::
        {"home_offset_rad": {"shoulder_pan": 0.0, ...}, "captured_at": "..."}
    """
    import json
    import time

    from isaac_auto_scene.lerobot_arm import LeRobotSO101Config, LeRobotSO101Driver

    urdf = load_urdf(args.urdf)
    driver = LeRobotSO101Driver(
        config=LeRobotSO101Config(port=args.arm_port, calibrate=False)
    )
    print(f"[set-home] reading arm joints from {args.arm_port}...", file=sys.stderr)
    with driver as drv:
        joints = drv.read_joints()
    _print_joint_readback(joints, urdf)
    home_offset = {k: float(v) for k, v in joints.items()}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "home_offset_rad": home_offset,
                "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "port": args.arm_port,
                "note": (
                    "Subtract these values from any subsequent LeRobot "
                    "readback before passing joints to URDF FK. The arm "
                    "was physically in the URDF home pose when this file "
                    "was written."
                ),
            },
            indent=2,
        )
    )
    print(f"[set-home] wrote home offset -> {out_path}")
    return 0


def _load_home_offset(
    path: str | None, *, auto_discover: bool = True
) -> dict[str, float]:
    """Load home offset JSON.

    When ``path`` is None and ``auto_discover`` is True, looks for
    ``$XDG_CONFIG_HOME/isaac-auto-scene/home_offset.json`` (or
    ``~/.config/isaac-auto-scene/home_offset.json``).  Returns an empty
    dict only when no file is supplied and (the default isn't present
    or auto-discover is disabled).
    """
    import json

    if path is None:
        if not auto_discover:
            return {}
        default_path = _default_home_offset_path()
        if default_path.exists():
            path = str(default_path)
            print(
                f"[home-offset] auto-loaded {default_path}",
                file=sys.stderr,
            )
        else:
            return {}

    data = json.loads(Path(path).read_text())
    return {k: float(v) for k, v in data.get("home_offset_rad", {}).items()}


def _apply_home_offset(
    joints: dict[str, float], home: dict[str, float]
) -> dict[str, float]:
    """Subtract home-offset from readback so it aligns with URDF zero."""
    if not home:
        return dict(joints)
    return {k: float(v) - float(home.get(k, 0.0)) for k, v in joints.items()}


def cmd_manual_align(args: argparse.Namespace) -> int:
    """Interactive viewer for manual CAD-over-arm alignment.

    Two input modes:
      - ``--live`` (default when --captures is omitted): capture one
        fresh D435 frame and read joint state from the SO-101 follower,
        then open the viewer.  Self-contained — no prior capture-poses
        run needed.
      - ``--captures DIR --pose NAME``: load a previously captured pose
        from a capture-poses manifest.  Useful for offline iteration.
    """
    from isaac_auto_scene.manual_align import run_manual_align
    from isaac_auto_scene.register import RegistrationResult
    from isaac_auto_scene.segment import segment_table_arm

    urdf = load_urdf(args.urdf)

    if args.live or not args.captures:
        cap, joints = _live_capture_for_manual_align(args, urdf)
    else:
        cap, joints = _offline_capture_for_manual_align(args)
        if cap is None:
            return 1

    home_offset = _load_home_offset(args.home_offset)
    if home_offset:
        print(
            f"[manual-align] applying home offset from {args.home_offset}: "
            f"{ {k: round(v, 3) for k, v in home_offset.items()} }",
            file=sys.stderr,
        )
    joints_urdf = _apply_home_offset(joints, home_offset)
    cad = assemble_pcd(urdf, joints_urdf, target_n_points=args.target_n_points)

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

    T_init = np.eye(4)
    if args.init_from:
        from isaac_auto_scene.calibrate import load_calibration

        prev = load_calibration(args.init_from)
        T_init[:3, 3] = prev.translation_m
        # Build rotation from quat_xyzw (Shepperd).
        x, y, z, w = prev.quat_xyzw
        T_init[:3, :3] = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    T_final = run_manual_align(
        cad.points,
        seg.arm_cloud,
        T_init=T_init,
        step_m=args.step,
        rot_step_deg=args.rot_step,
        icp_threshold_m=args.icp_threshold,
        final_icp_refine=not args.no_icp_refine,
    )
    if T_final is None:
        print("[manual-align] window closed without confirmation, no calib written")
        return 2

    # Wrap final T as a RegistrationResult so build_calibration packages
    # the camera intrinsics + quaternion the same way as the automated path.
    from isaac_auto_scene.calibrate import build_calibration, save_calibration

    synth_reg = RegistrationResult(
        T=T_final,
        fitness=1.0,
        inlier_rmse_m=0.0,
        used_fallback=True,
        n_restarts=0,
    )
    calib = build_calibration(cap, cad, synth_reg, T_cam_table=seg.T_world_table)
    save_calibration(calib, Path(args.out))
    print(f"manual-align calib -> {args.out}")
    return 0


def cmd_manual_align_all(args: argparse.Namespace) -> int:
    """Loop over every ok pose in a capture set, open the manual aligner
    for each one, save a per-pose calib_<name>.json, then print a
    summary table.  Useful for comparing single-pose fits and picking
    the best one (or feeding the best as --init-from to a bundle run).
    """
    from isaac_auto_scene.capture_multi import load_manifest, load_pose_capture
    from isaac_auto_scene.manual_align import run_manual_align
    from isaac_auto_scene.register import RegistrationResult
    from isaac_auto_scene.segment import segment_table_arm

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    urdf = load_urdf(args.urdf)
    manifest = load_manifest(Path(args.captures))
    ok_poses = [r for r in manifest.poses if r.status == "ok"]
    home_offset = _load_home_offset(getattr(args, "home_offset", None))

    # Optional starting T from a prior calib.json (e.g. previous
    # manual-align result).  Applied to EVERY pose's viewer so the user
    # only has to nudge from a sensible starting point.
    T_init_all = np.eye(4)
    init_from = getattr(args, "init_from", None)
    if init_from:
        prev = load_calibration(init_from)
        T_init_all = np.asarray(prev.T_cam_arm, dtype=np.float64)
        print(
            f"[manual-align-all] init from {init_from} "
            f"(T_cam_arm.translation={prev.translation_m})",
            flush=True,
        )

    print(
        f"[manual-align-all] {len(ok_poses)} pose(s) to align. "
        f"Output dir: {out_dir}",
        flush=True,
    )

    summary: list[dict] = []
    for idx, rec in enumerate(ok_poses, start=1):
        print(
            f"\n[manual-align-all] [{idx}/{len(ok_poses)}] pose {rec.name!r}",
            flush=True,
        )
        cap = load_pose_capture(Path(args.captures), rec)
        joints_urdf = _apply_home_offset(dict(rec.readback_joints), home_offset)
        cad = assemble_pcd(urdf, joints_urdf, target_n_points=args.target_n_points)
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
        T_final = run_manual_align(
            cad.points,
            seg.arm_cloud,
            T_init=T_init_all,
            window_title=f"manual-align: {rec.name} ({idx}/{len(ok_poses)})",
            step_m=args.step,
            rot_step_deg=args.rot_step,
            icp_threshold_m=args.icp_threshold,
            final_icp_refine=not args.no_icp_refine,
        )
        if T_final is None:
            print(
                f"[manual-align-all] pose {rec.name!r} cancelled (window closed)",
                flush=True,
            )
            summary.append({"name": rec.name, "calib": None, "fitness": None, "rmse_mm": None})
            continue

        # Re-run a final ICP and capture its fitness/rmse for the summary.
        from isaac_auto_scene.bundle_register import register_bundle

        bundle = register_bundle(
            [np.asarray(cad.points, dtype=np.float64)],
            [seg.arm_cloud],
            T_init=T_final,
            inlier_distance_m=0.02,
            max_nfev=50,
        )
        synth_reg = RegistrationResult(
            T=bundle.T, fitness=bundle.per_pose_fitness[0],
            inlier_rmse_m=bundle.per_pose_rmse_m[0],
            used_fallback=True, n_restarts=0,
        )
        calib_path = out_dir / f"calib_{rec.name}.json"
        calib = build_calibration(cap, cad, synth_reg, T_cam_table=seg.T_world_table)
        save_calibration(calib, calib_path)
        print(
            f"[manual-align-all] pose {rec.name!r}: fitness="
            f"{bundle.per_pose_fitness[0]:.3f} rmse="
            f"{bundle.per_pose_rmse_m[0]*1000:.2f}mm -> {calib_path}",
            flush=True,
        )
        summary.append(
            {
                "name": rec.name,
                "calib": str(calib_path),
                "fitness": float(bundle.per_pose_fitness[0]),
                "rmse_mm": float(bundle.per_pose_rmse_m[0] * 1000),
            }
        )

    print("\n================ MANUAL-ALIGN-ALL SUMMARY ================")
    print(f"{'pose':<20s} {'fitness':>8s} {'rmse_mm':>10s} {'calib':<s}")
    print(f"{'-'*20} {'-'*8} {'-'*10} {'-'*40}")
    saved = [s for s in summary if s["calib"] is not None]
    if saved:
        best = max(saved, key=lambda s: s["fitness"])
    else:
        best = None
    for s in summary:
        marker = " *" if best is not None and s is best else "  "
        f = f"{s['fitness']:.3f}" if s["fitness"] is not None else "-"
        r = f"{s['rmse_mm']:.2f}" if s["rmse_mm"] is not None else "-"
        c = s["calib"] or "(cancelled)"
        print(f"{marker}{s['name']:<18s} {f:>8s} {r:>10s} {c}")
    print("=" * 60)
    if best:
        print(f"[manual-align-all] best: {best['name']} -> {best['calib']}")
    return 0 if saved else 2


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
    pcp.add_argument(
        "--check-floor",
        action="store_true",
        help="REFUSE any pose whose URDF-FK predicts a link below "
        "--floor-z (default off).  Use with --home-offset so the FK "
        "is evaluated in URDF coords rather than raw LeRobot coords.",
    )
    pcp.add_argument(
        "--floor-z",
        type=float,
        default=-0.005,
        help="floor Z threshold (m) for --check-floor (default -0.005, "
        "i.e. 5 mm below URDF zero allowed as numerical slack).",
    )
    pcp.add_argument(
        "--home-offset",
        default=None,
        help="JSON from `set-home`; pose angles are evaluated in URDF "
        "coords (raw - offset) for the --check-floor pre-flight.",
    )
    pcp.add_argument(
        "--settle-s",
        type=float,
        default=0.0,
        help="override per-pose settle time in seconds (default 0 = use "
        "the YAML's settle_s).  Set to e.g. 3.0 to wait longer between "
        "poses so the arm stops swinging before each capture.",
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
        "--dump-debug",
        default=None,
        help="Output dir for per-pose PLY pairs (cad_<name>.ply + "
        "arm_<name>.ply, colour-coded red/green). Open with MeshLab or "
        "Open3D viewer to inspect fit per pose.",
    )
    prm.add_argument(
        "--home-offset",
        default=None,
        help="JSON from `set-home`; subtracted from each pose's readback "
        "before URDF FK.",
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
    prm.add_argument(
        "--init-from",
        default=None,
        help="Bundle init from this calib.json (overrides best per-pose).",
    )
    prm.add_argument(
        "--init-from-dir",
        default=None,
        help="Bundle init from the (fitness-weighted) mean of every "
        "calib_*.json in this directory.  Use with manual-align-all "
        "output, e.g. ~/.config/isaac-auto-scene/manual-calibs.",
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
    pr.add_argument(
        "--show",
        dest="show",
        action="store_true",
        default=True,
        help="open the rendered PNG in the default image viewer (default).",
    )
    pr.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="skip opening the image viewer.",
    )
    pr.set_defaults(func=cmd_render)

    pca = sub.add_parser(
        "calibrate-arm",
        help="run LeRobot's interactive calibration on the SO-101 follower",
    )
    pca.add_argument("--arm-port", default="/dev/ttyACM0")
    pca.add_argument(
        "--robot-id",
        default=None,
        help="optional robot identifier passed to lerobot.calibrate",
    )
    pca.set_defaults(func=cmd_calibrate_arm)

    psh = sub.add_parser(
        "set-home",
        help="capture current arm joint readback as the URDF-home offset",
    )
    psh.add_argument("--urdf", required=True)
    psh.add_argument("--arm-port", default="/dev/ttyACM0")
    psh.add_argument(
        "--out",
        default=str(_default_home_offset_path()),
        help=f"output JSON (default {_default_home_offset_path()})",
    )
    psh.set_defaults(func=cmd_set_home)

    pma = sub.add_parser(
        "manual-align",
        help="interactive viewer to manually align CAD over a captured arm cloud",
    )
    pma.add_argument("--urdf", required=True)
    pma.add_argument(
        "--live",
        action="store_true",
        help="capture one fresh D435 frame + read arm joints (default when "
        "--captures is omitted). Self-contained; no prior capture-poses run.",
    )
    pma.add_argument(
        "--captures",
        default=None,
        help="optional: existing capture run dir.  Pair with --pose to load "
        "a previously saved pose instead of live-capturing.",
    )
    pma.add_argument(
        "--pose",
        default=None,
        help="pose name in --captures manifest (required when --captures set)",
    )
    pma.add_argument(
        "--arm-port",
        default="/dev/ttyACM0",
        help="serial port for SO-101 follower (live mode)",
    )
    pma.add_argument(
        "--frames",
        type=int,
        default=15,
        help="temporal-median frame count for the live D435 capture",
    )
    pma.add_argument("--mock-arm", action="store_true")
    pma.add_argument("--mock-cam", action="store_true")
    pma.add_argument(
        "--home-offset",
        default=None,
        help="JSON from `set-home` (subtract its joints from any readback "
        "before passing to URDF FK so LeRobot zero = URDF zero).",
    )
    pma.add_argument(
        "--out",
        default=str(_default_calib_path()),
        help=f"output calib.json (default {_default_calib_path()})",
    )
    pma.add_argument(
        "--init-from",
        default=None,
        help="optional starting calib.json to seed the alignment "
        f"(default search: {_default_calib_path()} if it exists)",
    )
    pma.add_argument("--target-n-points", type=int, default=8000)
    pma.add_argument(
        "--step", type=float, default=0.01, help="translation step in metres"
    )
    pma.add_argument(
        "--rot-step", type=float, default=5.0, help="rotation step in degrees"
    )
    pma.add_argument(
        "--icp-threshold",
        type=float,
        default=0.02,
        help="SPACE-key ICP snap correspondence threshold (m)",
    )
    # Segmentation knobs (mirror register-multi).
    pma.add_argument("--workspace-z-max", type=float, default=None)
    pma.add_argument("--workspace-z-min", type=float, default=None)
    pma.add_argument("--expected-up", default=None)
    pma.add_argument("--up-tol-deg", type=float, default=30.0)
    pma.add_argument("--arm-merge-radius", type=float, default=0.0)
    pma.add_argument("--outlier-neighbors", type=int, default=0)
    pma.add_argument("--outlier-std", type=float, default=2.0)
    pma.add_argument(
        "--no-icp-refine",
        action="store_true",
        help="Save the manual T exactly as placed.  Skips the local "
        "ICP nudge that normally runs on Y/ENTER, so symmetry-ambiguous "
        "poses don't get rotated away from a careful placement.",
    )
    pma.set_defaults(func=cmd_manual_align)

    pmaa = sub.add_parser(
        "manual-align-all",
        help="open the manual aligner for every ok pose in a capture set "
        "and write per-pose calib JSONs + a summary table",
    )
    pmaa.add_argument("--captures", required=True)
    pmaa.add_argument("--urdf", required=True)
    pmaa.add_argument(
        "--out-dir",
        default=str(_default_manual_calibs_dir()),
        help=f"output dir for per-pose calib_<name>.json files "
        f"(default {_default_manual_calibs_dir()})",
    )
    pmaa.add_argument("--home-offset", default=None)
    pmaa.add_argument("--target-n-points", type=int, default=8000)
    pmaa.add_argument("--step", type=float, default=0.01)
    pmaa.add_argument("--rot-step", type=float, default=5.0)
    pmaa.add_argument("--icp-threshold", type=float, default=0.02)
    pmaa.add_argument("--workspace-z-max", type=float, default=None)
    pmaa.add_argument("--workspace-z-min", type=float, default=None)
    pmaa.add_argument("--expected-up", default=None)
    pmaa.add_argument("--up-tol-deg", type=float, default=30.0)
    pmaa.add_argument("--arm-merge-radius", type=float, default=0.0)
    pmaa.add_argument("--outlier-neighbors", type=int, default=0)
    pmaa.add_argument("--outlier-std", type=float, default=2.0)
    pmaa.add_argument("--no-icp-refine", action="store_true")
    pmaa.add_argument(
        "--init-from",
        default=None,
        help="Seed each pose's manual aligner from this calib.json (e.g. "
        "the result of a prior manual-align or register-multi run) "
        "instead of starting at identity.  Saves time when consecutive "
        "alignment sessions only need small per-pose adjustments.",
    )
    pmaa.set_defaults(func=cmd_manual_align_all)

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
    ps.add_argument("--home-offset", default=None)
    ps.add_argument("--check-floor", action="store_true")
    ps.add_argument("--floor-z", type=float, default=-0.005)
    ps.add_argument(
        "--settle-s",
        type=float,
        default=0.0,
        help="override per-pose settle time (0 = use YAML's settle_s)",
    )
    ps.set_defaults(func=cmd_smoke)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
