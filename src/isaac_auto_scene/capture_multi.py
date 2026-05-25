"""Multi-pose capture pipeline.

Public API
----------
PoseCaptureRecord       — per-pose result + status
MultiCaptureManifest    — full run manifest written to disk
capture_pose_set        — drive arm through poses, capture RGB-D per pose
load_manifest           — reload a manifest written by capture_pose_set

Design
------
This module is *capture only*.  It commands the arm, reads back servo
angles, grabs a synchronised RGB-D frame, and persists per-pose artefacts.
Registration consumes the manifest later — keeping the two concerns split
lets capture run in the hardware pixi env without dragging Open3D-heavy
registration into the loop.

Directory layout (out_dir):

```
out_dir/
  manifest.yaml         # this file's MultiCaptureManifest
  pose_<name>/
    rgb.png
    depth_median.npy
    pcd.ply
    intrinsics.json
    joints.json         # {"commanded": {...}, "readback": {...}}
```

Failure modes per pose are isolated: a single bad pose marks itself
``status="failed"`` in the manifest without aborting the whole run.  The
caller (register-multi) decides whether to drop or retry.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml
import yourdfpy

from isaac_auto_scene.capture import D435Source, capture, save_capture
from isaac_auto_scene.poses import (
    ArmDriver,
    JointPose,
    PoseValidationReport,
    validate_pose_set,
)


@dataclass(frozen=True)
class PoseCaptureRecord:
    """One pose's capture result + on-disk artefact paths."""

    name: str
    status: str  # "ok" | "failed" | "skipped"
    commanded_joints: dict[str, float]
    readback_joints: dict[str, float]
    pose_dir: str  # relative to manifest dir
    error: str | None = None


@dataclass(frozen=True)
class MultiCaptureManifest:
    """Manifest written at the root of a capture run."""

    run_id: str
    urdf_hash: str
    num_poses: int
    num_ok: int
    poses: tuple[PoseCaptureRecord, ...] = field(default=())
    capture_frames_per_pose: int = 30

    def to_yaml(self) -> str:
        return yaml.safe_dump(
            {
                "run_id": self.run_id,
                "urdf_hash": self.urdf_hash,
                "num_poses": self.num_poses,
                "num_ok": self.num_ok,
                "capture_frames_per_pose": self.capture_frames_per_pose,
                "poses": [asdict(p) for p in self.poses],
            },
            sort_keys=False,
        )


def _hash_urdf(urdf_path: Path) -> str:
    """Stable short hash of a URDF file's bytes."""
    import hashlib

    data = Path(urdf_path).read_bytes()
    return hashlib.sha256(data).hexdigest()[:16]


def capture_pose_set(
    poses: list[JointPose],
    driver: ArmDriver,
    source: D435Source,
    urdf: yourdfpy.URDF,
    urdf_path: Path,
    out_dir: Path,
    *,
    frames_per_pose: int = 30,
    run_id: str | None = None,
    sleep: "callable[[float], None]" = time.sleep,
    validate: bool = True,
) -> MultiCaptureManifest:
    """Drive *driver* through *poses* and capture RGB-D per pose.

    Parameters
    ----------
    poses:
        Pose set to execute, in order.
    driver:
        Arm driver (mock or real).  Caller is responsible for connecting it
        via context-manager OR connecting manually before calling.
    source:
        D435 source (mock or real).  Must be already started or context-
        managed by the caller.
    urdf:
        Loaded URDF used for joint-limit validation.
    urdf_path:
        Path to the URDF file, used to record a hash in the manifest.
    out_dir:
        Output directory.  Created if missing.
    frames_per_pose:
        Frames per temporal-median capture (default 30).
    run_id:
        Optional run identifier.  Defaults to ``YYYYmmddTHHMMSSZ``.
    sleep:
        Injectable sleep (tests pass a no-op).
    validate:
        If True, validate poses against URDF before commanding any arm
        motion; raises ValueError on failure.  Set False only when the
        caller has already validated.

    Returns
    -------
    MultiCaptureManifest
        Result manifest.  Per-pose failures are captured as
        ``status="failed"`` records rather than raised exceptions.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if validate:
        report: PoseValidationReport = validate_pose_set(poses, urdf)
        if not report.ok:
            messages = "\n".join(
                f"  - {e.pose_name}: {e.reason}" for e in report.errors
            )
            raise ValueError(f"pose validation failed:\n{messages}")

    if run_id is None:
        run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    records: list[PoseCaptureRecord] = []
    for pose in poses:
        pose_dir = out_dir / f"pose_{pose.name}"
        rec = _capture_one_pose(
            pose=pose,
            driver=driver,
            source=source,
            pose_dir=pose_dir,
            frames_per_pose=frames_per_pose,
            sleep=sleep,
        )
        records.append(rec)

    num_ok = sum(1 for r in records if r.status == "ok")
    manifest = MultiCaptureManifest(
        run_id=run_id,
        urdf_hash=_hash_urdf(urdf_path),
        num_poses=len(poses),
        num_ok=num_ok,
        poses=tuple(records),
        capture_frames_per_pose=frames_per_pose,
    )
    (out_dir / "manifest.yaml").write_text(manifest.to_yaml())
    return manifest


def _capture_one_pose(
    *,
    pose: JointPose,
    driver: ArmDriver,
    source: D435Source,
    pose_dir: Path,
    frames_per_pose: int,
    sleep: "callable[[float], None]",
) -> PoseCaptureRecord:
    pose_dir = Path(pose_dir)
    rel = pose_dir.name
    try:
        driver.command_joints(pose.joints)
        sleep(pose.settle_s)
        readback = driver.read_joints()

        cap = capture(source=source, num_frames=frames_per_pose)
        save_capture(cap, pose_dir)
        (pose_dir / "joints.json").write_text(
            json.dumps(
                {
                    "commanded": pose.joints,
                    "readback": readback,
                    "settle_s": pose.settle_s,
                },
                indent=2,
            )
        )
        return PoseCaptureRecord(
            name=pose.name,
            status="ok",
            commanded_joints=dict(pose.joints),
            readback_joints=readback,
            pose_dir=rel,
        )
    except Exception as exc:  # capture per-pose, do not abort whole run
        readback = _safe_readback(driver)
        return PoseCaptureRecord(
            name=pose.name,
            status="failed",
            commanded_joints=dict(pose.joints),
            readback_joints=readback,
            pose_dir=rel,
            error=f"{type(exc).__name__}: {exc}",
        )


def _safe_readback(driver: ArmDriver) -> dict[str, float]:
    try:
        return driver.read_joints()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_manifest(out_dir: Path) -> MultiCaptureManifest:
    """Read a manifest written by :func:`capture_pose_set`."""
    out_dir = Path(out_dir)
    data = yaml.safe_load((out_dir / "manifest.yaml").read_text())
    poses = tuple(
        PoseCaptureRecord(
            name=p["name"],
            status=p["status"],
            commanded_joints={k: float(v) for k, v in p["commanded_joints"].items()},
            readback_joints={k: float(v) for k, v in p["readback_joints"].items()},
            pose_dir=p["pose_dir"],
            error=p.get("error"),
        )
        for p in data["poses"]
    )
    return MultiCaptureManifest(
        run_id=str(data["run_id"]),
        urdf_hash=str(data["urdf_hash"]),
        num_poses=int(data["num_poses"]),
        num_ok=int(data["num_ok"]),
        poses=poses,
        capture_frames_per_pose=int(data.get("capture_frames_per_pose", 30)),
    )


def load_pose_capture(manifest_dir: Path, record: PoseCaptureRecord):
    """Reload a single pose's CaptureResult from disk.

    Returns ``isaac_auto_scene.capture.CaptureResult``.
    """
    from isaac_auto_scene.capture import load_capture

    return load_capture(Path(manifest_dir) / record.pose_dir)
