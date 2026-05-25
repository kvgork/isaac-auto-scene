"""LeRobot-backed SO-101 ArmDriver for hardware capture (Phase 7).

Wraps ``lerobot.robots.so_follower.SO101Follower`` behind the
:class:`isaac_auto_scene.poses.ArmDriver` protocol so the multi-pose
capture pipeline can drive the real arm with the same interface used by
:class:`isaac_auto_scene.poses.MockArmDriver`.

Soft-import contract
--------------------
The :mod:`lerobot` package is imported lazily inside ``connect()`` so this
module can be imported in environments without LeRobot installed.
Instantiating :class:`LeRobotSO101Driver` is also safe — only the actual
``connect()`` call raises if LeRobot is missing.

Units
-----
LeRobot's Feetech driver returns joint positions in **degrees**.  The rest
of this codebase uses **radians** throughout.  The driver converts at the
boundary so ``command_joints`` / ``read_joints`` are radians-on-the-wire,
matching :class:`MockArmDriver`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LeRobotSO101Config:
    """Connection parameters for an SO-101 follower.

    Attributes
    ----------
    port:
        Serial device path (e.g., ``/dev/ttyACM0``).
    calibrate:
        Whether to run LeRobot's calibration prompt at ``connect()``.
        Set to False once the arm has been calibrated and the calibration
        JSON is on disk in LeRobot's cache.
    id:
        Optional robot ID for multi-arm setups.  None = single arm.
    joint_name_map:
        Optional mapping from LeRobot motor name -> URDF joint name.
        Default (None) assumes both use the canonical SO-101 names
        (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll,
        gripper) — true when the URDF is ``so101_new_calib.urdf``.
    """

    port: str = "/dev/ttyACM0"
    calibrate: bool = False
    id: str | None = None
    joint_name_map: dict[str, str] | None = None
    joint_sign_flip: tuple[str, ...] = ()
    """Per-joint sign flip applied at the LeRobot <-> URDF boundary.

    Some SO-101 servos are wired so that positive servo angle
    corresponds to URDF-negative motion (and vice versa).  Adding a
    joint name to this tuple negates BOTH the value sent to the servo
    on command_joints AND the value returned by read_joints, so the
    rest of the codebase can treat the joint as if it were
    URDF-aligned.

    Observed defaults on the test SO-101:
      shoulder_lift  — flipped (positive servo = lower physically)
      elbow_flex     — flipped (likely; verify per arm)

    Set via ``LeRobotSO101Config(joint_sign_flip=("shoulder_lift",
    "elbow_flex"))``.
    """


@dataclass
class LeRobotSO101Driver:
    """ArmDriver implementation that drives the real SO-101 via LeRobot.

    Satisfies :class:`isaac_auto_scene.poses.ArmDriver` so it can be
    plugged into :func:`isaac_auto_scene.capture_multi.capture_pose_set`.
    """

    config: LeRobotSO101Config = field(default_factory=LeRobotSO101Config)
    _robot: Any = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # ArmDriver protocol
    # ------------------------------------------------------------------

    def connect(self) -> None:
        try:
            # lerobot 0.3.x uses lerobot.robots.so101_follower; 0.4+ moved to
            # lerobot.robots.so_follower. Try the newer path first.
            try:
                from lerobot.robots.so_follower import (  # type: ignore[import]
                    SO101Follower,
                    SO101FollowerConfig,
                )
            except ImportError:
                from lerobot.robots.so101_follower import (  # type: ignore[import]
                    SO101Follower,
                    SO101FollowerConfig,
                )
        except ImportError as exc:
            raise RuntimeError(
                "lerobot is not installed in this pixi env. "
                "Install via `.pixi/envs/hardware/bin/pip install "
                "'lerobot[feetech]>=0.3.2,<0.4'` (see pixi.toml for resolver "
                "caveat), or invoke from the sibling lerobot-isaac-training "
                "env which already has it."
            ) from exc

        cfg_kwargs: dict[str, Any] = {"port": self.config.port}
        if self.config.id is not None:
            cfg_kwargs["id"] = self.config.id
        robot_cfg = SO101FollowerConfig(**cfg_kwargs)
        self._robot = SO101Follower(robot_cfg)
        self._robot.connect(calibrate=self.config.calibrate)

    def disconnect(self) -> None:
        if self._robot is not None and self._robot.is_connected:
            self._robot.disconnect()
        self._robot = None

    def command_joints(self, joints: dict[str, float]) -> None:
        """Command joints (radians).  Non-blocking on the wire."""
        if self._robot is None:
            raise RuntimeError("LeRobotSO101Driver: call connect() first")
        action = {}
        flips = set(self.config.joint_sign_flip)
        for jname, rad in joints.items():
            motor = self._urdf_to_motor(jname)
            value = float(rad)
            if jname in flips or motor in flips:
                value = -value
            action[f"{motor}.pos"] = math.degrees(value)
        self._robot.send_action(action)

    def read_joints(self) -> dict[str, float]:
        """Read joint angles (radians)."""
        if self._robot is None:
            raise RuntimeError("LeRobotSO101Driver: call connect() first")
        obs = self._robot.get_observation()
        flips = set(self.config.joint_sign_flip)
        out: dict[str, float] = {}
        for key, value in obs.items():
            if not key.endswith(".pos"):
                continue  # camera channels and other non-joint observations
            motor = key[: -len(".pos")]
            urdf_joint = self._motor_to_urdf(motor)
            rad = math.radians(float(value))
            if urdf_joint in flips or motor in flips:
                rad = -rad
            out[urdf_joint] = rad
        return out

    def __enter__(self) -> "LeRobotSO101Driver":
        self.connect()
        return self

    def __exit__(self, *args: object) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _urdf_to_motor(self, urdf_joint: str) -> str:
        mapping = self.config.joint_name_map
        if mapping is None:
            return urdf_joint
        # config maps motor -> urdf; invert
        inverse = {v: k for k, v in mapping.items()}
        return inverse.get(urdf_joint, urdf_joint)

    def _motor_to_urdf(self, motor: str) -> str:
        mapping = self.config.joint_name_map
        if mapping is None:
            return motor
        return mapping.get(motor, motor)


__all__ = ["LeRobotSO101Config", "LeRobotSO101Driver"]
