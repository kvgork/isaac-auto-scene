# isaac-auto-scene — troubleshooting

Catalogue of pitfalls observed during real-hardware bring-up, grouped by
subsystem.  Most have fail-fast checks wired in — see the "remedy" entry
for the CLI flag or command that resolves it.

## Hardware connection

### `Permission denied: '/dev/ttyACM0'`

Cause: your user is in the `dialout` group at the OS level, but the
current terminal session predates the group addition — group
membership only activates on a new login.

Remedy:

```bash
# In the failing terminal:
newgrp dialout

# Then re-run.  Or, for a single command:
sg dialout -c 'pixi run -e hardware isaac-auto-scene calibrate-arm'
```

Lasting fix: log out and back in.

### `Frame didn't arrive within 5000` (D435 timeout)

Cause: the D435 enumerated on USB 2.x.  `RealSenseD435Source.start()`
now fail-fasts with an explicit message; the underlying timeout means
the colour + depth pipeline can't sync within the budget at USB 2.0
bandwidth.

Diagnose:

```bash
lsusb -t                                            # check device speed
pixi run -e hardware python -c "
import pyrealsense2 as rs
print(next(iter(rs.context().query_devices())).get_info(rs.camera_info.usb_type_descriptor))"
```

Remedy: replug into a USB3 port (blue connector / SS-marked).  The
fail-fast check refuses to start if `usb_type_descriptor` is not 3.x.

### D435 RGB sensor missing (single `Stereo Module` only)

Cause: firmware glitch; sometimes only one sensor enumerates after a
warm boot.

Remedy:

```bash
pixi run -e hardware python -c "
import pyrealsense2 as rs
next(iter(rs.context().query_devices())).hardware_reset()"
sleep 5
```

Then retry the capture.

## Isaac Sim renderer

### `vkAllocateMemory failed: ERROR_OUT_OF_DEVICE_MEMORY`

Cause: GPU VRAM exhaustion.  Render product allocation fails silently
inside the Vulkan layer; the script then loops forever waiting for an
rgb buffer that will never arrive.

Diagnose:

```bash
nvidia-smi --query-gpu=memory.free --format=csv
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
```

Remedy: the render script pre-flights `nvidia-smi` before booting
Isaac Sim and refuses to start if free VRAM is below the budget (1500
MiB baseline, 2500 MiB with `--ros2`).  Stop the competing workload
(a training run holding ~7 GiB is the usual suspect) and retry.

### `CPU performance profile is set to powersave` warning

Cosmetic on modern Intel CPUs with `intel_pstate` in active mode.  The
kernel governor is named `powersave` but cooperates with HWP and the
Energy-Performance-Preference (EPP) hint — when EPP is `performance`,
the CPU still hits turbo + full clocks.  Verify:

```bash
cat /sys/devices/system/cpu/intel_pstate/status     # expect: active
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor   # may say powersave
powerprofilesctl get                                # expect: performance
```

To silence the carb warning explicitly:

```bash
sudo cpupower frequency-set -g performance
```

### `Failed to get a valid attached USD stage id from PhysX simulation for kinematic bodies`

Cosmetic on Isaac Sim 6.0 with the SO-101 USD reference path.  The
render still completes correctly.  Was a real hang trigger on earlier
test branches when the Articulation runtime tried to re-spawn an
already-spawned prim; current `_build_so101_articulation` spawns once
via `cfg.spawn.func()` then binds the Articulation wrapper to the
existing prim.

## ICP / registration

### Per-pose fitness stuck around 0.27–0.32 on real captures

Cause: partial-view fitness ceiling.  D435 sees only the front-facing
hemisphere of the arm; the URDF-FK CAD has points on all sides.  ICP
fitness = fraction of CAD points with a target match — physically
capped at ~0.50 for half-visible objects.

Compounded by:
- Cylindrical-link symmetry — rotation about a link's long axis is
  unobservable from a single view, so different per-pose ICP runs land
  in different ambiguous basins.
- ICP random-restart wedge was too narrow (±0.6 rad) before commit
  `6b413d8`; now uses full SO(3) sampling.

Remedies:
- Use `--backend bundle` to solve one shared SE(3) across all poses;
  joint-angle diversity disambiguates the symmetry.
- Lower the quality gate for hardware bring-up:
  `--gate-fitness 0.40 --gate-rmse 0.015`.
- Use `manual-align-all` to seed bundle from human-placed Ts — the
  five manual fits average to a much better init than per-pose ICP.

### Bundle drops one pose to fitness 0.12 while others ≥0.5

Cause: FK error compounding.  URDF kinematics + LeRobot servo zero
have a constant offset; the offset is invisible at home but
accumulates along the chain for poses with large joint excursions.

Diagnose by computing per-pose centroid offset between the
bundle-predicted CAD position and the segmented arm cloud:

```python
cad_cam = cad_pts @ R.T + t       # CAD-after-bundle-T in camera frame
arm_cam = np.asarray(arm_cloud.points)
offset_mm = np.linalg.norm(cad_cam.mean(0) - arm_cam.mean(0)) * 1000
```

If `offset_mm` correlates with joint excursion magnitude, the
LeRobot↔URDF offset is the cause.

Remedies:
- `--backend bundle_joints` — bundle + per-joint Δθ offset (experimental).
- Re-run `set-home` with the arm placed as close to URDF home as you
  can manage physically.
- For visualization-only use, accept the per-pose floor and use
  individual manual-align calibs at each pose of interest.

### Manual alignment "rotated 90°" / arm pose visibly wrong

Two distinct causes:

1. Render uses URDF home (all zeros) but T_cam_arm was fit to the
   captured (folded) CAD.  Fixed in commit `110d4a9` —
   `joint_angles_at_capture` from calib.json is now applied via
   `Articulation.write_joint_state_to_sim()`.

2. Camera placement uses world-origin → arm-target convention.  Earlier
   versions aimed the render camera at the world origin (where the
   table was — and where the arm wasn't).  Fixed in commit `94527d6`
   — camera now sits at origin (world == camera frame) and aims at
   `spec.arm_position_m`.

If your render still shows the wrong orientation after a fresh
`manual-align-all`: regenerate the calib (older calibs lacking
`T_cam_table` or `arm_joint_angles_rad` will fall back to defaults
that may not match).

## SO-101 sign convention

### CAD orientation inverted vs point cloud

Cause: servo direction not aligned with URDF joint axis on some joints
(observed on `shoulder_lift` + `elbow_flex` on the test arm).  When
the CAD-FK and physical arm disagree about which direction a joint
"raises", the rendered model appears mirrored against the captured
cloud.

Test direction by commanding small positive vs negative angle and
watching the physical motion.  If positive-servo physically lowers
while URDF FK predicts a raise (or vice versa), declare the joint
in the sign flip list:

```bash
isaac-auto-scene set-home --urdf $URDF \
  --sign-flip shoulder_lift,elbow_flex
```

The driver then negates both the command sent to the servo AND the
readback returned, so the rest of the pipeline treats the joint as
URDF-aligned.  The flip persists in `home_offset.json`.

If you set sign-flip and observe the inversion FLIPPING DIRECTION,
the URDF and servo were already agreeing — clear the flip:

```bash
isaac-auto-scene set-home --urdf $URDF --sign-flip ""
```

## Open3D pitfalls on this build (Python 3.11 + Open3D 0.18)

Per the project CLAUDE.md, the following legacy registration paths
segfault and must be avoided:

- `o3d.pipelines.registration.registration_fgr_based_on_feature_matching`
- `o3d.pipelines.registration.registration_ransac_based_on_feature_matching`
- `o3d.pipelines.registration.registration_icp` (legacy)
- `o3d.pipelines.registration.registration_generalized_icp`

The pipeline uses `o3d.t.pipelines.registration.icp` (tensor API) for
all ICP, and rolls its own FPFH+RANSAC+Kabsch fallback for the
"robust" path (see `learned_register.py`).

`PointCloud.create_from_rgbd_image(...)` also segfaults — capture uses
manual back-projection `(u - cx) * z / fx`, `(v - cy) * z / fy`.

## Pose validation

### Arm slams the table during `capture-poses`

Cause: pose YAML angles are LeRobot servo values; on an arm where
LeRobot zero ≠ URDF zero, the URDF-derived "raise/lower" intuition
breaks.  Aggressive YAMLs (e.g. `so101_smoke_5pose.yaml`) can command
shoulder_lift / elbow_flex configurations that physically slam the
gripper through the table.

Remedy: pass `--check-floor --home-offset ~/.config/...`.  The flag
runs URDF FK on every pose before any motor target is sent and
refuses any pose whose link AABB falls below `--floor-z` (default
-5 mm).  Use `assets/poses/so101_safe_5pose.yaml` for first-time
hardware bring-up.

## Misc

### `pixi install` warns about unused `[feature.lerobot]` / `[feature.ros2]`

Fixed in commit `ef64352` — both dangling feature blocks removed.
LeRobot is installed via `pixi run install-lerobot` (pip into the
active env, bypassing the resolver) because its rerun-sdk wheel
doesn't match pixi's reported manylinux baseline.

### Calibration files vanish after reboot

Earlier defaults wrote to `/tmp/...`.  Fixed in commit `1782a3d` —
all `set-home`, `manual-align`, `manual-align-all`, and `register-multi`
defaults now point to `~/.config/isaac-auto-scene/`.  Files survive
reboots and stay out of the repo.

If you have leftover `/tmp/calib.json` from before the fix:

```bash
mv /tmp/calib.json ~/.config/isaac-auto-scene/calib.json
```
