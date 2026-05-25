# isaac-auto-scene — usage

End-to-end reference for the real-to-sim calibration pipeline.  Every
command runs from the repo root inside the `hardware` pixi env unless
noted otherwise.

```bash
PIXI=`pixi run -e hardware`   # shorthand below assumes you've activated
                              # the env; otherwise prepend `pixi run -e hardware`
URDF=/path/to/so101_new_calib.urdf
```

## Persistent state

Per-arm calibration files live under `~/.config/isaac-auto-scene/`
(XDG-respecting; uses `$XDG_CONFIG_HOME` when set):

```
~/.config/isaac-auto-scene/
├── home_offset.json            # set-home output: per-joint URDF offset + sign-flip
├── calib.json                  # manual-align default output
├── calib_bundle.json           # register-multi bundle output
└── manual-calibs/              # manual-align-all per-pose calibs
    ├── calib_home.json
    └── calib_<pose>.json
```

All persist across reboots.  `home_offset.json` is auto-loaded by every
command that talks to the live arm — no need to pass `--home-offset`
unless overriding.

## Workflow

### 1. Servo calibration (once per robot)

```bash
isaac-auto-scene calibrate-arm --arm-port /dev/ttyACM0
```

Spawns `python -m lerobot.calibrate` which walks you through each servo.
Calibration JSON lands in LeRobot's cache (~/.cache/huggingface/lerobot/).

### 2. Home offset (once per robot, redo after re-mounting)

Place the arm physically in the URDF home pose (look at
`assemble_pcd(urdf, all-zeros)` output for the expected geometry — straight
arm extended forward, all links coplanar).  Then:

```bash
isaac-auto-scene set-home --urdf $URDF
# Optional: declare any joints with inverted servo direction
# isaac-auto-scene set-home --urdf $URDF --sign-flip shoulder_lift,elbow_flex
```

Writes `~/.config/isaac-auto-scene/home_offset.json` with:
- `home_offset_rad` — current joint readback (subtracted from any future
  readback before being passed to URDF FK).
- `joint_sign_flip` — list of joints whose servo direction must be
  negated to match URDF axis convention.

### 3. Single-pose calibration (fastest path)

```bash
isaac-auto-scene calibrate --live --urdf $URDF \
  --workspace-z-max 1.5 --expected-up 0,1,0 --up-tol-deg 45 \
  --arm-merge-radius 0.30 --outlier-neighbors 20 --outlier-std 1.5 \
  --restarts 20
```

Reads current joints + captures D435 frame + segments + auto-registers in
one shot.  Writes to `~/.config/isaac-auto-scene/calib.json`.  Use when
the arm is in a single canonical pose and you trust ICP.

### 4. Multi-pose calibration (production path)

#### 4a. Capture

```bash
isaac-auto-scene capture-poses --urdf $URDF \
  --poses assets/poses/so101_safe_5pose.yaml \
  --out /tmp/cap \
  --check-floor              # refuses poses that would slam the table
```

The `safe_5pose` set uses small excursions (±0.30 rad max).  Each pose:
move → settle 3 s → 15-frame median capture.  Output: per-pose RGB-D +
depth median + joint readback under `/tmp/cap/pose_<name>/`.

#### 4b. Manual alignment

```bash
isaac-auto-scene manual-align-all --urdf $URDF \
  --captures /tmp/cap \
  --no-icp-refine \
  --workspace-z-max 1.5 --expected-up 0,1,0 --up-tol-deg 45 \
  --arm-merge-radius 0.30 --outlier-neighbors 20 --outlier-std 1.5
```

Opens an Open3D viewer per pose.  Controls in-window:

| Key | Action |
|---|---|
| W/A/S/D/Q/E | translate CAD ±X/±Y/±Z (camera frame) |
| I/K/J/L/U/O | rotate around CAD centroid (pitch / yaw / roll) |
| `+` / `-` | double / halve translation step |
| `]` / `[` | double / halve rotation step |
| SPACE | snap to local ICP |
| R / Z | reset to identity / to `--init-from` |
| Y / ENTER | save (final ICP refine unless `--no-icp-refine`) |
| close X | cancel |

Each Y saves `~/.config/isaac-auto-scene/manual-calibs/calib_<name>.json`.
After the loop a summary table marks the highest-fitness pose with `*`.

#### 4c. Bundle (combined ICP)

```bash
isaac-auto-scene register-multi --captures /tmp/cap --urdf $URDF \
  --backend bundle \
  --init-from-dir ~/.config/isaac-auto-scene/manual-calibs \
  --workspace-z-max 1.5 --expected-up 0,1,0 --up-tol-deg 45 \
  --arm-merge-radius 0.30 --outlier-neighbors 20 --outlier-std 1.5
```

Bundle adjustment: one shared SE(3) jointly optimised across all 5
poses via scipy LM on the se(3) Lie algebra.  Initialised from the
fitness-weighted Markley mean of the manual calibs.

Backend options:
- `per_pose` (default) — independent ICP per pose, weighted average.
- `bundle` — single shared SE(3), best for partial-view captures.
- `bundle_joints` — bundle + per-joint Δθ offset (12-DOF; experimental,
  often over-parameterised for 5 poses).

Output: `~/.config/isaac-auto-scene/calib_bundle.json`.

### 5. Render

```bash
isaac-auto-scene render \
  --calib ~/.config/isaac-auto-scene/calib_bundle.json \
  --out /tmp/frame.png
```

Spawns Isaac Sim 6.0 (via the sibling `lerobot-isaac-training` env
Python), builds the scene with SO-101 USD + table at calibrated pose +
dome light + D435 pinhole camera, sets the articulation to the captured
joint config, renders one frame.

Auto-opens in your default image viewer; pass `--no-show` to skip.

### 6. Generate USD stub

```bash
isaac-auto-scene generate --calib ~/.config/isaac-auto-scene/calib_bundle.json \
  --out /tmp/scene.usda
```

Writes a USDA file with camera + SO-101 reference + table at calibrated
pose.  Loadable in usdview / Isaac Sim for inspection.

### 7. Validate

```bash
isaac-auto-scene validate \
  --calib ~/.config/isaac-auto-scene/calib_bundle.json \
  --scene /tmp/scene.usda
```

Prints calib summary JSON + quality_gate pass/fail.

## Pose YAML schema

```yaml
poses:
  - name: home
    joints:
      shoulder_pan: 0.0
      shoulder_lift: 0.0
      elbow_flex: 0.0
      wrist_flex: 0.0
      wrist_roll: 0.0
      gripper: 0.0
    settle_s: 3.0
  - name: pan_left
    joints: { shoulder_pan: 0.30, ... }
    settle_s: 3.0
```

Values are LeRobot servo angles in radians.  `settle_s` is the dwell
time after each command before reading joints + capturing frames.

Built-in sets in `assets/poses/`:
- `so101_safe_5pose.yaml` — conservative, ≤0.30 rad excursions, default.
- `so101_smoke_5pose.yaml` — aggressive, larger angles; needs the
  home-offset workflow + `--check-floor` to avoid table impact.

## Tuning knobs

Segmentation (all registration commands):

| flag | default | meaning |
|---|---|---|
| `--workspace-z-max` | none | drop points beyond this Z (m) in camera frame before plane fit |
| `--workspace-z-min` | none | drop points closer than this |
| `--expected-up x,y,z` | none | expected table normal direction (`0,1,0` for D435 looking down at table); rejects wall/desk-side planes |
| `--up-tol-deg` | 30 | angular tolerance for `--expected-up` |
| `--arm-merge-radius` | 0 | merge DBSCAN clusters within this radius (m); `0.30` for SO-101 |
| `--outlier-neighbors` | 0 | statistical outlier removal nb_neighbors; `20` typical |
| `--outlier-std` | 2.0 | outlier filter std-dev multiplier (lower = more aggressive) |

Quality gate:

| flag | default | meaning |
|---|---|---|
| `--gate-fitness` | 0.65 | minimum ICP fitness |
| `--gate-rmse` | 0.005 (5 mm) | maximum inlier RMSE in metres |

Bundle:

| flag | default | meaning |
|---|---|---|
| `--init-from <calib.json>` | — | single explicit init |
| `--init-from-dir <dir>` | — | fitness-weighted mean of `calib_*.json` |
| `--bundle-inlier-distance` | 0.02 | residual clamp (m) — robustifies against missing back-half-of-arm |
| `--bundle-max-nfev` | 200 | scipy LM evaluation budget |
| `--optimize-joints` | all | csv of joints to free in `bundle_joints` backend |
| `--joint-offset-bound` | 0.35 | ±rad bound on each joint offset |
