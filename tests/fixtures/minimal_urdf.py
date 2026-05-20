"""Tiny URDF fixture: 2 box links connected by a revolute joint.

Used by Phase 4 (cad) tests; avoids vendoring real SO-101 STLs in CI.
"""

from __future__ import annotations

from pathlib import Path

_URDF_XML = """<?xml version="1.0"?>
<robot name="two_box">
  <link name="base">
    <visual>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry><box size="0.10 0.10 0.10"/></geometry>
    </visual>
  </link>
  <link name="arm">
    <visual>
      <origin xyz="0.10 0 0" rpy="0 0 0"/>
      <geometry><box size="0.20 0.05 0.05"/></geometry>
    </visual>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="base"/>
    <child link="arm"/>
    <origin xyz="0 0 0.10" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
</robot>
"""


def write_minimal_urdf(out_dir: Path) -> Path:
    """Write a minimal 2-link revolute URDF and return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "two_box.urdf"
    path.write_text(_URDF_XML)
    return path
