# Predefined Environment Spheres (Offline Record/Playback)

This document describes how to **record** tabletop obstacle spheres once (from depth), then **replay** them during experiments to speed up Safety Filter + Intent Prediction testing.

All published spheres are in `torso_lift_link` frame and are published to:
- `/detected_spheres` as `std_msgs/Float64MultiArray` in the flat format `[x,y,z,r, x,y,z,r, ...]`.

## 0) What You Get

- **Record mode**: runs perception online and writes a JSON file with the filtered spheres.
- **Playback mode**: does NOT use camera topics; it only loads the JSON and republishes `/detected_spheres` at a chosen rate.
- **Edit tool**: batch-adjust the saved spheres (translate/scale/clamp/filter) and see changes immediately in playback (file mtime hot-reload).

## 1) Record Spheres Once (Offline Capture)

Run on the robot/server machine with camera + TF available:

```bash
rosrun tiago_server pointcloud_to_sphere.py \
  _mode:=record \
  _record_path:=/tmp/table_env.json \
  _z_min:=0.0
```

Notes:
- `_z_min` is applied in `torso_lift_link` frame. For a tabletop scene, you typically want `_z_min:=table_height` (see below).
- If the scene looks mirrored, see **Axis flips**.

### Axis flips (only if needed)
The node projects depth into `xtion_rgb_optical_frame` (ROS optical convention). If your setup is mirrored, you can flip axes:

```bash
# Example: flip x
rosrun tiago_server pointcloud_to_sphere.py _mode:=record _record_path:=/tmp/table_env.json _negate_x:=true

# Example: flip y
rosrun tiago_server pointcloud_to_sphere.py _mode:=record _record_path:=/tmp/table_env.json _negate_y:=true
```

## 2) Playback Spheres During Experiments (Fast)

```bash
rosrun tiago_server pointcloud_to_sphere.py \
  _mode:=playback \
  _record_path:=/tmp/table_env.json \
  _playback_rate_hz:=20 \
  _z_min:=0.0
```

Playback mode reloads the JSON when the file changes (mtime cache), so edits take effect without restarting.

## 3) Tune the Saved Spheres (Adjustments)

Use the provided utility:

```bash
python3 tiago_server/tiago_server/perception/edit_predefined_spheres.py /tmp/table_env.json \
  --dx 0.02 --dy 0.00 --dz 0.00 \
  --radius-scale 1.05 \
  --z-min 0.0
```

Common operations:
- `--dx/--dy/--dz`: shift all spheres if there is a consistent offset.
- `--radius-scale`: make obstacles more conservative (e.g. `1.1`).
- `--z-min`: drop spheres below a plane.

## 4) Setting Table Height (Recommended)

If your tabletop is at height `table_height` in `torso_lift_link`, set:
- Record: `_z_min:=<table_height>`
- Playback: `_z_min:=<table_height>`

Example (table at 0.12m):
```bash
rosrun tiago_server pointcloud_to_sphere.py _mode:=record _record_path:=/tmp/table_env.json _z_min:=0.12
rosrun tiago_server pointcloud_to_sphere.py _mode:=playback _record_path:=/tmp/table_env.json _z_min:=0.12
```

## 5) How Safety Filter Consumes This

The right-arm safety filter expects obstacles as `[[x,y,z,r], ...]` in `torso_lift_link` frame.
When the server publishes `/detected_spheres`, the server state exposes it as `obs['obstacles']`, which the client can pass into the safety filter.
