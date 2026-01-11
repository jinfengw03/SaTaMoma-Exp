#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Edit predefined spheres JSON.

Input/Output format:
{
  "frame_id": "torso_lift_link",
  "timestamp": 1234567890.0,
  "spheres": [[x,y,z,r], ...]
}

This tool applies simple bulk edits so you can tune the offline environment.
"""

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("path", help="Path to predefined spheres json")
    p.add_argument("--out", help="Output path (default: overwrite input)")
    p.add_argument("--dx", type=float, default=0.0)
    p.add_argument("--dy", type=float, default=0.0)
    p.add_argument("--dz", type=float, default=0.0)
    p.add_argument("--radius-scale", type=float, default=1.0)
    p.add_argument("--radius-min", type=float, default=None)
    p.add_argument("--radius-max", type=float, default=None)
    p.add_argument("--z-min", type=float, default=None, help="Drop spheres with z < z-min")
    args = p.parse_args()

    in_path = Path(args.path)
    out_path = Path(args.out) if args.out else in_path

    data = load_json(in_path)
    spheres = data.get("spheres", [])

    edited = []
    dropped = 0
    for s in spheres:
        if not isinstance(s, (list, tuple)) or len(s) < 4:
            continue
        x, y, z, r = float(s[0]), float(s[1]), float(s[2]), float(s[3])
        x += args.dx
        y += args.dy
        z += args.dz
        r *= args.radius_scale
        if args.radius_min is not None:
            r = max(args.radius_min, r)
        if args.radius_max is not None:
            r = min(args.radius_max, r)
        if args.z_min is not None and z < args.z_min:
            dropped += 1
            continue
        edited.append([x, y, z, r])

    data["spheres"] = edited
    data["edited_by"] = "edit_predefined_spheres.py"
    data["edit"] = {
        "dx": args.dx,
        "dy": args.dy,
        "dz": args.dz,
        "radius_scale": args.radius_scale,
        "radius_min": args.radius_min,
        "radius_max": args.radius_max,
        "z_min": args.z_min,
        "dropped": dropped,
        "kept": len(edited),
    }

    save_json(out_path, data)
    print(f"Wrote {len(edited)} spheres to {out_path} (dropped {dropped})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
