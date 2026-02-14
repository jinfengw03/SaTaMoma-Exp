#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Visual editor for predefined spheres JSON.

This is a lightweight, offline editor for the JSON format used by
`edit_predefined_spheres.py`:

{
  "frame_id": "torso_lift_link",
  "timestamp": 1234567890.0,
  "spheres": [[x,y,z,r], ...]
}

UI (Matplotlib):
- 3D scatter of sphere centers (marker size ~ radius)
- Click a point to select a sphere
- Sliders to edit x/y/z/r
- Buttons: Prev/Next, Add, Delete, Save

Notes:
- Requires `matplotlib` and `numpy`.
- This editor operates on centers/radii only (no mesh).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _as_spheres_array(data: dict[str, Any]) -> np.ndarray:
    spheres = data.get("spheres", [])
    arr = []
    for s in spheres:
        if not isinstance(s, (list, tuple)) or len(s) < 4:
            continue
        try:
            arr.append([float(s[0]), float(s[1]), float(s[2]), float(s[3])])
        except Exception:
            continue
    if not arr:
        return np.zeros((0, 4), dtype=float)
    return np.asarray(arr, dtype=float)


def _nice_limits(vals: np.ndarray, pad: float) -> tuple[float, float]:
    if vals.size == 0:
        return -pad, pad
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if abs(hi - lo) < 1e-9:
        mid = 0.5 * (hi + lo)
        return mid - pad, mid + pad
    return lo - pad, hi + pad


def _marker_sizes(r: np.ndarray) -> np.ndarray:
    # Matplotlib scatter size is points^2; map radius (meters) to a readable size.
    # This is a heuristic; adjust if needed.
    r = np.asarray(r, dtype=float)
    r_clip = np.clip(r, 0.0, 1.0)
    return 50.0 + (r_clip / 0.10) ** 2 * 150.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("path", help="Path to predefined spheres json")
    p.add_argument("--out", help="Output path (default: overwrite input)")
    p.add_argument("--pad", type=float, default=0.10, help="Plot padding (meters)")
    p.add_argument("--default-radius", type=float, default=0.06, help="Radius for newly added spheres")
    args = p.parse_args()

    in_path = Path(args.path)
    out_path = Path(args.out) if args.out else in_path

    data = _load_json(in_path)
    spheres = _as_spheres_array(data)

    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "This tool requires matplotlib. Install with `pip install matplotlib`. "
            f"Original error: {exc}"
        )

    matplotlib.rcParams["toolbar"] = "toolmanager"  # nicer default

    selected: int = 0 if spheres.shape[0] > 0 else -1
    updating_sliders = False

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    fig.canvas.manager.set_window_title("Visual Edit Predefined Spheres")

    # Layout: reserve space at bottom for controls
    fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.30)

    def set_axes_limits() -> None:
        if spheres.shape[0] == 0:
            ax.set_xlim(-args.pad, args.pad)
            ax.set_ylim(-args.pad, args.pad)
            ax.set_zlim(-args.pad, args.pad)
            return
        xs, ys, zs = spheres[:, 0], spheres[:, 1], spheres[:, 2]
        x0, x1 = _nice_limits(xs, args.pad)
        y0, y1 = _nice_limits(ys, args.pad)
        z0, z1 = _nice_limits(zs, args.pad)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_zlim(z0, z1)

        # Best-effort equal aspect (matplotlib 3.3+)
        try:
            ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))
        except Exception:
            pass

    def colors() -> np.ndarray:
        if spheres.shape[0] == 0:
            return np.zeros((0, 4))
        c = np.tile(np.array([0.20, 0.55, 0.90, 0.85]), (spheres.shape[0], 1))
        if 0 <= selected < spheres.shape[0]:
            c[selected] = np.array([0.95, 0.30, 0.30, 1.0])
        return c

    scat = ax.scatter(
        spheres[:, 0] if spheres.shape[0] else [],
        spheres[:, 1] if spheres.shape[0] else [],
        spheres[:, 2] if spheres.shape[0] else [],
        s=_marker_sizes(spheres[:, 3]) if spheres.shape[0] else [],
        c=colors(),
        depthshade=True,
        picker=True,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"frame_id={data.get('frame_id', 'unknown')}  spheres={spheres.shape[0]}")

    status_text = fig.text(0.06, 0.26, "", fontsize=10)

    # Sliders
    ax_x = fig.add_axes([0.10, 0.20, 0.80, 0.03])
    ax_y = fig.add_axes([0.10, 0.16, 0.80, 0.03])
    ax_z = fig.add_axes([0.10, 0.12, 0.80, 0.03])
    ax_r = fig.add_axes([0.10, 0.08, 0.80, 0.03])

    # Slider ranges: initial bounding box +/- pad; radius range is generic.
    if spheres.shape[0]:
        x0, x1 = _nice_limits(spheres[:, 0], args.pad)
        y0, y1 = _nice_limits(spheres[:, 1], args.pad)
        z0, z1 = _nice_limits(spheres[:, 2], args.pad)
    else:
        x0, x1 = -0.5, 0.5
        y0, y1 = -0.5, 0.5
        z0, z1 = 0.0, 1.0

    s_x = Slider(ax_x, "x", x0, x1, valinit=float(spheres[selected, 0]) if selected >= 0 else 0.0)
    s_y = Slider(ax_y, "y", y0, y1, valinit=float(spheres[selected, 1]) if selected >= 0 else 0.0)
    s_z = Slider(ax_z, "z", z0, z1, valinit=float(spheres[selected, 2]) if selected >= 0 else 0.0)
    s_r = Slider(ax_r, "r", 0.0, 0.30, valinit=float(spheres[selected, 3]) if selected >= 0 else float(args.default_radius))

    # Buttons
    ax_prev = fig.add_axes([0.10, 0.01, 0.10, 0.05])
    ax_next = fig.add_axes([0.21, 0.01, 0.10, 0.05])
    ax_add = fig.add_axes([0.42, 0.01, 0.10, 0.05])
    ax_del = fig.add_axes([0.53, 0.01, 0.10, 0.05])
    ax_save = fig.add_axes([0.80, 0.01, 0.10, 0.05])

    b_prev = Button(ax_prev, "Prev")
    b_next = Button(ax_next, "Next")
    b_add = Button(ax_add, "Add")
    b_del = Button(ax_del, "Delete")
    b_save = Button(ax_save, "Save")

    def set_status(msg: str) -> None:
        status_text.set_text(msg)

    def redraw() -> None:
        # Update scatter data
        if spheres.shape[0] == 0:
            scat._offsets3d = ([], [], [])
            scat.set_sizes([])
            scat.set_facecolors([])
        else:
            scat._offsets3d = (spheres[:, 0], spheres[:, 1], spheres[:, 2])
            scat.set_sizes(_marker_sizes(spheres[:, 3]))
            scat.set_facecolors(colors())

        ax.set_title(f"frame_id={data.get('frame_id', 'unknown')}  spheres={spheres.shape[0]}")
        set_axes_limits()

        if selected < 0 or selected >= spheres.shape[0]:
            set_status("No sphere selected")
        else:
            x, y, z, r = spheres[selected].tolist()
            set_status(f"Selected #{selected}   x={x:.3f} y={y:.3f} z={z:.3f} r={r:.3f}")

        fig.canvas.draw_idle()

    def select_index(idx: int) -> None:
        nonlocal selected, updating_sliders
        if spheres.shape[0] == 0:
            selected = -1
            redraw()
            return

        selected = int(np.clip(idx, 0, spheres.shape[0] - 1))

        updating_sliders = True
        try:
            s_x.set_val(float(spheres[selected, 0]))
            s_y.set_val(float(spheres[selected, 1]))
            s_z.set_val(float(spheres[selected, 2]))
            s_r.set_val(float(spheres[selected, 3]))
        finally:
            updating_sliders = False

        redraw()

    def on_pick(event: Any) -> None:
        if getattr(event, "artist", None) is not scat:
            return
        inds = getattr(event, "ind", None)
        if inds is None or len(inds) == 0:
            return
        select_index(int(inds[0]))

    def on_slider_change(_: float) -> None:
        nonlocal spheres
        if updating_sliders:
            return
        if selected < 0 or selected >= spheres.shape[0]:
            return
        spheres[selected, 0] = float(s_x.val)
        spheres[selected, 1] = float(s_y.val)
        spheres[selected, 2] = float(s_z.val)
        spheres[selected, 3] = max(0.0, float(s_r.val))
        redraw()

    def on_prev(_: Any) -> None:
        if spheres.shape[0] == 0:
            return
        select_index((selected - 1) % spheres.shape[0])

    def on_next(_: Any) -> None:
        if spheres.shape[0] == 0:
            return
        select_index((selected + 1) % spheres.shape[0])

    def on_add(_: Any) -> None:
        nonlocal spheres
        if spheres.shape[0] == 0:
            spheres = np.asarray([[0.0, 0.0, 0.0, float(args.default_radius)]], dtype=float)
            select_index(0)
            return

        base = spheres[selected].copy() if 0 <= selected < spheres.shape[0] else spheres[-1].copy()
        base[3] = float(args.default_radius)
        spheres = np.vstack([spheres, base.reshape(1, 4)])
        select_index(spheres.shape[0] - 1)

    def on_delete(_: Any) -> None:
        nonlocal spheres, selected
        if spheres.shape[0] == 0 or selected < 0:
            return
        spheres = np.delete(spheres, selected, axis=0)
        if spheres.shape[0] == 0:
            selected = -1
        else:
            selected = int(np.clip(selected, 0, spheres.shape[0] - 1))
        select_index(selected)

    def on_save(_: Any) -> None:
        # Keep non-sphere fields intact.
        data["spheres"] = spheres.tolist()
        data["timestamp"] = float(time.time())
        data["edited_by"] = "visual_edit_predefined_spheres.py"
        _save_json(out_path, data)
        set_status(f"Saved {spheres.shape[0]} spheres to {out_path}")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", on_pick)
    s_x.on_changed(on_slider_change)
    s_y.on_changed(on_slider_change)
    s_z.on_changed(on_slider_change)
    s_r.on_changed(on_slider_change)

    b_prev.on_clicked(on_prev)
    b_next.on_clicked(on_next)
    b_add.on_clicked(on_add)
    b_del.on_clicked(on_delete)
    b_save.on_clicked(on_save)

    redraw()
    if spheres.shape[0] > 0:
        select_index(selected)

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
