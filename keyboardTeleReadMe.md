# Keyboard Teleoperation Guide

This guide explains how to use the Keyboard Teleoperation mode for the TIAGo robot, which serves as an alternative to the Oculus VR controller.

## How to Enable

To use keyboard control instead of VR, set the environment variable `TIAGO_TELEOP_TYPE` to `KEYBOARD` before running your client script.

```bash
# For Simulation
export TIAGO_TELEOP_TYPE=KEYBOARD
python tiago_client/run_tiago_real.py
```

## Controls

The keyboard interface runs in the terminal where you launched the script. Ensure that terminal window has focus.

### General Controls
| Key | Action |
| --- | --- |
| `Tab` | Switch between **Cartesian** and **Joint** modes (Currently only Cartesian affects arm path) |
| `Ctrl+C` | Exit |

### Mobile Base (Always Active)
| Key | Action |
| --- | --- |
| `W` / `S` | Move Forward / Backward |
| `A` / `D` | Rotate Left / Right |
| `Q` / `E` | Rotate Left / Right (Faster) |
| `Space` | Stop Base |

### Torso
| Key | Action |
| --- | --- |
| `M` | Move Torso **Up** |
| `N` | Move Torso **Down** |

### Gripper (Right Hand)
| Key | Action |
| --- | --- |
| `P` | **Open** Gripper |
| `;` | **Close** Gripper |

### Right Arm Control (Cartesian Mode)
Controls the end-effector position relative to the base.

| Axis | Increase (+) | Decrease (-) | Description |
| --- | --- | --- | --- |
| **X** (Forward/Back) | `I` | `K` | Move arm forward/backward |
| **Y** (Left/Right) | `J` | `L` | Move arm left/right |
| **Z** (Up/Down) | `U` | `O` | Move arm up/down |

*(Note: Currently, keyboard orientation control (Roll/Pitch/Yaw) is not mapped in `HybridTeleopPolicy`)*

---

## Limitations & Missing Features

The current implementation of `HybridTeleopPolicy` (`tiago_client/tiago_client/oculus_teleop/hybrid_teleop_policy.py`) has the following limitations compared to the standalone `hybrid_teleop.py`:

1.  **Joint Mode Incompatibility**:
    *   Although you can switch to `JOINT` mode with `Tab`, the `TiagoClient`'s internal logic (`get_teleop_action` method) currently expects Cartesian deltas (`[x, y, z, r, p, y]`) to perform IK.
    *   Direct joint angle inputs from the keyboard are currently **ignored** or not processed correctly by the main client loop. Only Cartesian inputs work for the arm.

2.  **No Orientation Control**:
    *   The keyboard mapping supports X, Y, Z translation but **does not** currently map keys for Roll, Pitch, or Yaw rotation of the end-effector.

3.  **Head Control**:
    *   The standalone script supports arrow keys for head control. This feature has **not** been ported to `HybridTeleopPolicy` yet. The head will remain in its default or server-controlled position.

4.  **Left Arm**:
    *   Controls are currently mapped **only for the Right Arm**. The Left Arm cannot be controlled via keyboard in this version.

## Development Status
This module is experimental. Future updates should address:
- [ ] Mapping keys for End-Effector Orientation (R/P/Y).
- [ ] Modifying `TiagoClient` to accept direct Joint commands (bypassing IK) for Joint Mode support.
- [ ] Adding Head control support.
- [ ] Adding Left Arm toggle support.
