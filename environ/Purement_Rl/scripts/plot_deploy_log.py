from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY"))


def _parse_args():
    p = argparse.ArgumentParser(description="Plot deployment log saved by isaaclab_deploy_policy_npz.py")
    p.add_argument("--npz", type=str, required=True, help="Path to deploy_log_*.npz")
    p.add_argument("--env", type=int, default=0, help="Env index inside the log (default: 0)")
    p.add_argument("--out", type=str, default=None, help="Output image path (.png). Default: <npz>.png")
    p.add_argument("--show", action="store_true", default=False, help="Show interactively (requires DISPLAY)")
    p.add_argument(
        "--axis-order",
        type=str,
        default="xyz",
        choices=["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"],
        help=(
            "Permute EE axes for plotting (default: xyz). "
            "Applies to both desired and true unless overridden by --axis-order-des/--axis-order-true."
        ),
    )
    p.add_argument(
        "--axis-order-des",
        type=str,
        default=None,
        choices=["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"],
        help="Axis permutation for desired EE only (ee_des_*). Overrides --axis-order.",
    )
    p.add_argument(
        "--axis-order-true",
        type=str,
        default=None,
        choices=["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"],
        help="Axis permutation for true/measured EE only (ee_true/ee_meas). Overrides --axis-order.",
    )
    p.add_argument(
        "--axis-flip",
        type=str,
        default="none",
        help=(
            "Flip sign of selected axes for plotting. "
            "Use a combination of letters from {x,y,z} (e.g., 'x', 'yz', 'xyz') or 'none'. "
            "Applies to both desired and true unless overridden by --axis-flip-des/--axis-flip-true."
        ),
    )
    p.add_argument(
        "--axis-flip-des",
        type=str,
        default=None,
        help="Axis sign flips for desired EE only (ee_des_*). Overrides --axis-flip.",
    )
    p.add_argument(
        "--axis-flip-true",
        type=str,
        default=None,
        help="Axis sign flips for true/measured EE only (ee_true/ee_meas). Overrides --axis-flip.",
    )
    return p.parse_args()


def _normalize_flip_spec(s: str | None) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    if s in ("", "none", "0", "false", "no"):
        return ""
    # allow separators like "," or whitespace
    s = s.replace(",", "").replace(" ", "")
    for ch in s:
        if ch not in ("x", "y", "z"):
            raise ValueError(f"Invalid axis flip spec: {s!r}. Expected combination of x/y/z or 'none'.")
    # de-duplicate while keeping order x,y,z for determinism
    out = "".join([c for c in "xyz" if c in set(s)])
    return out


def _apply_axis_flips(arr: np.ndarray, labels: list[str], flip_spec: str) -> np.ndarray:
    if arr is None:
        return arr
    if flip_spec == "":
        return arr
    flip_set = set(flip_spec)
    out = arr.copy()
    for i, lab in enumerate(labels):
        if lab in flip_set:
            out[:, i] *= -1.0
    return out


def main():
    args = _parse_args()

    # Non-interactive backend when headless
    if not args.show or not _has_display():
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    path = Path(args.npz).expanduser().resolve()
    data = np.load(str(path), allow_pickle=True)

    # Required-ish keys
    t = data["t"]

    env_ids = data.get("env_ids", None)
    if env_ids is not None:
        env_ids = env_ids.tolist()
    else:
        # Fall back to [0..N-1]
        env_ids = list(range(int(data["ee_true_local"].shape[1])))

    if args.env not in env_ids:
        raise ValueError(f"env={args.env} not present in log env_ids={env_ids}")

    env_idx = env_ids.index(args.env)

    ee_true_local = data["ee_true_local"][:, env_idx, :]
    ee_des_clean_local = data["ee_des_clean_local"][:, env_idx, :]
    ee_des_obs_local = data["ee_des_obs_local"][:, env_idx, :]

    # Some datasets / viewers use different axis conventions.
    # Allow permuting axes at plot time without touching the logger.
    axis_map = {"x": 0, "y": 1, "z": 2}
    order_des = args.axis_order_des if args.axis_order_des is not None else args.axis_order
    order_true = args.axis_order_true if args.axis_order_true is not None else args.axis_order
    perm_des = [axis_map[c] for c in order_des]
    perm_true = [axis_map[c] for c in order_true]

    flip_des = _normalize_flip_spec(args.axis_flip_des if args.axis_flip_des is not None else args.axis_flip)
    flip_true = _normalize_flip_spec(args.axis_flip_true if args.axis_flip_true is not None else args.axis_flip)
    labels_des = list(order_des)

    ee_true_local = ee_true_local[:, perm_true]
    ee_des_clean_local = ee_des_clean_local[:, perm_des]
    ee_des_obs_local = ee_des_obs_local[:, perm_des]

    # Apply sign flips in the *plotted* axis order (labels_des).
    # This is useful when ee_true appears mirrored vs ee_des.
    ee_des_clean_local = _apply_axis_flips(ee_des_clean_local, labels_des, flip_des)
    ee_des_obs_local = _apply_axis_flips(ee_des_obs_local, labels_des, flip_des)
    ee_true_local = _apply_axis_flips(ee_true_local, labels_des, flip_true)
    e_true_clean_norm = data.get("e_true_clean_norm", None)
    e_true_obs_norm = data.get("e_true_obs_norm", None)

    if e_true_clean_norm is not None:
        e_true_clean_norm = e_true_clean_norm[:, env_idx]
    if e_true_obs_norm is not None:
        e_true_obs_norm = e_true_obs_norm[:, env_idx]

    ee_speed = data.get("ee_speed", None)
    if ee_speed is not None:
        ee_speed = ee_speed[:, env_idx]

    # Joint plots (optional keys)
    q_meas = data.get("q_meas", None)
    q_ref = data.get("q_ref", None)
    q_target = data.get("q_target", None)
    qd_meas = data.get("qd_meas", None)
    qd_target = data.get("qd_target", None)

    q_des = q_target if q_target is not None else q_ref
    q_des_label = "q_target" if q_target is not None else ("q_ref" if q_ref is not None else "q_des")

    # Determine number of joints to plot
    dof = None
    if q_meas is not None and getattr(q_meas, "ndim", 0) == 3:
        dof = int(q_meas.shape[2])
    elif q_des is not None and getattr(q_des, "ndim", 0) == 3:
        dof = int(q_des.shape[2])
    
    dof_to_plot = min(dof if dof is not None else 0, 6)  # Limit to 6 joints (UR10)

    # Create single figure with all subplots: EE (3) + error + speed + joints (6) = 11 subplots
    nrows = 2 + 3 + dof_to_plot  # error + speed + x/y/z + joints
    if nrows == 0:
        raise RuntimeError("No plottable signals found in the NPZ.")

    fig, axs = plt.subplots(nrows, 1, figsize=(14, max(10, 2.5 * nrows)), sharex=True)
    if nrows == 1:
        axs = [axs]

    r = 0
    
    # Plot 1: Error norm
    ax = axs[r]
    if e_true_clean_norm is not None:
        ax.plot(t, e_true_clean_norm, label="||e_true_clean||", linewidth=2)
    if e_true_obs_norm is not None:
        ax.plot(t, e_true_obs_norm, label="||e_true_obs||", alpha=0.7, linewidth=2)
    ax.set_ylabel("Error (m)", fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.set_title("Position Error Norm", fontsize=10, loc='left')
    r += 1

    # Plot 2: EE Speed
    ax = axs[r]
    if ee_speed is not None:
        ax.plot(t, ee_speed, label="ee_speed", linewidth=2, color='green')
        ax.set_ylabel("Speed (m/s)", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        ax.set_title("End-Effector Speed", fontsize=10, loc='left')
    else:
        ax.text(0.5, 0.5, "ee_speed not logged", transform=ax.transAxes, ha='center', va='center')
        ax.axis("off")
    r += 1

    # Plots 3-5: EE position (x, y, z)
    labels = list(order_des)
    colors = ['tab:blue', 'tab:orange', 'tab:red']
    for i in range(3):
        ax = axs[r]
        ax.plot(t, ee_des_clean_local[:, i], label=f"desired clean", linewidth=1.5, linestyle='--', alpha=0.8)
        ax.plot(t, ee_des_obs_local[:, i], label=f"desired obs", linewidth=1.5, linestyle=':', alpha=0.7)
        ax.plot(t, ee_true_local[:, i], label=f"measured", linewidth=2, color=colors[i])
        ax.set_ylabel(f"EE {labels[i].upper()} (m)", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        ax.set_title(f"End-Effector Position: {labels[i].upper()}-axis", fontsize=10, loc='left')
        r += 1

    # Plots 6-11: Joint positions (q0 to q5)
    if dof_to_plot > 0:
        Qm = q_meas[:, env_idx, :] if q_meas is not None and q_meas.ndim == 3 else None
        Qd = q_des[:, env_idx, :] if q_des is not None and q_des.ndim == 3 else None
        
        joint_colors = ['tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        for j in range(dof_to_plot):
            ax = axs[r]
            if Qd is not None:
                ax.plot(t, Qd[:, j], label=q_des_label, alpha=0.75, linewidth=1.5, linestyle='--')
            if Qm is not None:
                ax.plot(t, Qm[:, j], label="measured", linewidth=2, color=joint_colors[j % len(joint_colors)])
            ax.set_ylabel(f"q{j} (rad)", fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=9)
            ax.set_title(f"Joint {j} Position", fontsize=10, loc='left')
            r += 1

    # Set xlabel on the last subplot only
    axs[-1].set_xlabel("Time (s)", fontsize=12, fontweight='bold')
    
    # Overall title
    title = f"Deployment Tracking Results: {path.name} (env={args.env})"
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    # Save figure
    out = Path(args.out).expanduser() if args.out else path.with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=200, bbox_inches='tight')
    print(f"[INFO] Saved complete subplot figure: {out}")

    if args.show and _has_display():
        plt.show()
    elif args.show:
        print("[WARN] --show requested but DISPLAY is not set; saved image only.")


if __name__ == "__main__":
    main()
