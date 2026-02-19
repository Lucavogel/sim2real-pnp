#!/usr/bin/env python3
"""
📦 Viewer dataset.npz (sans arguments)
- Tu mets le chemin du dataset directement dans DATASET_PATH
- Affiche la première trajectoire:
  - ee_ref_pos (x,y,z) si dispo
  - joints q (6)
  - plots jolis

Lance juste:
  python3 load_traj.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ===================== A MODIFIER ICI =====================
DATASET_PATH = "/home/ajin/workspace/sim2real-pnp/Real_robot/dataset.npz"
TRAJ_INDEX = 0  # 0 = première trajectoire
# ==========================================================

UR10_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

def to_dense_paths(paths: np.ndarray) -> np.ndarray:
    if isinstance(paths, np.ndarray) and paths.dtype == object:
        paths = np.stack(list(paths), axis=0)
    if paths.ndim != 3 or paths.shape[-1] != 6:
        raise ValueError(f"Format paths invalide: shape={paths.shape} (attendu N,T,6)")
    return paths.astype(np.float32, copy=False)

def to_dense_xyz(xyz: np.ndarray) -> np.ndarray:
    if isinstance(xyz, np.ndarray) and xyz.dtype == object:
        xyz = np.stack(list(xyz), axis=0)
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"Format EE invalide: shape={xyz.shape} (attendu N,T,3)")
    return xyz.astype(np.float32, copy=False)

def main():
    ds_path = Path(DATASET_PATH).expanduser()
    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {ds_path}")

    data = np.load(str(ds_path), allow_pickle=True)
    keys = list(data.keys())

    print("\n" + "=" * 70)
    print("🔍 DATASET VIEWER (NO ARGS)")
    print("=" * 70)
    print(f"📂 Fichier: {ds_path}")
    print(f"🔑 Clés ({len(keys)}): {keys}")

    dt = float(data["dt"]) if "dt" in data else None
    hz = float(data["control_hz"]) if "control_hz" in data else (1.0 / dt if dt else None)
    if dt is not None:
        print(f"⏱️  dt={dt:.4f}s  (~{hz:.1f} Hz)")
    else:
        print("⏱️  dt: (absent)")

    if "workspace" in data:
        ws = data["workspace"].astype(float).ravel()
        if ws.size >= 5:
            print(f"📐 Workspace: X[{ws[0]:.3f},{ws[1]:.3f}]  Y[{ws[2]:.3f},{ws[3]:.3f}]  Z={ws[4]:.3f}")
        else:
            print(f"📐 Workspace: {ws}")
    else:
        print("📐 Workspace: (absent)")

    if "paths" not in data:
        raise ValueError(f"❌ Clé 'paths' absente. Clés dispo: {keys}")

    paths = to_dense_paths(data["paths"])
    N, T, _ = paths.shape

    if not (0 <= TRAJ_INDEX < N):
        raise IndexError(f"TRAJ_INDEX={TRAJ_INDEX} invalide (N={N})")

    q = paths[TRAJ_INDEX]  # (T,6)

    print(f"la premiere psoition est a : {np.degrees(q[0])}")
    t = np.arange(T) * (dt if dt is not None else 1.0)

    print(f"\n🧩 paths: shape={paths.shape}, dtype={paths.dtype}")
    print(f"🎛️  Traj {TRAJ_INDEX}: T={T} points")

    q_min = q.min(axis=0)
    q_max = q.max(axis=0)
    q_std = q.std(axis=0)
    for i, name in enumerate(UR10_JOINT_NAMES):
        print(f"  - {name:18s}  min={q_min[i]: .3f}  max={q_max[i]: .3f}  std={q_std[i]: .3f}")

    ee_key = None
    if "ee_ref_pos" in data:
        ee_key = "ee_ref_pos"
    elif "ee_pos" in data:
        ee_key = "ee_pos"

    ee = None
    if ee_key is not None:
        ee_all = to_dense_xyz(data[ee_key])
        if ee_all.shape[0] == N and ee_all.shape[1] == T:
            ee = ee_all[TRAJ_INDEX]
            print(f"\n🎯 {ee_key}: shape={ee_all.shape}, dtype={ee_all.dtype}")
            print(f"   x[{ee[:,0].min():.3f},{ee[:,0].max():.3f}] "
                  f"y[{ee[:,1].min():.3f},{ee[:,1].max():.3f}] "
                  f"z[{ee[:,2].min():.3f},{ee[:,2].max():.3f}]")
        else:
            print(f"\n⚠️ {ee_key} présent mais shape inattendue: {ee_all.shape}")
    else:
        print("\nℹ️  Pas de ee_ref_pos / ee_pos dans ce dataset.")

    print("\n" + "=" * 70)

    # ------------------- PLOTS -------------------
    if ee is not None:
        plt.figure()
        plt.title(f"Traj {TRAJ_INDEX} - EE (x,y,z) [{ee_key}]")
        plt.plot(t, ee[:, 0], label="x (m)")
        plt.plot(t, ee[:, 1], label="y (m)")
        plt.plot(t, ee[:, 2], label="z (m)")
        plt.xlabel("time (s)" if dt is not None else "index")
        plt.ylabel("position")
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.legend()

        plt.figure()
        plt.title(f"Traj {TRAJ_INDEX} - EE XY path [{ee_key}]")
        plt.plot(ee[:, 0], ee[:, 1])
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.axis("equal")
        plt.grid(True, linestyle="--", linewidth=0.5)

    plt.figure()
    plt.title(f"Traj {TRAJ_INDEX} - Joints q (rad)")
    for j in range(6):
        plt.plot(t, q[:, j], label=UR10_JOINT_NAMES[j])
    plt.xlabel("time (s)" if dt is not None else "index")
    plt.ylabel("q (rad)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend(loc="best")

    plt.show()

if __name__ == "__main__":
    main()
