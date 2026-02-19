#!/usr/bin/env python3
import argparse
import pickle
import numpy as np
from pathlib import Path


def load_calib(path: Path):
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            data = pickle.load(f)
        K = np.array(data["camera_matrix"], dtype=float)
        D = np.array(data["dist_coeffs"], dtype=float).reshape(-1)
        return K, D
    elif path.suffix == ".npz":
        data = np.load(path)
        K = np.array(data["camera_matrix"], dtype=float)
        D = np.array(data["dist_coeffs"], dtype=float).reshape(-1)
        return K, D
    else:
        raise ValueError("Input must be .pkl or .npz")


def scale_K(K, w_old, h_old, w_new, h_new):
    sx = w_new / w_old
    sy = h_new / h_old
    K2 = K.copy()
    K2[0, 0] *= sx  # fx
    K2[1, 1] *= sy  # fy
    K2[0, 2] *= sx  # cx
    K2[1, 2] *= sy  # cy
    return K2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="camera_calibration.pkl or calibration_data.npz")
    ap.add_argument("--out", dest="out", required=True, help="output YAML path")
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)

    # Si la calibration a été faite dans une autre résolution, indique-la ici
    ap.add_argument("--calib_width", type=int, default=None)
    ap.add_argument("--calib_height", type=int, default=None)

    # modèle de distortion ROS
    ap.add_argument("--model", choices=["rational_polynomial", "plumb_bob"], default="rational_polynomial")

    args = ap.parse_args()
    inp = Path(args.inp)
    out = Path(args.out)

    K, D = load_calib(inp)

    # Scale K si besoin
    if args.calib_width and args.calib_height:
        K = scale_K(K, args.calib_width, args.calib_height, args.width, args.height)

    # Choix des coeffs D selon le modèle
    if args.model == "plumb_bob":
        # k1 k2 p1 p2 k3
        if D.size < 5:
            raise ValueError(f"D has {D.size} coeffs, need >= 5 for plumb_bob")
        D_use = D[:5]
        model = "plumb_bob"
    else:
        # rational: k1 k2 p1 p2 k3 k4 k5 k6
        if D.size < 8:
            raise ValueError(f"D has {D.size} coeffs, need >= 8 for rational_polynomial")
        D_use = D[:8]
        model = "rational_polynomial"

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    yaml = f"""image_width: {args.width}
image_height: {args.height}
camera_name: c930e
camera_matrix:
  rows: 3
  cols: 3
  data: [{fx:.12f}, 0.0, {cx:.12f},
         0.0, {fy:.12f}, {cy:.12f},
         0.0, 0.0, 1.0]
distortion_model: {model}
distortion_coefficients:
  rows: 1
  cols: {D_use.size}
  data: [{", ".join([f"{x:.12f}" for x in D_use])}]
rectification_matrix:
  rows: 3
  cols: 3
  data: [1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, 0.0, 1.0]
projection_matrix:
  rows: 3
  cols: 4
  data: [{fx:.12f}, 0.0, {cx:.12f}, 0.0,
         0.0, {fy:.12f}, {cy:.12f}, 0.0,
         0.0, 0.0, 1.0, 0.0]
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml)
    print(f"✅ Wrote ROS camera_info YAML to: {out}")
    print(f"   model={model}, D_used={D_use.size} coeffs (from original {D.size})")
    print(f"   fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")


if __name__ == "__main__":
    main()
