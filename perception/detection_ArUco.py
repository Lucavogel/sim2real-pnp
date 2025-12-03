"""
Standalone ArUco/AprilTag (TAG36H11) detector (non-ROS).
- Detecte tags DICT_APRILTAG_36h11 (Tag36h11)
- Supporte tailles par défaut et mapping par ID
- Utilise camera_calibration.pkl si présent (keys: camera_matrix, dist_coeffs)
- CLI: webcam, video file ou image file. Affiche debug window et imprime poses.
Usage examples:
  python detection_ArUco.py --webcam 0 --default-size-mm 72
  python detection_ArUco.py --image test.jpg --sizes-map "5:80,6:100" --calib camera_calibration.pkl
"""
import argparse
import pickle
import time
import os

import cv2
import numpy as np

# Try modern/openCV API then fallback
try:
    ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
    ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()
    detect_func = cv2.aruco.detectMarkers
    estimate_pose = cv2.aruco.estimatePoseSingleMarkers
    draw_axes = cv2.drawFrameAxes
except Exception:
    ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    ARUCO_PARAMS = cv2.aruco.DetectorParameters()
    detect_func = cv2.aruco.detectMarkers
    estimate_pose = cv2.aruco.estimatePoseSingleMarkers
    draw_axes = None  # optional

def rotmat_to_quat(R):
    """Convert rotation matrix to quaternion (x,y,z,w)."""
    q = np.empty((4,))
    trace = np.trace(R)
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q[3] = 0.25 / s
        q[0] = (R[2,1] - R[1,2]) * s
        q[1] = (R[0,2] - R[2,0]) * s
        q[2] = (R[1,0] - R[0,1]) * s
    else:
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            q[3] = (R[2,1] - R[1,2]) / s
            q[0] = 0.25 * s
            q[1] = (R[0,1] + R[1,0]) / s
            q[2] = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            q[3] = (R[0,2] - R[2,0]) / s
            q[0] = (R[0,1] + R[1,0]) / s
            q[1] = 0.25 * s
            q[2] = (R[1,2] + R[2,1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            q[3] = (R[1,0] - R[0,1]) / s
            q[0] = (R[0,2] + R[2,0]) / s
            q[1] = (R[1,2] + R[2,1]) / s
            q[2] = 0.25 * s
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])

def load_calibration(calib_file):
    """Load camera_matrix and dist_coeffs from pickle file. Return (K, D) or (None, None)."""
    if not calib_file or not os.path.isfile(calib_file):
        return None, None
    try:
        with open(calib_file, 'rb') as f:
            data = pickle.load(f)
        K = data.get('camera_matrix') or data.get('camera_intrinsics') or data.get('K')
        D = data.get('dist_coeffs') or data.get('distortion') or data.get('D')
        if K is None:
            return None, None
        return np.array(K, dtype=float), np.array(D, dtype=float)
    except Exception as e:
        print("Warning: failed to load calibration:", e)
        return None, None

def parse_sizes_map(sizes_map_str):
    """Parse string like '1:80,2:100' -> dict{id: size_mm}"""
    sizes = {}
    if not sizes_map_str:
        return sizes
    for pair in sizes_map_str.split(','):
        if ':' in pair:
            k, v = pair.split(':')
            try:
                sizes[int(k.strip())] = float(v.strip())
            except ValueError:
                pass
    return sizes

def estimate_and_draw(frame, corners_list, ids, K, D, sizes_mm, default_size_mm):
    """Estimate pose for each detected marker and annotate frame. Returns list of detections."""
    detections = []
    ids_flat = ids.flatten() if ids is not None else []
    for idx, corners in zip(ids_flat, corners_list):
        size_mm = sizes_mm.get(int(idx), default_size_mm)
        marker_length = float(size_mm) / 1000.0
        # estimatePoseSingleMarkers wants corners shape (N,4,2) -> pass np.array([corners])
        try:
            rvecs, tvecs, _ = estimate_pose(np.array([corners]), marker_length, K, D)
        except Exception:
            # fallback if API differs
            rvecs, tvecs = estimate_pose(np.array([corners]), marker_length, K, D)[:2]
        if rvecs is None:
            continue
        rvec = rvecs[0][0]
        tvec = tvecs[0][0]
        R, _ = cv2.Rodrigues(rvec)
        qx, qy, qz, qw = rotmat_to_quat(R)
        detections.append({
            'id': int(idx),
            'tvec': tvec.tolist(),
            'rvec': rvec.tolist(),
            'quat': (qx, qy, qz, qw),
            'size_m': marker_length
        })
        # draw marker outline and id
        corners_np = np.array(corners).reshape(-1,2).astype(int)
        cv2.polylines(frame, [corners_np], True, (0,255,0), 2)
        center = corners_np.mean(axis=0).astype(int)
        cv2.putText(frame, str(int(idx)), tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        # draw axis if possible
        if draw_axes is not None and K is not None:
            try:
                draw_axes(frame, K, D, rvec, tvec, marker_length * 0.5)
            except Exception:
                pass
    return detections

def create_default_camera_matrix(width, height, fx=None):
    """Create approximate intrinsics if calibration not provided."""
    if fx is None:
        fx = 525.0
    K = np.array([[fx, 0.0, width/2.0],
                  [0.0, fx, height/2.0],
                  [0.0, 0.0, 1.0]], dtype=float)
    D = np.zeros((5,), dtype=float)
    return K, D

def main():
    p = argparse.ArgumentParser(prog="detection_ArUco.py", description="Standalone ArUco/AprilTag detector")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--webcam', type=int, help='webcam device id (e.g. 0)')
    group.add_argument('--video', type=str, help='video file path')
    group.add_argument('--image', type=str, help='single image file to process and exit')
    p.add_argument('--calib', type=str, default="camera_calibration.pkl", help='pickle calibration file (optional)')
    p.add_argument('--default-size-mm', type=float, default=72.0, help='default marker size in mm (default 72)')
    p.add_argument('--sizes-map', type=str, default="", help='map marker id to size, e.g. "5:80,6:100"')
    p.add_argument('--show', action='store_true', help='show debug window')
    p.add_argument('--save-debug', type=str, default="", help='save debug image to this path for image mode')
    args = p.parse_args()

    sizes_map = parse_sizes_map(args.sizes_map)
    K, D = load_calibration(args.calib)
    cap = None
    single_image = None

    if args.image:
        if not os.path.isfile(args.image):
            print("Image not found:", args.image)
            return
        single_image = cv2.imread(args.image)
        if single_image is None:
            print("Failed to read image:", args.image)
            return
        h, w = single_image.shape[:2]
        if K is None:
            K, D = create_default_camera_matrix(w, h)
    else:
        if args.webcam is not None:
            cap = cv2.VideoCapture(args.webcam)
        else:
            cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print("Failed to open camera/video")
            return
        # read one frame to get size
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from source")
            return
        h, w = frame.shape[:2]
        if K is None:
            K, D = create_default_camera_matrix(w, h)

    print("Using camera intrinsics K:\n", K)
    print("Dist coeffs:\n", D)
    print("Default marker size (mm):", args.default_size_mm)
    if sizes_map:
        print("Sizes map:", sizes_map)

    def process_frame(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detect_func(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)
        detections = []
        if ids is not None:
            detections = estimate_and_draw(frame, corners, ids, K, D, sizes_map, args.default_size_mm)
        timestamp = time.time()
        # print simple summary
        if detections:
            for det in detections:
                print(f"[{timestamp:.3f}] id={det['id']} t={np.array(det['tvec'])} quat={det['quat']} size_m={det['size_m']:.3f}")
        else:
            print(f"[{timestamp:.3f}] no markers")
        return frame, detections

    if single_image is not None:
        out_frame, dets = process_frame(single_image)
        if args.show:
            cv2.imshow("aruco_debug", out_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if args.save_debug:
            cv2.imwrite(args.save_debug, out_frame)
        return

    # video/webcam loop
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out_frame, dets = process_frame(frame)
            if args.show:
                cv2.imshow("aruco_debug", out_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        if cap is not None:
            cap.release()
        if args.show:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()