"""Test calibration accuracy with known distance measurement"""
import cv2
import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--calib', default='camera_calibration.pkl')
parser.add_argument('--marker-size-mm', type=float, default=72.0, help='Real marker size in mm')
parser.add_argument('--real-distance-mm', type=float, help='Real measured distance camera->marker in mm')
args = parser.parse_args()

# Load calibration
with open(args.calib, 'rb') as f:
    calib = pickle.load(f)
K = calib['camera_matrix']
D = calib['dist_coeffs']

# ArUco setup
aruco = cv2.aruco
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# Open webcam
cap = cv2.VideoCapture(2)
marker_length = args.marker_size_mm / 1000.0  # Convert to meters

print("=" * 60)
print("CALIBRATION METRIC TEST")
print("=" * 60)
print(f"Marker size: {args.marker_size_mm} mm")
if args.real_distance_mm:
    print(f"Expected distance: {args.real_distance_mm} mm")
print("\nPlace marker at known distance and press 's' to measure")
print("Press 'q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    
    if ids is not None and len(ids) > 0:
        # Estimate pose
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, K, D
        )
        
        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            marker_id = ids[i][0]
            distance_m = np.linalg.norm(tvec[0])
            distance_mm = distance_m * 1000
            
            # Draw
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.drawFrameAxes(frame, K, D, rvec, tvec, marker_length * 0.5)
            
            # Info
            text = f"ID {marker_id}: {distance_mm:.1f} mm"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            
            if args.real_distance_mm:
                error_mm = abs(distance_mm - args.real_distance_mm)
                error_pct = (error_mm / args.real_distance_mm) * 100
                text2 = f"Error: {error_mm:.1f} mm ({error_pct:.1f}%)"
                color = (0, 255, 0) if error_pct < 5 else (0, 165, 255)
                cv2.putText(frame, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, color, 2)
    
    cv2.imshow('Calibration Metric Test', frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('s') and ids is not None:
        print(f"\n📏 Measurement snapshot:")
        for i, tvec in enumerate(tvecs):
            marker_id = ids[i][0]
            distance_mm = np.linalg.norm(tvec[0]) * 1000
            print(f"  Marker {marker_id}: {distance_mm:.2f} mm")
            if args.real_distance_mm:
                error_mm = abs(distance_mm - args.real_distance_mm)
                error_pct = (error_mm / args.real_distance_mm) * 100
                print(f"  Error: {error_mm:.2f} mm ({error_pct:.2f}%)")

cap.release()
cv2.destroyAllWindows()