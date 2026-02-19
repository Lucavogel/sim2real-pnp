import cv2
import time
import numpy as np
from pupil_apriltags import Detector

def benchmark_vision_tag16h5():
   # 1. Setup Camera
   cap = cv2.VideoCapture(2) # Change index to 1 or 2 if using external cam
   
   # 640x480 is recommended for "Weak PC" speed
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
   
   # 2. Setup Detector (SWITCHED TO TAG16H5)
   # nthreads=1 is usually best for Python to avoid overhead on weak CPUs
   at_detector = Detector(families='tag36h11', nthreads=1, quad_decimate=1.0)
   
   # --- FILTERS (Essential for Tag16h5) ---
   # Tag16h5 is noisy. We must ignore low-confidence detections.
   # If you still see random green dots, INCREASE MIN_MARGIN to 40.0
   MIN_MARGIN = 30.0   
   MIN_PIXELS = 15.0  

   print("⚡ Starting Vision Benchmark (Tag16h5)...")
   print(f"   Filtering: Margin > {MIN_MARGIN}, Width > {MIN_PIXELS}px")
   print("   Press 'q' to quit.")
   
   frame_count = 0
   times = []
   
   try:
       while True:
           t0 = time.time()
           
           # A. Capture
           ret, frame = cap.read()
           if not ret:
               print("Failed to grab frame")
               break
           
           # B. Convert to Grayscale
           gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
           
           # C. Detect
           tags = at_detector.detect(gray)
           
           dt = (time.time() - t0) * 1000 
           times.append(dt)
           frame_count += 1
           
           valid_tags_found = 0
           
           if len(tags) > 0:
               for tag in tags:
                   # --- FILTERS ---
                   # 1. Check Confidence (Decision Margin)
                   if tag.decision_margin < MIN_MARGIN:
                       continue # Skip noise/ghosts
                   
                   # 2. Check Size (Width in pixels)
                   width_px = np.linalg.norm(tag.corners[1] - tag.corners[0])
                   if width_px < MIN_PIXELS:
                       continue # Skip tiny specs

                   # --- REAL TAG FOUND ---
                   valid_tags_found += 1
                   
                   # Fix for OpenCV Crash: Convert to float32
                   corners_f32 = tag.corners.astype(np.float32)
                   area = cv2.contourArea(corners_f32)
                   
                   print(f"✅ [Frame {frame_count}] TAG FOUND: ID {tag.tag_id} | "
                         f"Time: {dt:.1f}ms | "
                         f"Width: {width_px:.1f}px | "
                         f"Margin: {tag.decision_margin:.0f}")
                   
                   # Draw Green Box (Valid)
                   corners_int = tag.corners.astype(np.int32)
                   cv2.polylines(frame, [corners_int], isClosed=True, color=(0, 255, 0), thickness=2)
                   
                   # Draw Center Dot & ID
                   cx, cy = int(tag.center[0]), int(tag.center[1])
                   cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                   cv2.putText(frame, f"ID:{tag.tag_id}", 
                               (corners_int[0][0], corners_int[0][1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
           
           # Show the image
           cv2.imshow("Tag16h5 Debugger", frame)
           
           if cv2.waitKey(1) & 0xFF == ord('q'):
               break

   except KeyboardInterrupt:
       pass
       
   finally:
       if len(times) > 0:
           avg_time = np.mean(times)
           fps = 1000.0 / avg_time
           
           print("\n" + "="*40)
           print("BENCHMARK SUMMARY (Tag16h5)")
           print("="*40)
           print(f"Avg Process Time: {avg_time:.1f} ms")
           print(f"Real-World FPS:   {fps:.1f} Hz")
           print("="*40)
           
           if avg_time > 35.0:
                print("⚠️  WARNING: Too slow for 25Hz. Use 12.5 Hz.")
           else:
                print("🚀 SUCCESS: Fast enough for 25Hz.")

       cap.release()
       cv2.destroyAllWindows()

if __name__ == "__main__":
   benchmark_vision_tag16h5()
