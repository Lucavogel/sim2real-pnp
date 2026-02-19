import time
import numpy as np
import socket
from urx.urrtmon import URRTMonitor

# --- CONFIGURATION ---
ROBOT_IP = "192.168.0.60"
PORT = 30002          
TEST_DURATION = 10.0  # Run for 10 seconds

def main():
   print(f"Connecting to {ROBOT_IP} for FULL CONTROL LOOP test...")
   
   # 1. Setup Reader
   rt = URRTMonitor(ROBOT_IP)
   rt.start()
   rt.wait()
   
   # 2. Setup Writer (Raw Socket)
   sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
   sock.settimeout(1.0)
   sock.connect((ROBOT_IP, PORT))
   
   print("Socket connected. Measuring Round-Trip Frequency...")
   
   # Get initial position to hold steady (Safety)
   init_data = rt.get_all_data(wait=True)
   q_hold = list(init_data['qActual'])
   q_str = str(q_hold)
   
   # Pre-format the command string to minimize Python overhead during loop
   # We send a "Stay Here" command
   cmd_bytes = f"servoj({q_str}, 0, 0, 0.08)\n".encode('utf-8')
   
   timestamps = []
   start_time = time.time()
   loop_count = 0
   
   while time.time() - start_time < TEST_DURATION:
       # --- STEP 1: READ (Wait for Robot) ---
       rt.get_all_data(wait=True)
       
       # --- STEP 2: WRITE (Send Action) ---
       sock.sendall(cmd_bytes)
       
       # Record time
       timestamps.append(time.time())
       loop_count += 1
       
   # --- ANALYSIS ---
   deltas = np.diff(timestamps)
   avg_dt = np.mean(deltas)
   std_dt = np.std(deltas)
   freq = 1.0 / avg_dt
   
   print("\n" + "="*40)
   print(f"RESULTS: FULL CONTROL LOOP FREQUENCY")
   print("="*40)
   print(f"Loop Type:        Read + Write")
   print(f"Total Cycles:     {loop_count}")
   print(f"Working Freq:     {freq:.2f} Hz")
   print(f"Avg Interval:     {avg_dt*1000:.2f} ms")
   print(f"Jitter (StdDev):  {std_dt*1000:.3f} ms")
   print("="*40)
   
   # --- INTERPRETATION ---
   if 120 < freq < 130:
       print("🚀 ELITE PERFORMANCE: 125 Hz Loop.")
       print("Set Isaac Lab Physics dt=0.008")
   elif 60 < freq < 65:
       print("✅ STANDARD PERFORMANCE: 62.5 Hz Loop.")
       print("Set Isaac Lab Physics dt=0.016 (Decimation 2)")
   elif 40 < freq < 45:
       print("⚠️ SLOW: 41 Hz Loop.")
       print("Set Isaac Lab Physics dt=0.024")
   else:
       print("❓ UNSTABLE: Frequency is erratic.")

   # Cleanup
   sock.sendall(b"stopj(1.0)\n")
   sock.close()
   rt.stop()

if __name__ == "__main__":
   main()