import numpy as np
import time
import socket
import logging
import sys
from urx.urrtmon import URRTMonitor

# --- CONFIGURATION ---
ROBOT_IP = "192.168.0.60"
PORT = 30002
DATASET_FILE = "/home/ajin/workspace/sim2real-pnp/Real_robot/dataset.npz"

# Servo Parameters
# SERVO_T: Lookahead time. 
# 0.1 is very smooth/safe. 0.04 is sharper tracking.
SERVO_T = 0.08 
ROBOT_FREQ = 125.0
ROBOT_DT = 1.0 / ROBOT_FREQ

def load_and_process_trajectory(traj_index=0):
   """
   Loads the npz, picks a trajectory, and upsamples it from 25Hz to 125Hz.
   """
   print(f"📂 Loading {DATASET_FILE}...")
   try:
       data = np.load(DATASET_FILE)
       paths = data["paths"]       # Shape (N, 64, 6)
       file_hz = data["control_hz"] # Should be 25
       file_dt = data["dt"]         # Should be 0.04
       
       print(f"   Data Shape: {paths.shape}")
       print(f"   Recorded Freq: {file_hz} Hz (dt={file_dt:.3f}s)")
       
   except Exception as e:
       print(f"❌ Error loading file: {e}")
       sys.exit(1)

   # 1. Select Trajectory
   if traj_index >= len(paths):
       print(f"❌ Index {traj_index} out of bounds (Max {len(paths)-1})")
       sys.exit(1)
       
   raw_path = paths[traj_index] # Shape (64, 6)
   
   # 2. Upsample to 125 Hz (Interpolation)
   # We need to turn 64 points (25Hz) into ~320 points (125Hz)
   num_points = raw_path.shape[0]
   duration = num_points * file_dt
   
   # Create time grids
   t_original = np.linspace(0, duration, num_points)
   
   # Target grid: 125Hz steps
   target_steps = int(duration * ROBOT_FREQ)
   t_target = np.linspace(0, duration, target_steps)
   
   print(f"🔄 Upsampling Trajectory {traj_index}...")
   print(f"   Original: {num_points} steps ({file_hz} Hz)")
   print(f"   Target:   {target_steps} steps ({ROBOT_FREQ} Hz)")
   
   # Interpolate each joint
   smooth_path = np.zeros((target_steps, 6))
   for j in range(6):
       # Linear interpolation is safe and efficient
       smooth_path[:, j] = np.interp(t_target, t_original, raw_path[:, j])
       
   return smooth_path

def main():
   # 1. Prepare Data
   # Change index to execute different paths
   traj_index = 0 
   path_125hz = load_and_process_trajectory(traj_index)
   
   start_q = path_125hz[0]
   end_q = path_125hz[-1]

   # 2. Connect to Robot
   print(f"\n🔌 Connecting to {ROBOT_IP}...")
   rt = URRTMonitor(ROBOT_IP)
   rt.start()
   rt.wait() # Wait for first packet
   
   sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
   sock.settimeout(2.0)
   sock.connect((ROBOT_IP, PORT))
   print("✅ Connected.")

   try:
       # 3. Move to Start Position (Safety First)
       print("\n🚀 Moving to START position (Slow movej)...")
       # Check distance to avoid huge jumps
       current_q = rt.get_all_data(wait=True)['qActual']
       dist = np.max(np.abs(np.array(current_q) - np.array(start_q)))
       print(f"   Distance to start: {dist:.3f} rad")
       
       # Send movej command string
       # movej(q, a=0.5, v=0.5)
       start_str = str(list(start_q))
       cmd = f"movej({start_str}, 0.5, 0.5)\n"
       sock.sendall(cmd.encode('utf-8'))
       
       # Wait until we are there (Simple blocking check)
       time.sleep(dist / 0.3 + 1.0) # Rough estimate duration
       
       # Verify arrival
       current_q = rt.get_all_data(wait=True)['qActual']
       if np.max(np.abs(np.array(current_q) - np.array(start_q))) > 0.05:
           print("⚠️ Robot didn't reach start perfectly. Check controller.")
       
       # 4. Execute Trajectory (Streaming)
       print("\n▶️  Executing Trajectory (125 Hz Streaming)...")
       print("   HOLD E-STOP.")
       
       # Pre-format strings to save CPU during loop
       cmds = []
       for q in path_125hz:
           # servoj(q, a=0, v=0, t=SERVO_T, lookahead_time=0.1, gain=300)
           # t=0.08 allows 0.08s to reach target (smooths jitter)
           # lookahead_time and gain are standard UR tuning
           cmd = f"servoj({list(q)}, 0, 0, {SERVO_T})\n"
           cmds.append(cmd.encode('utf-8'))
           
       # The Real-Time Loop
       start_time = time.time()
       for i, cmd_bytes in enumerate(cmds):
           # A. Wait for robot heartbeat (Sync)
           rt.get_all_data(wait=True)
           
           # B. Send Command
           sock.sendall(cmd_bytes)
           
           # Progress bar
           if i % 125 == 0:
               print(f"   Step {i}/{len(cmds)}")

       total_time = time.time() - start_time
       print(f"\n✅ Finished in {total_time:.2f}s (Expected: {len(cmds)*ROBOT_DT:.2f}s)")

   except KeyboardInterrupt:
       print("\n🛑 STOPPING (Ctrl+C)")
       sock.sendall(b"stopj(2.0)\n")
       
   except Exception as e:
       print(f"❌ Error: {e}")
       sock.sendall(b"stopj(2.0)\n")

   finally:
       print("Closing connections.")
       sock.close()
       rt.stop()

if __name__ == "__main__":
   main()