import urx
import time
import logging
import numpy as np
# --- CONFIGURATION ---
ROBOT_IP = "192.168.0.60"

# Target Joint Positions (Radians)
# Format: [Base, Shoulder, Elbow, Wrist1, Wrist2, Wrist3]
# Example below is a common "Upright / Home" position
TARGET_JOINTS = [np.radians(-1.3466358), np.radians(-48.786118), np.radians(85.919304), np.radians(-127.13319), np.radians(-89.999985), np.radians(-91.346634)]

ACCELERATION = 0.3       # rad/s^2 (Keep low for safety)
VELOCITY = 0.3           # rad/s   (Keep low for safety)

def move_to_absolute_position():
   logging.basicConfig(level=logging.INFO)
   
   rob = None
   try:
       print(f"Connecting to {ROBOT_IP}...")
       rob = urx.Robot(ROBOT_IP)
       
       # 1. Print Current Position (for verification)
       current_joints = rob.getj()
       print(f"Current Joints: {current_joints}")
       print("-" * 40)
       print(f"Moving to Target: {TARGET_JOINTS}")
       print(f"Expected Duration: ~{(max(abs(x - y) for x, y in zip(current_joints, TARGET_JOINTS)) / VELOCITY):.1f} seconds")
       print("-" * 40)

       # 2. Execute the Move
       # wait=True means the code will pause here until the robot finishes moving
       rob.movej(TARGET_JOINTS, acc=ACCELERATION, vel=VELOCITY, wait=True)
       
       print("✅ Arrived at target position.")
       
       # Verify final position
       final_joints = rob.getj()
       print(f"Final Joints:   {final_joints}")

   except Exception as e:
       print(f"❌ Error: {e}")
       
   finally:
       if rob:
           rob.close()
           print("Connection closed.")

if __name__ == "__main__":
   move_to_absolute_position()