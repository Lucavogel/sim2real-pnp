import urx
import math
import time
import logging

# --- CONFIGURATION ---
ROBOT_IP = "192.168.0.60"
JOINT_INDEX = 5          # 0=Base, 1=Shoulder, 2=Elbow, 3=Wrist1, 4=Wrist2, 5=Wrist3
DEG_CHANGE = -20.0       # Move +20 degrees relative to current
ACCELERATION = 0.5       # rad/s^2 (Safe, slow acceleration)
VELOCITY = 0.3           # rad/s   (Safe, slow speed)

def move_relative():
   # Setup logging to see what URX is doing
   logging.basicConfig(level=logging.INFO)
   
   try:
       print(f"Connecting to {ROBOT_IP}...")
       rob = urx.Robot(ROBOT_IP)
       
       # 1. Get Current Joint Positions (in Radians)
       current_joints = rob.getj()
       print(f"Current Joints (Rad): {current_joints}")
       
       # 2. Calculate the Target
       # Convert -20 degrees to radians: -20 * (pi / 180) approx -0.349 rad
       rad_change = math.radians(DEG_CHANGE)
       
       target_joints = current_joints[:] # Create a copy
       target_joints[JOINT_INDEX] += rad_change
       
       print(f"Moving Joint {JOINT_INDEX} by {DEG_CHANGE} degrees...")
       print(f"Target Joints (Rad):  {target_joints}")

       # 3. Move the robot (Blocking call - waits until finished)
       # movej moves all joints to the target position
       rob.movej(target_joints, acc=ACCELERATION, vel=VELOCITY)
       
       print("✅ Movement completed.")
       
   except Exception as e:
       print(f"❌ Error: {e}")
       
   finally:
       # Always close the connection
       if 'rob' in locals():
           rob.close()

if __name__ == "__main__":
   move_relative()