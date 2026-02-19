
import time
import numpy as np
import torch
import socket
from urx.urrtmon import URRTMonitor

# --- CONFIG ---
ROBOT_IP = "192.168.0.60"
CONTROL_FREQ = 125.0       # Robot "Body" Frequency
POLICY_FREQ = 25.0         # RL "Brain" Frequency
DECIMATION = int(CONTROL_FREQ / POLICY_FREQ)  # 125 / 25 = 5 ticks

def run_inference(model):
   print(f"Connecting to {ROBOT_IP}...")
   
   # 1. Setup Low-Level Connections
   rt = URRTMonitor(ROBOT_IP)
   rt.start()
   rt.wait()
   sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
   sock.connect((ROBOT_IP, 30002))
   
   print(f"✅ Body running at {CONTROL_FREQ} Hz")
   print(f"🧠 Brain running at {POLICY_FREQ} Hz (Decimation: {DECIMATION})")
   
   # Initialize variables
   current_action = np.zeros(6) # The last action chosen by RL
   tick_counter = 0             # Counts 0, 1, 2, 3, 4, 0...
   
   try:
       while True:
           # A. Wait for Robot Heartbeat (Strict 125 Hz)
           # This keeps the socket alive and smooth
           data = rt.get_all_data(wait=True)
           
           # B. Is it time for the Brain to think? (Every 5th tick)
           if tick_counter % DECIMATION == 0:
               
               # --- RL INFERENCE BLOCK ---
               # 1. Get Observation
               obs = process_observation(data) 
               
               # 2. Run Model (This can take 5-20ms safely now!)
               # Because we have the buffer of the next few ticks
               with torch.no_grad():
                    action = model(obs)
               
               # 3. Update the Current Action
               current_action = action.cpu().numpy()
               # --------------------------

           # C. Send Command to Robot (Happens EVERY tick, 125 Hz)
           # We keep sending the SAME action for 5 ticks, or interpolate.
           # Ideally, use servoj with t=0.1 to smooth the steps.
           send_servoj_command(sock, current_action, t=0.08)
           
           # Increment counter
           tick_counter += 1

   finally:
       stop_robot(sock, rt)