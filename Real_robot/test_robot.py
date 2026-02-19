import time
import numpy as np
import socket
import urx
from urx.urrtmon import URRTMonitor

# --- CONFIGURATION ---
ROBOT_IP = "192.168.0.60"
PORT = 30002          
DURATION = 10.0        
AMPLITUDE = 0.1       # +/- 0.1 rad (~6 degrees)

# --- SMOOTHNESS vs. LAG ---
# t=0.1 is VERY smooth and safe.
# It means "take 0.1s to get there". Since we send a new command every 0.008s,
# the robot is constantly chasing a moving target 0.1s ahead.
# If this works, you can lower it to 0.08 later for faster reaction.
SERVO_T = 0.1        

# --- SAFETY LIMITS ---
# If the target is more than 0.05 rad (~3 deg) away from current position
# in a single timestep, we ASSUME A CRASH/ERROR and stop.
MAX_STEP_CHANGE = 0.05 

def main():
    print(f"🔒 Connecting to {ROBOT_IP} with SAFETY CHECKS...")
    
    rt = URRTMonitor(ROBOT_IP)
    rt.start()
    rt.wait() 
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1.0)
    sock.connect((ROBOT_IP, PORT))
    print("✅ Connection Established.")

    try:
        # Get Initial Position
        init_data = rt.get_all_data(wait=True)
        start_q = np.array(init_data['qActual'])
        
        print(f"Initial Q: {np.round(start_q, 3)}")
        print("Starting Smooth Sine Wave on Wrist 3...")
        print("⚠️  KEEP HAND ON E-STOP BUTTON.")
        
        start_time = time.time()
        loop_count = 0
        
        while True:
            elapsed = time.time() - start_time
            if elapsed >= DURATION:
                print("Time limit reached.")
                break
            
            # 1. SYNCHRONIZATION (Crucial for Safety)
            # This ensures we don't spam the robot faster than it can think.
            rt.get_all_data(wait=True) 

            # 2. CALCULATE TARGET
            offset = AMPLITUDE * np.sin(2 * np.pi * 0.5 * elapsed)
            target_q = start_q.copy()
            target_q[0] += offset 

            # 3. SAFETY CHECK: JUMP DETECTION
            # Read fresh current state
            # Note: qActual is what the robot is DOING. target_q is what we WANT.
            current_q = np.array(rt.get_all_data()['qActual'])
            
            # Check deviation (Are we asking for a teleport?)
            diff = np.max(np.abs(target_q - current_q))
            
            # Allow deviation because target_q is "future", but cap it.
            # If we are > 0.2 rad away, we are likely out of sync or unstable.
            if diff > 0.2: 
                print(f"⛔ EMERGENCY STOP: Target deviation too high ({diff:.3f} rad)")
                break

            # 4. SEND COMMAND
            q_str = str(list(target_q))
            # Send command with newline \n
            command = f"servoj({q_str}, 0, 0, {SERVO_T})\n"
            sock.sendall(command.encode('utf-8'))
            
            loop_count += 1

        # Diagnostics
        total_time = time.time() - start_time
        freq = loop_count / total_time
        print(f"Finished. Frequency: {freq:.1f} Hz")

    except KeyboardInterrupt:
        print("\nUser Interrupted (Ctrl+C). Stopping...")

    except Exception as e:
        print(f"ERROR: {e}")

    finally:
        # --- SAFE SHUTDOWN ---
        print("Sending STOP command...")
        # stopj(2.0) stops the robot with deceleration of 2.0 rad/s^2
        sock.sendall(b"stopj(2.0)\n")
        time.sleep(0.5)
        
        print("Closing connections.")
        rt.stop()
        sock.close()

if __name__ == "__main__":
    main()