#!/usr/bin/env python3
import time
from urx.urrtmon import URRTMonitor

ROBOT_IP = "192.168.0.60"

def main():
    rt = URRTMonitor(ROBOT_IP)
    rt.start()                 # thread qui lit en continu sur 30003
    rt.wait()                  # attend le 1er paquet

    print("Measuring realtime stream (30003) for 5 seconds...")
    t0 = time.time()
    n = 0
    last_ctrl_ts = None

    while time.time() - t0 < 5.0:
        data = rt.get_all_data(wait=True)
        ctrl_ts = float(data["ctrltimestamp"])  # timestamp du contrôleur
        if last_ctrl_ts is None or ctrl_ts != last_ctrl_ts:
            n += 1
            last_ctrl_ts = ctrl_ts

    freq = n / 5.0
    print(f"Realtime frequency (30003): {freq:.1f} Hz")

    rt.close()

if __name__ == "__main__":
    main()
