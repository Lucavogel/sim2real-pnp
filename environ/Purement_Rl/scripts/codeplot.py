import glob
import numpy as np
import matplotlib.pyplot as plt

files = sorted(glob.glob("./logs_errors/errors_env000_ep*.npz"))

for f in files:
    d = np.load(f)
    plt.figure()
    plt.plot(d["e_true"], label="e_true (Cible-EE)")
    plt.plot(d["e_cam"], label="e_cam (caméra)", alpha=0.7)
    plt.xlabel("step")
    plt.ylabel("erreur (m)")
    plt.title(f)
    plt.grid(True)
    plt.legend()
    plt.show()


