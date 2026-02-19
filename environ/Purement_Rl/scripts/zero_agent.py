#!/usr/bin/env python3
import argparse

from isaaclab.app import AppLauncher

# ---------------------------------------------------------
# 1) CLI + lancement Isaac (Kit)
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description="Ouvrir un USD avec IsaacLab")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument(
    "--usd",
    type=str,
    required=True,
    help="/home/ajin/work2/sim2real-pnp/environ/my_env/source/env_v1.usd",
)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app  # IMPORTANT: garde l'app vivante

# ---------------------------------------------------------
# 2) Imports après lancement de l'app
# ---------------------------------------------------------
import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

# ---------------------------------------------------------
# 3) Main
# ---------------------------------------------------------
def main():
    usd_path = args_cli.usd
    print(f"[INFO] Ouverture du USD: {usd_path}")

    # Ouvre le stage USD dans Kit
    omni.usd.get_context().open_stage(usd_path)

    # Configure + démarre le contexte de simulation
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.01,
        device=args_cli.device,   # ex: "cuda:0" ou "cpu" via flags IsaacLab
    )
    sim = SimulationContext(sim_cfg)

    # Optionnel: place la vue caméra de l'éditeur (ça aide à voir le stage)
    sim.set_camera_view(eye=[1.5, 1.2, 1.2], target=[0.0, 0.0, 0.4])

    # Reset (applique init, stabilise, etc.)
    sim.reset()
    print("[INFO] Simulation reset OK. Lancement boucle... (Ctrl+C pour arrêter)")

    # Boucle de simulation
    try:
        while simulation_app.is_running():
            sim.step()
    except KeyboardInterrupt:
        print("\n[INFO] Arrêt utilisateur (Ctrl+C).")

    # Clean exit
    sim.close()
    print("[INFO] Simulation fermée proprement.")

if __name__ == "__main__":
    main()
    simulation_app.close()
