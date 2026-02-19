#!/usr/bin/env python3
"""
DEBUG: Tracking pur de la trajectoire du dataset (Sans Policy)
Objectif : Vérifier que le robot MuJoCo peut physiquement suivre la trajectoire .npz
"""
import mujoco
import mujoco.viewer
import numpy as np
import time
import os

# --- PARAMETRES ---
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

XML_PATH = "ur10_complete.xml"
NPZ_PATH = "/home/ajin/workspace/sim2real-pnp/Real_robot/validation_mujoco/tag2_to_tag3.npz"

# Fréquences
DT_CTRL = 1.0 / 60.0    # 60 Hz (Boucle de commande)
FREQ_DATASET = 12.0     # 12 Hz (Données enregistrées)
SIM_DT = 0.002          # 500 Hz (Physique)

N_SUBSTEPS = int(DT_CTRL / SIM_DT)

class TrajectoryTracker:
    def __init__(self, xml_path, npz_path):
        print("\n" + "="*70)
        print("📈 TEST DE SUIVI DE TRAJECTOIRE PUR (SANS IA)")
        print(f"   XML: {xml_path}")
        print(f"   NPZ: {npz_path}")
        print("="*70 + "\n")

        # 1. MuJoCo
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names]
        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'tool0')

        # 2. Dataset
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Dataset introuvable: {npz_path}")
            
        data = np.load(npz_path, allow_pickle=True)
        self.paths = data["paths"]        
        self.ee_refs = data["ee_ref_pos"] 
        self.num_total_steps = self.paths.shape[1]
        
        # Listes pour le stockage
        self.history_time = []
        self.history_q_ref = []
        self.history_q_actual = []
        self.history_ee_ref = []
        self.history_ee_actual = []
        self.history_errors_ee = []

        print(f"Dataset chargé: {self.num_total_steps} points à {FREQ_DATASET}Hz")

    def get_ref_at_step(self, sim_step_count):
        """Récupère la position désirée (interpolée) pour le step actuel"""
        # Conversion step 60Hz -> index 12Hz
        ratio = (1.0 / DT_CTRL) / FREQ_DATASET
        idx_float = sim_step_count / ratio
        
        idx_prev = int(np.floor(idx_float))
        idx_next = int(np.ceil(idx_float))
        alpha = idx_float - idx_prev
        
        # Clamp aux limites
        idx_prev = min(idx_prev, self.num_total_steps - 1)
        idx_next = min(idx_next, self.num_total_steps - 1)
        
        q_prev = self.paths[0, idx_prev]
        q_next = self.paths[0, idx_next]
        
        # Interpolation linéaire
        q_target = (1 - alpha) * q_prev + alpha * q_next
        
        ee_target = self.ee_refs[0, idx_prev] # Pas besoin d'interpoler l'EE pour le tracking joint
        
        return q_target, ee_target

    def plot_results(self, t, q_ref, q_real, ee_ref, ee_real):
        import matplotlib.pyplot as plt
        
        # 1. Joint Tracking
        plt.figure(figsize=(15, 10))
        plt.suptitle("DEBUG: Suivi Moteur Pur (Sans Policy)", fontsize=16)
        
        for i in range(6):
            ax = plt.subplot(2, 3, i+1)
            plt.title(self.joint_names[i])
            plt.plot(t, q_ref[:, i], 'k--', label="Ref (Dataset)")
            plt.plot(t, q_real[:, i], 'g', label="Sim (Physics)")
            plt.grid(True)
            
            # --- CORRECTION ECHELLE ---
            # On force une échelle minime de 0.1 rad pour éviter le zoom sur le bruit
            ymin, ymax = ax.get_ylim()
            if (ymax - ymin) < 0.1:
                mid = (ymax + ymin) / 2
                plt.ylim(mid - 0.05, mid + 0.05)
            # --------------------------

            if i==0: plt.legend()
            
        plt.tight_layout()
        plt.savefig("debug_tracking_joints.png")
        print("✅ Plot sauvegardé : debug_tracking_joints.png")

        # 2. Cartesian Error
        plt.figure(figsize=(10, 6))
        err = np.linalg.norm(ee_ref - ee_real, axis=1) * 1000
        plt.title("Erreur Cartésienne (mm)")
        plt.plot(t, err, 'r')
        plt.xlabel("Temps (s)")
        plt.ylabel("Erreur (mm)")
        plt.grid(True)
        plt.savefig("debug_tracking_error.png")
        print("✅ Plot sauvegardé : debug_tracking_error.png")

    def run(self):
        # Position Initiale
        q_start = self.paths[0, 0]
        for i, jid in enumerate(self.joint_ids):
            self.data.qpos[jid] = q_start[i]
            self.data.qvel[jid] = 0.0
            self.data.ctrl[i] = q_start[i]
        
        # Stabilisation
        print("⏳ Stabilisation...")
        mujoco.mj_forward(self.model, self.data)
        for _ in range(100): 
            mujoco.mj_step(self.model, self.data)
            
        print("🎮 Lancement Viewer...")
        viewer = mujoco.viewer.launch_passive(self.model, self.data)
        time.sleep(1.0)

        sim_step_count = 0
        max_sim_steps = int(self.num_total_steps * ((1.0/DT_CTRL) / FREQ_DATASET))
        
        while sim_step_count < max_sim_steps:
            if not viewer.is_running(): break
            t_start = time.time()

            # 1. RECUPERER LA CIBLE (DATASET)
            target_q, ee_des = self.get_ref_at_step(sim_step_count)
            
            # 2. COMMANDE MOTEUR (Action = Position Désirée)
            # On force le robot à aller à la position target
            self.data.ctrl[:6] = target_q

            # 3. PHYSIQUE
            for _ in range(N_SUBSTEPS):
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
            
            # 4. MONITORING
            # Position actuelle
            current_q = np.array([self.data.qpos[i] for i in self.joint_ids])
            current_ee = np.array(self.data.xpos[self.ee_body_id])
            
            # Stockage pour plot
            self.history_time.append(sim_step_count * DT_CTRL)
            self.history_q_ref.append(target_q.copy())
            self.history_q_actual.append(current_q.copy())
            self.history_ee_ref.append(ee_des.copy())
            self.history_ee_actual.append(current_ee.copy())

            # Erreur Joints (Max error sur les 6 joints)
            err_q_max = np.max(np.abs(current_q - target_q))
            # Erreur Cartésienne
            err_ee = np.linalg.norm(current_ee - ee_des) * 1000
            
            if sim_step_count % 60 == 0:
                print(f"[{sim_step_count}] Target Joint Err: {err_q_max:.4f} rad | Cartesian Err: {err_ee:.1f} mm")

            # Synchro temps réel
            t_elapsed = time.time() - t_start
            if t_elapsed < DT_CTRL:
                time.sleep(DT_CTRL - t_elapsed)
                
            sim_step_count += 1
            
        viewer.close()
        
        # Generation des plots
        self.plot_results(
            np.array(self.history_time),
            np.array(self.history_q_ref),
            np.array(self.history_q_actual),
            np.array(self.history_ee_ref),
            np.array(self.history_ee_actual)
        )


if __name__ == "__main__":
    tracker = TrajectoryTracker(XML_PATH, NPZ_PATH)
    tracker.run()
