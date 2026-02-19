#!/usr/bin/env python3
"""
Validation Sim-to-Sim : Isaac Lab → MuJoCo
CONFIGURATION : Identique à purement_rl_env_cfg.py (Decimation 5, 60Hz Sim)
"""
import mujoco
import mujoco.viewer
import torch
import numpy as np
import time
import os

# --- CONFIGURATION IDENTIQUE A L'ENTRAINEMENT ---
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

XML_PATH = "ur10_complete.xml"
NPZ_PATH = "/home/ajin/workspace/sim2real-pnp/Real_robot/validation_mujoco/tag2_to_tag3.npz"
MODEL_PATH = "/home/ajin/workspace/sim2real-pnp/Real_robot/validation_mujoco/policy(2).pt"

# --- PARAMETRES CRITIQUES ---
DT_CTRL = 1.0 / 12.0    # 0.0833s (12 Hz)
FREQ_DATASET = 12.0     # 12 Hz

# Simulation MuJoCo alignée sur Isaac Lab
SIM_DT = 1/60           # 60 Hz (Isaac Lab: sim.dt=1/60)
N_SUBSTEPS = int(DT_CTRL / SIM_DT) # 5 substeps (Isaac Lab: decimation=5)

ACTION_SCALE = 0.05      # my_env_env_cfg.py ligne 40 (valeur 0.5)
LOOKAHEAD_STEPS = 0      # Isaac Lab deploy default: 0 (start exactly at first waypoint)

# Limites physiques (pour éviter les sauts violents)
MAX_SPEED_RAD_S = 1.5    

class MuJoCoValidator:
    def __init__(self, xml_path, model_path, npz_path):
        print("\n" + "="*70)
        print("🤖 VALIDATION STRICTE")
        print(f"   Control Freq : {1/DT_CTRL:.1f} Hz")
        print(f"   Physics Freq : {1/SIM_DT:.1f} Hz ({N_SUBSTEPS} substeps)")
        print("="*70 + "\n")
        
        # 1. MuJoCo
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names]
        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'wrist_3_link')
        print(f"ℹ️  Using EE body: wrist_3_link (id={self.ee_body_id})")

        # 2. Dataset
        data = np.load(npz_path, allow_pickle=True)
        self.paths = data["paths"]        
        self.ee_refs = data["ee_ref_pos"] 
        self.num_total_steps = self.paths.shape[1]
        print(f"📦 Dataset: {self.paths.shape[0]} trajs, {self.num_total_steps} steps")
        print(f"📍 First EE ref: {self.ee_refs[0, 0]}")
        print(f"📍 Last EE ref:  {self.ee_refs[0, -1]}")
        print(f"📍 First q_ref: {self.paths[0, 0]}")
        
        # ⚠️ VÉRIFICATION CRITIQUE: Isaac Lab utilise des mètres, pas millimètres
        # Si les coordonnées sont > 10, c'est probablement en millimètres
        if np.max(np.abs(self.ee_refs)) > 10.0:
            print("⚠️  WARNING: Dataset semble être en millimètres, conversion en mètres...")
            self.ee_refs = self.ee_refs / 1000.0
            print(f"📍 First EE ref (converti): {self.ee_refs[0, 0]}")
            print(f"📍 Last EE ref (converti):  {self.ee_refs[0, -1]}")
        
        # 3. Modèle
        self.device = torch.device("cpu")
        print(f"🧠 Chargement modèle : {model_path}")
        
        # CORRECTION : On autorise explicitement le chargement complet (weights_only=False)
        try:
            # Option 1 : Si c'est un modèle exporté via JIT (souvent le cas avec Isaac Lab)
            self.policy = torch.jit.load(model_path, map_location=self.device)
        except Exception:
            # Option 2 : Si c'est un checkpoint standard (.pt)
            self.policy = torch.load(model_path, map_location=self.device, weights_only=False)
            
        self.policy.eval()
        self.last_action = torch.zeros(6, device=self.device)

        # --- SIMULATION DELAI ACTION (Comme dans Isaac) ---
        # Config Training: action_delay_min=1, max=3. 
        # On utilise buffer pour simuler ce retard.
        self.latency_steps = 1  # Valeur min sûre
        self.action_queue = [torch.zeros(6, device=self.device) for _ in range(self.latency_steps + 1)]


    def get_obs(self, sim_step_count):
        # Récup données simu
        q = torch.tensor([self.data.qpos[i] for i in self.joint_ids], device=self.device).float()
        qd = torch.tensor([self.data.qvel[i] for i in self.joint_ids], device=self.device).float()
        ee_meas = torch.tensor(self.data.xpos[self.ee_body_id], device=self.device).float()

        # Récup référence (pas d'interpolation comme Isaac Lab)
        # Isaac Lab: avance directement dans le dataset à la fréquence de contrôle
        idx_ref = sim_step_count
        
        # Lookahead : Isaac Lab deploy default = 0 (start exactly at first waypoint)
        idx_ref = min(idx_ref + LOOKAHEAD_STEPS, self.num_total_steps - 1)
        
        q_ref = torch.tensor(self.paths[0, idx_ref], device=self.device).float()
        ee_des = torch.tensor(self.ee_refs[0, idx_ref], device=self.device).float()

        # Observation (40 dims) - EXACT Isaac Lab format
        obs = torch.cat([
            q, qd, q_ref,
            (q_ref - q),           # err_q
            ee_meas, ee_des,
            (ee_des - ee_meas),    # err_ee
            torch.norm(ee_des - ee_meas).unsqueeze(0), # e_meas_norm
            self.last_action
        ])
        return obs, q_ref, ee_des, ee_meas

    def run(self):
        # Position Initiale
        q_start = self.paths[0, 0]
        
        # --- PROGRESSIVE WARMUP AU LIEU DE HARD RESET ---
        # Si on force instantanément q_start, il peut y avoir un gap FK entre MoveIt et MuJoCo
        # Solution: démarrer d'une pose home puis interpoler progressivement vers q_start
        print("⏳ Warm-up progressif vers pose de départ...")
        
        # Pose home UR10 (position neutre stable)
        q_home = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
        
        # Interpolation progressive: home -> q_start en 100 steps
        n_warmup_steps = 100
        for step in range(n_warmup_steps + 1):
            alpha = step / n_warmup_steps
            q_interp = q_home + alpha * (q_start - q_home)
            
            for i, jid in enumerate(self.joint_ids):
                self.data.qpos[jid] = q_interp[i]
                self.data.qvel[jid] = 0.0
                self.data.ctrl[i] = q_interp[i]
            
            mujoco.mj_step(self.model, self.data)
        
        # Stabilisation finale à q_start
        for _ in range(50): 
            for i, jid in enumerate(self.joint_ids):
                self.data.qpos[jid] = q_start[i]
                self.data.qvel[jid] = 0.0
                self.data.ctrl[i] = q_start[i]
            mujoco.mj_step(self.model, self.data)
        
        mujoco.mj_forward(self.model, self.data) # Met à jour xpos, sensors
        
        ee_start_stabilized = np.array(self.data.xpos[self.ee_body_id])
        ee_ref_0 = self.ee_refs[0, 0]
        ee_error_start = np.linalg.norm(ee_start_stabilized - ee_ref_0) * 1000
        
        print(f"📍 Pos T=0 (After Reset) : {ee_start_stabilized}")
        print(f"📍 Ref T=0 (Dataset)     : {ee_ref_0}")
        print(f"📍 Erreur initiale       : {ee_error_start:.1f} mm")
        
        if ee_error_start > 10.0:
            print("⚠️  ATTENTION: Erreur initiale élevée! Problème possible:")
            print("   - Dataset généré avec un autre robot/frame")
            print("   - Unités incorrectes (mm vs m)")
            print("   - Cinématique MuJoCo différente de MoveIt")

        print("🎮 Lancement Viewer...")
        viewer = mujoco.viewer.launch_passive(self.model, self.data)
        time.sleep(1.0) # Attente ouverture

        # BOUCLE PRINCIPALE
        sim_step_count = 0
        max_sim_steps = self.num_total_steps  # Isaac Lab: 1 step = 1 dataset point (64 steps @ 12Hz)
        
        # Histore pour plot - ON COMMENCE À ENREGISTRER SEULEMENT APRÈS LE RESET
        h_time, h_q_ref, h_q_real, h_vel, h_action, h_ee_ref, h_ee_real = [], [], [], [], [], [], []
        
        print(f"\n🚀 Début exécution trajectoire ({max_sim_steps} steps)...\n")

        while sim_step_count < max_sim_steps:
            if not viewer.is_running():
                print("⚠️ Viewer fermé par l'utilisateur")
                break
            
            t_start = time.time()

            # 1. OBS & INFERENCE (A la fréquence de contrôle 12Hz)
            obs, q_ref, ee_des, ee_meas = self.get_obs(sim_step_count)
            
            with torch.no_grad():
                action = self.policy(obs.unsqueeze(0)).squeeze(0)
               
            
            # --- DEBUG ACTION SHAPE ONCE ---
            if sim_step_count == 0:
                 print(f"ℹ️  Action Shape: {action.shape} | Obs Shape: {obs.shape}")
            # -------------------------------

            # --- ZONE DE TEST DU SIGNE ---
            # action = -action  # <--- DECOMMENTE ICI SI BESOIN D'INVERSER
            # -----------------------------

            action = torch.clamp(action, -1.0, 1.0)
            
            # --- GESTION LATENCE ---
            self.action_queue.append(action)
            action_delayed = self.action_queue.pop(0) # FIFO: On sort l'action d'il y a N steps
            # -----------------------

            self.last_action = action_delayed # On donne à l'obs l'action qui est *vraiment* appliquée

            # Calcul de la cible (JOINTS) avec l'action RETARDÉE
            target_q = q_ref.cpu().numpy() + action_delayed.cpu().numpy() * ACTION_SCALE

            # --- CORRECTION VISCOUS DRAG (IMPEDANCE MATCHING ISAAC) ---
            # Isaac utilise un PD implicite : F = Kp(q_targ - q) + Kd(qd_targ - qd)
            # MuJoCo Position actuator : F = Kp(ctrl - q) - Kd(qd)
            # On doit compenser le terme Kd(qd_targ) via ctrl.
            # ctrl = q_targ + (Kd/Kp) * qd_targ
            
            # 1. Calcul de la vitesse cible comme dans l'env d'entrainement
            # purement_rl_env.py L479: qd_cmd = (q_des - q) / dt
            current_q = self.data.qpos[:6]
            qd_target = (target_q - current_q) / DT_CTRL
            
            # 2. Injection dans le contrôleur (Kp=10000, Kd=1000 => ratio 0.1)
            # valeurs issues de ur10_complete.xml
            kd_kp_ratio = 1000.0 / 10000.0 
            
            ctrl_adjusted = target_q + kd_kp_ratio * qd_target
            self.data.ctrl[:6] = ctrl_adjusted

            # 2. PHYSIQUE
            # On applique la MEME action pendant N steps de simu (500Hz -> 60Hz)
            
            for _ in range(N_SUBSTEPS):
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
            
            # 3. Synchronisation Temps Réel
            t_elapsed = time.time() - t_start
            if t_elapsed < DT_CTRL:
                time.sleep(DT_CTRL - t_elapsed)

            # Debug console ponctuel
            if sim_step_count % 12 == 0:  # Toutes les secondes (12Hz)
                err = np.linalg.norm(ee_des.cpu().numpy() - ee_meas.cpu().numpy()) * 1000
                print(f"Step {sim_step_count}/{max_sim_steps} | Err: {err:.1f} mm")
            
            # --- LOGGING (enregistrement pour plot) ---
            h_time.append(sim_step_count * DT_CTRL)
            h_q_ref.append(q_ref.cpu().numpy())
            h_q_real.append(self.data.qpos[:6].copy())
            h_vel.append(self.data.qvel[:6].copy())
            h_action.append(action.cpu().numpy())
            h_ee_ref.append(ee_des.cpu().numpy())
            h_ee_real.append(ee_meas.cpu().numpy())
            # ---------------

            sim_step_count += 1
            
        viewer.close()

        print("\n" + "="*70)
        print(f"✅ Trajectoire complète exécutée ({sim_step_count}/{max_sim_steps} steps)")
        print("="*70 + "\n")

        # Sauvegarde
        print("💾 Sauvegarde des données...")
        np.savez("rapport_mujoco.npz",
                 time=np.array(h_time),
                 q_ref=np.array(h_q_ref),
                 q_actual=np.array(h_q_real),
                 velocities=np.array(h_vel),
                 actions=np.array(h_action),
                 ee_ref=np.array(h_ee_ref),
                 ee_actual=np.array(h_ee_real)
        )
        print("✅ Données sauvegardées dans rapport_mujoco.npz")

if __name__ == "__main__":
    v = MuJoCoValidator(XML_PATH, MODEL_PATH, NPZ_PATH)
    v.run()