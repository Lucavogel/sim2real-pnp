#!/usr/bin/env python3
"""
Génère UNE trajectoire du tag 2 vers tag 3 (MoveIt2) avec:
- Lecture TF world->tag2 et world->tag3
- IK start (/compute_ik)
- compute_cartesian_path
- resample à 64 points
- unwrap + filtre anti-sauts joint-space (max dq en rad/s)
- sauvegarde .npz format identique à generate_single_line.py

⚙️ IMPORTANT: Dataset réglé pour 25 Hz
- dt = 0.04 s
- 64 points => 2.56 s par trajectoire

Usage:
  # Terminal 1: lancer ROS2 + apriltag detection
  ros2 launch apriltag_webcam_viewer webcam_apriltag_rviz.launch.py video_device:=/dev/video0
  
  # Terminal 2: lancer MoveIt2
  ros2 launch ur_coppeliasim ur_isaaclab_moveit.launch.py
  
  # Terminal 3: générer trajectoire
  python3 generate_tag_to_tag_trajectory.py --output tag2_to_tag3.npz
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

import numpy as np
import time
import argparse
from pathlib import Path

from moveit_msgs.srv import GetCartesianPath, GetPositionIK
from moveit_msgs.msg import RobotState, PositionIKRequest
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


UR10_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


class TagToTagTrajectoryGenerator(Node):
    def __init__(self, output_file="tag2_to_tag3.npz"):
        super().__init__("tag_to_tag_trajectory_generator")

        self.output_file = Path(output_file)

        # ---------------- Dataset rate: 12 Hz (IDENTIQUE À ISAAC LAB) ----------------
        # Isaac Lab: sim.dt=1/60, decimation=5 => control freq = 60/5 = 12 Hz
        self.control_hz = 12
        self.dt = 1.0 / float(self.control_hz)  # 0.0833 s

        # Workspace (identique à generate_single_line.py)
        self.workspace = {
            "x_min": 0.7,
            "x_max": 1.000,
            "y_min": -0.2,
            "y_max": 0.2,
            "z_fixed": 0.20,
        }

        # Dataset constraints (identiques)
        self.resample_len = 64  # ✅ 64 points à 12 Hz => 5.33 s
        self.min_line_len = 0.20  # distance minimale tag2->tag3

        # ---------------- dq filter en rad/s (identique) ----------------
        self.max_dq_rad_s = 6.0
        self.max_dq_per_step = float(self.max_dq_rad_s * self.dt)  # rad/step
        self.max_dq_rad_s_joints = None  # option: [thr0..thr5] en rad/s

        # TF frames (référence = world où sont tous les frames)
        self.world_frame = "world"  # ✅ Frame world du robot
        self.tag0_frame = "tag36h11:2"
        self.tag1_frame = "tag36h11:3"

        # TF buffer + listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Clients MoveIt
        self._cartesian_path_client = self.create_client(GetCartesianPath, "/compute_cartesian_path")
        self._ik_client = self.create_client(GetPositionIK, "/compute_ik")

        self.executor = None

        # Orientation tool vers le bas (x,y,z,w) - identique
        self.fixed_orientation = [1.0, 0.0, 0.0, 0.0]

        # Seed IK (home)
        self.seed_joint_state = [0.0, -1.309, 2.14675, -2.44346, -1.5708, 0.0]

        # Waypoints demandés
        self.num_waypoints = 64

        self.get_logger().info("🏷️  Générateur trajectoire tag2 → tag3 (IK + resample + unwrap + dq-filter)")
        self.get_logger().info(f"   Control rate: {self.control_hz} Hz (dt={self.dt:.3f}s)")
        self.get_logger().info(f"   Resample:     {self.resample_len} points")
        self.get_logger().info(
            f"   dq max:       {self.max_dq_rad_s:.2f} rad/s => {self.max_dq_per_step:.3f} rad/step"
        )
        self.get_logger().info("")

    def set_executor(self, executor):
        self.executor = executor

    def _spin_until_done(self, future, timeout_sec=5.0):
        t0 = time.time()
        while not future.done():
            if (time.time() - t0) > timeout_sec:
                return False
            if self.executor:
                self.executor.spin_once(timeout_sec=0.1)
            else:
                time.sleep(0.05)
        return True

    def _robot_state_from_q(self, q):
        rs = RobotState()
        rs.joint_state = JointState()
        rs.joint_state.name = list(UR10_JOINT_NAMES)
        rs.joint_state.position = list(map(float, q))
        return rs

    # ---------------- TF lookup ----------------
    def get_tag_position(self, tag_frame, timeout_sec=2.0):
        """
        Récupère la position (x,y,z) d'un tag dans le frame 'world'.
        Retourne (x,y,z) ou None si échec.
        """
        try:
            trans = self.tf_buffer.lookup_transform(
                self.world_frame,  # world (frame du robot)
                tag_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=timeout_sec)
            )
            x = float(trans.transform.translation.x)
            y = float(trans.transform.translation.y)
            z = float(trans.transform.translation.z)
            return (x, y, z)
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().error(f"❌ TF lookup échoué pour {tag_frame}: {e}")
            return None

    # ---------------- Workspace validation ----------------
    def validate_workspace(self, pos_xyz, tag_name):
        """
        Vérifie que la position est dans le workspace.
        Retourne True si valide, False sinon.
        NE BLOQUE PAS la génération - juste pour info/warning.
        """
        x, y, z = pos_xyz
        valid = True
        
        # Check X
        if not (self.workspace["x_min"] <= x <= self.workspace["x_max"]):
            self.get_logger().warn(
                f"⚠️  {tag_name} hors workspace X: {x:.3f} (limites: [{self.workspace['x_min']:.2f}, {self.workspace['x_max']:.2f}])"
            )
            valid = False
        
        # Check Y
        if not (self.workspace["y_min"] <= y <= self.workspace["y_max"]):
            self.get_logger().warn(
                f"⚠️  {tag_name} hors workspace Y: {y:.3f} (limites: [{self.workspace['y_min']:.2f}, {self.workspace['y_max']:.2f}])"
            )
            valid = False
        
        # Check Z (tolérance ±5cm)
        z_tolerance = 0.05
        if abs(z - self.workspace["z_fixed"]) > z_tolerance:
            self.get_logger().warn(
                f"⚠️  {tag_name} Z éloigné: {z:.3f} (attendu: {self.workspace['z_fixed']:.2f} ±{z_tolerance:.2f})"
            )
            valid = False
        
        return valid

    # ---------------- Resampling (identique) ----------------
    def resample_q(self, q_array, out_len):
        if q_array is None or len(q_array) < 2:
            return None
        T = q_array.shape[0]
        x_old = np.linspace(0.0, 1.0, T, dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, out_len, dtype=np.float32)

        q_new = np.zeros((out_len, q_array.shape[1]), dtype=np.float32)
        for j in range(q_array.shape[1]):
            q_new[:, j] = np.interp(x_new, x_old, q_array[:, j]).astype(np.float32)
        return q_new

    # ---------------- unwrap (identique) ----------------
    def unwrap_joints(self, q_res):
        return np.unwrap(q_res, axis=0).astype(np.float32)

    # ---------------- dq filter (identique) ----------------
    def passes_dq_filter(self, q_res_unwrapped):
        if q_res_unwrapped is None or len(q_res_unwrapped) < 2:
            return False
        dq = np.abs(np.diff(q_res_unwrapped, axis=0))  # (T-1,6)

        if self.max_dq_rad_s_joints is not None:
            thr = (np.array(self.max_dq_rad_s_joints, dtype=np.float32) * self.dt).reshape(1, 6)
            return bool(np.all(dq <= thr))
        else:
            return bool(np.max(dq) <= float(self.max_dq_per_step))

    # ---------------- IK (identique) ----------------
    def compute_ik(self, x, y, z, seed_q=None):
        req = GetPositionIK.Request()
        req.ik_request = PositionIKRequest()
        req.ik_request.group_name = "ur_manipulator"
        req.ik_request.ik_link_name = "wrist_3_link"

        ps = PoseStamped()
        ps.header.frame_id = "world"  # ✅ Frame du robot (pas world_tag!)
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = float(z)
        ps.pose.orientation.x = self.fixed_orientation[0]
        ps.pose.orientation.y = self.fixed_orientation[1]
        ps.pose.orientation.z = self.fixed_orientation[2]
        ps.pose.orientation.w = self.fixed_orientation[3]
        req.ik_request.pose_stamped = ps

        if seed_q is None:
            seed_q = self.seed_joint_state
        req.ik_request.robot_state = self._robot_state_from_q(seed_q)

        future = self._ik_client.call_async(req)
        if not self._spin_until_done(future, timeout_sec=5.0):
            self.get_logger().error("   ⏱️ IK timeout")
            return None

        res = future.result()
        if res is None or res.error_code.val != 1:
            code = None if res is None else res.error_code.val
            self.get_logger().warn(f"   ⚠️ IK failed, error_code={code}")
            return None

        js = res.solution.joint_state
        name_to_pos = dict(zip(js.name, js.position))
        try:
            q = [float(name_to_pos[n]) for n in UR10_JOINT_NAMES]
        except KeyError:
            self.get_logger().error("   ❌ IK: joints manquants")
            return None
        return q

    # ---------- Waypoints cartésiens (identique) ----------
    def generate_cartesian_waypoints(self, start_pose_xyz, end_pose_xyz):
        sx, sy, sz = start_pose_xyz
        ex, ey, ez = end_pose_xyz

        waypoints = []
        for i in range(self.num_waypoints + 1):
            t = i / self.num_waypoints
            pose = PoseStamped()
            pose.header.frame_id = "world"  # ✅ Frame du robot (pas world_tag!)
            pose.pose.position.x = sx + t * (ex - sx)
            pose.pose.position.y = sy + t * (ey - sy)
            pose.pose.position.z = sz + t * (ez - sz)
            pose.pose.orientation.x = self.fixed_orientation[0]
            pose.pose.orientation.y = self.fixed_orientation[1]
            pose.pose.orientation.z = self.fixed_orientation[2]
            pose.pose.orientation.w = self.fixed_orientation[3]
            waypoints.append(pose.pose)
        return waypoints

    # ---------- Cartesian path (identique) ----------
    def plan_cartesian_path(self, start_pose_xyz, end_pose_xyz, q_start):
        request = GetCartesianPath.Request()
        request.header.frame_id = "world"  # ✅ Frame du robot (pas world_tag!)
        request.group_name = "ur_manipulator"
        request.link_name = "wrist_3_link"

        request.start_state = self._robot_state_from_q(q_start)
        request.waypoints = self.generate_cartesian_waypoints(start_pose_xyz, end_pose_xyz)

        request.max_step = 0.005
        request.jump_threshold = 0.0
        request.avoid_collisions = True

        future = self._cartesian_path_client.call_async(request)
        if not self._spin_until_done(future, timeout_sec=10.0):
            self.get_logger().error("   ⏱️ Timeout /compute_cartesian_path (10s)")
            return None

        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"   ❌ Erreur service cartésien: {e}")
            return None

        if response is None:
            return None

        if response.fraction < 1.0:
            self.get_logger().warn(
                f"   ⚠️ Trajectoire incomplète ({response.fraction*100:.1f}%) - REJETÉE"
            )
            return None

        return response

    def trajectory_to_numpy(self, trajectory_solution):
        if trajectory_solution is None or not hasattr(trajectory_solution, "joint_trajectory"):
            return None
        points = trajectory_solution.joint_trajectory.points
        if len(points) == 0:
            return None
        joint_angles = [list(p.positions) for p in points]
        return np.array(joint_angles, dtype=np.float32)

    # ---------------- Main generation ----------------
    def generate_trajectory(self):
        self.get_logger().info("🚀 Début génération trajectoire tag2 → tag3...")
        self.get_logger().info("=" * 70)

        # 1. Attendre services MoveIt
        self.get_logger().info("🔍 Recherche service /compute_cartesian_path...")
        if not self._cartesian_path_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("❌ Service /compute_cartesian_path indisponible!")
            return False
        self.get_logger().info("✅ Service cartésien trouvé")

        self.get_logger().info("🔍 Recherche service /compute_ik...")
        if not self._ik_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("❌ Service IK /compute_ik indisponible!")
            return False
        self.get_logger().info("✅ Service IK trouvé\n")

        # 2. Lire positions tags depuis TF
        self.get_logger().info("🏷️  Lecture positions des tags depuis /tf...")
        
        tag0_pos = self.get_tag_position(self.tag0_frame, timeout_sec=3.0)
        if tag0_pos is None:
            self.get_logger().error(f"❌ Impossible de lire TF pour {self.tag0_frame}")
            self.get_logger().error("   Assure-toi que apriltag_ros tourne et voit le tag 2!")
            return False
        
        tag1_pos = self.get_tag_position(self.tag1_frame, timeout_sec=3.0)
        if tag1_pos is None:
            self.get_logger().error(f"❌ Impossible de lire TF pour {self.tag1_frame}")
            self.get_logger().error("   Assure-toi que apriltag_ros tourne et voit le tag 3!")
            return False

        self.get_logger().info(f"✅ Tag 2 position: ({tag0_pos[0]:.3f}, {tag0_pos[1]:.3f}, {tag0_pos[2]:.3f})")
        self.get_logger().info(f"✅ Tag 3 position: ({tag1_pos[0]:.3f}, {tag1_pos[1]:.3f}, {tag1_pos[2]:.3f})")

        # 3. Info workspace (pas de rejet, juste info)
        self.get_logger().info("\n📦 Info workspace:")
        self.get_logger().info(f"   X: [{self.workspace['x_min']:.2f}, {self.workspace['x_max']:.2f}]")
        self.get_logger().info(f"   Y: [{self.workspace['y_min']:.2f}, {self.workspace['y_max']:.2f}]")
        self.get_logger().info(f"   Z fixe: {self.workspace['z_fixed']:.2f}")
        
        in_ws_0 = self.validate_workspace(tag0_pos, "Tag 2")
        in_ws_1 = self.validate_workspace(tag1_pos, "Tag 3")
        
        if in_ws_0 and in_ws_1:
            self.get_logger().info("✅ Tags dans le workspace (optimal)")
        else:
            self.get_logger().warn("⚠️  Tags hors workspace (trajectoire générée quand même)")

        # 4. Vérifier distance minimale
        line_len = float(np.linalg.norm(np.array(tag1_pos) - np.array(tag0_pos)))
        self.get_logger().info(f"\n📏 Distance tag2 → tag3: {line_len:.3f} m")
        
        if line_len < self.min_line_len:
            self.get_logger().error(
                f"❌ Distance trop courte: {line_len:.3f}m < {self.min_line_len:.3f}m"
            )
            return False
        self.get_logger().info(f"✅ Distance suffisante (min: {self.min_line_len:.3f}m)\n")

        # 5. Utiliser z_fixed au lieu des z mesurés (comme generate_single_line.py)
        tag0_pos_fixed = (tag0_pos[0], tag0_pos[1], self.workspace["z_fixed"])
        tag1_pos_fixed = (tag1_pos[0], tag1_pos[1], self.workspace["z_fixed"])
        
        self.get_logger().info(f"🔧 Utilisation z fixe: {self.workspace['z_fixed']:.3f}m")
        self.get_logger().info(f"   Tag 2 corrigé: ({tag0_pos_fixed[0]:.3f}, {tag0_pos_fixed[1]:.3f}, {tag0_pos_fixed[2]:.3f})")
        self.get_logger().info(f"   Tag 3 corrigé: ({tag1_pos_fixed[0]:.3f}, {tag1_pos_fixed[1]:.3f}, {tag1_pos_fixed[2]:.3f})\n")

        # 6. Calculer IK pour tag2
        self.get_logger().info("🎯 Calcul IK pour position tag2...")
        q_start = self.compute_ik(tag0_pos_fixed[0], tag0_pos_fixed[1], tag0_pos_fixed[2], seed_q=self.seed_joint_state)
        if q_start is None:
            self.get_logger().error("❌ IK impossible pour tag2")
            return False
        self.get_logger().info("✅ IK tag2 trouvée\n")

        # 7. Planifier trajectoire cartésienne
        self.get_logger().info("🛤️  Planification trajectoire cartésienne tag2 → tag3...")
        response = self.plan_cartesian_path(tag0_pos_fixed, tag1_pos_fixed, q_start=q_start)
        if response is None:
            self.get_logger().error("❌ Planification cartésienne échouée")
            return False

        q_array = self.trajectory_to_numpy(response.solution)
        if q_array is None or len(q_array) < 2:
            self.get_logger().error("❌ Trajectoire trop courte")
            return False

        self.get_logger().info(f"✅ Trajectoire brute: {len(q_array)} points\n")

        # 5. Resample à 64 points
        self.get_logger().info(f"📊 Resampling à {self.resample_len} points...")
        q_res = self.resample_q(q_array, self.resample_len)
        if q_res is None:
            self.get_logger().error("❌ Resample échoué")
            return False
        self.get_logger().info(f"✅ Trajectoire resample: {self.resample_len} points\n")

        # 6. Unwrap + dq filter
        self.get_logger().info("🔍 Vérification contraintes dq (unwrap + filter)...")
        q_unwrap = self.unwrap_joints(q_res)
        if not self.passes_dq_filter(q_unwrap):
            dq_max = float(np.max(np.abs(np.diff(q_unwrap, axis=0))))
            self.get_logger().error(
                f"❌ dq trop grand: {dq_max:.3f} rad/step | seuil={self.max_dq_per_step:.3f}"
            )
            return False
        self.get_logger().info("✅ Contraintes dq respectées\n")

        # 7. Créer référence cartésienne (interpolation linéaire)
        xs = np.linspace(tag0_pos_fixed[0], tag1_pos_fixed[0], self.resample_len, dtype=np.float32)
        ys = np.linspace(tag0_pos_fixed[1], tag1_pos_fixed[1], self.resample_len, dtype=np.float32)
        zs = np.full(self.resample_len, self.workspace["z_fixed"], dtype=np.float32)
        ee_ref = np.stack([xs, ys, zs], axis=1).astype(np.float32)

        # 8. Sauvegarder (format identique à generate_single_line.py)
        self.get_logger().info("💾 Sauvegarde format Isaac Lab...")

        # Format: (1, 64, 6) pour compatibilité avec le dataset multi-trajectoires
        q_paths_array = q_res[np.newaxis, :, :].astype(np.float32)  # (1,64,6)
        ee_ref_array = ee_ref[np.newaxis, :, :].astype(np.float32)  # (1,64,3)

        duration = self.resample_len * self.dt

        dataset = {
            "paths": q_paths_array,            # (1,64,6) float32
            "ee_ref_pos": ee_ref_array,        # (1,64,3) float32
            "tag0_pos": np.array(tag0_pos, dtype=np.float32),       # position TF mesurée
            "tag1_pos": np.array(tag1_pos, dtype=np.float32),       # position TF mesurée
            "tag0_pos_fixed": np.array(tag0_pos_fixed, dtype=np.float32),  # position corrigée (z fixe)
            "tag1_pos_fixed": np.array(tag1_pos_fixed, dtype=np.float32),  # position corrigée (z fixe)
            "line_length": np.float32(line_len),
            "workspace": np.array(
                [
                    self.workspace["x_min"],
                    self.workspace["x_max"],
                    self.workspace["y_min"],
                    self.workspace["y_max"],
                    self.workspace["z_fixed"],
                ],
                dtype=np.float32,
            ),
            "num_trajectories": np.int32(1),
            "resample_len": np.int32(self.resample_len),
            "min_line_len": np.float32(self.min_line_len),
            
            # dq info
            "max_dq_rad_s": np.float32(self.max_dq_rad_s),
            "max_dq_per_step": np.float32(self.max_dq_per_step),

            # timing
            "dt": np.float32(self.dt),
            "control_hz": np.int32(self.control_hz),
        }

        np.savez_compressed(self.output_file, **dataset)
        self.get_logger().info(f"✅ Sauvegardé: {self.output_file.absolute()}\n")

        self.get_logger().info("📊 Statistiques:")
        self.get_logger().info(f"   Points: {self.resample_len}")
        self.get_logger().info(f"   Durée: {duration:.2f}s à {self.control_hz}Hz")
        self.get_logger().info(f"   Distance: {line_len:.3f}m")
        self.get_logger().info(f"   Workspace: X=[{self.workspace['x_min']:.2f}, {self.workspace['x_max']:.2f}], "
                              f"Y=[{self.workspace['y_min']:.2f}, {self.workspace['y_max']:.2f}], "
                              f"Z={self.workspace['z_fixed']:.2f}\n")

        self.get_logger().info("=" * 70)
        self.get_logger().info("✅ TERMINÉ")
        self.get_logger().info("=" * 70)

        return True


def main():
    parser = argparse.ArgumentParser(
        description="Générateur trajectoire tag2 → tag3 (MoveIt2) @12Hz (Isaac Lab compatible)"
    )
    parser.add_argument("--output", type=str, default="tag2_to_tag3.npz", help="Fichier de sortie")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🏷️  GÉNÉRATEUR TRAJECTOIRE TAG2 → TAG3 (MOVEIT2) @12Hz")
    print("=" * 70)
    print(f"Fichier de sortie: {args.output}")
    print("=" * 70 + "\n")

    print("⚠️  Prérequis:")
    print("   1. ROS2 + apriltag detection actifs (voir tags 2 et 3)")
    print("   2. MoveIt2 actif (ur_isaaclab_moveit.launch.py)")
    print("=" * 70 + "\n")

    input("Appuyez sur ENTRÉE pour continuer...")

    rclpy.init()
    node = TagToTagTrajectoryGenerator(output_file=args.output)

    from rclpy.executors import SingleThreadedExecutor
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    node.set_executor(executor)

    print("\n⏳ Initialisation ROS2 + TF buffer (3 secondes)...")
    start_time = time.time()
    while time.time() - start_time < 3.0:
        executor.spin_once(timeout_sec=0.1)
    print("✅ ROS2 + TF prêt\n")

    success = node.generate_trajectory()

    executor.shutdown()
    rclpy.shutdown()

    if success:
        print("\n✅ Succès!")
    else:
        print("\n❌ Échec de la génération")
        exit(1)


if __name__ == "__main__":
    main()
