#!/usr/bin/env python3
"""
Générateur de dataset - lignes droites aléatoires 2D (avec IK MoveIt2)
+ filtre longueur minimale
+ détection aller-retour robuste
+ resampling à longueur fixe (64 points)

Usage:
   cd ~/workspace/sim2real-pnp/environ/ur10
   source install/setup.bash

   # Terminal 1 - MoveIt
   ros2 launch ur_coppeliasim ur_isaaclab_moveit.launch.py

   # Terminal 2 - Générer dataset
   python3 src/ur_coppeliasim/scripts/generate_single_line.py --num-traj 100 --output random_lines_dataset.npz
"""

import rclpy
from rclpy.node import Node

import numpy as np
import time
import argparse
from pathlib import Path

from moveit_msgs.srv import GetCartesianPath, GetPositionIK
from moveit_msgs.msg import RobotState, PositionIKRequest
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped


UR10_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


class RandomLinesGenerator(Node):
    def __init__(self, num_trajectories=100, output_file="random_lines_dataset.npz"):
        super().__init__("random_lines_generator")

        self.num_trajectories = num_trajectories
        self.output_file = Path(output_file)

        # Workspace fixe
        self.workspace = {
            "x_min": 0.8,
            "x_max": 1.164,
            "y_min": -0.2,
            "y_max": 0.2,
            "z_fixed": 0.15,
        }

        # --- NOUVEAU: contraintes dataset ---
        self.min_line_len = 0.20     # m (10 cm) : rejette les lignes trop courtes
        self.resample_len = 64       # points fixes dans le dataset final

        # Clients MoveIt
        self._cartesian_path_client = self.create_client(
            GetCartesianPath, "/compute_cartesian_path"
        )
        self._ik_client = self.create_client(GetPositionIK, "/compute_ik")

        self.executor = None

        # Orientation fixe (outil vers le bas). Quaternion (x,y,z,w)
        self.fixed_orientation = [1.0, 0.0, 0.0, 0.0]

        # Seed joints (home UR10) pour aider l'IK
        self.seed_joint_state = [0.0, -1.309, 2.14675, -2.44346, -1.5708, 0.0]

        # Waypoints demandés à MoveIt (ça n'impose pas le nb de points retournés)
        self.num_waypoints = 64

        self.get_logger().info("📏 Générateur Dataset - Lignes Droites Aléatoires (avec IK)")
        self.get_logger().info(f"   Nombre: {self.num_trajectories} trajectoires")
        self.get_logger().info(
            f"   Workspace: X=[{self.workspace['x_min']:.3f}, {self.workspace['x_max']:.3f}] "
            f"Y=[{self.workspace['y_min']:.3f}, {self.workspace['y_max']:.3f}] "
            f"Z={self.workspace['z_fixed']:.3f}"
        )
        self.get_logger().info(f"   Min line length: {self.min_line_len:.2f} m")
        self.get_logger().info(f"   Resample length: {self.resample_len} points")
        self.get_logger().info(f"   Output: {self.output_file}")
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

    # ---------- NOUVEAU: resampling ----------
    def resample_q(self, q_array, out_len):
        """
        Resample une trajectoire joint-space (T,6) vers (out_len,6) par interpolation linéaire.
        """
        if q_array is None or len(q_array) < 2:
            return None

        T = q_array.shape[0]
        x_old = np.linspace(0.0, 1.0, T, dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, out_len, dtype=np.float32)

        q_new = np.zeros((out_len, q_array.shape[1]), dtype=np.float32)
        for j in range(q_array.shape[1]):
            q_new[:, j] = np.interp(x_new, x_old, q_array[:, j]).astype(np.float32)

        return q_new

    # ---------- NOUVEAU: aller-retour robuste ----------
    def cut_if_roundtrip(self, q_array):
        """
        Détecte un aller-retour en regardant la distance au start.
        Si la traj s'éloigne puis revient près du start, on coupe au point le plus éloigné.
        """
        if q_array is None or len(q_array) < 3:
            return q_array, False

        start = q_array[0]
        d = np.linalg.norm(q_array - start, axis=1)  # distance au start à chaque point

        max_idx = int(np.argmax(d))
        max_d = float(d[max_idx])
        end_d = float(d[-1])

        # Seuils (moins agressifs que ton ancien 0.5 rad)
        min_excursion = 0.15     # il faut s'éloigner au moins un peu
        return_ratio = 0.50      # "revient" si fin < 50% du max

        if max_d > min_excursion and end_d < (return_ratio * max_d) and max_idx >= 2:
            return q_array[: max_idx + 1], True

        return q_array, False

    # ---------- IK ----------
    def compute_ik(self, x, y, z, seed_q=None):
        req = GetPositionIK.Request()
        req.ik_request = PositionIKRequest()
        req.ik_request.group_name = "ur_manipulator"
        req.ik_request.ik_link_name = "tool0"

        ps = PoseStamped()
        ps.header.frame_id = "world"
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
        ok = self._spin_until_done(future, timeout_sec=5.0)
        if not ok:
            self.get_logger().error("   ⏱️ IK timeout (5s)")
            return None

        res = future.result()
        if res is None:
            self.get_logger().warn("   ⚠️ IK: réponse None")
            return None

        if res.error_code.val != 1:
            self.get_logger().warn(f"   ⚠️ IK failed, error_code={res.error_code.val}")
            return None

        js = res.solution.joint_state
        name_to_pos = dict(zip(js.name, js.position))

        try:
            q = [float(name_to_pos[n]) for n in UR10_JOINT_NAMES]
        except KeyError:
            self.get_logger().error("   ❌ IK: joint_state incomplet")
            return None

        return q

    # ---------- CARTESIAN PATH ----------
    def generate_cartesian_waypoints(self, start_pose_xyz, end_pose_xyz):
        waypoints = []
        sx, sy, sz = start_pose_xyz
        ex, ey, ez = end_pose_xyz

        for i in range(self.num_waypoints + 1):
            t = i / self.num_waypoints
            pose = PoseStamped()
            pose.header.frame_id = "world"

            pose.pose.position.x = sx + t * (ex - sx)
            pose.pose.position.y = sy + t * (ey - sy)
            pose.pose.position.z = self.workspace["z_fixed"]

            pose.pose.orientation.x = self.fixed_orientation[0]
            pose.pose.orientation.y = self.fixed_orientation[1]
            pose.pose.orientation.z = self.fixed_orientation[2]
            pose.pose.orientation.w = self.fixed_orientation[3]

            waypoints.append(pose.pose)

        return waypoints

    def plan_cartesian_path(self, start_pose_xyz, end_pose_xyz, q_start):
        request = GetCartesianPath.Request()
        request.header.frame_id = "world"
        request.group_name = "ur_manipulator"
        request.link_name = "tool0"

        request.start_state = self._robot_state_from_q(q_start)
        request.waypoints = self.generate_cartesian_waypoints(start_pose_xyz, end_pose_xyz)

        request.max_step = 0.005
        request.jump_threshold = 0.0
        request.avoid_collisions = True

        future = self._cartesian_path_client.call_async(request)
        ok = self._spin_until_done(future, timeout_sec=10.0)
        if not ok:
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

    def generate_dataset(self):
        self.get_logger().info("🚀 Début génération...")

        self.get_logger().info("🔍 Recherche service /compute_cartesian_path...")
        if not self._cartesian_path_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("❌ Service /compute_cartesian_path indisponible!")
            return
        self.get_logger().info("✅ Service cartésien trouvé")

        self.get_logger().info("🔍 Recherche service /compute_ik...")
        if not self._ik_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("❌ Service IK /compute_ik indisponible!")
            return
        self.get_logger().info("✅ Service IK trouvé\n")

        valid_trajectories = []     # list of (resample_len,6)
        valid_ee_trajectories = []  # list of (resample_len,3)

        attempts = 0
        max_attempts = self.num_trajectories * 5  # un peu plus car on filtre des lignes courtes

        self.get_logger().info(f"🎲 Génération de {self.num_trajectories} trajectoires aléatoires...")
        self.get_logger().info("=" * 70)

        while len(valid_trajectories) < self.num_trajectories and attempts < max_attempts:
            attempts += 1

            start_x = np.random.uniform(self.workspace["x_min"], self.workspace["x_max"])
            start_y = np.random.uniform(self.workspace["y_min"], self.workspace["y_max"])
            end_x = np.random.uniform(self.workspace["x_min"], self.workspace["x_max"])
            end_y = np.random.uniform(self.workspace["y_min"], self.workspace["y_max"])

            # --- NOUVEAU: filtre longueur minimale ---
            line_len = float(np.hypot(end_x - start_x, end_y - start_y))
            if line_len < self.min_line_len:
                continue

            start_pose = (float(start_x), float(start_y), float(self.workspace["z_fixed"]))
            end_pose = (float(end_x), float(end_y), float(self.workspace["z_fixed"]))

            self.get_logger().info(f"\n[{len(valid_trajectories)+1}/{self.num_trajectories}] Tentative {attempts}")
            self.get_logger().info(f"   Start: ({start_pose[0]:.3f}, {start_pose[1]:.3f})")
            self.get_logger().info(f"   End:   ({end_pose[0]:.3f}, {end_pose[1]:.3f})")
            self.get_logger().info(f"   Line length: {line_len:.3f} m")

            # 1) IK start
            q_start = self.compute_ik(start_pose[0], start_pose[1], start_pose[2], seed_q=self.seed_joint_state)
            if q_start is None:
                self.get_logger().warn("   ⚠️ REJETÉE (IK start impossible)")
                continue

            # 2) Cartesian path
            response = self.plan_cartesian_path(start_pose, end_pose, q_start=q_start)
            if response is None:
                self.get_logger().warn("   ⚠️ REJETÉE (collision/singularité/cartésien)")
                continue

            q_array = self.trajectory_to_numpy(response.solution)
            if q_array is None or len(q_array) < 2:
                self.get_logger().warn("   ⚠️ REJETÉE (traj trop courte)")
                continue

            # --- NOUVEAU: couper si aller-retour réel ---
            q_cut, did_cut = self.cut_if_roundtrip(q_array)
            if did_cut:
                self.get_logger().info(f"   ✂️ Aller-retour détecté: coupé à {len(q_cut)} points")
            q_array = q_cut

            if len(q_array) < 2:
                self.get_logger().warn("   ⚠️ REJETÉE (après coupe, trop courte)")
                continue

            # --- NOUVEAU: resample à longueur fixe ---
            q_res = self.resample_q(q_array, self.resample_len)
            if q_res is None:
                self.get_logger().warn("   ⚠️ REJETÉE (resample échoué)")
                continue

            # ee_pos idéale à longueur fixe (toujours une belle ligne)
            xs = np.linspace(start_pose[0], end_pose[0], self.resample_len, dtype=np.float32)
            ys = np.linspace(start_pose[1], end_pose[1], self.resample_len, dtype=np.float32)
            zs = np.full(self.resample_len, self.workspace["z_fixed"], dtype=np.float32)
            ee_res = np.stack([xs, ys, zs], axis=1)

            valid_trajectories.append(q_res)
            valid_ee_trajectories.append(ee_res)

            duration = self.resample_len * 0.02
            self.get_logger().info(f"   ✅ VALIDE: raw={len(q_array)} pts -> resampled={self.resample_len} pts")
            self.get_logger().info(f"   📊 ~{duration:.2f}s à 50Hz")

            # Seed IK = dernier point resamplé (aide la stabilité)
            self.seed_joint_state = list(map(float, q_res[-1]))

        self.get_logger().info("")
        self.get_logger().info("=" * 70)

        if len(valid_trajectories) < self.num_trajectories:
            self.get_logger().error(f"❌ Échec: seulement {len(valid_trajectories)}/{self.num_trajectories} générées")
            self.get_logger().error(f"   Tentatives totales: {attempts}")
            return

        self.get_logger().info(f"✅ {len(valid_trajectories)} trajectoires valides générées!\n")

        total_points = len(valid_trajectories) * self.resample_len
        avg_points = float(self.resample_len)
        total_duration = total_points * 0.02

        self.get_logger().info("📊 Statistiques du dataset:")
        self.get_logger().info(f"   Trajectoires: {len(valid_trajectories)}")
        self.get_logger().info(f"   Points totaux: {total_points}")
        self.get_logger().info(f"   Points/trajectoire (fixe): {avg_points:.1f}")
        self.get_logger().info(f"   Durée totale: {total_duration:.1f}s à 50Hz\n")

        self.get_logger().info("💾 Sauvegarde format Isaac Lab...")

        q_paths_array = np.array(valid_trajectories, dtype=np.float32)   # (N,64,6)
        ee_paths_array = np.array(valid_ee_trajectories, dtype=np.float32)  # (N,64,3)

        dataset = {
            "paths": q_paths_array,
            "ee_pos": ee_paths_array,
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
            "num_trajectories": int(len(q_paths_array)),
            "resample_len": int(self.resample_len),
            "min_line_len": float(self.min_line_len),
        }

        np.savez_compressed(self.output_file, **dataset)
        self.get_logger().info(f"✅ Sauvegardé: {self.output_file.absolute()}\n")
        self.get_logger().info("=" * 70)
        self.get_logger().info("✅ TERMINÉ")
        self.get_logger().info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Générateur de dataset - lignes droites aléatoires (avec IK + resample)")
    parser.add_argument("--num-traj", type=int, default=100, help="Nombre de trajectoires à générer")
    parser.add_argument("--output", type=str, default="random_lines_dataset.npz", help="Fichier de sortie")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("📏 GÉNÉRATEUR DATASET - LIGNES DROITES (IK + MIN LEN + RESAMPLE)")
    print("=" * 70)
    print(f"Nombre de trajectoires: {args.num_traj}")
    print(f"Fichier de sortie: {args.output}")
    print("=" * 70 + "\n")

    input("Appuyez sur ENTRÉE pour continuer...")

    rclpy.init()
    node = RandomLinesGenerator(num_trajectories=args.num_traj, output_file=args.output)

    from rclpy.executors import SingleThreadedExecutor
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    node.set_executor(executor)

    print("\n⏳ Initialisation ROS2 (3 secondes)...")
    start_time = time.time()
    while time.time() - start_time < 3.0:
        executor.spin_once(timeout_sec=0.1)
    print("✅ ROS2 prêt\n")

    node.generate_dataset()

    executor.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
