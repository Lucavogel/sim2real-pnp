#!/usr/bin/env python3
"""
Dataset lignes droites 2D (MoveIt2) avec:
- IK start (/compute_ik)
- compute_cartesian_path
- filtre min longueur
- coupe aller-retour robuste
- resample à longueur fixe (64 points)
- unwrap + filtre anti-sauts joint-space après resample (max dq en rad/s)
- sauvegarde .npz ROBUSTE (np.stack) pour éviter dtype=object

⚙️ IMPORTANT: Dataset réglé pour 25 Hz
- dt = 0.04 s
- 64 points => 2.56 s par trajectoire

Sortie:
- paths:      (N, 64, 6) float32
- ee_ref_pos: (N, 64, 3) float32  (référence cartésienne, pas FK réel)
- dt/control_hz + metadata

Usage:
  ros2 launch ur_coppeliasim ur_isaaclab_moveit.launch.py
  python3 generate_moveit_dataset_v2.py --num-traj 100 --output dataset.npz
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
    def __init__(self, num_trajectories=50, output_file="dataset.npz"):
        super().__init__("random_lines_generator")

        self.num_trajectories = int(num_trajectories)
        self.output_file = Path(output_file)

        # ---------------- Dataset rate: 12 Hz (IDENTIQUE À ISAAC LAB) ----------------
        # Isaac Lab: sim.dt=1/60, decimation=5 => control freq = 60/5 = 12 Hz
        self.control_hz = 12
        self.dt = 1.0 / float(self.control_hz)  # 0.0833 s

        # Workspace (adapte si ton frame "world" est différent)
        self.workspace = {
            "x_min": 0.7,
            "x_max": 1.000,
            "y_min": -0.2,
            "y_max": 0.2,
            "z_fixed": 0.20,
        }

        # Dataset constraints
        self.min_line_len = 0.20  # ✅ Réduit de 0.30 à 0.20 pour plus de flexibilité
        self.resample_len = 64  # ✅ 64 points à 25 Hz => 2.56 s
        self.num_segments = 2  # ✅ Nombre de droites par trajectoire

        # ---------------- dq filter en rad/s (stable quel que soit Hz) ----------------
        # dq/step = dq_rad_s * dt
        self.max_dq_rad_s = 8.0  # ✅ Augmenté de 6.0 à 8.0 pour accepter plus de variations
        self.max_dq_per_step = float(self.max_dq_rad_s * self.dt)  # rad/step
        self.max_dq_rad_s_joints = None  # option: [thr0..thr5] en rad/s

        # Clients MoveIt
        self._cartesian_path_client = self.create_client(GetCartesianPath, "/compute_cartesian_path")
        self._ik_client = self.create_client(GetPositionIK, "/compute_ik")

        self.executor = None

        # Orientation tool vers le bas (x,y,z,w)
        self.fixed_orientation = [1.0, 0.0, 0.0, 0.0]

        # Seed IK (home)
        self.seed_joint_state = [0.0, -1.309, 2.14675, -2.44346, -1.5708, 0.0]

        # Waypoints demandés (n’impose pas le nb de points retournés)
        self.num_waypoints = 64

        self.get_logger().info("📏 Dataset lignes droites (IK + resample + unwrap + dq-filter + stack-save)")
        self.get_logger().info(f"   Control rate: {self.control_hz} Hz (dt={self.dt:.3f}s)")
        self.get_logger().info(f"   Min line len: {self.min_line_len:.2f} m")
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

    # ---------------- Resampling ----------------
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

    def resample_ee(self, ee_array, out_len):
        """Resample cartesian positions (same logic as joints)"""
        if ee_array is None or len(ee_array) < 2:
            return None
        T = ee_array.shape[0]
        x_old = np.linspace(0.0, 1.0, T, dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, out_len, dtype=np.float32)

        ee_new = np.zeros((out_len, 3), dtype=np.float32)
        for j in range(3):
            ee_new[:, j] = np.interp(x_new, x_old, ee_array[:, j]).astype(np.float32)
        return ee_new

    # ---------------- unwrap pour éviter faux sauts 2π ----------------
    def unwrap_joints(self, q_res):
        return np.unwrap(q_res, axis=0).astype(np.float32)

    # ---------------- dq filter ----------------
    def passes_dq_filter(self, q_res_unwrapped):
        if q_res_unwrapped is None or len(q_res_unwrapped) < 2:
            return False
        dq = np.abs(np.diff(q_res_unwrapped, axis=0))  # (T-1,6)

        if self.max_dq_rad_s_joints is not None:
            thr = (np.array(self.max_dq_rad_s_joints, dtype=np.float32) * self.dt).reshape(1, 6)
            return bool(np.all(dq <= thr))
        else:
            return bool(np.max(dq) <= float(self.max_dq_per_step))

    # ------------- Aller-retour robuste -------------
    def cut_if_roundtrip(self, q_array):
        if q_array is None or len(q_array) < 3:
            return q_array, False

        start = q_array[0]
        d = np.linalg.norm(q_array - start, axis=1)
        max_idx = int(np.argmax(d))
        max_d = float(d[max_idx])
        end_d = float(d[-1])

        min_excursion = 0.15
        return_ratio = 0.50

        if max_d > min_excursion and end_d < (return_ratio * max_d) and max_idx >= 2:
            return q_array[: max_idx + 1], True
        return q_array, False

    # ---------------- IK ----------------
    def compute_ik(self, x, y, z, seed_q=None):
        req = GetPositionIK.Request()
        req.ik_request = PositionIKRequest()
        req.ik_request.group_name = "ur_manipulator"
        req.ik_request.ik_link_name = "wrist_3_link"

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

    # ---------- Waypoints cartésiens ----------
    def generate_cartesian_waypoints(self, start_pose_xyz, end_pose_xyz):
        sx, sy, sz = start_pose_xyz
        ex, ey, ez = end_pose_xyz

        waypoints = []
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

    # ---------- Cartesian path ----------
    def plan_cartesian_path(self, start_pose_xyz, end_pose_xyz, q_start):
        request = GetCartesianPath.Request()
        request.header.frame_id = "world"
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

    # ---------------- Save helpers (anti dtype=object) ----------------
    def _stack_or_die(self, seq, expected_shape, name):
        """
        Force un tableau dense float32:
        - vérifie que chaque élément est np.ndarray de shape expected_shape
        - cast float32 si besoin
        - np.stack => garantit dtype numérique (jamais object)
        """
        fixed = []
        for i, arr in enumerate(seq):
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"{name}[{i}] n'est pas np.ndarray: {type(arr)}")
            if arr.shape != expected_shape:
                raise ValueError(f"{name}[{i}] shape={arr.shape}, attendu {expected_shape}")
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32, copy=False)
            fixed.append(arr)
        return np.stack(fixed, axis=0).astype(np.float32, copy=False)

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

        valid_trajectories = []
        valid_ee_ref_trajectories = []

        attempts = 0
        max_attempts = self.num_trajectories * 20  # ✅ Augmenté de 12 à 20 tentatives par trajectoire

        self.get_logger().info(f"🎲 Génération de {self.num_trajectories} trajectoires aléatoires...")
        self.get_logger().info("=" * 70)

        while len(valid_trajectories) < self.num_trajectories and attempts < max_attempts:
            attempts += 1

            # Générer N+1 points pour N segments (ici 3 points pour 2 segments)
            waypoints = []
            for i in range(self.num_segments + 1):
                x = np.random.uniform(self.workspace["x_min"], self.workspace["x_max"])
                y = np.random.uniform(self.workspace["y_min"], self.workspace["y_max"])
                waypoints.append((float(x), float(y), float(self.workspace["z_fixed"])))

            # Vérifier que chaque segment respecte la longueur minimale
            valid_segments = True
            for i in range(len(waypoints) - 1):
                line_len = float(np.hypot(waypoints[i+1][0] - waypoints[i][0], 
                                         waypoints[i+1][1] - waypoints[i][1]))
                if line_len < self.min_line_len:
                    valid_segments = False
                    break
            
            if not valid_segments:
                continue

            self.get_logger().info(f"\n[{len(valid_trajectories)+1}/{self.num_trajectories}] Tentative {attempts}")
            for i, wp in enumerate(waypoints):
                self.get_logger().info(f"   Point {i}: ({wp[0]:.3f}, {wp[1]:.3f})")

            # Planifier chaque segment et les concaténer
            q_segments = []
            current_seed = self.seed_joint_state
            
            for i in range(len(waypoints) - 1):
                start_pose = waypoints[i]
                end_pose = waypoints[i + 1]
                
                # IK pour le point de départ
                q_start = self.compute_ik(start_pose[0], start_pose[1], start_pose[2], seed_q=current_seed)
                if q_start is None:
                    self.get_logger().warn(f"   ⚠️ REJETÉE (IK segment {i} start impossible)")
                    break

                # Planifier le segment
                response = self.plan_cartesian_path(start_pose, end_pose, q_start=q_start)
                if response is None:
                    self.get_logger().warn(f"   ⚠️ REJETÉE (segment {i} cartésien échoué)")
                    break

                q_array = self.trajectory_to_numpy(response.solution)
                if q_array is None or len(q_array) < 2:
                    self.get_logger().warn(f"   ⚠️ REJETÉE (segment {i} trop court)")
                    break
                
                q_segments.append(q_array)
                current_seed = list(map(float, q_array[-1]))
            
            # Vérifier que tous les segments ont été planifiés
            if len(q_segments) != self.num_segments:
                continue
            
            # Concaténer tous les segments
            q_array = np.vstack(q_segments)
            self.get_logger().info(f"   ✅ {self.num_segments} segments planifiés: total {len(q_array)} points")

            q_cut, did_cut = self.cut_if_roundtrip(q_array)
            if did_cut:
                self.get_logger().info(f"   ✂️ Aller-retour détecté: coupé à {len(q_cut)} points")
            q_array = q_cut

            if len(q_array) < 2:
                self.get_logger().warn("   ⚠️ REJETÉE (après coupe, trop courte)")
                continue

            q_res = self.resample_q(q_array, self.resample_len)
            if q_res is None:
                self.get_logger().warn("   ⚠️ REJETÉE (resample échoué)")
                continue

            # ✅ unwrap puis dq-filter
            q_unwrap = self.unwrap_joints(q_res)
            if not self.passes_dq_filter(q_unwrap):
                dq_max = float(np.max(np.abs(np.diff(q_unwrap, axis=0))))
                self.get_logger().warn(
                    f"   ⚠️ REJETÉE (dq trop grand: {dq_max:.3f} rad/step | thr={self.max_dq_per_step:.3f})"
                )
                continue

            # ✅ Référence cartésienne (interpolation linéaire pour tous les segments)
            ee_ref_list = []
            for i in range(len(waypoints) - 1):
                num_points = len(q_segments[i])
                xs = np.linspace(waypoints[i][0], waypoints[i+1][0], num_points, dtype=np.float32)
                ys = np.linspace(waypoints[i][1], waypoints[i+1][1], num_points, dtype=np.float32)
                zs = np.full(num_points, self.workspace["z_fixed"], dtype=np.float32)
                ee_ref_list.append(np.stack([xs, ys, zs], axis=1).astype(np.float32))
            
            ee_ref = np.vstack(ee_ref_list)
            
            # ✅ Resample ee_ref à la même longueur que q_res
            ee_ref_resampled = self.resample_ee(ee_ref, self.resample_len)
            if ee_ref_resampled is None:
                self.get_logger().warn("   ⚠️ REJETÉE (resample ee_ref échoué)")
                continue

            # IMPORTANT: on stocke q_res (pas unwrap) comme angles "standards"
            valid_trajectories.append(q_res.astype(np.float32, copy=False))
            valid_ee_ref_trajectories.append(ee_ref_resampled)

            duration = self.resample_len * self.dt
            self.get_logger().info(f"   ✅ VALIDE: raw={len(q_array)} pts -> resampled={self.resample_len} pts")
            self.get_logger().info(f"   📊 ~{duration:.2f}s à {self.control_hz}Hz")

            self.seed_joint_state = list(map(float, q_res[-1]))

        self.get_logger().info("")
        self.get_logger().info("=" * 70)

        if len(valid_trajectories) < self.num_trajectories:
            self.get_logger().error(f"❌ Échec: seulement {len(valid_trajectories)}/{self.num_trajectories} générées")
            self.get_logger().error(f"   Tentatives totales: {attempts}")
            return

        self.get_logger().info(f"✅ {len(valid_trajectories)} trajectoires valides générées!\n")

        # ✅ Stack robuste => jamais dtype=object
        q_paths_array = self._stack_or_die(valid_trajectories, (self.resample_len, 6), "paths")
        ee_ref_array = self._stack_or_die(valid_ee_ref_trajectories, (self.resample_len, 3), "ee_ref_pos")

        total_points = q_paths_array.shape[0] * q_paths_array.shape[1]
        total_duration = total_points * self.dt

        self.get_logger().info("📊 Statistiques du dataset:")
        self.get_logger().info(f"   Trajectoires: {q_paths_array.shape[0]}")
        self.get_logger().info(f"   Points totaux: {total_points}")
        self.get_logger().info(f"   Points/trajectoire (fixe): {q_paths_array.shape[1]}")
        self.get_logger().info(f"   Durée totale: {total_duration:.1f}s à {self.control_hz}Hz\n")

        self.get_logger().info("💾 Sauvegarde format Isaac Lab...")

        dataset = {
            "paths": q_paths_array,            # (N,64,6) float32
            "ee_ref_pos": ee_ref_array,        # (N,64,3) float32
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
            "num_trajectories": np.int32(q_paths_array.shape[0]),
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
        self.get_logger().info("=" * 70)
        self.get_logger().info("✅ TERMINÉ")
        self.get_logger().info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Générateur dataset - lignes droites (MoveIt2) @25Hz (stack-save anti object dtype)"
    )
    parser.add_argument("--num-traj", type=int, default=500, help="Nombre de trajectoires à générer")
    parser.add_argument("--output", type=str, default="dataset_2droites.npz", help="Fichier de sortie")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("📏 GÉNÉRATEUR DATASET - LIGNES DROITES (MOVEIT2) @25Hz")
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
