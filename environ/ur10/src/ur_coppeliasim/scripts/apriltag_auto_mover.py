#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import tf2_ros

from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import Bool

from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import (
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    JointConstraint,
    BoundingVolume,
    RobotTrajectory
)
from moveit_msgs.srv import GetCartesianPath
from shape_msgs.msg import SolidPrimitive

import traceback


class AprilTagLineDrawer(Node):
    def __init__(self):
        super().__init__("apriltag_line_drawer_cartesian")

        # Param
        self.declare_parameter("group_name", "ur_manipulator")
        self.declare_parameter("planning_time", 15.0)
        self.declare_parameter("velocity_scaling", 0.2)  # Plus lent pour traçage précis
        self.declare_parameter("z_offset", 0.50)
        self.declare_parameter("num_tags", 2)
        self.declare_parameter("cartesian_steps", 50)  # Nombre de points intermédiaires

        self.group_name = self.get_parameter("group_name").value
        self.planning_time = self.get_parameter("planning_time").value
        self.velocity_scaling = self.get_parameter("velocity_scaling").value
        self.z_offset = self.get_parameter("z_offset").value
        self.num_tags = self.get_parameter("num_tags").value
        self.cartesian_steps = self.get_parameter("cartesian_steps").value

        self.current_target = 0
        self.motion_trigger = False
        self.moving = False
        self.fixed_z = None
        self.fixed_orientation = None
        self.first_move_done = False
        self.last_pose = None

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Trigger Isaac
        self.create_subscription(Bool, "/start_motion", self.trigger_cb, 10)

        # MoveIt Action
        self.client = ActionClient(self, MoveGroup, "/move_action")
        
        # Action client pour exécution de trajectoire
        self.execute_client = ActionClient(self, ExecuteTrajectory, "/execute_trajectory")
        
        # Service pour Cartesian Path
        self.cartesian_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')

        # Loop timer
        self.create_timer(0.5, self.loop)

        self.get_logger().info("🖊️ Line Drawer CARTESIAN (ligne droite X-Y) prêt !")


    def trigger_cb(self, msg):
        if msg.data:
            self.motion_trigger = True
            self.get_logger().info("🎮 Trigger reçu !")


    def loop(self):
        if not self.motion_trigger or self.moving:
            return
        
        if not self.client.server_is_ready():
            self.get_logger().warn("MoveIt pas prêt...")
            return

        self.motion_trigger = False

        try:
            # cible = Tag0 ou Tag1
            tag = f"Tag{self.current_target}"
            self.get_logger().info(f"📡 Lecture TF : {tag}")

            try:
                tf = self.tf_buffer.lookup_transform(
                    "world", tag, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.3)
                )
            except:
                self.get_logger().warn(f"Pas de TF pour {tag}")
                return

            # Premier mouvement : fixer Z et orientation
            if not self.first_move_done:
                self.fixed_z = tf.transform.translation.z + self.z_offset
                # Orientation par défaut (stylo vertical vers le bas)
                self.fixed_orientation = Pose().orientation
                self.fixed_orientation.x = 1.0
                self.fixed_orientation.y = 0.0
                self.fixed_orientation.z = 0.0
                self.fixed_orientation.w = 0.0
                self.first_move_done = True
                self.get_logger().info(f"🔒 Z fixé = {self.fixed_z:.3f}")

            # Pose cible
            target_pose = PoseStamped()
            target_pose.header.frame_id = "world"
            target_pose.pose.position.x = tf.transform.translation.x
            target_pose.pose.position.y = tf.transform.translation.y
            target_pose.pose.position.z = self.fixed_z
            target_pose.pose.orientation = self.fixed_orientation

            # Si c'est le premier mouvement, pas de ligne à tracer
            if self.last_pose is None:
                self.get_logger().info("➡️ Premier point : mouvement direct")
                self.send_goal_direct(target_pose)
            else:
                # Tracer une ligne droite X-Y depuis last_pose vers target_pose
                self.get_logger().info("📏 Traçage ligne cartésienne...")
                self.send_cartesian_line(self.last_pose, target_pose)
            
            self.last_pose = target_pose

        except Exception as e:
            self.get_logger().error(traceback.format_exc())


    def send_cartesian_line(self, start_pose, end_pose):
        """Trace une ligne droite cartésienne de start à end (Z et orientation fixes)"""
        self.moving = True
        
        # Générer des waypoints intermédiaires
        waypoints = []
        for i in range(self.cartesian_steps + 1):
            t = i / self.cartesian_steps
            waypoint = Pose()
            waypoint.position.x = start_pose.pose.position.x + t * (end_pose.pose.position.x - start_pose.pose.position.x)
            waypoint.position.y = start_pose.pose.position.y + t * (end_pose.pose.position.y - start_pose.pose.position.y)
            waypoint.position.z = self.fixed_z  # Z toujours fixe
            waypoint.orientation = self.fixed_orientation  # Orientation toujours fixe
            waypoints.append(waypoint)
        
        self.get_logger().info(f"📐 {len(waypoints)} waypoints générés")
        
        # Appeler le service GetCartesianPath
        request = GetCartesianPath.Request()
        request.header.frame_id = "world"
        request.group_name = self.group_name
        request.link_name = "tool0"
        request.waypoints = waypoints
        request.max_step = 0.01  # 1cm de résolution
        request.jump_threshold = 0.0  # Pas de saut
        request.avoid_collisions = True
        
        future = self.cartesian_client.call_async(request)
        future.add_done_callback(self.cartesian_response)


    def cartesian_response(self, future):
        """Callback après calcul du chemin cartésien"""
        try:
            response = future.result()
            fraction = response.fraction
            
            self.get_logger().info(f"📊 Chemin cartésien calculé : {fraction*100:.1f}% couvert")
            
            if fraction < 0.95:  # Si moins de 95% du chemin est faisable
                self.get_logger().warn(f"⚠️ Chemin incomplet ({fraction*100:.1f}%), mouvement direct")
                # Fallback : mouvement direct vers la cible
                self.moving = False
                return
            
            # Exécuter la trajectoire cartésienne via ExecuteTrajectory action
            goal = ExecuteTrajectory.Goal()
            goal.trajectory = response.solution
            
            self.get_logger().info(f"🚀 Exécution trajectoire ({len(response.solution.joint_trajectory.points)} points)...")
            fut = self.execute_client.send_goal_async(goal)
            fut.add_done_callback(self.execute_goal_response)
            
        except Exception as e:
            self.get_logger().error(f"❌ Erreur cartesian path: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            self.moving = False


    def execute_goal_response(self, future):
        """Callback après acceptation du goal ExecuteTrajectory"""
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error("❌ Trajectoire rejetée !")
            self.moving = False
            return

        self.get_logger().info("⏳ Exécution en cours...")
        handle.get_result_async().add_done_callback(self.execute_result_cb)


    def execute_result_cb(self, future):
        """Callback après exécution de la trajectoire"""
        result = future.result().result.error_code.val

        if result == 1:
            self.get_logger().info("✅ Ligne tracée avec succès !")
            self.current_target = (self.current_target + 1) % self.num_tags
        else:
            self.get_logger().error(f"❌ Erreur exécution : {result}")

        self.moving = False


    def send_goal_direct(self, pose):
        """Mouvement direct (sans ligne) vers une pose"""
        self.moving = True
        goal = MoveGroup.Goal()

        goal.request.group_name = self.group_name
        goal.request.allowed_planning_time = self.planning_time
        goal.request.max_velocity_scaling_factor = self.velocity_scaling
        goal.request.num_planning_attempts = 20

        goal.request.planner_id = "RRTConnect"
        
        # Goal simple : position + orientation fixe
        goal_constraint = Constraints()
        
        pos_constraint = PositionConstraint()
        pos_constraint.header = pose.header
        pos_constraint.link_name = "tool0"
        
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.03]
        
        target_region = BoundingVolume()
        target_region.primitives.append(sphere)
        target_region.primitive_poses.append(pose.pose)
        
        pos_constraint.constraint_region = target_region
        pos_constraint.weight = 1.0
        goal_constraint.position_constraints.append(pos_constraint)
        
        ori_constraint = OrientationConstraint()
        ori_constraint.header = pose.header
        ori_constraint.link_name = "tool0"
        ori_constraint.orientation = pose.pose.orientation
        ori_constraint.absolute_x_axis_tolerance = 0.1
        ori_constraint.absolute_y_axis_tolerance = 0.1
        ori_constraint.absolute_z_axis_tolerance = 0.1
        ori_constraint.weight = 1.0
        goal_constraint.orientation_constraints.append(ori_constraint)
        
        goal.request.goal_constraints.append(goal_constraint)
        goal.planning_options.plan_only = False

        fut = self.client.send_goal_async(goal)
        fut.add_done_callback(self.goal_response)


    def goal_response(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error("❌ Goal rejeté !")
            self.moving = False
            return

        self.get_logger().info("🧠 MoveIt planifie…")
        handle.get_result_async().add_done_callback(self.result_cb)


    def result_cb(self, future):
        result = future.result().result.error_code.val

        if result == 1:
            self.get_logger().info("✅ Point atteint")
            self.current_target = (self.current_target + 1) % self.num_tags
        else:
            self.get_logger().error(f"❌ MoveIt erreur : {result}")

        self.moving = False



def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(AprilTagLineDrawer())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
