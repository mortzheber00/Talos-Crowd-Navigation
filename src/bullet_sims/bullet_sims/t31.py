import pybullet as pb
import numpy as np
from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot
import pinocchio as pin
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseStamped
from tf2_ros import TransformBroadcaster
from numpy.linalg import norm, solve


class JointSpaceController:
    def __init__(self, robot, Kp, Kd):
        self.robot = robot
        self.Kp = Kp
        self.Kd = Kd
        self.model = robot.wrapper.model
        self.data = robot.wrapper.data

    def update(self, q_r, q_r_dot, q_r_ddot):
        q = self.robot.q()
        v = self.robot.v()

        pin.crba(self.model, self.data, q)
        mass_matrix = np.array(self.data.M)
        h = np.array(pin.nonLinearEffects(self.model, self.data, q, v)).reshape(-1)

        feedback = q_r_ddot - self.Kd @ (v - q_r_dot) - self.Kp @ (q - q_r)
        tau = mass_matrix @ feedback + h
        return tau


class CartesianSpaceController:
    """Operational space (SE3) controller for end-effector tracking with position and orientation."""

    def __init__(self, robot, joint_name, Kp, Kd):
        """
        Initialize Cartesian space (SE3) controller.
        
        Args:
            robot: TalosRobot instance with Pinocchio wrapper
            joint_name: Name of the frame to control (e.g., 'arm_right_7_link')
            Kp: 6x6 proportional gain matrix (or scalar for identity). 
                Top-left 3x3 for position, bottom-right 3x3 for orientation.
            Kd: 6x6 derivative gain matrix (or scalar for identity).
                Top-left 3x3 for linear velocity, bottom-right 3x3 for angular velocity.
        """
        self.robot = robot
        self.joint_name = joint_name
        self.model = robot.wrapper.model
        self.data = robot.wrapper.data

        # Get joint ID from joint name (as required in the assignment)
        self.joint_id = self.model.getJointId(joint_name)
        if self.joint_id >= self.model.njoints:
            raise ValueError(f"Joint '{joint_name}' not found in Pinocchio model.")

        # Ensure Kp and Kd are 6x6 matrices
        if isinstance(Kp, (int, float)):
            self.Kp = Kp * np.eye(6)
        else:
            self.Kp = np.array(Kp)
            if self.Kp.shape != (6, 6):
                raise ValueError(f"Kp must be 6x6, got shape {self.Kp.shape}")
            
        if isinstance(Kd, (int, float)):
            self.Kd = Kd * np.eye(6)
        else:
            self.Kd = np.array(Kd)
            if self.Kd.shape != (6, 6):
                raise ValueError(f"Kd must be 6x6, got shape {self.Kd.shape}")

    def update(self, X_r, X_dot_r, X_ddot_r):
        """
        Compute torques for SE(3) Cartesian space trajectory tracking using inverse dynamics.
        
        Implements Eq. (7): τ = M(q) J^# (¨X_d - Ḋ_J q̇) + h
        
        Uses Joint Jacobian with Jlog6 transformation for proper SE(3) error handling.
        
        Where:
        - M(q) is the mass matrix
        - J^# is the Moore-Penrose pseudo-inverse of the transformed Jacobian
        - ¨X_d is the desired operational space acceleration (6D)
        - Ḋ_J is the time derivative of the Jacobian
        - h includes gravity and Coriolis/centrifugal effects
        
        Args:
            X_r: Desired end-effector pose (pin.SE(3))
            X_dot_r: Desired end-effector spatial twist (6,) [v_linear; omega]
            X_ddot_r: Desired end-effector spatial acceleration (6,) [a_linear; alpha]
        
        Returns:
            tau: Joint torques (n_actuated,)
        """
        q = self.robot.q()
        v = self.robot.v()

        # Compute forward kinematics
        pin.forwardKinematics(self.model, self.data, q, v)
        pin.updateFramePlacements(self.model, self.data)
        
        # Get current controlled-joint pose and compute pose error (joint frame)
        X_current = self.data.oMi[self.joint_id]
        iMd = X_current.actInv(X_r)

        # Joint Jacobian in joint frame
        # Use the actual joint Jacobian for the inverse-dynamics law.
        # The Jlog6 mapping is useful for IK error handling, but mixing it into
        # the torque law makes the velocity/acceleration terms inconsistent.
        J = pin.computeJointJacobian(self.model, self.data, q, self.joint_id)
        
        # Get current Cartesian pose and velocity
        X = self.data.oMi[self.joint_id]
        X_dot = J @ v  # Current end-effector spatial velocity (6D)
        
        # Compute desired Cartesian acceleration with feedback
        # Pose error in SE(3) using logarithmic map
        err_pose = pin.log(iMd).vector  # 6D pose error from logarithmic map
        
        # Velocity error
        err_velocity = X_dot_r - X_dot  # 6D velocity error
        
        # Desired acceleration with PD feedback: X_ddot_d = X_ddot_r + Kp*err_pose + Kd*err_velocity
        X_ddot_d = X_ddot_r + self.Kp @ err_pose + self.Kd @ err_velocity
        
        # Compute the term dJ*qdot using Pinocchio classical acceleration on the same joint
        # This keeps the term consistent with the joint-frame Jacobian used above.
        classical_acc = pin.getClassicalAcceleration(self.model, self.data, self.joint_id)
        dJ_v = classical_acc.vector
        
        # Compute mass matrix M(q)
        pin.crba(self.model, self.data, q)
        M = np.array(self.data.M)
        
        # Compute Coriolis/centrifugal and gravity effects
        h = np.array(pin.nonLinearEffects(self.model, self.data, q, v)).reshape(-1)
        
        # Use damped least-squares solve to compute J# * b via solving (J J^T + λI) x = b
        # v = - J.T.dot(solve(J.dot(J.T) + damp * I, err))
        damping = 1e-4  # Damping parameter λ
        JJT = J @ J.T

        # ---- IK mapping (task error -> joint velocity -> integrate q) ----
        # Compute task-space error (6D)
        err = pin.log(iMd).vector

        # Solve (J J^T + λ I) y = err and map back to joint velocities v = -J^T y
        y_err = solve(JJT + damping * np.eye(JJT.shape[0]), err)
        v_joint = - J.T.dot(y_err)

        # Integrate to get a candidate new configuration (optional debug/monitoring)
        DT = 1.0 / 1000.0
        _q_candidate = pin.integrate(self.model, q, v_joint * DT)

        # Map desired Cartesian acceleration to joint acceleration via damped LS
        control_input = X_ddot_d - dJ_v
        y_ctrl = solve(JJT + damping * np.eye(JJT.shape[0]), control_input)
        q_ddot_d = J.T.dot(y_ctrl)

        # Pull the mapped acceleration through the dynamics to get torques
        # τ = M(q) q_ddot_d + h
        tau = M.dot(q_ddot_d) + h

        return tau


class TalosRobot(Robot, Node):
    """Robot specialization that loads Talos URDF and stores Pinocchio wrapper."""

    URDF_PATH = "src/talos_description/robots/talos_reduced.urdf"
    MESHES_PATH = "src/talos_description/meshes/../.."

    def __init__(
        self,
        simulator,
        base_position,
        base_quaternion,
        q=None,
        verbose=True,
        print_pinocchio_details=True,
    ):
        Node.__init__(self, 'talos_robot_publisher')

        self.wrapper = pin.RobotWrapper.BuildFromURDF(
            self.URDF_PATH,
            self.MESHES_PATH,
            None,
            print_pinocchio_details,
            None,
        )
        self.model = self.wrapper.model
        self.data = self.wrapper.data

        super().__init__(
            simulator,
            filename=self.URDF_PATH,
            model=self.model,
            basePosition=base_position,
            baseQuationerion=base_quaternion,
            q=q,
            useFixedBase=True,
            verbose=verbose,
        )

        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

    def publish(self):
        stamp = self.get_clock().now().to_msg()

        joint_state_msg = JointState()
        joint_state_msg.header.stamp = stamp
        joint_state_msg.header.frame_id = 'world'
        joint_state_msg.name = self.actuatedJointNames()
        # Use actuatedJointPosition/Velocity (PyBullet order) not q()/v() (Pinocchio order)
        joint_state_msg.position = self.actuatedJointPosition().tolist()
        joint_state_msg.velocity = self.actuatedJointVelocity().tolist()
        joint_state_msg.effort = self._tau_cmd.tolist()
        self.joint_state_pub.publish(joint_state_msg)

        world_position, world_orientation = self.baseWorldPose()
        transform_msg = TransformStamped()
        transform_msg.header.stamp = stamp
        transform_msg.header.frame_id = 'world'
        transform_msg.child_frame_id = self.baseName()
        transform_msg.transform.translation.x = float(world_position[0])
        transform_msg.transform.translation.y = float(world_position[1])
        transform_msg.transform.translation.z = float(world_position[2])
        transform_msg.transform.rotation.x = float(world_orientation[0])
        transform_msg.transform.rotation.y = float(world_orientation[1])
        transform_msg.transform.rotation.z = float(world_orientation[2])
        transform_msg.transform.rotation.w = float(world_orientation[3])
        self.tf_broadcaster.sendTransform(transform_msg)

        # Publish all joint frames from Pinocchio
        q = self.q()
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        transforms = []
        for frame_id in range(len(self.model.names)):
            # Get frame name from Pinocchio model
            frame_name = self.model.names[frame_id]
            # Skip the universe frame
            if frame_name == 'universe':
                continue
            
            # Get frame pose from Pinocchio
            frame_placement = self.data.oMi[frame_id]
            position = frame_placement.translation
            quaternion = pin.Quaternion(frame_placement.rotation)
            
            # Create transform message
            transform_msg = TransformStamped()
            transform_msg.header.stamp = stamp
            transform_msg.header.frame_id = 'world'
            transform_msg.child_frame_id = frame_name
            transform_msg.transform.translation.x = float(position[0])
            transform_msg.transform.translation.y = float(position[1])
            transform_msg.transform.translation.z = float(position[2])
            transform_msg.transform.rotation.x = float(quaternion.x)
            transform_msg.transform.rotation.y = float(quaternion.y)
            transform_msg.transform.rotation.z = float(quaternion.z)
            transform_msg.transform.rotation.w = float(quaternion.w)
            
            transforms.append(transform_msg)
        
        # Send all transforms
        if transforms:
            self.tf_broadcaster.sendTransform(transforms)

    def update(self):
        super().update()
        self.wrapper.forwardKinematics(self.q(), self.v())

'''
Joint ordering confusion resolved:
- robot.q() stores internally in PINOCCHIO order
- controller.update() works with PINOCCHIO order (q_d from trajectory)
- Only when publishing to ROS (JointState) do we need PyBullet order

Internally use PINOCCHIO order:
0-5:    left leg (leg_left_1-6)
6-11:   right leg (leg_right_1-6)
12-13:  torso (torso_1, torso_2)
14-20:  left arm (arm_left_1-7)
21:     gripper_left
22-28:  right arm (arm_right_1-7)
29:     gripper_right
30-31:  head (head_1, head_2)
'''

# Initial condition for the simulator (in PINOCCHIO order)
z_init = 1.15
q_actuated_home = np.zeros(32)  # PINOCCHIO order

# Leg configurations (stable posture for Talos)
q_leg = np.array([0, 0, -0.44, 0.9, -0.45, 0])
q_actuated_home[0:6] = q_leg       # left leg (leg_left_1-6)
q_actuated_home[6:12] = q_leg      # right leg (leg_right_1-6)

# Torso stays at zero
# q_actuated_home[12:14] = [0, 0]

# Arm configurations (stable posture for Talos)
q_arm = np.array([0, -0.24, 0, -1, 0, 0, 0])
q_actuated_home[14:21] = q_arm     # left arm (arm_left_1-7)
q_actuated_home[21] = 0            # gripper_left
q_actuated_home[22:29] = q_arm     # right arm (arm_right_1-7)
q_actuated_home[29] = 0            # gripper_right

# Head stays at zero
# q_actuated_home[30:32] = [0, 0]

# Fixed-base initialization contains only actuated joints
q_home = q_actuated_home.copy()

# DEBUG: Print the home configuration
print("\n=== HOME CONFIGURATION ===")
print("q_actuated_home:")
print(q_actuated_home)
print("\nq_home (fixed base):")
print(q_home)

# setup the simulator
simulator = PybulletWrapper(sim_rate=1000)

rclpy.init()

# Create Talos robot with pybullet + pinocchio wrapper
robot = TalosRobot(simulator,
                   base_position=[0, 0, z_init],
                   base_quaternion=[0, 0, 0, 1],
                   q=q_home,
                   verbose=True,
                   print_pinocchio_details=True)

model = robot.wrapper.model
data = robot.wrapper.data
print(f"wrapper.model.nq: {model.nq}")

# Adjust q_actuated_home to match actual model dimension
n_model = model.nq
if n_model != len(q_actuated_home):
    print(f"Warning: q_actuated_home has {len(q_actuated_home)} joints, but model has {n_model}.")
    print(f"Resizing q_actuated_home from {len(q_actuated_home)} to {n_model} with zero padding.")
    q_actuated_home_new = np.zeros(n_model)
    q_actuated_home_new[:len(q_actuated_home)] = q_actuated_home
    q_actuated_home = q_actuated_home_new
    q_home = q_actuated_home.copy()
    print(f"New q_actuated_home size: {len(q_actuated_home)}")

pin.crba(model, data, robot.q())
mass_matrix = np.array(data.M)
non_linear_effects = pin.nonLinearEffects(model, data, robot.q(), robot.v())

print("Complete Inertia Matrix:")
print(mass_matrix)

print("Non Linear Effects:")
print(np.array(non_linear_effects))

#Needed for compatibility
simulator.addLinkDebugFrame(-1,-1)

# Setup pybullet camera
pb.resetDebugVisualizerCamera(
    cameraDistance=1.2,
    cameraYaw=90,
    cameraPitch=-20,
    cameraTargetPosition=[0.0, 0.0, 0.8])

# Spline interpolation function
def spline_trajectory(q_ini_full, q_home_full, t_total, t_current, dt, model):
    """
    Interpolate joint positions from initial state to home posture.
    Returns both position and velocity for trajectory tracking.
    
    Args:
        q_ini_full: Initial full robot configuration (size model.nq)
        q_home_full: Home/target full robot configuration (size model.nq)
        t_total: Total time for the interpolation (seconds)
        t_current: Current time elapsed (seconds)
        dt: Time step for numerical differentiation
        model: Pinocchio robot model
    
    Returns:
        q_d_actuated: Desired actuated joint positions
        v_d_actuated: Desired actuated joint velocities
        a_d_actuated: Desired actuated joint accelerations
    """
    # Clamp time to [0, t_total]
    t = np.clip(t_current, 0.0, t_total)
    
    # Compute interpolation parameter (0 to 1)
    alpha = t / t_total
    
    # Use Pinocchio's interpolate function for smooth interpolation
    q_interp = pin.interpolate(model, q_ini_full, q_home_full, alpha)
    
    # Compute desired velocity using finite differences
    alpha_next = np.clip(t_current + dt, 0.0, t_total) / t_total
    q_next = pin.interpolate(model, q_ini_full, q_home_full, alpha_next)
    alpha_next2 = np.clip(t_current + 2.0 * dt, 0.0, t_total) / t_total
    q_next2 = pin.interpolate(model, q_ini_full, q_home_full, alpha_next2)

    # Use Pinocchio difference for generalized velocity (size model.nv)
    v_interp = pin.difference(model, q_interp, q_next) / dt
    v_next = pin.difference(model, q_next, q_next2) / dt
    a_interp = (v_next - v_interp) / dt

    # Fixed-base model: full vectors are actuated vectors
    q_d_actuated = q_interp
    v_d_actuated = v_interp
    a_d_actuated = a_interp

    return q_d_actuated, v_d_actuated, a_d_actuated

# PD Controller parameters
n_actuated = len(q_actuated_home)  # Number of actuated joints

# Create gain matrices with correct dimension
Kp = np.eye(n_actuated)
Kd = np.eye(n_actuated)

# Talos joint layout (PINOCCHIO order - internal to robot.q()):
# 0-5: left leg, 6-11: right leg
# 12-13: torso, 14-20: left arm, 21: gripper_left
# 22-28: right arm, 29: gripper_right
# 30-31: head

# Legs (0-11): higher gains for stiff support
Kp[0:12, 0:12] *= 1200.0
Kd[0:12, 0:12] *= 3.0

# Torso, arms, grippers (12-29): medium-to-lower gains
Kp[12:30, 12:30] *= 400.0
Kd[12:30, 12:30] *= 1.0

# Head (30-31): lower gains
Kp[30:32, 30:32] *= 100.0
Kd[30:32, 30:32] *= 0.5

q_d = np.zeros(n_actuated)          # Desired joint configuration (zero)

controller = JointSpaceController(robot, Kp, Kd)

# Cartesian controller for right hand (will be activated after spline finishes)
Kp_cart = 1000.0
Kd_cart = 100.0
cart_controller = CartesianSpaceController(robot, 'arm_right_7_joint', Kp_cart, Kd_cart)
cart_mode = False
X_goal = None


# Subscriber callback to update Cartesian goal from interactive marker
def goal_pose_callback(msg: PoseStamped):
    global X_goal, cart_mode
    p = msg.pose.position
    q = msg.pose.orientation
    try:
        # Try using Pinocchio helper if available
        if hasattr(pin, 'XYZQUATToSE3'):
            X_goal = pin.XYZQUATToSE3([p.x, p.y, p.z, q.x, q.y, q.z, q.w])
        else:
            # Build SE3 manually: rotation matrix from quaternion
            quat = np.array([q.x, q.y, q.z, q.w])
            R = pin.Quaternion(quat).matrix
            X_goal = pin.SE3(R, np.array([p.x, p.y, p.z]))
        cart_mode = True
        print(f"Received goal pose from topic: ({p.x:.4f}, {p.y:.4f}, {p.z:.4f})")
    except Exception as e:
        print(f"Failed to convert PoseStamped to SE3: {e}")

# Create subscription on the robot node to receive goal poses
# `robot` is an rclpy Node (TalosRobot inherits Node)
robot.create_subscription(PoseStamped, 'goal_pose', goal_pose_callback, 10)

# Trajectory parameters
q_ini_full = robot.q().copy()             # Initial full robot configuration
q_home_full = q_home.copy()               # Target full robot configuration
q_posture_ref = q_home_full.copy()        # Posture task reference qr
trajectory_time = 0.9                    # Time to reach home posture (seconds)
t_elapsed = 0.0                          # Elapsed time
dt = 1.0 / 1000.0                        # Simulation timestep (1000 Hz)

# Joint command vector
tau = q_actuated_home*0

done = False
iteration = 0
print_interval = 1000  # Print every 1000 iterations (every 1 second at 1000 Hz)
cart_debug_interval = 100  # Debug prints every 100 iterations in Cartesian mode

while not done:
    # update the simulator and the robot
    simulator.step()
    simulator.debug()
    robot.update()
    # Process incoming ROS messages so subscription callbacks run
    rclpy.spin_once(robot, timeout_sec=0.0)
    
    # Generate desired trajectory using spline interpolation
    q_d, v_d, q_dd = spline_trajectory(q_ini_full, q_home_full, trajectory_time, t_elapsed, dt, model)
    
    # Switch from joint-space spline to Cartesian control after trajectory_time
    if t_elapsed < trajectory_time:
        # Computed-torque controller in Eq. (4) during spline
        tau = controller.update(q_d, v_d, q_dd)
    else:
        # On first entry to Cartesian mode, capture current hand transform as goal
        if not cart_mode:
            cart_mode = True
            q_goal = q_home_full.copy()
            v_zero = np.zeros(model.nv)
            pin.forwardKinematics(model, data, q_goal, v_zero)
            pin.updateFramePlacements(model, data)
            X_goal = data.oMi[cart_controller.joint_id].copy()
            print("\n=== CARTESIAN MODE ACTIVATED ===")
            print(f"joint_name: {cart_controller.joint_name}")
            print(f"joint_id: {cart_controller.joint_id}")
            print(f"X_goal: {X_goal}")
        # Hold the hand at X_goal (primary task)
        X_r = X_goal
        X_dot_r = np.zeros(6)
        X_ddot_r = np.zeros(6)

        # Primary Cartesian tracking torque τ_cart
        tau_cart = cart_controller.update(X_r, X_dot_r, X_ddot_r)

        # Secondary posture torque τ_posture = -Kp(q-qr) - Kd*qdot
        q_current = robot.q()
        v_current = robot.v()
        tau_posture = -Kp @ (q_current - q_posture_ref) - Kd @ v_current

        # Nullspace projector N = I - J^T (J#)^T (damped pseudo-inverse)
        pin.forwardKinematics(model, data, q_current, v_current)
        pin.updateFramePlacements(model, data)
        J = pin.computeJointJacobian(model, data, q_current, cart_controller.joint_id)
        damping_ns = 1e-4
        JJT = J @ J.T
        J_pinv = J.T @ solve(JJT + damping_ns * np.eye(JJT.shape[0]), np.eye(JJT.shape[0]))
        N = np.eye(model.nv) - J.T @ J_pinv.T

        # Total torque with nullspace posture task
        tau = tau_cart + N @ tau_posture

        if iteration % cart_debug_interval == 0:
            X_current_dbg = data.oMi[cart_controller.joint_id].copy()
            iMd_dbg = X_current_dbg.actInv(X_r)
            err_dbg = pin.log(iMd_dbg).vector
            err_trans_dbg = norm(err_dbg[:3])
            err_rot_dbg = norm(err_dbg[3:])
            print(f"\n--- CART DEBUG iter={iteration} t={t_elapsed:.3f}s ---")
            print(f"X_current: {X_current_dbg}")
            print(f"X_goal:    {X_goal}")
            print(f"err_trans norm: {err_trans_dbg:.6f}")
            print(f"err_rot norm:   {err_rot_dbg:.6f}")
            print(f"err_pose norm: {norm(err_dbg):.6f}")
            print(f"err_pose: {err_dbg}")
            print(f"X_dot_r: {X_dot_r}")
            print(f"X_ddot_r: {X_ddot_r}")
            print(f"tau_cart norm: {norm(tau_cart):.6f}")
            print(f"tau_posture norm: {norm(tau_posture):.6f}")
            print(f"tau min/max: {np.min(tau):.6f} / {np.max(tau):.6f}")
            print(f"tau norm: {norm(tau):.6f}")
            print(f"tau: {tau}")

    # command to the robot
    robot.setActuatedJointTorques(tau)

    robot.publish()
    
    # Update elapsed time
    t_elapsed += dt
    iteration += 1