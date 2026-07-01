import pybullet as pb
import numpy as np
from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot
import pinocchio as pin

# For REEM-C robot
#urdf = "src/reemc_description/robots/reemc.urdf"
#path_meshes = "src/reemc_description/meshes/../.."

# For Talos robot
urdf = "src/talos_description/robots/talos_reduced.urdf"
path_meshes = "src/talos_description/meshes/../.."

'''
Talos
0, 1, 2, 3, 4, 5, 			    # left leg
6, 7, 8, 9, 10, 11, 			# right leg
12, 13,                         # torso
14, 15, 16, 17, 18, 19, 20, 21  # left arm
22, 23, 24, 25, 26, 27, 28, 29  # right arm
30, 31                          # head

REEMC
0, 1, 2, 3, 4, 5, 			    # left leg
6, 7, 8, 9, 10, 11, 			# right leg
12, 13,                         # torso
14, 15, 16, 17, 18, 19, 20,     # left arm
21, 22, 23, 24, 25, 26, 27,     # right arm
28, 29                          # head
'''

# Initial condition for the simulator an model
z_init = 1.15
q_actuated_home = np.zeros(32)

# Leg configurations (stable posture for Talos)
q_leg = np.array([0, 0, -0.44, 0.9, -0.45, 0])
q_actuated_home[:6] = q_leg        # Left leg
q_actuated_home[6:12] = q_leg      # Right leg

# Torso stays at zero (already initialized)
# q_actuated_home[12:14] = [0, 0]

# Arm configurations (stable posture for Talos)
q_arm = np.array([0, -0.24, 0, -1, 0, 0, 0, 0])
q_actuated_home[14:22] = q_arm     # Left arm
q_actuated_home[22:30] = q_arm     # Right arm

# Head stays at zero (already initialized)
# q_actuated_home[30:32] = [0, 0]

# Initialization position including floating base
q_home = np.hstack([np.array([0, 0, z_init, 0, 0, 0, 1]), q_actuated_home])

# DEBUG: Print the home configuration
print("\n=== HOME CONFIGURATION ===")
print("q_actuated_home:")
print(q_actuated_home)
print("\nq_home (with floating base):")
print(q_home)

# setup the task stack
modelWrap = pin.RobotWrapper.BuildFromURDF(urdf,                        # Model description
                                           path_meshes,                 # Model geometry descriptors 
                                           pin.JointModelFreeFlyer(),   # Floating base model. Use "None" if fixed
                                           True,                        # Printout model details
                                           None)                        # Load meshes different from the descripor
# Get model from wrapper
model = modelWrap.model

# setup the simulator
simulator = PybulletWrapper(sim_rate=1000)

#Create Pybullet-Pinocchio map
robot = Robot(simulator,            # The Pybullet wrapper
              urdf,                 # Robot descriptor
              model,                # Pinocchio model
              [0, 0, z_init],       # Floating base initial position
              [0,0,0,1],            # Floating base initial orientation [x,y,z,w]
              q=q_home,             # Initial state
              useFixedBase=False,   # Fixed base or not
              verbose=True)         # Printout details

data = robot._model.createData()

pin.crba(robot._model, data, robot.q())
mass_matrix = np.array(data.M)
non_linear_effects = pin.nonLinearEffects(robot._model, data, robot.q(), robot.v())

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

    # Use Pinocchio difference for generalized velocity (size model.nv)
    v_interp = pin.difference(model, q_interp, q_next) / dt

    # Extract actuated components (exclude floating base)
    q_d_actuated = q_interp[7:]
    v_d_actuated = v_interp[6:]

    return q_d_actuated, v_d_actuated

# PD Controller parameters
n_actuated = len(q_actuated_home)  # Number of actuated joints

# Create gain matrices with different values for legs and upper body
Kp = np.eye(n_actuated)
Kd = np.eye(n_actuated)

# Talos joint layout:
# 0-5: left leg, 6-11: right leg
# 12-13: torso, 14-21: left arm, 22-29: right arm, 30-31: head

# Legs (0-11): higher gains for stiff support
Kp[0:12, 0:12] *= 1200.0
Kd[0:12, 0:12] *= 3.0

# Upper body (12-31): lower gains
Kp[12:32, 12:32] *= 400.0
Kd[12:32, 12:32] *= 1.0

q_d = np.zeros(n_actuated)          # Desired joint configuration (zero)

# Trajectory parameters
q_ini_full = robot.q().copy()             # Initial full robot configuration
q_home_full = q_home.copy()               # Target full robot configuration
trajectory_time = 0.9                    # Time to reach home posture (seconds)
t_elapsed = 0.0                          # Elapsed time
dt = 1.0 / 1000.0                        # Simulation timestep (1000 Hz)

# Joint command vector
tau = q_actuated_home*0

done = False
iteration = 0
print_interval = 1000  # Print every 1000 iterations (every 1 second at 1000 Hz)

while not done:
    # update the simulator and the robot
    simulator.step()
    simulator.debug()
    robot.update()
    
    # Get current joint state (excluding floating base)
    q_current = robot.q()[7:]        # Skip floating base (7 DOF: 3 pos + 4 quat)
    v_current = robot.v()[6:]        # Skip floating base velocity (6 DOF)
    
    # Generate desired trajectory using spline interpolation
    q_d, v_d = spline_trajectory(q_ini_full, q_home_full, trajectory_time, t_elapsed, dt, model)
    
    # PD Controller with velocity feedforward: tau = Kp (q_d - q) + Kd (v_d - v)
    tau = Kp @ (q_d - q_current) + Kd @ (v_d - v_current)
    # 
    # command to the robot
    robot.setActuatedJointTorques(tau)
    
    # DEBUG: Print values at intervals
    if iteration % print_interval == 0:
        print(f"\n=== ITERATION {iteration} (t={t_elapsed:.2f}s) ===")
        print(f"q_d (Desired): {q_d}")
        print(f"q_current (Actual): {q_current}")
        print(f"Error (q_d - q): {q_d - q_current}")
        print(f"v_d (Desired vel): {v_d}")
        print(f"v_current (Actual vel): {v_current}")
    
    # Update elapsed time
    t_elapsed += dt
    iteration += 1