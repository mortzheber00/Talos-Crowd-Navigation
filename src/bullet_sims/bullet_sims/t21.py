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
q_actuated_home[:6] = np.array([0, 0, 0, 0, 0, 0])
q_actuated_home[6:12] = np.array([0, 0, 0, 0, 0, 0])
q_actuated_home[14:22] = np.array([0, 0, 0, 0, 0, 0, 0, 0 ])
q_actuated_home[22:30] = np.array([0, 0, 0, 0, 0, 0, 0, 0 ])

# Initialization position including floating base
q_home = np.hstack([np.array([0, 0, z_init, 0, 0, 0, 1]), q_actuated_home])

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

# PD Controller parameters
n_actuated = len(q_actuated_home)  # Number of actuated joints

# Create gain matrices with different values for legs and upper body
Kp = np.eye(n_actuated)
Kd = np.eye(n_actuated)

# Talos joint layout:
# 0-5: left leg, 6-11: right leg
# 12-13: torso, 14-21: left arm, 22-29: right arm, 30-31: head

# Legs (0-11): higher gains for stiff support
Kp[0:12, 0:12] *= 300.0
Kd[0:12, 0:12] *= 2.5

# Upper body (12-31): lower gains
Kp[12:32, 12:32] *= 100.0
Kd[12:32, 12:32] *= 1.2

q_d = np.zeros(n_actuated)          # Desired joint configuration (zero)

# Joint command vector
tau = q_actuated_home*0

done = False
while not done:
    # update the simulator and the robot
    simulator.step()
    simulator.debug()
    robot.update()
    
    # Get current joint state (excluding floating base)
    q_current = robot.q()[7:]        # Skip floating base (7 DOF: 3 pos + 4 quat)
    v_current = robot.v()[6:]        # Skip floating base velocity (6 DOF)
    
    # PD Controller: f_i = Kp (q_d - q) - Kd * v_dot
    tau = Kp @ (q_d - q_current) - Kd @ v_current
    
    # command to the robot
    robot.setActuatedJointTorques(tau)