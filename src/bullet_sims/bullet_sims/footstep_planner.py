import numpy as np
import pinocchio as pin
from enum import Enum
import time
import pybullet as pb
import pybullet_data
import bullet_sims.talos_conf as conf
import simulator.utilities as sim_utils


class Side(Enum):
    """Side
    Describes which foot to use
    """
    LEFT=0
    RIGHT=1

def other_foot_id(id):
    if id == Side.LEFT:
        return Side.RIGHT
    else:
        return Side.LEFT
        
class FootStep:
    """FootStep
    Holds all information describing a single footstep
    """
    def __init__(self, pose, footprint, side=Side.LEFT):
        """inti FootStep

        Args:
            pose (pin.SE3): the pose of the footstep
            footprint (np.array): 3 by n matrix of foot vertices
            side (_type_, optional): Foot identifier. Defaults to Side.LEFT.
        """
        self.pose = pose
        self.footprint = footprint
        self.side = side
        
    def poseInWorld(self):
        return self.pose
        
    def plot(self, simulation):

        Q = sim_utils.matrixToQuaternion(self.pose.rotation)

        # footprint extents in the foot-local frame: column 0 is the
        # positive-side corner, column 1 the negative-side corner
        length = self.footprint[0, 0] + self.footprint[0, 1]
        width = self.footprint[1, 0] + self.footprint[1, 1]
        local_center = np.array([
            (self.footprint[0, 0] - self.footprint[0, 1]) / 2,
            (self.footprint[1, 0] - self.footprint[1, 1]) / 2,
            0.0,
        ])
        rect_center = self.pose.translation + self.pose.rotation @ local_center

        # plot the footprint as a rectangle on the ground
        simulation.addGlobalDebugRectancle(rect_center, Q=Q, length=length, width=width)

        # display which foot this step belongs to
        #simulation.addUserDebugText(-1, -1, self.side.name, self.pose.translation + np.array([0, 0, 0.05]))

        # plot the target position of the step
        simulation.addSphereMarker(self.pose.translation, orientation=Q)

        return None

class FootStepPlanner:
    """FootStepPlanner
    Creates footstep plans (list of right and left steps)
    """
    
    def __init__(self, conf):
        self.conf = conf
        self.steps = []
        
    def planLine(self, T_0_w, side, no_steps):
        """plan a sequence of steps in a strait line

        Args:
            T_0_w (pin.SE3): The inital starting position of the plan
            side (Side): The intial foot for starting the plan
            no_steps (_type_): The number of steps to take

        Returns:
            list: sequence of steps
        """
        
        # the displacement between steps in x and y direction
        dx = self.conf.step_size_x
        dy = 2*self.conf.step_size_y
        
        # the footprint of the robot
        lfxp, lfxn = self.conf.lfxp, self.conf.lfxn
        lfyp, lfyn = self.conf.lfyp, self.conf.lfyn
        
        steps=[]

        for i in range(no_steps):
            # Compute the pose of the current step based on the initial pose and step size
            step_pose = pin.SE3(T_0_w.rotation, T_0_w.translation + np.array([dx * i, dy * (i % 2), 0]))
            # Determine the footprint based on the side of the step
            if side == Side.LEFT:
                footprint = np.array([[lfxp, lfxn], [lfyp, lfyn], [0, 0]])
            else:
                footprint = np.array([[lfxn, lfxp], [lfyn, lfyp], [0, 0]])
            # Create a FootStep object and add it to the steps list
            steps.append(FootStep(step_pose, footprint, side))
            # Alternate sides for the next step
            side = other_foot_id(side)
                                
        self.steps = steps
        return steps

    
    def plot(self, simulation):
        for step in self.steps:
            step.plot(simulation)

            
if __name__=='__main__':
    """ Test footstep planner
    """
    from types import SimpleNamespace
    from simulator.pybullet_wrapper import PybulletWrapper

    conf = SimpleNamespace(
        step_size_x=0.25, step_size_y=0.096,
        lfxp=0.12, lfxn=0.08, lfyp=0.065, lfyn=0.065,
    )

    sim = PybulletWrapper(render=True)

    planner = FootStepPlanner(conf)
    steps = planner.planLine(pin.SE3.Identity(), Side.LEFT, no_steps=6)
    planner.plot(sim)

    while True:
        sim.step()
