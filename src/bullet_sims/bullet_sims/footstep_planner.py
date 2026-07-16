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

    def _step_z(self, ground_z=None):
        """Sole-frame z to place footsteps at.

        This MUST be pinned to the ground, not read off the measured support
        foot. Both planners used to do ``z0 = T_0_w.translation[2]``, and since
        _plan_from re-anchors from the measured support foot every step, that
        made the step height a unity-gain integrator with no reference to the
        floor at all:

            z_target(n+1) = z_measured(n)

        so any systematic landing bias compounded instead of being corrected.
        Measured at ~2 mm/step: CoM z rose 0.854 -> 0.892 over 15 s of straight
        walking, and _report_drift could not see it because its CoM norm is
        xy-only. The contact task (w=1e5) outweighs the CoM task (w=1e1) by 1e4,
        so the CoM task can never pull the robot back down -- the feet decide the
        height, which is why no kp_com/w_com value fixed this.
        """
        if ground_z is not None:
            return float(ground_z)
        return float(getattr(self.conf, 'ground_z', 0.0))
        
    def planLine(self, T_0_w, side, no_steps, ground_z=None):
        """plan a sequence of steps in a straight line along T_0_w's heading

        Args:
            T_0_w (pin.SE3): The inital starting position of the plan
            side (Side): The intial foot for starting the plan
            no_steps (_type_): The number of steps to take
            ground_z (float): sole-frame z of a foot flat on the floor. Defaults
                to conf.ground_z (0.0). See _step_z for why this must NOT come
                from the measured support foot.

        Returns:
            list: sequence of steps
        """
        
        # Structured like planCircle with zero curvature: a centre path advances
        # step_size_x per step along the INITIAL heading, and each foot is placed
        # step_size_y to its OWN side of that path.
        #
        # The previous version offset each step by dy * (i % 2), which is never
        # negative -- the feet never straddled the path. On the one-shot plan that
        # merely sat the whole gait 0.5*dy off to one side, but _plan_from
        # re-anchors from the measured support foot EVERY step and always takes
        # plan[1], where i = 1 -> +dy. So every step shoved the swing foot a
        # further +2*step_size_y in world +y, and the robot walked a permanent
        # atan2(2*sy, sx) = 37.5 deg diagonal. (It looked like goal-seeking only
        # because ROBOT_GOAL=(4,3) happens to sit at 37.5 deg.) The offset was
        # also added in WORLD axes, never rotated by T_0_w.rotation, so the "line"
        # ignored the robot's heading.
        step_x = self.conf.step_size_x          # advance of the centre per step
        sy = self.conf.step_size_y              # lateral offset of each foot
        
        # the footprint of the robot
        lfxp, lfxn = self.conf.lfxp, self.conf.lfxn
        lfyp, lfyn = self.conf.lfyp, self.conf.lfyn
        
        # Heading + position from the initial pose. The step orientation is the
        # YAW-ONLY rotation: a footstep target is flat on the ground, whereas
        # T_0_w is the measured support foot, so taking its rotation whole would
        # propagate that foot's roll/pitch into every planned step.
        yaw0 = np.arctan2(T_0_w.rotation[1, 0], T_0_w.rotation[0, 0])
        p0 = T_0_w.translation[:2]
        z0 = self._step_z(ground_z)
        c, s = np.cos(yaw0), np.sin(yaw0)
        forward = np.array([c, s])
        left = np.array([-s, c])
        R0 = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

        # place the centre path so that step 0 lands on the support foot
        lateral0 = sy if side == Side.LEFT else -sy
        centre = p0 - lateral0 * left

        steps = []
        cur_side = side

        for i in range(no_steps):
            # advance the centre along the heading, then offset to this foot's side
            lateral = sy if cur_side == Side.LEFT else -sy
            foot_xy = centre + i * step_x * forward + lateral * left
            step_pose = pin.SE3(R0, np.array([foot_xy[0], foot_xy[1], z0]))
            # Determine the footprint based on the side of the step
            if cur_side == Side.LEFT:
                footprint = np.array([[lfxp, lfxn], [lfyp, lfyn], [0, 0]])
            else:
                footprint = np.array([[lfxn, lfxp], [lfyn, lfyp], [0, 0]])
            # Create a FootStep object and add it to the steps list
            steps.append(FootStep(step_pose, footprint, cur_side))
            # Alternate sides for the next step
            cur_side = other_foot_id(cur_side)
                                
        self.steps = steps
        return steps

    def planCircle(self, T_0_w, side, no_steps, radius=1.5, clockwise=False,
                   face_path=False, ground_z=None):
        """plan a sequence of steps along a circular arc.

        The robot's centre path follows a circle of the given radius; each foot is
        placed step_size_y to its own side of that path, offset perpendicular to
        the path so the feet always straddle the direction of travel.

        Two modes:
        * face_path=False (default, robust): the feet keep their INITIAL fixed
          orientation (no yaw). The robot follows the curved path while facing a
          constant direction (omnidirectional / side-stepping). This reuses the
          proven straight-line gait and does not require rotating the body, so it
          is much more stable — ideal for bounded curves like obstacle avoidance.
        * face_path=True: each foot is yawed to face the direction of travel, so
          the robot turns its whole body around the curve. More natural for
          sustained turning but requires the (harder to stabilise) body rotation.

        plan[0] coincides with the initial (support) foot pose, like planLine.

        Args:
            T_0_w (pin.SE3): initial pose (the current support foot).
            side (Side): the initial (support) foot.
            no_steps (int): number of steps to take.
            radius (float): radius of the centre path [m]. Larger = gentler curve.
            clockwise (bool): curve direction. False = left/CCW, True = right/CW.
            face_path (bool): rotate the feet/body to face the path (see above).
            ground_z (float): sole-frame z of a foot flat on the floor. Defaults
                to conf.ground_z (0.0). See _step_z for why this must NOT come
                from the measured support foot.

        Returns:
            list: sequence of FootStep.
        """
        step_x = self.conf.step_size_x          # arc advance of the centre per step
        sy = self.conf.step_size_y              # lateral offset of each foot
        lfxp, lfxn = self.conf.lfxp, self.conf.lfxn
        lfyp, lfyn = self.conf.lfyp, self.conf.lfyn

        # yaw increment per step = arc_length / radius; sign sets the curve side
        turn = -1.0 if clockwise else 1.0
        dpsi = turn * step_x / radius

        # starting heading + position taken from the initial pose; the height is
        # pinned to the ground (see _step_z), NOT read off the support foot
        yaw0 = np.arctan2(T_0_w.rotation[1, 0], T_0_w.rotation[0, 0])
        p0 = T_0_w.translation[:2]
        z0 = self._step_z(ground_z)

        def left_dir(psi):
            # unit vector 90 deg to the left of heading psi
            return np.array([-np.sin(psi), np.cos(psi)])

        def rot_z(psi):
            c, s = np.cos(psi), np.sin(psi)
            return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

        # foot orientation: track the path heading, or stay fixed at the start
        R_fixed = rot_z(yaw0)

        # place the centre path so that step 0 lands on the support foot
        lateral0 = sy if side == Side.LEFT else -sy
        centre = p0 - lateral0 * left_dir(yaw0)

        steps = []
        cur_side = side
        psi = yaw0
        for i in range(no_steps):
            lateral = sy if cur_side == Side.LEFT else -sy
            foot_xy = centre + lateral * left_dir(psi)
            foot_R = rot_z(psi) if face_path else R_fixed
            step_pose = pin.SE3(foot_R, np.array([foot_xy[0], foot_xy[1], z0]))
            if cur_side == Side.LEFT:
                footprint = np.array([[lfxp, lfxn], [lfyp, lfyn], [0, 0]])
            else:
                footprint = np.array([[lfxn, lfxp], [lfyn, lfyp], [0, 0]])
            steps.append(FootStep(step_pose, footprint, cur_side))
            # advance the centre along the current heading, then turn
            centre = centre + step_x * np.array([np.cos(psi), np.sin(psi)])
            psi += dpsi
            cur_side = other_foot_id(cur_side)

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
