import numpy as np
import pinocchio as pin

# import ndcurves, scipy, numpy, etc... to do your splines
from scipy.interpolate import BPoly

class SwingFootTrajectory:
    """SwingFootTrajectory
    Interpolate Foot trajectory between SE3 T0 and T1
    """
    def __init__(self, T0, T1, duration, height=0.09):
        """initialize SwingFootTrajectory

        Args:
            T0 (pin.SE3): Inital foot pose
            T1 (pin.SE3): Final foot pose
            duration (float): step duration
            height (float, optional): setp height. Defaults to 0.05.
        """
        self._height = height
        self._t_elapsed = 0.0
        self._duration = duration
        self.reset(T0, T1)

    def reset(self, T0, T1):
        '''reset back to zero, update poses, plan the spline
        '''
        self._t_elapsed = 0.0
        self._T0, self._T1 = T0, T1 
        d = self._duration

        # extract position from pin objects 
        p0, p1 = T0.translation, T1.translation 

        # spline in XY-plane 
        # defined from [0 to d]
        # the second argument is the starting and end points (or inbetweens),
        # which includes the first and second derivatives as constraints 
        self._spline_xy = [
            BPoly.from_derivatives(
                [0, d], [[p0[i], 0, 0], [p1[i], 0, 0]]
            )
            for i in range(2)
        ]

        z_mid = 0.5 * (p0[2]+p1[2]) + self._height
        self._spline_z = BPoly.from_derivatives(
            [0, d/2, d],
            [[p0[2], 0, 0], [z_mid, 0], [p1[2], 0, 0]] # zero vertical velocity at mid-point
        )

        # yaw spline: interpolate the foot heading from T0 to T1 along the
        # shortest angular path, with zero angular velocity/acceleration at both
        # ends. This lets the swing foot turn (needed for curved/circular walking);
        # for straight walking T0 and T1 share a heading, so yaw stays constant.
        yaw0 = np.arctan2(T0.rotation[1, 0], T0.rotation[0, 0])
        yaw1 = np.arctan2(T1.rotation[1, 0], T1.rotation[0, 0])
        dyaw = (yaw1 - yaw0 + np.pi) % (2 * np.pi) - np.pi   # wrap to [-pi, pi]
        self._spline_yaw = BPoly.from_derivatives(
            [0, d], [[yaw0, 0, 0], [yaw0 + dyaw, 0, 0]]
        )

    def isDone(self):
        return self._t_elapsed >= self._duration 
    
    def evaluate(self, t):
        """evaluate at time t
        """
        t = np.clip(t, 0.0, self._duration)

        pos = np.array([self._spline_xy[0](t), self._spline_xy[1](t), self._spline_z(t)])
        vel = np.array([self._spline_xy[0](t, 1), self._spline_xy[1](t, 1), self._spline_z(t, 1)])
        acc = np.array([self._spline_xy[0](t, 2), self._spline_xy[1](t, 2), self._spline_z(t, 2)])

        # heading (yaw) and its derivatives from the yaw spline
        yaw = float(self._spline_yaw(t))
        yaw_dot = float(self._spline_yaw(t, 1))
        yaw_ddot = float(self._spline_yaw(t, 2))
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

        T = pin.SE3(R, pos)
        v = pin.Motion(vel, np.array([0.0, 0.0, yaw_dot]))
        a = pin.Motion(acc, np.array([0.0, 0.0, yaw_ddot]))

        return T, v, a

################################################################################
# Tests 
################################################################################
if __name__=="__main__":
    T0 = pin.SE3(np.eye(3), np.array([0, 0, 0]))
    T1 = pin.SE3(np.eye(3), np.array([0.2, 1.0, 0]))
    d = 1.0
    import matplotlib.pyplot as plt

    # initiate object 
    swing_foot = SwingFootTrajectory(T0, T1, d, height=0.05)

    t_vals = np.linspace(0, d, 100)
    pos_vals = np.array([swing_foot.evaluate(t)[0].translation for t in t_vals])
    vel_vals = np.array([swing_foot.evaluate(t)[1].linear for t in t_vals])
    acc_vals = np.array([swing_foot.evaluate(t)[2].linear for t in t_vals])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(pos_vals[:, 0], pos_vals[:, 1], pos_vals[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Foot Trajectory')
    plt.show()
