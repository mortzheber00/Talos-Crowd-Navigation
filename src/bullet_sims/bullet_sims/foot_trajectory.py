import numpy as np
import pinocchio as pin

# import ndcurves, scipy, numpy, etc... to do your splines
from scipy.interpolate import BPoly

class SwingFootTrajectory:
    """SwingFootTrajectory
    Interpolate Foot trajectory between SE3 T0 and T1
    """
    def __init__(self, T0, T1, duration, height=0.05):
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

    def isDone(self):
        return self._t_elapsed >= self._duration 
    
    def evaluate(self, t):
        """evaluate at time t
        """
        t = np.clip(t, 0.0, self._duration)

        pos = np.array([self._spline_xy[0](t), self._spline_xy[1](t), self._spline_z(t)])
        vel = np.array([self._spline_xy[0](t, 1), self._spline_xy[1](t, 1), self._spline_z(t, 1)])
        acc = np.array([self._spline_xy[0](t, 2), self._spline_xy[1](t, 2), self._spline_z(t, 2)])

        T = pin.SE3(np.eye(3), pos)
        v = pin.Motion(vel, np.zeros(3))
        a = pin.Motion(acc, np.zeros(3))

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
