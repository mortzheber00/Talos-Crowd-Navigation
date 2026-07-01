import numpy as np
import pinocchio as pin

# import ndcurves, scipy, numpy, etc... to do your splines

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
        '''reset back to zero, update poses
        '''
                #>>>>TODO: plan the spline

        self._T0 = T0
        self._T1 = T1
        self._t_elapsed = 0.0

        self._p0 = np.array(T0.translation).copy()
        self._p1 = np.array(T1.translation).copy()
        self._R0 = np.array(T0.rotation).copy()
        self._R1 = np.array(T1.rotation).copy()

        self._dp = self._p1 - self._p0
        self._w_rel = pin.log3(self._R0.T @ self._R1)

    def isDone(self):
        return self._t_elapsed >= self._duration 
    
    def evaluate(self, t):
        """evaluate at time t
        """
        #>>>>TODO: evaluate the spline at time t, return pose, velocity, acceleration
        t = np.clip(t, 0.0, self._duration)
        s = t / self._duration
        ds_dt = 1.0 / self._duration

        # Smooth cubic blend for the horizontal motion.
        h = 3.0 * s**2 - 2.0 * s**3
        dh_ds = 6.0 * s - 6.0 * s**2
        d2h_ds2 = 6.0 - 12.0 * s

        # # Parabolic swing height: zero at ends, max at mid-swing.
        # z_bump = 4.0 * self._height * s * (1.0 - s)
        # dzb_ds = 4.0 * self._height * (1.0 - 2.0 * s)
        # d2zb_ds2 = -8.0 * self._height

        # Nutze eine Sinus-Funktion: Bei s=0 und s=1 ist die Höhe 0.
        # Da die Ableitung cos(pi * s) ist, wird sie bei s=1 (Landung) zu cos(pi) = -1.
        # Um auch die Geschwindigkeit auf 0 zu bringen, nutzen wir sin^2:
        z_bump = self._height * (np.sin(np.pi * s))**2
        dzb_ds = 2.0 * self._height * np.sin(np.pi * s) * np.cos(np.pi * s) * np.pi
        d2zb_ds2 = 2.0 * self._height * np.pi**2 * ((np.cos(np.pi * s))**2 - (np.sin(np.pi * s))**2)

        pos = self._p0 + h * self._dp
        pos[2] += z_bump

        vel = dh_ds * ds_dt * self._dp
        vel[2] += dzb_ds * ds_dt

        acc = d2h_ds2 * (ds_dt ** 2) * self._dp
        acc[2] += d2zb_ds2 * (ds_dt ** 2)

        R = self._R0 @ pin.exp3(self._w_rel * s)
        pose = pin.SE3(R, pos)

        self._t_elapsed = t
        return pose, vel, acc

if __name__=="__main__":
    import matplotlib.pyplot as plt

    # 1. Setup der Start- und Endkonfiguration (wie in deiner Vorgabe)
    T0 = pin.SE3(np.eye(3), np.array([0, 0, 0]))
    T1 = pin.SE3(np.eye(3), np.array([0.2, 0, 0]))

#     #>>>>TODO: plot to make sure everything is correct

    # duration = 1.0
    # height = 0.05

    # # Instanziierung des Planers
    # # (Funktioniert sowohl mit deiner mathematischen als auch der ndcurves-Variante)
    # planner = SwingFootTrajectory(T0, T1, duration=duration, height=height)

    # # 2. Trajektorie über diskrete Zeitschritte evaluieren
    # dt = 0.01
    # time_steps = np.arange(0.0, duration + dt, dt)
    
    # positions = []
    # velocities = []
    # accelerations = []

    # for t in time_steps:
    #     pose, vel, acc = planner.evaluate(t)
    #     positions.append(pose.translation.copy())
    #     velocities.append(vel.copy())
    #     accelerations.append(acc.copy())

    # # Konvertierung in NumPy-Arrays für einfaches Slicing ([Zeit, Achse])
    # positions = np.array(positions)
    # velocities = np.array(velocities)
    # accelerations = np.array(accelerations)

    # # ==========================================
    # # PLOTTING
    # # ==========================================
    # fig = plt.figure(figsize=(14, 10))
    # fig.suptitle("Swing Foot Trajectory Verification", fontsize=16, fontweight='bold')

    # # --- Plot 1: 3D Raumkurve des Fußes ---
    # ax3d = fig.add_subplot(2, 2, 1, projection='3d')
    # ax3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', label='Foot Path', linewidth=2)
    # ax3d.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='green', s=50, label='Start (T0)')
    # ax3d.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='red', s=50, label='Goal (T1)')
    # ax3d.set_title("3D Trajectory Space")
    # ax3d.set_xlabel("X [m]")
    # ax3d.set_ylabel("Y [m]")
    # ax3d.set_zlabel("Z [m]")
    # ax3d.legend()
    # ax3d.set_zlim(0, height * 1.2) # Skalierung für die Höhe optimieren

    # # --- Plot 2: Positionen über die Zeit ---
    # ax_pos = fig.add_subplot(2, 2, 2)
    # ax_pos.plot(time_steps, positions[:, 0], 'r-', label='X (Vorwärts)')
    # ax_pos.plot(time_steps, positions[:, 2], 'g-', label='Z (Höhe)')
    # ax_pos.set_title("Position vs Time")
    # ax_pos.set_xlabel("Time [s]")
    # ax_pos.set_ylabel("Position [m]")
    # ax_pos.grid(True)
    # ax_pos.legend()

    # # --- Plot 3: Geschwindigkeiten über die Zeit ---
    # ax_vel = fig.add_subplot(2, 2, 3)
    # ax_vel.plot(time_steps, velocities[:, 0], 'r--', label='v_x')
    # ax_vel.plot(time_steps, velocities[:, 2], 'g--', label='v_z')
    # ax_vel.set_title("Velocity vs Time")
    # ax_vel.set_xlabel("Time [s]")
    # ax_vel.set_ylabel("Velocity [m/s]")
    # ax_vel.grid(True)
    # ax_vel.legend()

    # # --- Plot 4: Beschleunigungen über die Zeit ---
    # ax_acc = fig.add_subplot(2, 2, 4)
    # ax_acc.plot(time_steps, accelerations[:, 0], 'r:', label='a_x')
    # ax_acc.plot(time_steps, accelerations[:, 2], 'g:', label='a_z')
    # ax_acc.set_title("Acceleration vs Time")
    # ax_acc.set_xlabel("Time [s]")
    # ax_acc.set_ylabel("Acceleration [m/s²]")
    # ax_acc.grid(True)
    # ax_acc.legend()

    # plt.tight_layout()
    # plt.show()