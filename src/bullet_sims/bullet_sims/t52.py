import numpy as np
from numpy import nan
from numpy.linalg import norm as norm
import matplotlib.pyplot as plt

# pinocchio
import pinocchio as pin

# simulator
import pybullet as pb
from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot

# robot and controller
from bullet_sims.tsid_wrapper import TSIDWrapper
import bullet_sims.config as conf

# ROS
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, TransformStamped

################################################################################
# settings
################################################################################

DO_PLOT = True

################################################################################
# Robot
################################################################################

class Talos(Robot, Node):
    def __init__(self, simulator, urdf, model, q=None, basePosition=None, baseQuaternion=None, verbose=True, useFixedBase=False):
        Node.__init__(self, 'talos_standing_node')

        if basePosition is None:
            basePosition = np.zeros(3)
        if baseQuaternion is None:
            baseQuaternion = np.array([0.0, 0.0, 0.0, 1.0])

        # initialize simulator/robot base
        super().__init__(
            simulator,
            filename=urdf,
            model=model,
            basePosition=basePosition,
            baseQuationerion=baseQuaternion,
            q=q,
            useFixedBase=useFixedBase,
            verbose=verbose,
        )

        # ROS publishers
        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Store TSID base state for publishing
        self.tsid_base_T = None
        self.tsid_base_v = None

        # Parameter für die Push-Sequenz
        self.f_push = 100.0       # Stärke der Kraft in Newton
        self.t_push = 0.5        # Dauer des Schubs in Sekunden
        self.t_period = 4.0      # Wartezeit zwischen den Schüben
        
        # Zustandstracking für die State Machine
        # Zustände: 0=Idle, 1=PushRight, 2=WaitLeft, 3=PushLeft, 4=WaitBack, 5=PushBack, 6=Finished
        self.push_state = 0
        self.push_timer = self.t_period  # Wir starten mit der ersten Wartezeit
        self.push_line_id = -1           # Für die PyBullet-Visualisierung
    
        # Aktivierungs-Flags für die Balancestrategies
        self.enable_ankle_strategy = False
        self.enable_hip_strategy =  True
        
        # Zustandsspeicher für die Ankle Strategy 
        self.ankle_x_d = None
        self.ankle_v_d = np.zeros(3)
        
    def update(self):
        super().update()

    def publish(self):
        stamp = self.get_clock().now().to_msg()

        js = JointState()
        js.header.stamp = stamp
        js.name = self.actuatedJointNames()
        js.position = self.actuatedJointPosition().tolist()
        js.velocity = self.actuatedJointVelocity().tolist()
        js.effort = self._tau_cmd.tolist()
        self.joint_state_pub.publish(js)

        # publish base transform from TSID solution
        if self.tsid_base_T is not None:
            world_pos = self.tsid_base_T.translation
            world_orient = pin.Quaternion(self.tsid_base_T.rotation)
        else:
            # fallback to pybullet if TSID not available
            world_pos, world_orient = self.baseWorldPose()
            world_orient = pin.Quaternion(world_orient)

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'world'
        t.child_frame_id = self.baseName()
        t.transform.translation.x = float(world_pos[0])
        t.transform.translation.y = float(world_pos[1])
        t.transform.translation.z = float(world_pos[2])
        t.transform.rotation.x = float(world_orient.x)
        t.transform.rotation.y = float(world_orient.y)
        t.transform.rotation.z = float(world_orient.z)
        t.transform.rotation.w = float(world_orient.w)
        self.tf_broadcaster.sendTransform(t)

    def update_push_sequence(self, dt, sim):
        """
        Führt die zeitgesteuerte Push-State-Machine aus.
        Muss in jedem Schleifendurchlauf aufgerufen werden.
        """
        # Wenn wir fertig sind, machen wir gar nichts mehr
        if self.push_state == 6:
            return

        # Timer herunterzählen (Zeitschritt von der Simulationsfrequenz abziehen)
        self.push_timer -= dt

        # Zustand 0: Idle (Warten auf den ersten Push von rechts)
        if self.push_state == 0:
            if self.push_timer <= 0:
                self.push_state = 1
                self.push_timer = self.t_push
                # Visualisierung anmachen: Kraft von rechts (wirkt in negative Y-Richtung)
                self._draw_push_line(sim, np.array([0, -self.f_push, 0]))

        # Zustand 1: Push von rechts aktiv
        elif self.push_state == 1:
            force = np.array([0, -self.f_push, 0])
            self.applyForce(force)
            
            if self.push_timer <= 0:
                self._remove_push_line(sim)
                self.push_state = 2
                self.push_timer = self.t_period

        # Zustand 2: Warten auf Push von links
        elif self.push_state == 2:
            if self.push_timer <= 0:
                self.push_state = 3
                self.push_timer = self.t_push
                # Visualisierung anmachen: Kraft von links (wirkt in positive Y-Richtung)
                self._draw_push_line(sim, np.array([0, self.f_push, 0]))

        # Zustand 3: Push von links aktiv
        elif self.push_state == 3:
            force = np.array([0, self.f_push, 0])
            self.applyForce(force)
            
            if self.push_timer <= 0:
                self._remove_push_line(sim)
                self.push_state = 4
                self.push_timer = self.t_period

        # Zustand 4: Warten auf Push von hinten
        elif self.push_state == 4:
            if self.push_timer <= 0:
                self.push_state = 5
                self.push_timer = self.t_push
                # Visualisierung anmachen: Kraft von hinten (wirkt in positive X-Richtung)
                self._draw_push_line(sim, np.array([self.f_push, 0, 0]))

        # Zustand 5: Push von hinten aktiv
        elif self.push_state == 5:
            force = np.array([self.f_push, 0, 0])
            self.applyForce(force)
            
            if self.push_timer <= 0:
                self._remove_push_line(sim)
                self.push_state = 6
                print("Push-Sequenz beendet!")

                # repeat cycle:
                self.f_push += 10
                self.push_state = 0
                print("Kraft erhoeht: ", self.f_push)
                self.push_timer = self.t_period
                
    # --- Helper functions for visualizing the pushes in PyBullet---
    def get_hip_position(self):
        """
        Gibt die 3D-Position der Hüfte (Base Link) zurück.
        """
        # pb ist laut deinem Codeausschnitt das pybullet-Modul
        position, _ = pb.getBasePositionAndOrientation(self._id)
        return np.array(position)

    def _draw_push_line(self, sim, force_vector):
        hip_pos = self.get_hip_position() # Deine Methode, um die Hüftposition zu bekommen
        # Linie startet leicht versetzt und zeigt zur Hüfte
        p1 = hip_pos - force_vector * 0.1  
        p2 = hip_pos
        self.push_line_id = sim.addGlobalDebugLine(p1, p2, self.push_line_id, color=[1, 0, 0])

    def _remove_push_line(self, sim):
        if self.push_line_id != -1:
            sim.removeDebugItem(self.push_line_id)
            self.push_line_id = -1

    # Balancestrategies

    def apply_ankle_strategy(self, dt, p_global, x_ref, tw):
        """Berechnet die Ankle Strategy und modifiziert die CoM-Sollwerte."""
        if not self.enable_ankle_strategy:
            return x_ref, np.zeros(3) # Wenn inaktiv, gib einfach die Standard-Mitte zurück

        # Erster Schritt: Initialisierung
        if self.ankle_x_d is None:
            self.ankle_x_d = x_ref.copy()

        # Gains (0 < Kp < omega < Kx)
        omega = np.sqrt(9.81 / x_ref[2])
        Kp = 0.9 * omega  
        Kx = 500 * omega  

        p_ref = np.array([x_ref[0], x_ref[1], 0.0])
        v_ref = np.zeros(3)

        # Regelgesetz (Gleichung 11)
        self.ankle_v_d[0] = v_ref[0] - Kx * (self.ankle_x_d[0] - x_ref[0]) + Kp * (p_global[0] - p_ref[0])
        self.ankle_v_d[1] = v_ref[1] - Kx * (self.ankle_x_d[1] - x_ref[1]) + Kp * (p_global[1] - p_ref[1])
        
        # Integration
        self.ankle_x_d += self.ankle_v_d * dt
        self.ankle_x_d[2] = x_ref[2] # Höhe halten

        return self.ankle_x_d, self.ankle_v_d

    def apply_hip_strategy(self, dt, cmp_global, x_ref, tw):
        """Berechnet die Hip Strategy nach Gleichung (12) und übergibt den

        gewünschten Drehimpuls (AM) direkt an die AM-Task des Wrappers.
        """
        if not self.enable_hip_strategy:
            # Wenn deaktiviert, regeln wir den Drehimpuls starr auf Null
            tw.setAmRefState(np.zeros(3), np.zeros(3))
            return

        # Gain definieren (K_gamma)
        K_gamma = 15.2 
        
        # CMP-Referenz bestimmen (2D auf dem Boden, z = 0.0)
        r_ref = np.array([x_ref[0], x_ref[1], 0.0])
        r     = np.array([cmp_global[0], cmp_global[1], 0.0])

        # Drehimpuls berechnen:
        cmp_error = r - r_ref
        
        gamma_d = np.zeros(3)
        gamma_d[0] = -K_gamma * cmp_error[1] # Roll-Moment dämpft Lateral-Fehler
        gamma_d[1] =  K_gamma * cmp_error[0] # Pitch-Moment dämpft Sagittal-Fehler
        gamma_d[2] =  0.0                     # Yaw bleibt meist unangetastet

        # Ableitung des Drehimpulses (d_gamma_d) berechnen
        if not hasattr(self, 'prev_gamma_d'):
            self.prev_gamma_d = np.zeros(3)
            
        d_gamma_d = (gamma_d - self.prev_gamma_d) / dt
        self.prev_gamma_d = gamma_d.copy()

        # AM-Task in Wrappers 
        tw.setAmRefState(gamma_d, d_gamma_d)

    def execute_balance_control(self, dt, p_global, cmp_global, x_ref, tw):
        """
        Der 'Manager' (Deine Mini-State-Machine).
        Wird in der main aufgerufen und koordiniert beide Strategien.
        """

        x_target, v_target = self.apply_ankle_strategy(dt, p_global, x_ref, tw)
        
        tw.setComRefState(x_target, v_target)
        
        self.apply_hip_strategy(dt, cmp_global, x_ref, tw)

################################################################################
# Helper functions for calculating ground reference points
################################################################################

def compute_foot_zmp(f_ankle, tau_ankle, d=0.1):
    """
    Berechnet den lokalen ZMP eines Fußes relativ zum Fuß-Koordinatensystem.
    Gleichungen (1), (2), (3)
    """
    # Vermeide Division durch 0, falls der Fuß keinen Bodenkontakt hat
    if f_ankle[2] > 1e-3:
        px_local = (-tau_ankle[1] - f_ankle[0] * d) / f_ankle[2]
        py_local = ( tau_ankle[0] - f_ankle[1] * d) / f_ankle[2]
    else:
        px_local, py_local = 0.0, 0.0
        
    return np.array([px_local, py_local, 0.0])


def compute_global_zmp(p_R_local, p_L_local, H_w_rsole, H_w_lsole, fz_R, fz_L):
    """
    Transformiert die lokalen ZMPs in die Welt und berechnet den globalen ZMP.
    Gleichungen (4), (5), (6)
    """
    # Lokale ZMPs in Weltkoordinaten transformieren
    p_R_world = H_w_rsole.act(p_R_local)
    p_L_world = H_w_lsole.act(p_L_local)
    
    fz_total = fz_R + fz_L
    if fz_total > 1.0:
        px = (p_R_world[0] * fz_R + p_L_world[0] * fz_L) / fz_total
        py = (p_R_world[1] * fz_R + p_L_world[1] * fz_L) / fz_total
    else:
        px, py = 0.0, 0.0
        
    return np.array([px, py, 0.0])


def compute_cmp(x_com, f_R, f_L, H_w_rsole, H_w_lsole):
    """
    Berechnet den Centroidal Moment Pivot (CMP) im Weltkoordinatensystem.
    Gleichungen (7), (8), (9)
    """
    # Lokale Kräfte in das Weltkoordinatensystem rotieren
    f_R_world = H_w_rsole.rotation @ f_R
    f_L_world = H_w_lsole.rotation @ f_L
    f_total_world = f_R_world + f_L_world
    
    if f_total_world[2] > 1.0:
        rx = x_com[0] - (f_total_world[0] / f_total_world[2]) * x_com[2]
        ry = x_com[1] - (f_total_world[1] / f_total_world[2]) * x_com[2]
    else:
        rx, ry = x_com[0], x_com[1]
        
    return np.array([rx, ry, 0.0])


def compute_capture_point(x_com, v_com, g=9.81):
    """
    Berechnet den Capture Point / Divergent Component of Motion (DCM).
    Gleichung (10)
    """
    # Eigenfrequenz des inversen Pendels omega = sqrt(g / z_com)
    omega = np.sqrt(g / x_com[2])
    
    # Projektion auf den flachen Boden (z=0)
    x_p = np.array([x_com[0], x_com[1], 0.0])
    v_p = np.array([v_com[0], v_com[1], 0.0])
    
    xi = x_p + (v_p / omega)
    return xi

def plot_ground_reference_points(data_history):
    """
    Plottet die aufgezeichneten Trajektorien von CoM, ZMP, CMP und CP.
    """
    print("Generiere Zustand-Plots...")
    t = data_history['time']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 1. Plot für die X-Komponente (Sagittal)
    ax1.plot(t, data_history['com_x'], label='CoM X', color='black', linestyle='--')
    ax1.plot(t, data_history['zmp_x'], label='ZMP X', color='red')
    ax1.plot(t, data_history['cmp_x'], label='CMP X', color='blue', linestyle=':')
    ax1.plot(t, data_history['cp_x'],  label='CP/DCM X', color='green')

    ax1.set_ylim(-0.1, 0.1)

    ax1.set_ylabel('Position [m]')
    ax1.set_title('Bodenreferenzpunkte - X-Komponente (Vorwärts/Rückwärts)')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Plot für die Y-Komponente (Lateral)
    ax2.plot(t, data_history['com_y'], label='CoM Y', color='black', linestyle='--')
    ax2.plot(t, data_history['zmp_y'], label='ZMP Y', color='red')
    ax2.plot(t, data_history['cmp_y'], label='CMP Y', color='blue', linestyle=':')
    ax2.plot(t, data_history['cp_y'],  label='CP/DCM Y', color='green')

    ax2.set_ylim(-0.1, 0.1)

    ax2.set_xlabel('Zeit [s]')
    ax2.set_ylabel('Position [m]')
    ax2.set_title('Bodenreferenzpunkte - Y-Komponente (Seitwärts)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

################################################################################
# main
################################################################################

def main(): 
    rclpy.init()

    print('[standing] config module file:', conf.__file__)
    print('[standing] loaded q_home shape:', conf.q_home.shape)
    print('[standing] loaded q_home[:10]:', np.round(conf.q_home[:10], 4))
    print('[standing] loaded q_home z:', float(conf.q_home[2]))

    # The robot starts from zero base position and zero joint angles, while
    # TSID still uses q_home as the standing target.
    tsid_conf = conf
    q_home = tsid_conf.q_home.copy()
    q_init = np.zeros_like(q_home)      
    q_init[:3] = q_home[:3]
    q_init[3:7] = np.array([0.0, 0.0, 0.0, 1.0])

    # instantiate TSID wrapper with the exact same initial state
    try:
        tw = TSIDWrapper(tsid_conf)
    except Exception as e:
        print('Failed to create TSIDWrapper:', e)
        return

    # simulator
    simulator = PybulletWrapper(sim_rate=1000)

    # create robot using the pinocchio model from TSID
    pin_model = tw.robot.model()
    robot = Talos(
        simulator,
        tsid_conf.urdf,
        pin_model,
        q=q_init,
        basePosition=q_init[:3],
        baseQuaternion=q_init[3:7],
        verbose=True,
        useFixedBase=False,
    )

    # activate ankle-Torque Sensors in PyBullet Sim
    pb.enableJointForceTorqueSensor(robot.id(), robot.jointNameIndexMap()['leg_right_6_joint'], True)
    pb.enableJointForceTorqueSensor(robot.id(), robot.jointNameIndexMap()['leg_left_6_joint'], True)

    # Dictionary for ground ref. points data 
    history = {
        'time': [], 
        'com_x': [], 'com_y': [], 
        'zmp_x': [], 'zmp_y': [], 
        'cmp_x': [], 'cmp_y': [], 
        'cp_x': [],  'cp_y': []
    }

    # For Balance control
    x_ref = np.array(tw.comState().pos()).copy()

    # --- NEU: Virtuelle TSID-Zustände mit Startwerten initialisieren ---
    q_tsid = q_init.copy()
    v_tsid = np.zeros(robot._model.nv) 

    print('[standing] URDF:', tsid_conf.urdf)
    print('[standing] q_home base xyz:', q_home[:3])
    print('[standing] q_init base xyz:', q_init[:3])
    print('[standing] basePosition passed to Robot:', q_init[:3])
    print('[standing] baseQuaternion passed to Robot:', q_init[3:7])
    print('[standing] robot base name:', robot.baseName())

    t_publish = 0.0
    debug_counter = 0
    dt = 1 / 1000.0 # periodtime for simulation
    current_time = 0.0

    try:
        while simulator.isRunning():
            # elapsed time
            t = simulator.simTime()

            # update robot state from simulator
            robot.update()

            # get pinocchio-style states (für Sensorik/Plots weiterhin nützlich)
            q = robot.q()
            v = robot.v()

            # Pinocchio Kinematik-Daten aktualisieren
            data = robot._model.createData()
            pin.framesForwardKinematics(robot._model, data, q)
            
            # Transformationen (Position & Orientierung) im Weltkoordinatensystem holen
            H_w_lsole = data.oMf[robot._model.getFrameId("left_sole_link")]
            H_w_rsole = data.oMf[robot._model.getFrameId("right_sole_link")]
            H_w_lankle = data.oMf[robot._model.getFrameId("leg_left_6_joint")]
            H_w_rankle = data.oMf[robot._model.getFrameId("leg_right_6_joint")]
            
            # Kraft-Drehmoment-Sensoren auslesen (Rechtes Sprunggelenk)
            wren_r = pb.getJointState(robot.id(), robot.jointNameIndexMap()['leg_right_6_joint'])[2]
            wnp_r = np.array([-wren_r[0], -wren_r[1], -wren_r[2], -wren_r[3], -wren_r[4], -wren_r[5]])
            wr_rankle = pin.Force(wnp_r)
            
            # Kraft-Drehmoment-Sensoren auslesen (Linkes Sprunggelenk)
            wren_l = pb.getJointState(robot.id(), robot.jointNameIndexMap()['leg_left_6_joint'])[2]
            wnp_l = np.array([-wren_l[0], -wren_l[1], -wren_l[2], -wren_l[3], -wren_l[4], -wren_l[5]])
            wl_lankle = pin.Force(wnp_l)

            # --- Schwerpunkt (CoM) aus dem Wrapper auslesen ---
            com_state = tw.comState()
            x_com = np.array(com_state.pos())
            v_com = np.array(com_state.vel())

            # --- BERECHNUNG ÜBER HELPER FUNKTIONEN ---
            p_R_local = compute_foot_zmp(wr_rankle.linear, wr_rankle.angular, d=0.1)
            p_L_local = compute_foot_zmp(wl_lankle.linear, wl_lankle.angular, d=0.1)
            p_global = compute_global_zmp(p_R_local, p_L_local, H_w_rsole, H_w_lsole, wr_rankle.linear[2], wl_lankle.linear[2])
            r_global = compute_cmp(x_com, wr_rankle.linear, wl_lankle.linear, H_w_rsole, H_w_lsole)
            xi_global = compute_capture_point(x_com, v_com)

            # --- IN HISTORY SPEICHERN ---
            current_time += dt
            history['time'].append(current_time)
            history['com_x'].append(x_com[0]);   history['com_y'].append(x_com[1])
            history['zmp_x'].append(p_global[0]);history['zmp_y'].append(p_global[1])
            history['cmp_x'].append(r_global[0]);history['cmp_y'].append(r_global[1])
            history['cp_x'].append(xi_global[0]); history['cp_y'].append(xi_global[1])

            # Balance control
            robot.execute_balance_control(dt, p_global, r_global, x_ref, tw)

            # --- ANPASSUNG: Update TSID mit den VIRTUELLEN Zuständen ---
            try:
                tau, dv = tw.update(q_tsid, v_tsid, t, do_solve=True)
                # store TSID base state for publishing
                robot.tsid_base_T, robot.tsid_base_v = tw.baseState()
            except Exception as e:
                print('TSID update error:', e)
                tau = np.zeros(robot.numActuatedJoints())
                dv = np.zeros(robot._v.shape)

            if debug_counter % 100 == 0:
                pb_base_pos, pb_base_quat = robot.baseWorldPose()
                print(f'[standing] t={t:.3f}s')
                print(f'[standing]   PB base pos : {np.round(pb_base_pos, 4)}')
                print(f'[standing]   PB base quat: {np.round(pb_base_quat, 4)}')
                if robot.tsid_base_T is not None:
                    tsid_base_pos = robot.tsid_base_T.translation
                    tsid_base_quat = pin.Quaternion(robot.tsid_base_T.rotation)
                    print(f'[standing]   TSID base pos : {np.round(tsid_base_pos, 4)}')
                    print(f'[standing]   TSID base quat: {np.round([tsid_base_quat.x, tsid_base_quat.y, tsid_base_quat.z, tsid_base_quat.w], 4)}')
                    print(f'[standing]   base pos diff  : {np.round(pb_base_pos - tsid_base_pos, 4)}')
                print(f'[standing]   |tau|={norm(tau):.4f} |dv|={norm(dv):.4f}')

            # # --- ANPASSUNG: Integration im virtuellen Zustand via Pinocchio ---
            # # v_tsid integrieren: v_next = v + dv * dt
            # v_tsid += dv * dt
            # # q_tsid integrieren: nutzt die Lie-Gruppen-Integration von Pinocchio
            # q_tsid = pin.integrate(robot._model, q_tsid, v_tsid * dt)

            q_tsid, v_tsid = tw.integrate_dv(q_tsid, v_tsid, dv, 1*
            simulator.stepTime())

            # Hardware-Sollwerte auf Position setzen
            robot.setActuatedJointPositions(q_tsid, v_tsid)

            # step simulation
            simulator.step()
            simulator.debug()

            # publish to ROS at ~30Hz
            if t - t_publish > 1.0 / 30.0:
                t_publish = t
                rclpy.spin_once(robot, timeout_sec=0.0)
                robot.publish()
            
            # Hier wird die State Machine getaktet
            robot.update_push_sequence(dt, simulator)

            debug_counter += 1
    finally:
        simulator.disconnect()
        robot.destroy_node()
        rclpy.shutdown()
        plot_ground_reference_points(history)
    
if __name__ == '__main__': 
    main()
    
