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
import tf2_ros
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster

################################################################################
# settings
################################################################################

DO_PLOT = True

################################################################################
# Robot
################################################################################

class Talos(Robot, Node):
    def __init__(self, simulator, urdf, model, q=None, basePosition=None, baseQuaternion=None, verbose=True, useFixedBase=False):
        Node.__init__(self, 'tutorial_4_one_leg_stand_node')

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

    def update_com_sine_reference(self, tsid_wrapper, com_ref, t, amplitude=0.05, frequency=0.5, start_time=4.0):
        omega = 2.0 * np.pi * frequency
        p_com = com_ref.copy()
        v_com = np.zeros(3)
        a_com = np.zeros(3)

        if t >= start_time:
            t_rel = t - start_time
            p_com[2] = com_ref[2] + amplitude * np.sin(omega * t_rel)
            v_com[2] = amplitude * omega * np.cos(omega * t_rel)
            a_com[2] = -amplitude * omega**2 * np.sin(omega * t_rel)

        tsid_wrapper.setComRefState(p_com, v_com, a_com)
        return p_com, v_com, a_com

    def setup_pistol_squat(self, tsid_wrapper, q_home, lift_height=0.3):
        init_pos_LF = tsid_wrapper.get_placement_LF()
        init_pos_LF.translation = init_pos_LF.translation.copy()
        init_pos_LF.translation[2] += lift_height

        tsid_wrapper.remove_contact_LF()
        tsid_wrapper.set_LF_pose_ref(init_pos_LF)

        q_posture = q_home[7:].copy()
        q_posture[:6] = 0.0
        tsid_wrapper.setPostureRef(q_posture)

        return init_pos_LF, q_posture

    def update_right_hand_circle(self, tsid_wrapper, t, start_time=8.0):

        c = np.array([0.4, -0.2, 1.1])
        r = 0.2
        f = 0.1
        omega = 2.0 * np.pi * f

        p = c.copy()
        v = np.zeros(3)
        a = np.zeros(3)

        if t >= start_time:
            t_rel = t - start_time

            # circle in the Y-Z plane, x stays constant
            p = c + np.array([0.0, r * np.cos(omega * t_rel), r * np.sin(omega * t_rel)])
            v = np.array([0.0, -r * omega * np.sin(omega * t_rel), r * omega * np.cos(omega * t_rel)])
            a = np.array([0.0, -r * omega**2 * np.cos(omega * t_rel), -r * omega**2 * np.sin(omega * t_rel)])

        tsid_wrapper.set_RH_pos_ref(p, v, a)

        return p, v, a

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

def plot_com_trajectories(tsid_wrapper, robot, t, 
                          com_ref_history, com_ref_v_history, com_ref_a_history, 
                          com_tsid_history, com_pb_history, 
                          v_tsid_history, v_pb_history, 
                          a_tsid_history, a_pb_history):
    if not DO_PLOT:
        return

    # Daten in NumPy-Arrays konvertieren für saubereres Slicing [:, i]
    ref_p, tsid_p, pb_p = np.array(com_ref_history), np.array(com_tsid_history), np.array(com_pb_history)
    ref_v, tsid_v, pb_v = np.array(com_ref_v_history), np.array(v_tsid_history), np.array(v_pb_history)
    ref_a, tsid_a, pb_a = np.array(com_ref_a_history), np.array(a_tsid_history), np.array(a_pb_history)

    # Erstellt ein 3x3 Grid. sharex=True sorgt dafür, dass alle die Zeitachse teilen
    fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    
    axes_names = ['X', 'Y', 'Z']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Schickere Standardfarben (Blau, Orange, Grün)
    
    # Stil-Definitionen für die 3 Datenquellen
    styles = {
        'Ref':  {'color': 'black', 'linestyle': '-',  'alpha': 0.8, 'linewidth': 1.5},
        'TSID': {'color': 'red',   'linestyle': '--', 'alpha': 0.8, 'linewidth': 1.5},
        'PB':   {'color': 'blue',  'linestyle': ':',  'alpha': 0.8, 'linewidth': 2.0}
    }

    for i in range(3):
        # --- ZEILE 1: Position ---
        axs[0, i].plot(t, ref_p[:, i], label='Ref', **styles['Ref'])
        axs[0, i].plot(t, tsid_p[:, i], label='TSID', **styles['TSID'])
        axs[0, i].plot(t, pb_p[:, i], label='PB', **styles['PB'])
        axs[0, i].set_title(f'Position - Achse {axes_names[i]}')
        axs[0, i].grid(True, linestyle=':', alpha=0.6)
        if i == 0: axs[0, i].set_ylabel('Pos [m]')

        # --- ZEILE 2: Geschwindigkeit ---
        axs[1, i].plot(t, ref_v[:, i], **styles['Ref'])
        axs[1, i].plot(t, tsid_v[:, i], **styles['TSID'])
        axs[1, i].plot(t, pb_v[:, i], **styles['PB'])
        axs[1, i].set_title(f'Velocity - Achse {axes_names[i]}')
        axs[1, i].grid(True, linestyle=':', alpha=0.6)
        if i == 0: axs[1, i].set_ylabel('Vel [m/s]')

        # --- ZEILE 3: Beschleunigung ---
        axs[2, i].plot(t, ref_a[:, i], **styles['Ref'])
        axs[2, i].plot(t, tsid_a[:, i], **styles['TSID'])
        axs[2, i].plot(t, pb_a[:, i], **styles['PB'])
        axs[2, i].set_title(f'Acceleration - Achse {axes_names[i]}')
        axs[2, i].grid(True, linestyle=':', alpha=0.6)
        axs[2, i].set_xlabel('Time [s]')
        if i == 0: axs[2, i].set_ylabel('Acc [m/s²]')

    # Nur eine einzige, saubere Legende für das gesamte Fenster oben anzeigen
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.98))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Platz oben für die Gesamtlegende lassen
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

    # Move the COM reference above the right foot in XY and keep the current COM height.
    p_com0 = tw.comState().pos().copy()
    p_rf = tw.get_placement_RF().translation.copy()
    p_com0[:2] = p_rf[:2]
    tw.setComRefState(p_com0)

    # Debugprints
    print('[standing] COM reference moved to RF XY:', np.round(p_com0, 4))
    print('[standing] URDF:', tsid_conf.urdf)
    print('[standing] q_home base xyz:', q_home[:3])
    print('[standing] q_init base xyz:', q_init[:3])
    print('[standing] basePosition passed to Robot:', q_init[:3])
    print('[standing] baseQuaternion passed to Robot:', q_init[3:7])
    print('[standing] robot base name:', robot.baseName())

    t_publish = 0.0
    debug_counter = 0
    lift_left_foot_time = 2.0
    lift_left_foot_done = False
    hand_motion_started = False

    # Add this to the main function before the try block
    com_ref_history = []
    com_tsid_history = []
    com_pb_history = []
    v_tsid_history = []
    v_pb_history = []
    a_tsid_history = []
    a_pb_history = []
    # additional histories for reference velocity/acc
    com_ref_v_history = []
    com_ref_a_history = []
    time_history = []

    try:
        while simulator.isRunning():
            # elapsed time
            t = simulator.simTime()

            # update robot state from simulator
            robot.update()

            # get pinocchio-style states
            q = robot.q()
            v = robot.v()

            # After the COM has had time to shift, lift the left foot once.
            if (not lift_left_foot_done) and t >= lift_left_foot_time:
                # Prepare a pistol squat: free the left foot and keep the left leg straight. (not working yet, no straight leg)
                init_pos_LF, q_posture = robot.setup_pistol_squat(tw, q_home, lift_height=0.3)
                lift_left_foot_done = True
                print(f'[standing] left foot lifted at t={t:.3f}s')

            # Update the COM height trajectory directly from the helper on Talos.
            p_com, v_com, a_com = robot.update_com_sine_reference(tw, p_com0, t)

            if t >= 8.0 and not hand_motion_started:
                tw.add_motion_RH(transition_time=0.0)
                hand_motion_started = True

            robot.update_right_hand_circle(tw, t)

            # update TSID and solve for torques/accelerations
            try:
                tau, dv = tw.update(q, v, t, do_solve=True)
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

            # command to the robot (torques)
            robot.setActuatedJointTorques(tau)

            # step simulation
            simulator.step()
            simulator.debug()


            # Plotting related Information:
            time_history.append(t)
            # COM reference (from our sine helper)
            com_ref_history.append(p_com.copy())
            com_ref_v_history.append(v_com.copy())
            com_ref_a_history.append(a_com.copy())

            # TSID (model) COM
            com_tsid_history.append(tw.comState().pos().copy())
            v_tsid_history.append(tw.comState().vel().copy())
            a_tsid_history.append(tw.comState().acc().copy())

            # PyBullet COM (use base world position as approximation)
            com_pb = robot.baseWorldPosition()
            com_pb_history.append(com_pb)

            # try to get linear velocity from baseWorldVelocity (first 3 entries)
            try:
                v_pb = np.asarray(robot.baseWorldVelocity())[:3]
            except Exception:
                # fallback to (possibly misspelled) linear velocity method
                try:
                    v_pb = np.asarray(robot.baseWorldLinearVeloctiy())
                except Exception:
                    v_pb = np.zeros(3)
            v_pb_history.append(v_pb.copy())

            # numerical acceleration for pybullet
            if len(time_history) >= 2:
                dt = time_history[-1] - time_history[-2]
                if dt == 0:
                    a_pb = np.zeros(3)
                else:
                    a_pb = (v_pb_history[-1] - v_pb_history[-2]) / dt
            else:
                a_pb = np.zeros(3)
            a_pb_history.append(a_pb)

            # publish to ROS at ~30Hz
            if t - t_publish > 1.0 / 30.0:
                t_publish = t
                rclpy.spin_once(robot, timeout_sec=0.0)
                robot.publish()

            debug_counter += 1
    finally:
        simulator.disconnect()
        robot.destroy_node()
        rclpy.shutdown()

        # ensure all history lists have same length for plotting
        histories = {
            'time_history': time_history,
            'com_ref_history': com_ref_history,
            'com_ref_v_history': com_ref_v_history,
            'com_ref_a_history': com_ref_a_history,
            'com_tsid_history': com_tsid_history,
            'com_pb_history': com_pb_history,
            'v_tsid_history': v_tsid_history,
            'v_pb_history': v_pb_history,
            'a_tsid_history': a_tsid_history,
            'a_pb_history': a_pb_history,
        }

        lengths = {k: len(v) for k, v in histories.items()}
        print('[plot] history lengths before trim:', lengths)
        min_len = min(lengths.values()) if len(lengths) > 0 else 0
        if any(l != min_len for l in lengths.values()):
            print(f'[plot] trimming all histories to min length {min_len} to avoid plotting errors')
            for k in histories:
                histories[k] = histories[k][:min_len]

        plot_com_trajectories(
            tw,
            robot,
            histories['time_history'],
            histories['com_ref_history'],
            histories['com_ref_v_history'],
            histories['com_ref_a_history'],
            histories['com_tsid_history'],
            histories['com_pb_history'],
            histories['v_tsid_history'],
            histories['v_pb_history'],
            histories['a_tsid_history'],
            histories['a_pb_history'],
        )
    
if __name__ == '__main__': 
    main()
