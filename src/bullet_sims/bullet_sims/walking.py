"""
talos walking simulation
"""

import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from enum import Enum


# simulator
#>>>>TODO: Fix include
from simulator.robot import Robot
from simulator.mujoco_wrapper import MujocoWrapper as PybulletWrapper

# whole-body controller
#>>>>TODO: Fix include
from bullet_sims.tsid_wrapper import TSIDWrapper

# robot configs
#>>>>TODO: Fix include
import bullet_sims.talos_conf as conf

#>>>>TODO: Fix include
from bullet_sims.footstep_planner import Side, FootStepPlanner, FootStep, other_foot_id
from bullet_sims.foot_trajectory import SwingFootTrajectory
from bullet_sims.talos import Talos
from bullet_sims.lip_mpc import LIPMPC, LIPInterpolator, generate_zmp_reference

# ROS visualizations
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from visualization_msgs.msg import MarkerArray, Marker
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped


class ContactState(Enum):
    SUPPORT = "SUPPORT"
    SWING = "SWING"
    DOUBLE_SUPPORT = "DOUBLE_SUPPORT"


def _foot_joint_name(side):
    if side == Side.RIGHT:
        return 'leg_right_6_joint'
    return 'leg_left_6_joint'


def _read_foot_force_z(robot, side, client_id):
    state = robot._sim.getJointState(
        robot.id(),
        robot.jointNameIndexMap()[_foot_joint_name(side)],
        physicsClientId=client_id,
    )
    return float(state[2][2])


def main(): 
    
    ############################################################################
    # setup
    ############################################################################
    
    tsid_conf = conf

    # Roboter startet flach auf dem Boden (Gelenke auf 0)
    q_home = tsid_conf.q_home.copy()
    q_init = np.zeros_like(q_home)      
    q_init[:3] = q_home[:3]
    q_init[3:7] = np.array([0.0, 0.0, 0.0, 1.0])

    # Instanziierung des TSID Wrappers
    try:
        tw = TSIDWrapper(tsid_conf)
    except Exception as e:
        print('Failed to create TSIDWrapper:', e)
        return

    # Simulator initialisieren (1000 Hz basierend auf tsid_conf.dt = 0.001)
    sim_freq = int(1.0 / tsid_conf.dt)
    simulator = PybulletWrapper(sim_rate=sim_freq)

    # Erzeuge den Roboter mit dem Pinocchio-Modell aus TSID
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

    # Wir verknüpfen den TSIDWrapper mit der Roboter-Instanz
    robot.stack = tw

    # Initial Footsteps aus der Kinematik holen.
    # Wichtig: support_side ist die Seite des Fußes, der im Moment wirklich Kontakt hat.
    support_side = Side.RIGHT
    swing_side = other_foot_id(support_side)
    robot.setSupportFoot(support_side)
    # robot.setSwingFoot(swing_side)
    T_support_w = robot.supportFootPose()
    T_swing_w = robot.swingFootPose()

    # Kontakt-Überwachung: pro Bein ein Zustand plus gefilterte Fußkraft.
    contact_state = {
        Side.LEFT: ContactState.SWING,
        Side.RIGHT: ContactState.SUPPORT,
    }
    filtered_foot_force_z = {
        Side.LEFT: 0.0,
        Side.RIGHT: 0.0,
    }
    stable_contact_counter = {
        Side.LEFT: 0,
        Side.RIGHT: 0,
    }

    force_threshold = float(getattr(tsid_conf, 'force_threshold', 100.0))
    force_filter_alpha = float(getattr(tsid_conf, 'force_filter_alpha', 0.2))
    force_stable_samples = int(getattr(tsid_conf, 'force_stable_samples', 5))
    pending_support_side = None

    # --- Double-Support / sanfter Kontaktwechsel ---------------------------
    # ds_duration: Zeit (s), in der BEIDE Füße Kontakt halten, bevor der alte
    # Standfuß gelöst wird. release_transition: Zeitfenster, über das TSID die
    # Kontaktkraft des alten Fußes danach sanft abbaut (kein harter Sprung).
    ds_duration = float(getattr(tsid_conf, 'double_support_dur', 0.08))   # ~80 ms
    release_transition = float(getattr(tsid_conf, 'contact_release_dur', 0.05))
    double_support_active = False
    ds_t_elapsed = 0.0
    old_support_side = None
    # CoM-Referenz-Blending: Snapshot zu Beginn der Double-Support-Phase und
    # Ziel (über dem neuen Standfuß). Wird per Min-Jerk ruckfrei überblendet.
    com_blend_from = None
    com_blend_to = None
    
    # Setup the planner with 20 steps
    no_steps = 6
    planner = FootStepPlanner(tsid_conf)
    
    # Wichtig: der Plan startet immer mit dem aktuellen Support-Fuß.
    # Der erste Swing-Fuß ist dann automatisch die andere Seite.
    steps = planner.planLine(T_support_w,T_swing_w, support_side=support_side, swing_side=swing_side, no_steps=no_steps)
    
    # Append the two last steps once more to the plan so our mpc horizon will never run out 
    steps.append(steps[-2])
    steps.append(steps[-1])

    # Generate reference über deine Funktion aus lip_mpc.py und Variablen aus der Config
    # Verwende tsid_conf.no_mpc_samples_per_step (aus der Config)
    # Für die Berechnung des Schwerpunkts fügen wir den Start-Fuß einmalig als erstes Standbein ein
    zmp_steps = [planner.initial_support] + steps
    ZMP_ref = generate_zmp_reference(zmp_steps, tsid_conf.no_mpc_samples_per_step)
    
    # Plot the plan in the simulator GUI zur visuellen Kontrolle
    planner.plot(simulator)
    
    # Setup the mpc & interpolator aus lip_mpc.py
    mpc = LIPMPC(tsid_conf)
    
    # Assume the com is over the current support foot (LEFT) -> [cx, vx, cy, vy]
    x0 = np.array([T_support_w.translation[0], 0.0, T_support_w.translation[1], 0.0])
    interpolator = LIPInterpolator(x0, tsid_conf)
    
    # Set the com task reference to the initial support foot
    c, c_dot, c_ddot = interpolator.comState()
    tw.setComRefState(c, c_dot, c_ddot)

    ############################################################################
    # logging & timing setup
    ############################################################################

    pre_dur = 1.5                      # Zeit für die Schwerpunktsverlagerung vor dem Gehen
    dt_sim = tsid_conf.dt              # Controller Zeitschritt (0.001s)
    
    # Berechne Anzahl der Iterationen mit den korrekten Variablen
    N_pre = int(pre_dur / dt_sim)
    N_sim = int((no_steps * tsid_conf.step_dur) / dt_sim)
    
    # Erzeuge Vektoren für das Daten-Logging
    LOG_COM_POS_REF = np.zeros((N_sim, 3))
    LOG_COM_POS_PIN = np.zeros((N_sim, 3))
    LOG_ZMP_REF     = np.zeros((N_sim, 2))
    LOG_ZMP_EST     = np.zeros((N_sim, 3))
    LOG_DCM_EST     = np.zeros((N_sim, 3))
    TIME            = np.nan * np.empty(N_sim)
    
    ############################################################################
    # execution loop
    ############################################################################
    
    k = 0                                               # Aktueller MPC-Index                          
    plan_idx = 0                                        # Aktueller Schritt-Index im Plan
    t_step_elapsed = 0.0                                # Abgelaufene Zeit im aktuellen Schritt
    t_publish = 0.0                                     # Zeitstempel der letzten ROS-Ausgabe

    crash_triggered = False
    tau_last = np.zeros(robot.numActuatedJoints())

    # Nutzen deiner SwingFootTrajectory-Klasse für die Flugbahn des Fußes
    # Beachte Variablen-Mapping: duration=tsid_conf.step_dur

    # Falls step_height nicht in deiner Config definiert ist, nutzen wir einen sicheren Standardwert (z.B. 0.05)
    step_height_val = 0.3 
    # getattr(tsid_conf, 'h', 0.5)

    # Jetzt holen wir die Pose des aktuellen (noch in Kontakt stehenden) Swing-Fußes ab.
    # Bis der erste Schritt wirklich getriggert wird (i % no_sim_per_step == 0, i > 0),
    # steht der Fuß noch im Rigid-Contact. Wir bauen die Trajektorie daher zunächst als
    # "no-op" (T0 == T1), damit der Fuß nicht gegen einen falschen Sollwert gezogen wird.
    T_start_swing = robot.swingFootPose()

    # Flugbahn-Spline mit Start == Ziel initialisieren (wird beim ersten Schritt resettet)
    foot_trajectory = SwingFootTrajectory(
        T0=T_start_swing,
        T1=T_start_swing,
        duration=tsid_conf.step_dur,
        height=step_height_val
    )
    
    # Das anfängliche uk des MPC auf den aktuellen, realen Standfuß setzen!
    T_real_support = robot.supportFootPose()
    uk = np.array([T_real_support.translation[0], T_real_support.translation[1]])

    print("Starte TALOS Fortbewegungssimulation mit angepassten Config-Variablen...")

    try:
        for i in range(-N_pre, N_sim):
            t = simulator.simTime()
            dt = dt_sim

            ########################################################################
            # contact state machine
            ########################################################################
            for side in (Side.LEFT, Side.RIGHT):
                client_id = getattr(simulator, 'physicsClientId', 0)
                raw_force_z = _read_foot_force_z(robot, side, client_id)
                filtered_foot_force_z[side] = (
                    force_filter_alpha * raw_force_z
                    + (1.0 - force_filter_alpha) * filtered_foot_force_z[side]
                )

                if filtered_foot_force_z[side] >= force_threshold:
                    stable_contact_counter[side] += 1
                else:
                    stable_contact_counter[side] = 0

            contact_confirmed = {
                side: stable_contact_counter[side] >= force_stable_samples
                for side in (Side.LEFT, Side.RIGHT)
            }

            min_swing_time = 0.2 * tsid_conf.step_dur  # z.B. 100ms bei 500ms Schrittdauer

            if (pending_support_side is not None 
                    and t_step_elapsed > min_swing_time 
                    and contact_confirmed[pending_support_side]):
                
                support_side = pending_support_side
                swing_side = Side.RIGHT if support_side == Side.LEFT else Side.LEFT

                # Erst JETZT, bei echter Landung, umschalten!
                robot.setSupportFoot(support_side)
                robot.setSwingFoot(swing_side, transition_time=release_transition)

                contact_state[support_side] = ContactState.SUPPORT
                contact_state[swing_side] = ContactState.SWING

                print(f"[Kontakt] {support_side.name} ECHT gelandet nach {t_step_elapsed:.2f}s! Wechsel ausgeführt.")
                
                pending_support_side = None
            
            ########################################################################
            # update the mpc every no_sim_per_mpc steps
            ########################################################################
            if i >= 0 and (i % tsid_conf.no_sim_per_mpc == 0):
                # 1. Get current LIP state from interpolator
                xk = interpolator.x
                
                # 2. Extract ZMP reference over current horizon 
                ZMP_ref_k = ZMP_ref[k : k + tsid_conf.no_mpc_samples_per_horizon]
                
                # 3. Solve mpc and get first control uk
                terminal_idx = tsid_conf.no_mpc_samples_per_horizon - 1
                uk = mpc.buildSolveOCP(xk, ZMP_ref_k, terminal_idx)
                
                # 4. Increment mpc counter
                k += 1        

            ########################################################################
            # update the foot spline 
            ########################################################################
            if i >= 0 and (i % tsid_conf.no_sim_per_step == 0):
                next_step = steps[plan_idx]

                # 2. Der neue Swing-Fuß wird in den Schwungmodus versetzt.
                robot.setSwingFoot(next_step.side, transition_time=release_transition)
                pending_support_side = next_step.side
                contact_state[next_step.side] = ContactState.SWING

                # 4. Current location of the swing foot
                T_start_swing = robot.swingFootPose()

                # 5. Plan a foot trajectory via reset() method
                foot_trajectory.reset(T_start_swing, next_step.pose)

                # 6. Reset step clock
                t_step_elapsed = 0.0
                print(f"[Schritt-Trigger] t={t:.2f}s | Schritt {plan_idx}/{no_steps} | Swing: {next_step.side.name}")
                plan_idx += 1

            ########################################################################
            # in every iteration when walking
            ########################################################################
            if i >= 0:
                # 1. Update foot trajectory
                T_ref, V_ref, A_ref = foot_trajectory.evaluate(t_step_elapsed)
                robot.updateSwingFootRef(T_ref, V_ref, A_ref)
                
                # 2. Update interpolator IMMER ungestört mit MPC Output
                interpolator.integrate(uk)
                c, c_dot, c_ddot = interpolator.comState()

                # 3. TSID bekommt IMMER die reine MPC-Referenz
                tw.setComRefState(c, c_dot, c_ddot)
                
                zmp_ref_current = uk

                robot.execute_balance_control(
                    dt=dt_sim, 
                    x_ref=c, 
                    zmp_ref_current=zmp_ref_current, 
                    tw=tw
                )
                
                t_step_elapsed += dt

            ########################################################################
            # update the simulation
            ########################################################################

            # 1. Update simulator and robot
            robot.update()
            q = robot.q()
            v = robot.v()

            # 2. Ganzkörper-WBC berechnen und QP lösen
            if not crash_triggered:
                try:
                    tau, dv = tw.update(q, v, t, do_solve=True)
                    
                    # Prüfen, ob der Solver in diesem Schritt numerisch kollabiert ist
                    if np.isnan(tau).any() or np.isnan(dv).any():
                        raise ValueError("WBC-Solver hat NaN-Werte (Not-a-Number) zurückgegeben!")
                    
                    # Wenn alles stabil ist, sichern wir die Drehmomente für den Notfall
                    tau_last = tau.copy()
                    robot.tsid_base_T, robot.tsid_base_v = tw.baseState()
                    
                except Exception as e:
                    crash_triggered = True
                    print("\n" + "="*50)
                    print(f"!!! CRASH DETEKTIERT bei i={i}, Systemzeit t={t:.3f}s !!!")
                    print(f"Fehlerursache: {e}")
                    print("="*50)
                    print(f"Letztes stabiles MPC-Kommando (uk): {uk}")
                    print(f"Aktueller Soll-Schwerpunkt (c):      {c}")
                    print(f"Aktuelle CoM-Geschwindigkeit (c_dot):{c_dot}")
                    print(f"Aktuelle CoM-Beschleunigung (c_ddot):{c_ddot}")
                    print(f"Max. Gelenkgeschwindigkeit (v_max):  {np.max(np.abs(v)):.4f}")
                    print(f"Aktueller Schritt-Index im Plan:      {plan_idx} (Swing-Fuß: {steps[plan_idx].side.name})")
                    print(f"Abgelaufene Zeit im aktuellen Schritt:{t_step_elapsed:.4f}s")
                    print("="*50)
                    print("Der Roboter verharrt jetzt in der letzten stabilen Pose für die optische Analyse.\n")
                    
                    # Ab jetzt nutzen wir das letzte funktionierende Drehmoment
                    tau = tau_last.copy()
            else:
                # Im eingefrorenen Zustand halten wir die Drehmomente konstant
                tau = tau_last.copy()

            # Schätzer für Marker-Updates füttern (nur wenn wir nicht gecrasht sind)
            if not crash_triggered:
                robot._update_zmp_estimate()
                robot._update_dcm_estimate()

            # Torques einprägen und Physik simulieren
            robot.setActuatedJointTorques(tau)
            simulator.step()
            simulator.debug()

            # Publish to ROS (~30 Hz)
            if t - t_publish > 1.0 / 30.0:
                t_publish = t
                robot.publish()
                rclpy.spin_once(robot.ros_node, timeout_sec=0.0)
                
            # Store for visualizations / logging
            if i >= 0:
                TIME[i] = t
                LOG_COM_POS_REF[i] = c
                LOG_COM_POS_PIN[i] = robot.com()
                LOG_ZMP_REF[i]     = ZMP_ref[k][0:2] if k < len(ZMP_ref) else uk
                LOG_ZMP_EST[i]     = robot.zmp
                LOG_DCM_EST[i]     = robot.dcm
                
    finally:
        print("Schließe Verbindung zum Simulator und fahre ROS 2 Node herunter...")
        simulator.disconnect()
        if rclpy.ok():
            robot.ros_node.destroy_node()
            rclpy.shutdown()

        ############################################################################
    # POST-SIMULATION PLOTS: ZMP & CoM Balance-Analyse
    ############################################################################

    # Nur plotten, wenn wir überhaupt Daten gesammelt haben
    valid_idx = ~np.isnan(TIME)
    if np.any(valid_idx):
        t_plot = TIME[valid_idx]
        com_ref = LOG_COM_POS_REF[valid_idx]
        com_real = LOG_COM_POS_PIN[valid_idx]
        zmp_ref_log = LOG_ZMP_REF[valid_idx]
        zmp_est_log = LOG_ZMP_EST[valid_idx]

        fig = plt.figure(figsize=(14, 10))
        fig.suptitle("TALOS Geh-Stabilisierungs-Analyse (ZMP vs CoM)", fontsize=14, fontweight='bold')

        # --- Plot 1: Sagittalebene (X - Vorwärtsbewegung) ---
        ax_x = fig.add_subplot(2, 2, 1)
        ax_x.plot(t_plot, com_ref[:, 0], 'r-', label='CoM Soll (MPC)', linewidth=2)
        ax_x.plot(t_plot, com_real[:, 0], 'r--', label='CoM Ist (Pinocchio)', linewidth=1.5)
        ax_x.plot(t_plot, zmp_ref_log[:, 0], 'g-', label='ZMP Soll (MPC)', alpha=0.7)
        ax_x.plot(t_plot, zmp_est_log[:, 0], 'b.', label='ZMP Schätzer (Sensoren)', markersize=3, alpha=0.5)
        ax_x.set_title("Sagittal-Dynamik (X-Achse)")
        ax_x.set_xlabel("Zeit [s]")
        ax_x.set_ylabel("Position [m]")
        ax_x.grid(True, linestyle='--')
        ax_x.legend(loc='upper left')

        # --- Plot 2: Lateralebene (Y - Seitliche Balance) ---
        ax_y = fig.add_subplot(2, 2, 2)
        ax_y.plot(t_plot, com_ref[:, 1], 'r-', label='CoM Soll (MPC)', linewidth=2)
        ax_y.plot(t_plot, com_real[:, 1], 'r--', label='CoM Ist (Pinocchio)', linewidth=1.5)
        ax_y.plot(t_plot, zmp_ref_log[:, 1], 'g-', label='ZMP Soll (MPC)', alpha=0.7)
        ax_y.plot(t_plot, zmp_est_log[:, 1], 'b.', label='ZMP Schätzer (Sensoren)', markersize=3, alpha=0.5)
        ax_y.set_title("Lateral-Dynamik (Y-Achse) <- CRITICAL FOR FALLS")
        ax_y.set_xlabel("Zeit [s]")
        ax_y.set_ylabel("Position [m]")
        ax_y.grid(True, linestyle='--')
        ax_y.legend(loc='upper left')

        # --- Plot 3: 2D Top-Down Sicht der Standfläche (Boden-Ebene) ---
        ax_2d = fig.add_subplot(2, 2, (3, 4))
        ax_2d.plot(com_real[:, 0], com_real[:, 1], 'r--', label='CoM Ist-Pfad', linewidth=2)
        ax_2d.plot(zmp_ref_log[:, 0], zmp_ref_log[:, 1], 'g-', label='ZMP Soll-Pfad (MPC)', linewidth=1.5)
        ax_2d.scatter(zmp_est_log[:, 0], zmp_est_log[:, 1], c=t_plot, cmap='jet', label='ZMP Schätzer (Zeitverlauf)', s=6, alpha=0.6)
        
        # Einzeichnen der geplanten Fußpolygone zur visuellen Begrenzung
        for idx, step in enumerate(steps):
            if idx < 6: # Die ersten paar Schritte einzeichnen, um den Crash-Bereich zu sehen
                # Extrahiere die Ecken des Fußabdrucks im Weltkoordinatensystem
                corners_w = np.zeros((4, 2))
                for c_idx in range(4):
                    p_c = step.pose.translation + step.pose.rotation.dot(step.footprint[:, c_idx])
                    corners_w[c_idx] = p_c[0:2]
                # Polygon schließen
                polygon = np.vstack([corners_w, corners_w[0]])
                color = 'green' if step.side == Side.LEFT else 'red'
                ax_2d.plot(polygon[:, 0], polygon[:, 1], color=color, linestyle=':', alpha=0.7, label=f'Fuß {step.side.name}' if idx < 2 else "")

        ax_2d.set_title("Top-Down ZMP & CoM Spurverlauf auf dem Boden")
        ax_2d.set_xlabel("X (Vorwärts) [m]")
        ax_2d.set_ylabel("Y (Lateral) [m]")
        ax_2d.grid(True, linestyle='--')
        # Doppelte Einträge in der Legende verhindern
        handles, labels = ax_2d.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_2d.legend(by_label.values(), by_label.keys(), loc='lower left')
        ax_2d.axis('equal')

        plt.tight_layout()
        plt.show()
    else:
        print("[Plotter] Keine validen Logdaten zum Plotten gefunden (Simulation vor Geitstart gecrasht).")

if __name__ == '__main__': 
    rclpy.init()
    main()
