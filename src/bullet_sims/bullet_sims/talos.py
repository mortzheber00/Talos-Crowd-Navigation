import numpy as np
import pinocchio as pin
import pybullet as pb

# simulator
#>>>>TODO: Fix include
from simulator.robot import Robot

# whole-body controller
#>>>>TODO: Fix include
from bullet_sims.tsid_wrapper import TSIDWrapper

# robot configs
#>>>>TODO: Fix include
import bullet_sims.talos_conf as conf

#>>>>TODO: Fix include
from bullet_sims.footstep_planner import Side

# ROS visualizations
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from visualization_msgs.msg import MarkerArray, Marker
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class Talos(Robot):
    """Talos robot
    combines wbc with pybullet, functions to read and set
    sensor values.
    """
    def __init__(self, simulator, urdf, model, q=None, basePosition=None, baseQuaternion=None, verbose=True, useFixedBase=False):
        self.conf = conf
        self.sim = simulator
        
        # 1. Zuerst den TSID Wrapper erstellen
        self.stack = TSIDWrapper(self.conf)
        
        # 2. Das Pinocchio-Modell aus dem Wrapper extrahieren
        pin_model = self.stack.robot.model()
        
        # 3. Den Roboter (über die geerbte Robot-Klasse) initialisieren.
        q_init = self.conf.q_home  # Startpose aus der Config
        
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

        self.simulator = simulator
        self.robot = self  
        ########################################################################
        # state
        ########################################################################
        self.support_foot = Side.RIGHT
        self.swing_foot = Side.LEFT
        
        ########################################################################
        # estimators
        ########################################################################
        self.zmp = np.zeros(3)
        self.dcm = np.zeros(3)
        
        ########################################################################
        # sensors
        ########################################################################
        # ft sensors
        #>>>>TODO: Turn on the force torque sensor in the robots feet
            # activate ankle-Torque Sensors in PyBullet Sim
        pb.enableJointForceTorqueSensor(self.robot.id(), self.robot.jointNameIndexMap()['leg_right_6_joint'], True)
        pb.enableJointForceTorqueSensor(self.robot.id(), self.robot.jointNameIndexMap()['leg_left_6_joint'], True)

        self.enable_ankle_strategy = True
        self.enable_hip_strategy = True
        self.ankle_x_d = None
        self.ankle_v_d = np.zeros(3)
        self.prev_gamma_d = np.zeros(3)

        ########################################################################
        # visualizations
        ########################################################################
        
        # Initialize ROS node if not already done
        if not rclpy.ok():
            rclpy.init()
        self.ros_node = Node('talos_controller')
        
        # Joint state publisher
        self.joint_state_pub = self.ros_node.create_publisher(
            JointState, 'joint_states', 10
        )
        
        # Floating base broadcaster (TF2 for odometry)
        self.tf_broadcaster = TransformBroadcaster(self.ros_node)
        
        # ZMP and DCM point publisher (MarkerArray for visualization)
        self.marker_pub = self.ros_node.create_publisher(
            MarkerArray, 'markers', 10
        )
        
        # Wrench publishers for left and right foot
        self.wrench_rf_pub = self.ros_node.create_publisher(
            WrenchStamped, 'foot_forces/right', 10
        )
        self.wrench_lf_pub = self.ros_node.create_publisher(
            WrenchStamped, 'foot_forces/left', 10
        )
        
        # Store joint names for publishing
        self.joint_names = list(self.robot.jointNameIndexMap().keys())
        self.joint_names = self.joint_names[1:]  # Remove base link name if present
           
    def update(self):
        """updates the robot
        """
        super().update()


    def setSupportFoot(self, side):
        """sets the the support foot of the robot on given side
        """
        # The support foot is in rigid contact with the ground and should 
        # hold the weight of the robot
        self.support_foot = side
        
        # Aktiviert den harten Kontakt auf der Support-Seite.
        # In TSID blockiert ein aktiver RigidContact die Bewegung, 
        # wodurch der MotionTask für diesen Fuß quasi "deaktiviert" bzw. überschrieben ist.
        if side == Side.RIGHT:
            self.stack.add_contact_RF()
        elif side == Side.LEFT:
            self.stack.add_contact_LF()
    
    def setSwingFoot(self, side, transition_time=0.0):
        """sets the swing foot of the robot on given side

        transition_time : Zeit (s) über die TSID die Kontaktkraft sanft
                           abklingen lässt (removeRigidContact). > 0 vermeidet
                           den abrupten Kraftsprung beim Lösen des Standbeins.
        """
        # The swing foot is not in contact and can move
        self.swing_foot = side
        
        # Entfernt den Kontakt-Constraint. 
        # Intern setzt remove_contact_XX bereits die aktuelle Pose als Referenz 
        # für den MotionTask, damit der Fuß nicht springt.
        # transition_time > 0 -> Kraft wird linear über das Zeitfenster reduziert.
        if side == Side.RIGHT:
            self.stack.remove_contact_RF(transition_time)
        elif side == Side.LEFT:
            self.stack.remove_contact_LF(transition_time)
        
    def updateSwingFootRef(self, T_swing_w, V_swing_w, A_swing_w):
        """updates the swing foot motion reference

        T_swing_w : pin.SE3 desired swing foot pose
        V_swing_w : desired linear velocity (3D) of the swing foot
        A_swing_w : desired linear acceleration (3D) of the swing foot
        """
        # The SE3 foot task expects a 6D twist/acc (linear + angular).
        # Our spline only produces a linear part, so the angular part stays zero.
        v6 = np.zeros(6)
        a6 = np.zeros(6)
        v6[:3] = np.asarray(V_swing_w).ravel()[:3]
        a6[:3] = np.asarray(A_swing_w).ravel()[:3]

        # Use the wrapper helpers, they build the TrajectorySample correctly
        # (value() accepts a pin.SE3 directly, derivatives are 6D vectors).
        if self.swing_foot == Side.RIGHT:
            self.stack.set_RF_pose_ref(T_swing_w, v6, a6)
        elif self.swing_foot == Side.LEFT:
            self.stack.set_LF_pose_ref(T_swing_w, v6, a6)

    def swingFootPose(self):
        """return the pose of the current swing foot
        """
        # Greift direkt auf die Kinematikdaten des Roboters in TSID zu
        if self.swing_foot == Side.RIGHT:
            return self.stack.robot.framePosition(self.stack.formulation.data(), self.stack.RF)
        elif self.swing_foot == Side.LEFT:
            return self.stack.robot.framePosition(self.stack.formulation.data(), self.stack.LF)
        return None
    
    def supportFootPose(self):
        """return the pose of the current support foot
        """
        if self.support_foot == Side.RIGHT:
            return self.stack.robot.framePosition(self.stack.formulation.data(), self.stack.RF)
        elif self.support_foot == Side.LEFT:
            return self.stack.robot.framePosition(self.stack.formulation.data(), self.stack.LF)
        return None
    
    def publish(self):
        """Publish ROS messages for joint states, odometry, wrenches, and markers"""
        # Einheitlichen Zeitstempel generieren
        stamp = self.ros_node.get_clock().now().to_msg()
        
        # 1. Gelenkzustände (Exakt deine funktionierende Logik!)
        js = JointState()
        js.header.stamp = stamp
        js.name = self.actuatedJointNames()
        js.position = self.actuatedJointPosition().tolist()
        js.velocity = self.actuatedJointVelocity().tolist()
        js.effort = self._tau_cmd.tolist() if hasattr(self, '_tau_cmd') else []
        self.joint_state_pub.publish(js)
        
        # 2. Floating Base Odometry / TF (Exakt deine funktionierende Logik!)
        if hasattr(self, 'tsid_base_T') and self.tsid_base_T is not None:
            world_pos = self.tsid_base_T.translation
            world_orient = pin.Quaternion(self.tsid_base_T.rotation)
        else:
            # Fallback auf PyBullet, falls TSID (noch) keine Base-Daten liefert
            world_pos, world_orient = self.baseWorldPose()
            world_orient = pin.Quaternion(world_orient)

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'world'
        t.child_frame_id = self.baseName()  # Nutzt den korrekten Namen aus der URDF (z.B. base_link)
        t.transform.translation.x = float(world_pos[0])
        t.transform.translation.y = float(world_pos[1])
        t.transform.translation.z = float(world_pos[2])
        t.transform.rotation.x = float(world_orient.x)
        t.transform.rotation.y = float(world_orient.y)
        t.transform.rotation.z = float(world_orient.z)
        t.transform.rotation.w = float(world_orient.w)
        self.tf_broadcaster.sendTransform(t)
        
        # 3. F/T-Sensoren (Wrenches) an den Füßen publizieren
        client_id = getattr(self.sim, 'physicsClientId', 0)
        
        # Rechter Fuß
        try:
            state_RF = pb.getJointState(self.robot.id(), self.robot.jointNameIndexMap()['leg_right_6_joint'], physicsClientId=client_id)
            wrench_rf = WrenchStamped()
            wrench_rf.header.stamp = stamp
            wrench_rf.header.frame_id = 'leg_right_6_link'
            ft_RF = state_RF[2]
            wrench_rf.wrench.force.x, wrench_rf.wrench.force.y, wrench_rf.wrench.force.z = float(ft_RF[0]), float(ft_RF[1]), float(ft_RF[2])
            wrench_rf.wrench.torque.x, wrench_rf.wrench.torque.y, wrench_rf.wrench.torque.z = float(ft_RF[3]), float(ft_RF[4]), float(ft_RF[5])
            self.wrench_rf_pub.publish(wrench_rf)
        except:
            pass
        
        # Linker Fuß
        try:
            state_LF = pb.getJointState(self.robot.id(), self.robot.jointNameIndexMap()['leg_left_6_joint'], physicsClientId=client_id)
            wrench_lf = WrenchStamped()
            wrench_lf.header.stamp = stamp
            wrench_lf.header.frame_id = 'leg_left_6_link'
            ft_LF = state_LF[2]
            wrench_lf.wrench.force.x, wrench_lf.wrench.force.y, wrench_lf.wrench.force.z = float(ft_LF[0]), float(ft_LF[1]), float(ft_LF[2])
            wrench_lf.wrench.torque.x, wrench_lf.wrench.torque.y, wrench_lf.wrench.torque.z = float(ft_LF[3]), float(ft_LF[4]), float(ft_LF[5])
            self.wrench_lf_pub.publish(wrench_lf)
        except:
            pass
        
        # 4. ZMP und DCM Marker für RViz erzeugen
        marker_array = MarkerArray()
        marker_array.markers.append(self._create_marker(0, self.zmp, [1.0, 0.0, 0.0, 1.0], 0.04, 'world', 'zmp', stamp))
        marker_array.markers.append(self._create_marker(1, self.dcm, [0.0, 0.0, 1.0, 1.0], 0.04, 'world', 'dcm', stamp))
        # self.marker_pub.publish(marker_array)

    def _create_marker(self, marker_id, position, color, scale, frame_id, ns, timestamp):
        """Helper function to create a sphere marker"""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = timestamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2])
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        
        return marker

    ############################################################################
    # balance control
    ############################################################################

    def apply_ankle_strategy(self, dt, p_global, x_ref, zmp_ref_current):
        """Berechnet die Ankle Strategy dynamisch fürs Gehen."""
        if not self.enable_ankle_strategy:
            return x_ref, np.zeros(3)

        if self.ankle_x_d is None:
            self.ankle_x_d = x_ref.copy()

        omega = np.sqrt(9.81 / x_ref[2])
        Kp = 0.8 * omega  
        Kx = 70 * omega  
        
        # WICHTIG: Die Referenz ist jetzt der dynamische Soll-ZMP vom MPC/Interpolator!
        p_ref = np.array([zmp_ref_current[0], zmp_ref_current[1], 0.0])
        v_ref = np.zeros(3) # Kann um CoM_dot erweitert werden, falls nötig

        # Regelgesetz bleibt identisch, arbeitet aber mit wandernden Koordinaten
        self.ankle_v_d[0] = v_ref[0] - Kx * (self.ankle_x_d[0] - x_ref[0]) + Kp * (p_global[0] - p_ref[0])
        self.ankle_v_d[1] = v_ref[1] - Kx * (self.ankle_x_d[1] - x_ref[1]) + Kp * (p_global[1] - p_ref[1])
        
        self.ankle_x_d += self.ankle_v_d * dt
        self.ankle_x_d[2] = x_ref[2] # Soll-Höhe zwingend beibehalten

        return self.ankle_x_d, self.ankle_v_d

    def apply_hip_strategy(self, dt, cmp_global, zmp_ref_current, tw):
        """Berechnet die Hip Strategy basierend auf der Abweichung zum aktuellen MPC-Soll-ZMP."""
        if not self.enable_hip_strategy:
            tw.setAmRefState(np.zeros(3), np.zeros(3))
            return

        K_gamma = 9 # 4.2 
        
        # Auch hier: Abweichung zum aktuellen, wandernden ZMP berechnen
        r_ref = np.array([zmp_ref_current[0], zmp_ref_current[1], 0.0])
        r     = np.array([cmp_global[0], cmp_global[1], 0.0])

        cmp_error = r - r_ref
        
        gamma_d = np.zeros(3)
        gamma_d[0] = -K_gamma * cmp_error[1] 
        gamma_d[1] =  K_gamma * cmp_error[0] 
        gamma_d[2] =  0.0                     

        if not hasattr(self, 'prev_gamma_d'):
            self.prev_gamma_d = np.zeros(3)
            
        d_gamma_d = (gamma_d - self.prev_gamma_d) / dt
        self.prev_gamma_d = gamma_d.copy()

        tw.setAmRefState(gamma_d, d_gamma_d)

    def execute_balance_control(self, dt, x_ref, zmp_ref_current, tw):
        """Manager-Funktion, die in der main aufgerufen wird."""
        # Hol dir die echten physikalischen Schätzwerte direkt aus der Talos-Sensorik
        # (Ersetze dies durch deine echten Schätzer-Namen, z.B. self.zmp_estimate oder ähnlich)
        self._update_dcm_estimate()
        self._update_zmp_estimate()
        p_global = self.zmp
        cmp_global = p_global # Oft synonym verwendet, falls kein separater CMP-Schätzer da ist
        
        if p_global is None or np.isnan(p_global).any():
            # Fallback, falls Schätzer noch nicht eingeschaltet ist
            tw.setComRefState(x_ref, np.zeros(3))
            tw.setAmRefState(np.zeros(3), np.zeros(3))
            return

        # 1. Ankle Strategy
        x_target, v_target = self.apply_ankle_strategy(dt, p_global, x_ref, zmp_ref_current)
        tw.setComRefState(x_target, v_target)
        
        # 2. Hip Strategy
        self.apply_hip_strategy(dt, cmp_global, zmp_ref_current, tw)

    ############################################################################
    # private funcitons
    ############################################################################

    def _solve(self, t, dt):
        # get the current state
        q = self.robot.q()
        v = self.robot.v()
        
        # 1. Update den TSID-Wrapper mit dem aktuellen Zustand des Roboters
        self.stack.integrate_kinematics(q, v, dt)
        
        # 2. Löse das Whole-Body QP
        # Sie liefert die berechneten Drehmomente (tau) zurück
        H_data = self.stack.formulation.computeProblemData(t, q, v)
        sol = self.stack.solver.solve(H_data)
        tau = self.stack.formulation.getActuatorForces(sol)
        
        # 3. Drehmomente an den PyBullet-Simulator übergeben
        self.robot.setActuatedJointTorques(tau)
    
    def _update_zmp_estimate(self):
        """update the estimated zmp position
        """
        import pybullet as pb
        client_id = getattr(self.sim, 'physicsClientId', 0)
        
        # Gelenk-Zustände auslesen
        state_RF = pb.getJointState(self.robot.id(), self.robot.jointNameIndexMap()['leg_right_6_joint'], physicsClientId=client_id)
        state_LF = pb.getJointState(self.robot.id(), self.robot.jointNameIndexMap()['leg_left_6_joint'], physicsClientId=client_id)
        
        ft_RF = np.array(state_RF[2])
        ft_LF = np.array(state_LF[2])
        
        # Transformationen holen
        T_rf_w = self.supportFootPose() if self.support_foot == Side.RIGHT else self.swingFootPose()
        T_lf_w = self.supportFootPose() if self.support_foot == Side.LEFT else self.swingFootPose()
        
        # Lokale ZMPs berechnen (Sicherheits-Check gegen Division durch ~0)
        zmp_rf_local = np.array([-ft_RF[4]/ft_RF[2], ft_RF[3]/ft_RF[2], 0.0]) if abs(ft_RF[2]) > 1.0 else np.zeros(3)
        zmp_lf_local = np.array([-ft_LF[4]/ft_LF[2], ft_LF[3]/ft_LF[2], 0.0]) if abs(ft_LF[2]) > 1.0 else np.zeros(3)
        
        # Mit .act() sauber in Weltkoordinaten transformieren
        zmp_rf_w = T_rf_w.act(zmp_rf_local) if T_rf_w is not None else np.zeros(3)
        zmp_lf_w = T_lf_w.act(zmp_lf_local) if T_lf_w is not None else np.zeros(3)
        
        # Globalen ZMP mitteln
        fz_total = ft_RF[2] + ft_LF[2]
        if fz_total > 10.0:  # Schwelle etwas absenken
            self.zmp = (ft_RF[2] * zmp_rf_w + ft_LF[2] * zmp_lf_w) / fz_total
        else:
            # Fallback: Projektion des CoM auf den Boden (z=0)
            # Falls self.robot.com() nicht existiert, nutze pinocchio direkt über self.stack
            try:
                com_pos = self.robot.com()
            except AttributeError:
                com_pos = self.stack.robot.com(self.stack.formulation.data())
            
            self.zmp = np.array([com_pos[0], com_pos[1], 0.0])
        
    def com(self):
        """Gibt die aktuelle Center of Mass (CoM) Position (3D-Vektor) zurück."""
        # Der RobotWrapper in TSID hat meistens eine direkte com()-Methode,
        # die das Datenobjekt intern mitschickt:
        try:
            return self.stack.robot.com()
        except TypeError:
            # Fallback, falls deine Version das Daten-Objekt explizit fordert:
            return self.stack.robot.com(self.stack.formulation.data())

    def com_velocity(self):
        """Gibt die aktuelle Center of Mass (CoM) Geschwindigkeit (3D-Vektor) zurück."""
        # Da 'com_velocity' nicht direkt existiert, nutzen wir pinocchio.
        # Pinocchio speichert die CoM-Geschwindigkeit im 'vcom'-Attribut der Datenstruktur:
        data = self.stack.formulation.data()
        
        # In den allermeisten Pinocchio/TSID-Versionen greift man so darauf zu:
        if hasattr(data, 'vcom'):
            # vcom ist eine Liste/Array von Geschwindigkeiten (oft für jedes Gelenk),
            # der Gesamt-CoM des Roboters steht ganz vorne am Index 0
            return np.array(data.vcom[0])

    def _update_dcm_estimate(self):
        """update the estimated dcm position
        """
        # Holen der Center of Mass Position und Geschwindigkeit aus der Roboter-Kinematik
        com_pos = self.robot.com()
        com_vel = self.robot.com_velocity()
        
        g = 9.81
        z_c = max(com_pos[2], 0.1) # Höhe des CoM absichern gegen Nulldivision
        omega_0 = np.sqrt(g / z_c)
        
        # DCM Formel: xi = c + c_dot / omega_0
        self.dcm = com_pos + (com_vel / omega_0)

def run_test():
    from simulator.pybullet_wrapper import PybulletWrapper
    from footstep_planner import Side
    from talos import Talos

    print("Starte TALOS Walking WBC Test...")
    
# Konfiguration laden
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

    # Simulator initialisieren (1000 Hz)
    simulator = PybulletWrapper(sim_rate=1000)

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

    # p_com = tw.comState().pos().copy()
    # p_rf = tw.get_placement_RF().translation.copy()
    # p_com[:2] = p_rf[:2]
    # tw.setComRefState(p_com)

    # Wir verknüpfen den TSIDWrapper mit der Roboter-Instanz, 
    # damit 'robot.stack' innerhalb der Klasse auf den echten Wrapper zeigt!
    robot.stack = tw
    
    # Neue Zustands-Merker vor der Schleife definieren
    com_shifted = False
    walking_initialized = False
    T_start_swing = None
    motion_duration = 1.0  
    t_publish = 0.0  # Zeitstempel für die Publishing-Rate
    debug_counter = 0

    try:
        while simulator.isRunning():
            t = simulator.simTime()
            robot.update()
            q = robot.q()
            v = robot.v()

            # Schritt 1: Nach 1.0 Sekunde stabilen Stehens verschieben wir die CoM-Referenz 
            # über den rechten Fuß, damit der Roboter sein Gewicht verlagert.
            if t >= 1.0 and not com_shifted:
                p_com = tw.comState().pos().copy()
                p_rf = tw.get_placement_RF().translation.copy()
                p_com[:2] = p_rf[:2]  # XY-Koordinaten auf den rechten Fuß setzen
                tw.setComRefState(p_com)
                com_shifted = True
                print("[Walking Test] Verschiebe CoM-Referenz über den rechten Fuß...")

            # # Schritt 2: Erst nach 2.5 Sekunden (wenn das Gewicht physikalisch drüben ist)
            # # nehmen wir den linken Fuß sicher als Swing Foot hoch.
            # if t >= 2.5 and not walking_initialized:
            #     print("\n--- [Walking Test] Schalte um: Rechter Fuß = SUPPORT, Linker Fuß = SWING ---")
            #     robot.setSupportFoot(Side.RIGHT)
            #     robot.setSwingFoot(Side.LEFT)
                
            #     T_start_swing = robot.swingFootPose()
            #     t_start_walk = t
            #     walking_initialized = True

            # # Schritt 3: Trajektorie für den schwebenden Fuß füttern
            # if walking_initialized and (t - t_start_walk) <= motion_duration:
            #     t_local = t - t_start_walk
            #     z_offset = 0.08 * np.sin(np.pi * (t_local / motion_duration))
                
            #     # Nutzen von pin.SE3 zur sicheren Kopie
            #     T_target = pin.SE3(T_start_swing)
            #     T_target.translation[2] += z_offset
                
            #     v_target = pin.Motion.Zero()
            #     a_target = pin.Motion.Zero()
                
            #     robot.updateSwingFootRef(T_target, v_target, a_target)

            # 3. TSID Update und QP lösen
            try:
                tau, dv = tw.update(q, v, t, do_solve=True)
                robot.tsid_base_T, robot.tsid_base_v = tw.baseState()
            except Exception as e:
                print('TSID update error:', e)
                tau = np.zeros(robot.numActuatedJoints())
                dv = np.zeros(robot._v.shape)

            # Debug-Ausgabe zur Kontrolle der NaNs
            if debug_counter % 100 == 0:
                print(f't={t:.3f}s | norm(tau)={np.linalg.norm(tau):.2f} | ZMP: {np.round(robot.zmp[:2], 3)}')

            # --- ZMP & DCM SCHÄTZER UPDATEN ---
            # Wichtig, damit robot.publish() nicht die Initialwerte (0,0,0) publiziert!
            robot._update_zmp_estimate()
            robot._update_dcm_estimate()

            robot.setActuatedJointTorques(tau)
            simulator.step()
            simulator.debug()

            # --- ROS PUBLISH & SPIN (~30Hz) ---
            if t - t_publish > 1.0 / 30.0:
                t_publish = t
                # 1. Daten über die internen Publisher herausschicken
                robot.publish()
                # 2. Den ROS 2 Node kurz "atmen" lassen, um Callbacks/Daten zu verarbeiten
                rclpy.spin_once(robot.ros_node, timeout_sec=0.0)

            debug_counter += 1
            
    finally:
        simulator.disconnect()
        # Node sauber herunterfahren
        robot.ros_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    run_test()