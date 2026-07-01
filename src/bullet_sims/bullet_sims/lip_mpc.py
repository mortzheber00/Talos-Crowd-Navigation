import numpy as np
from scipy.linalg import expm
from pydrake.all import MathematicalProgram, Solve

################################################################################
# Helper fnc
################################################################################

def continuous_LIP_dynamics(g, h):
    """returns the matrices A,B of the continuous LIP dynamics

    Args:
        g (_type_): gravity
        h (_type_): fixed height

    Returns:
        np.array: A, B
    """
    #>>>>TODO: Generate A, B for the continous linear inverted pendulum
    #>>>>Hint: Look at Eq. 4 and rewrite as a system first order diff. eq.
    omega2 = g / h
    # state x = [c, c_dot], input u = p (ZMP)
    # c_ddot = (g/h) * (c - p)
    A = np.array([[0.0, 1.0],
                  [omega2, 0.0]])
    B = np.array([[0.0],
                  [-omega2]])
    return A, B

def discrete_LIP_dynamics(delta_t, g, h):
    """returns the matrices static Ad,Bd of the discretized LIP dynamics

    Args:
        delta_t (_type_): discretization steps
        g (_type_): gravity
        h (_type_): height

    Returns:
        _type_: _description_
    """
    # Generate Ad, Bd for the discretized linear inverted pendulum
    # continuous: x = [c, c_dot], x_dot = A x + B p with
    # A = [[0,1],[omega2,0]] and B = [[0],[-omega2]] where omega2 = g/h
    omega2 = g / h
    omega = np.sqrt(omega2)
    ct = np.cosh(omega * delta_t)
    st = np.sinh(omega * delta_t)

    Ad = np.array([[ct, (1.0 / omega) * st],
                   [omega * st, ct]])

    # For constant input p over the interval, Bd follows from the analytical
    # solution: c(t+dt) = Ad[0,:] x + (1 - cosh) * p, c_dot term includes -omega*sinh
    Bd = np.array([[1.0 - ct],
                   [-omega * st]])

    return Ad, Bd

################################################################################
# LIPInterpolator
################################################################################

class LIPInterpolator:
    """Integrates the linear inverted pendulum model using the 
    continous dynamics. To interpolate the solution to high frequency.
    """
    def __init__(self, x_inital, conf):
        self.conf = conf
        self.dt = conf.dt
        self.x = np.array(x_inital)
        
        self.g = 9.81
        self.h = conf.h
        self.omega = np.sqrt(self.g / self.h)
        
        # Hier merken wir uns den aktuellen ZMP-Steuerbefehl
        # Initial setzen wir ihn auf die Startposition (zMP liegt unter dem CoM)
        self.current_u = np.array([self.x[0], self.x[2]])

    def integrate(self, u):
        """Integrate the continuous LIP dynamics for one time step dt"""
        self.current_u = np.array(u)
        
        cx, vx, cy, vy = self.x[0], self.x[1], self.x[2], self.x[3]
        px, py = self.current_u[0], self.current_u[1]
        
        ax = (self.omega**2) * (cx - px)
        ay = (self.omega**2) * (cy - py)
        
        vx_next = vx + ax * self.dt
        cx_next = cx + vx_next * self.dt
        vy_next = vy + ay * self.dt
        cy_next = cy + vy_next * self.dt
        
        self.x = np.array([cx_next, vx_next, cy_next, vy_next])
        return self.x
    
    def comState(self):
        """Returns the center of mass state in 3D (No recursion!)"""
        cx, vx, cy, vy = self.x[0], self.x[1], self.x[2], self.x[3]
        
        c = np.array([cx, cy, self.h])
        c_dot = np.array([vx, vy, 0.0])
        
        px, py = self.current_u[0], self.current_u[1]
        ax = (self.omega**2) * (cx - px)
        ay = (self.omega**2) * (cy - py)
        c_ddot = np.array([ax, ay, 0.0])
        
        return c, c_dot, c_ddot
    
    def zmp(self):
        """Returns the current 3D ZMP position on the floor (No recursion!)"""
        # DIREKTZUGRIFF: Gibt einfach den gespeicherten Wert als 3D-Vektor zurück
        return np.array([self.current_u[0], self.current_u[1], 0.0])
        
    
################################################################################
# LIPMPC
################################################################################

class LIPMPC:
    def __init__(self, conf):
        self.conf = conf
        self.dt = conf.dt_mpc  # MPC-Schrittweite (oft gröber als Sim-dt)
        self.no_samples = conf.no_mpc_samples_per_horizon
        
        # Abmessungen der Fußsohle aus der Konfig
        self.foot_length = conf.lfxp - conf.lfxn
        self.foot_width = conf.lfyp - conf.lfyn
        
        # Systemmatrizen einmalig berechnen (Diskrete LIP-Dynamik)
        self.g = 9.81
        self.h = conf.h
        omega = np.sqrt(self.g / self.h)
        
        # Kontinuierliche Matrizen für x und y entkoppelt
        # Zustand: [x, x_dot]
        A_c = np.array([[0.0, 1.0], [omega**2, 0.0]])
        B_c = np.array([[0.0], [-omega**2]])
        
        # Exakte Diskretisierung via Matrix-Exponentialfunktion (oder Taylor)
        # Da A_c klein ist, nutzen wir die analytische Form oder expm
        M = expm(np.block([[A_c, B_c], [np.zeros((1, 2)), 0.0]]) * self.dt)
        self.A_discrete = M[0:2, 0:2]
        self.B_discrete = M[0:2, 2:3]
        
        # Zusammenbauen zu 4D (X und Y kombiniert)
        self.A_big = np.zeros((4, 4))
        self.A_big[0:2, 0:2] = self.A_discrete
        self.A_big[2:4, 2:4] = self.A_discrete
        
        self.B_big = np.zeros((4, 2))
        self.B_big[0:2, 0:1] = self.B_discrete
        self.B_big[2:4, 1:2] = self.B_discrete

        # Gewichte für die Kostenfunktion
        self.alpha = conf.alpha  # ZMP Tracking-Gewicht
        self.gamma = conf.gamma  # CoM Glättungs-Gewicht

        # Speicher für Lösungen
        self.X_k = None
        self.U_k = None
        self.ZMP_ref_k = None
        
    def buildSolveOCP(self, x_k, ZMP_ref_k, terminal_idx):
        """Builds and solves the local OCP (the MPC iteration)"""
        self.ZMP_ref_k = ZMP_ref_k
        
        nx = 4  # [cx, vx, cy, vy]
        nu = 2  # [px, py]
        
        prog = MathematicalProgram()
        
        # Variablen über den Horizont anlegen
        state = prog.NewContinuousVariables(self.no_samples, nx, 'state')
        control = prog.NewContinuousVariables(self.no_samples, nu, 'control')
        
        # 1. Startzustand festlegen
        prog.AddBoundingBoxConstraint(x_k, x_k, state[0])
    
        # 2. Dynamik-Constraints einpflegen
        for k in range(self.no_samples - 1):
            dynamics_error = state[k+1] - (self.A_big @ state[k] + self.B_big @ control[k])
            prog.AddLinearEqualityConstraint(dynamics_error, np.zeros(4))
        
        # 3. ZMP innerhalb der Fußsohle halten
        half_l = self.foot_length / 2.0
        half_w = self.foot_width / 2.0

        for k in range(self.no_samples):
            zmp_center_x = ZMP_ref_k[k, 0]
            zmp_center_y = ZMP_ref_k[k, 1]
            
            zmp_x_min = zmp_center_x - half_l
            zmp_x_max = zmp_center_x + half_l
            zmp_y_min = zmp_center_y - half_w
            zmp_y_max = zmp_center_y + half_w
            
            prog.AddBoundingBoxConstraint(zmp_x_min, zmp_x_max, control[k][0])
            prog.AddBoundingBoxConstraint(zmp_y_min, zmp_y_max, control[k][1])
    
        # 4. Terminal Constraint (wird später aktiviert, falls gefordert)
        if terminal_idx < self.no_samples:
            # Beispiel für Terminal-Constraint: Geschwindigkeit am Ende auf 0 zwingen
            # prog.AddBoundingBoxConstraint(0.0, 0.0, state[terminal_idx][1]) # vx = 0
            # prog.AddBoundingBoxConstraint(0.0, 0.0, state[terminal_idx][3]) # vy = 0
            pass
            
        # 5. Kostenfunktion hinzufügen
        for k in range(self.no_samples):
            # Tracking des Referenz-ZMP
            zmp_error_x = control[k][0] - ZMP_ref_k[k, 0]
            zmp_error_y = control[k][1] - ZMP_ref_k[k, 1]
            prog.AddCost(self.alpha * (zmp_error_x**2 + zmp_error_y**2))
            
            # Glättung der CoM-Geschwindigkeiten
            v_x = state[k][1]
            v_y = state[k][3]
            prog.AddCost(self.gamma * (v_x**2 + v_y**2))
            
        # solve
        result = Solve(prog)
        if not result.is_success:
            print("failure")
        
        if not result.is_success():
            print("Warnung: MPC konnte das Problem nicht lösen!")
            
        # Ergebnisse für das Logging sichern
        self.X_k = result.GetSolution(state)
        self.U_k = result.GetSolution(control)
        
        # Gib den allerersten Steuerbefehl an den Simulator zurück
        return self.U_k[0]
    

def generate_zmp_reference(foot_steps, no_samples_per_step):
    """Generates the ZMP reference trajectory from a list of FootStep objects"""
    no_steps = len(foot_steps)
    total_samples = no_steps * no_samples_per_step
    
    zmp_ref = np.zeros((total_samples, 2))
    
    for i in range(no_steps):
        start = i * no_samples_per_step
        end = start + no_samples_per_step
        
        # Da foot_steps jetzt Instanzen der Klasse FootStep sind, 
        # holen wir uns die Position via Pinocchio aus der SE3-Pose
        step_pose = foot_steps[i].poseInWorld()
        
        # Extrahiere X und Y aus dem 3D-Translationsvektor (x, y, z)
        zmp_ref[start:end, 0] = step_pose.translation[0]
        zmp_ref[start:end, 1] = step_pose.translation[1]
        
    return zmp_ref


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from pydrake.solvers import Solve
    from footstep_planner import FootStepPlanner, Side, FootStep    
    import pinocchio as pin

    # 1. Eine einfache Dummy-Konfiguration erstellen, die alle nötigen Parameter hält
    class DummyConfig:
        # Geometrie & Planer
        step_size_x = 0.2
        step_size_y = 0.09
        lfxp, lfxn = 0.05, -0.05  # Fußlänge = 0.10m
        lfyp, lfyn = 0.03, -0.03  # Fußbreite = 0.06m
        
        # MPC & Interpolator Zeiten
        dt = 0.01                  # Sim-Schrittweite (Interpolator)
        dt_mpc = 0.1              # MPC-Schrittweite (Gröber!)
        no_mpc_samples_per_horizon = 16
        
        # Physik & Gewichte
        h = 0.5                   # Schwerpunkthöhe
        alpha = 1000.0            # ZMP-Tracking Gewicht
        gamma = 1.0               # Geschwindigkeits-Glättung Gewicht

    conf = DummyConfig()

    print("=== 1. Teste FootStepPlanner ===")
    planner = FootStepPlanner(conf)
    # Startpose: Einheitstransformationsmatrix (Roboter steht im Ursprung)
    T_0_w = pin.SE3.Identity() 
    
    # Generiere einen Plan mit 6 Schritten, beginnend mit dem linken Fuß
    foot_steps = planner.planLine(T_0_w, support_side=Side.LEFT, no_steps=6)
    print(f"Erfolgreich {len(foot_steps)} Fußschritte generiert.")
    for idx, step in enumerate(foot_steps):
        print(f" Schritt {idx}: ({step.side.name}) -> x: {step.pose.translation[0]:.2f}, y: {step.pose.translation[1]:.2f}")

    print("\n=== 2. Teste ZMP-Referenz-Generierung ===")
    # Angenommen, ein Fußschritt dauert im MPC-Raster 4 Samples
    no_samples_per_step_mpc = 4
    ZMP_ref_MPC = generate_zmp_reference(foot_steps, no_samples_per_step_mpc)
    print(f"ZMP-Referenztrajektorie generiert. Shape: {ZMP_ref_MPC.shape} (Erwartet: ({len(foot_steps)*no_samples_per_step_mpc}, 2))")

    print("\n=== 3. Teste MPC-Initialisierung ===")
    mpc = LIPMPC(conf)
    print("MPC-Klasse erfolgreich geladen. Diskrete A-Matrix Form:")
    print(mpc.A_big)

    print("\n=== 4. Teste eine einzelne MPC-Optimierung ===")
    # Initialer Zustand: Stehend auf Startposition des ersten Fußes (y = -0.09)
    # Zustand: [cx, vx, cy, vy]
    x_initial = np.array([0.0, 0.0, -0.09, 0.0])
    
    # Schneide den ersten Horizont aus der Referenz aus
    ZMP_ref_k = ZMP_ref_MPC[0 : conf.no_mpc_samples_per_horizon]
    terminal_idx = conf.no_mpc_samples_per_horizon - 1

    # Rufe deinen optimierten Solver auf
    u_0 = mpc.buildSolveOCP(x_initial, ZMP_ref_k, terminal_idx)
    print(f"MPC-Optimierung abgeschlossen! Berechneter initialer ZMP-Stellbefehl (u_0): {u_0}")

    print("\n=== 5. Teste LIPInterpolator ===")
    interpolator = LIPInterpolator(x_initial, conf)
    
    # Integriere das System für 5 Zeitschritte mit dem berechneten u_0
    print("Integriere Zustand vorwärts...")
    for step in range(5):
        x_next = interpolator.integrate(u_0)
        c, c_dot, _ = interpolator.comState()
        print(f" Schritt {step+1} (t={((step+1)*conf.dt):.2f}s) -> CoM-Pos: [{c[0]:.3f}, {c[1]:.3f}], CoM-Vel: [{c_dot[0]:.3f}, {c_dot[1]:.3f}]")

    print("\n Alles läuft fehlerfrei durch! Das mathematische Gerüst steht.")