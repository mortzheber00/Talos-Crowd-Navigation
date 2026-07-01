"""Task2: Linear inverted pendulum MPC

The goal of this file is to formulate the optimal control problem (OCP)
in equation 12 but this time as a model predictive controller (MPC).

In this case we will solve the trajectory planning multiple times over 
a shorter horizon of just 2 steps (receding horizon).
Time between two MPC updates is called T_MPC.

In between MPC updates we simulate the Linear inverted pendulum at a smaller
step time T_SIM, with the lates MPC control ouput u.

Our state & control is the same as before
x = [cx, vx, cy, vy]
u = [px, py]

You will need to fill in the TODO to solve the task.
"""

import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import MathematicalProgram, Solve

import matplotlib.animation as animation

################################################################################
# settings
################################################################################

NO_STEPS                = 8         # total number of foot steps
STEP_TIME               = 0.8       # time needed for every step

# Robot Parameters:
# --------------
h                       = 0.80      # fixed CoM height (assuming walking on a flat terrain)
g                       = 9.81      # norm of the gravity vector
foot_length             = 0.10      # foot size in the x-direction
foot_width              = 0.06      # foot size in the y-direciton


# MPC Parameters:
# --------------
T_MPC                   = 0.1                                               # sampling time interval of the MPC
NO_MPC_SAMPLES_PER_STEP = int(round(STEP_TIME/T_MPC))                       # number of mpc updates per step

NO_STEPS_PER_HORIZON  = 2                                                   # how many steps in the horizon
T_HORIZON = NO_STEPS_PER_HORIZON*STEP_TIME                                  # duration of future horizon
NO_MPC_SAMPLES_HORIZON = int(round(NO_STEPS_PER_HORIZON*STEP_TIME/T_MPC))   # number of mpc updates per horizon

# Cost Parameters:
# ---------------
alpha       = 10**(-1)                                  # ZMP error squared cost weight (= tracking cost)
gamma       = 10**(-3)                                  # CoM velocity error squared cost weight (= smoothing cost)

# Simulation Parameters:
# --------------
T_SIM                   = 0.005                         # 200 Hz simulation time

NO_SIM_SAMPLES_PER_MPC = int(round(T_MPC/T_SIM))        # NO SIM samples between MPC updates
NO_MPC_SAMPLES = int(round(NO_STEPS*STEP_TIME/T_MPC))   # Total number of MPC samples
NO_SIM_SAMPLES = int(round(NO_STEPS*STEP_TIME/T_SIM))   # Total number of Simulator samples

################################################################################
# Helper fnc
################################################################################

def generate_foot_steps(foot_step_0, step_size_x, no_steps):
    """Write a function that generates footstep of step size = step_size_x in the 
    x direction starting from foot_step_0 located at (x0, y0).
    
    Args:
        foot_step_0 (_type_): first footstep position (x0, y0)
        step_size_x (_type_): step size in x direction
        no_steps (_type_): number of steps to take
    """

    #>>>>TODO: generate the foot step plan with no_steps
    #>>>>Hint: Check the pdf Fig.3 for inspiration

    step_width_y = 0.10 
    foot_steps = np.zeros((no_steps, 2))

    for i in range(no_steps):
        # Wenn i ungerade ist, setzen wir den Fuß nach links (+), wenn gerade, nach rechts (-)
        y_offset = step_width_y if (i % 2 == 1) else -step_width_y
        
        # Der Startpunkt foot_step_0 bestimmt das globale Zentrum
        foot_steps[i] = foot_step_0 + np.array([i * step_size_x, y_offset])
    return foot_steps


def plot_foot_steps(foot_steps, XY_foot_print, ax):
    """Write a function that plots footsteps in the xy plane using the given
    footprint (length, width)
    You can use the function ax.fill() to gerneate a colored rectanges.
    Color the left and right steps different and check if the step sequence makes sense.

    Args:
        foot_steps (_type_): the foot step plan
        XY_foot_print (_type_): the dimensions of the foot (x,y)
        ax (_type_): the axis to plot on
    """
    #>>>>TODO: Plot the the footsteps into ax 
    # foot_steps: array (N,2) with centers of footprints
    l, w = XY_foot_print
    half_l = l / 2.0
    half_w = w / 2.0
    colors = ["tab:blue", "tab:orange"]
    for i, (cx, cy) in enumerate(foot_steps):
        # rectangle corners (clockwise)
        xs = [cx - half_l, cx + half_l, cx + half_l, cx - half_l]
        ys = [cy - half_w, cy - half_w, cy + half_w, cy + half_w]
        ax.fill(xs, ys, color=colors[i % 2], alpha=0.6)
    

def generate_zmp_reference(foot_steps, no_samples_per_step):
    """generate a function that computes a referecne trajecotry for the ZMP
    (We need this for the tracking cost in the cost function of eq. 12)
    Remember: Our goal is to keep the ZMP at the footstep center within each step.
    So for the # of samples a step is active the zmp_ref should be at that step.
    
    Returns a vector of size (TOTAL_NO_SAMPLES, 2)

    Args:
        foot_steps (_type_): the foot step plan
        no_samples_per_step (_type_): number of sampes per step
    """
    # Generate the ZMP reference so that for each foot step the ZMP
    # reference equals the foot step center for the duration of the step
    no_steps = foot_steps.shape[0]
    
    # Der Horizont hat so viele Zeitschritte, wie die übergebenen Schritte mal Samples dauern
    total_samples_horizon = no_steps * no_samples_per_step
    
    zmp_ref = np.zeros((total_samples_horizon, 2))

    for i in range(no_steps):
        start = i * no_samples_per_step
        end = start + no_samples_per_step
        
        # Befülle das Samplestück mit der Position des i-ten Fußschritts
        zmp_ref[start:end, :] = foot_steps[i]

    return zmp_ref



################################################################################
# Dynamics of the simplified walking model
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
# Simulation
################################################################################

class Simulator:
    """Simulates the Linear inverted pendulum continous dynamics
    Uses simple euler integration to simulate LIP at sample time dt
    """
    def __init__(self, x_inital, dt):
        self.dt = dt
        self.x = x_inital
        
        self.A, self.B = continuous_LIP_dynamics(9.81, 0.5)
        self.D = np.array([[0, 0], [1, 0], [0, 0], [0, 1]])

        self.A_big = np.block([
            [self.A, np.zeros((2, 2))],
            [np.zeros((2, 2)), self.A]
        ])

        # B_big blockdiagonal zusammensetzen
        self.B_big = np.block([
            [self.B, np.zeros((2, 1))],
            [np.zeros((2, 1)), self.B]
        ])
        
    def simulate(self, u, d=np.zeros(2)):
        """updates the LIP state x using based on command u
        
        Optionally: Takes a disturbance acceleration d to simulate effect
        of external pushes on the LIP.
        """

        #>>>>TODO: Compute x_dot and use euler integration to approximate
        # the state at t+dt
        #>>>>TODO: The disturbance is added in x_dot as self.D@d
        x_dot = self.A_big @ self.x + self.B_big @ u + self.D @ d
        self.x = self.x + x_dot * self.dt
        
        return self.x    

################################################################################
# MPC
################################################################################

class MPC:
    """MPC for the Linear inverted pendulum
    """
    def __init__(self, dt, T_horizon):
        self.dt = dt                                        # mpc dt
        self.T_horizon = T_horizon                          # time of horizon
        self.no_samples = int(round(T_horizon/self.dt))     # mpc samples in horizon (nodes)

        self.Ad, self.Bd = discrete_LIP_dynamics(dt, 9.81, 0.5)

        self.A_big = np.block([
            [self.Ad, np.zeros((2, 2))],
            [np.zeros((2, 2)), self.Ad]
        ])

        # B_big blockdiagonal zusammensetzen
        self.B_big = np.block([
            [self.Bd, np.zeros((2, 1))],
            [np.zeros((2, 1)), self.Bd]
        ])
        
        self.X_k = None                                     # state over current horizon
        self.U_k = None                                     # control over current horizon
        self.ZMP_ref_k = None                               # ZMP reference over current horizon
        
    def buildSolveOCP(self, x_k, ZMP_ref_k, terminal_idx):
        """ build the MathematicalProgram that solves the mpc problem and 
        returns the first command of U_k

        Args:
            x_k (_type_): the current state of the lip when starting the mpc
            ZMP_ref_k (_type_): the reference over the current horizon, shape=(no_samples, 2)
            terminal_idx (_type_): index of the terminal constraint within horizon (or bigger than horizon if no constraint)
            
        """
        
        # variables
        nx = 4 #>>>>TODO: State dimension = ?
        nu = 2 #>>>>TODO: control dimension = ?
        prog = MathematicalProgram()
        
        state = prog.NewContinuousVariables(self.no_samples, nx, 'state')
        control = prog.NewContinuousVariables(self.no_samples, nu, 'control')
        
        # 1. intial constraint
        #>>>>TODO: Add inital state constraint, Hint: x_k
        prog.AddBoundingBoxConstraint(x_k, x_k, state[0])
    
        # 2. at each time step: respect the LIP descretized dynamics
        #>>>>TODO: Enforce the dynamics at every time step

        for k in range(self.no_samples - 1):
            # Berechne die Differenz: x_{k+1} - (A*x_k + B*u_k)
            # Wenn die Dynamik erfüllt ist, muss dieser Vektor exakt 0 sein
            dynamics_error = state[k+1] - (self.A_big @ state[k] + self.B_big @ control[k])
            
            # Wir zwingen jede Komponente der Differenz dazu, 0 zu sein
            prog.AddLinearEqualityConstraint(dynamics_error, np.zeros(4))
        
        # 3. at each time step: keep the ZMP within the foot sole (use the footprint and planned step position)
        #>>>>TODO: Add ZMP upper and lower bound to keep the control (ZMP) within each footprints
        #Hint: first compute upper and lower bound based on zmp_ref then add constraints.
        #Hint: Add constraints at every time step

        # Halbe Abmessungen der Fußsohle berechnen
        half_l = foot_length / 2.0
        half_w = foot_width / 2.0

        for k in range(len(control)):
            # 1. Aktuelles Zentrum des Fußes aus der Referenz holen
            zmp_center_x = ZMP_ref_k[k, 0]
            zmp_center_y = ZMP_ref_k[k, 1]
            
            # 2. Grenzen für die X-Richtung (Länge) berechnen
            zmp_x_min = zmp_center_x - half_l
            zmp_x_max = zmp_center_x + half_l
            
            # 3. Grenzen für die Y-Richtung (Breite) berechnen
            zmp_y_min = zmp_center_y - half_w
            zmp_y_max = zmp_center_y + half_w
            
            # 4. Constraints in Drake hinzufügen (control[k] ist [px, py])
            prog.AddBoundingBoxConstraint(zmp_x_min, zmp_x_max, control[k][0]) # X-Grenze
            prog.AddBoundingBoxConstraint(zmp_y_min, zmp_y_max, control[k][1]) # Y-Grenze

    
        # 4. if terminal_idx < self.no_samples than we have the terminal state within
        # the current horizon. In this case create the terminal state (foot step pos + zero vel)
        # and apply the state constraint to all states >= terminal_idx within the horizon
        #>>>>TODO: Add the terminal constraint if requires
        #Hint: If you are unsure, you can start testing without this first!
    
        # setup our cost: minimize zmp error (tracking), minimize CoM velocity (smoothing)
        #>>>>TODO: add the cost at each timestep, hint: prog.AddCost
            
        for k in range(self.no_samples):
            # 1. Tracking cost: alpha * ||p_k - p_ref,k||^2
            zmp_error_x = control[k][0] - ZMP_ref_k[k][0]
            zmp_error_y = control[k][1] - ZMP_ref_k[k][1]
            
            prog.AddCost(alpha * (zmp_error_x**2 + zmp_error_y**2))
            
            # 2. Smoothing cost: gamma * ||v_k||^2
            # state[k][1] ist vx, state[k][3] ist vy
            v_x = state[k][1]
            v_y = state[k][3]
            
            prog.AddCost(gamma * (v_x**2 + v_y**2))

        # solve
        result = Solve(prog)
        if not result.is_success:
            print("failure")
            
        self.X_k = result.GetSolution(state)
        self.U_k = result.GetSolution(control)
        if np.isnan(self.X_k).any():
            print("failure")
        
        self.ZMP_ref_k = ZMP_ref_k
        return self.U_k[0]
    
################################################################################
# run the simulation
################################################################################

# inital state in x0 = [px0, vx0]
x_0 = np.array([0.0, 0.0])
# inital state in y0 = [py0, vy0]
y_0 = np.array([-0.09, 0.0])

# footprint
footprint = np.array([foot_length, foot_width])

# generate the footsteps
step_size = 0.2
#>>>>TODO: 1. generate the foot step plan using generate_foot_steps
foot_steps = generate_foot_steps(np.array([0.0, 0.0]), step_size, NO_STEPS)

# reapeat the last two foot steps (so the mpc horizon never exceeds the plan!)
foot_steps = np.vstack([
    foot_steps, foot_steps[-1], foot_steps[-1]])

# zmp reference trajecotry
#>>>>TODO: 2. generate the complete ZMP reference using generate_zmp_reference
ZMP_ref = generate_zmp_reference(foot_steps, NO_SIM_SAMPLES_PER_MPC)

# generate mpc
mpc = MPC(T_MPC, T_HORIZON)

# generate the pendulum simulator
state_0 = np.concatenate([x_0, y_0])
sim = Simulator(state_0, T_SIM)

# setup some vectors for plotting stuff
TIME_VEC = np.nan*np.ones(NO_SIM_SAMPLES)
STATE_VEC = np.nan*np.ones([NO_SIM_SAMPLES, 4])
ZMP_REF_VEC = np.nan*np.ones([NO_SIM_SAMPLES, 2])
ZMP_VEC = np.nan*np.ones([NO_SIM_SAMPLES, 2])

# time to add some disturbance
t_push = 3.2

# execution loop

ZMP_ref_MPC = generate_zmp_reference(foot_steps, NO_MPC_SAMPLES_PER_STEP)

k = 0   # the number of mpc update
for i in range(NO_SIM_SAMPLES):
    
    # simulation time
    t = i*T_SIM
        
    if i % NO_SIM_SAMPLES_PER_MPC == 0:
        # time to update the mpc
        
        # current state
        #>>>>TODO: get current state from the simulator
        x_k = sim.x
    
        #>>>>TODO: extract the current horizon from the complete reference trajecotry ZMP_ref
        # Bestimme das aktuelle Sample im groben MPC-Raster
        # mpc_sample_start = k * NO_MPC_SAMPLES_PER_STEP
        mpc_sample_start = k

        # Schneide den prädiktiven Horizont korrekt aus
        ZMP_ref_k = ZMP_ref_MPC[mpc_sample_start : mpc_sample_start + NO_MPC_SAMPLES_HORIZON]

        # check if we have terminal constraint
        idx_terminal_k = NO_MPC_SAMPLES - k
        #>>>>TODO: Update the mpc, get new command
        u_k = mpc.buildSolveOCP(x_k, ZMP_ref_k, idx_terminal_k)
        
        k += 1
    
    # simulate a push for 0.05 sec with 1.0 m/s^2 acceleration 
    x_ddot_ext = np.array([0, 0])
    
    #>>>>TODO: when you got everything working try adding a small disturbance
    if i > int(t_push/T_SIM) and i < int((t_push + 0.05)/T_SIM):
       x_ddot_ext = np.array([0, 1.0])
    
    #>>>>TODO: Update the simulation using the current command
    x_k = sim.simulate(u_k, x_ddot_ext)
    
    # save some stuff
    TIME_VEC[i] = t
    STATE_VEC[i] = x_k
    ZMP_VEC[i] = u_k
    # ZMP_REF_VEC[i] = mpc.ZMP_ref_k[0]
    ZMP_REF_VEC[i] = ZMP_VEC[i]
    
ZMP_LB_VEC = ZMP_REF_VEC - footprint[None,:]
ZMP_UB_VEC = ZMP_REF_VEC + footprint[None,:]

#>>>>TODO: Use the recodings in STATE_VEC and ZMP_VEC to compute the 
# LIP acceleration
#>>>>Hint: Use the continious dynamic matrices
omega2 = 9.81 / 0.5 # g / h

# Extrahiere CoM und ZMP über alle aufgezeichneten Zeitschritte
com_x_vec = STATE_VEC[:, 0]
com_y_vec = STATE_VEC[:, 2]
zmp_x_vec = ZMP_VEC[:, 0]
zmp_y_vec = ZMP_VEC[:, 1]

# Berechne die Beschleunigungen analytisch
com_ddot_x = omega2 * (com_x_vec - zmp_x_vec)
com_ddot_y = omega2 * (com_y_vec - zmp_y_vec)

STATE_DOT_VEC = np.column_stack((com_ddot_x, com_ddot_y))

################################################################################
# plot something

#>>>>TODO: plot everything in x-axis
#>>>>TODO: plot everything in y-axis
#>>>>TODO: plot everything in xy-plane

# ------------------------------------------------------------------------------
# PLOTTING
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# PLOTTING WITH HIGHLIGHTED DISTURBANCE
# ------------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 10))

# Definiere das Störungs-Zeitfenster für die Visualisierung
t_start_push = t_push
t_end_push = t_push + 0.05

# Subplot 1: Alles auf der X-Achse über die Zeit
ax1 = fig.add_subplot(3, 1, 1)
ax1.plot(TIME_VEC, ZMP_REF_VEC[:, 0], 'r--', label='ZMP Ref X (Füße)', drawstyle='steps-post')
ax1.plot(TIME_VEC, zmp_x_vec, 'tab:orange', label='ZMP Optimiert X', linewidth=2)
ax1.plot(TIME_VEC, com_x_vec, 'tab:blue', label='CoM Position X', linewidth=2)
# HIGHLIGHT: Da der Schubser nur in Y-Richtung geht, kannst du hier optional einen dünnen Strich machen
ax1.axvspan(t_start_push, t_end_push, color='red', alpha=0.15, label='Schubser (Zeitfenster)')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.set_title("Trajektorien auf der X-Achse")
ax1.set_ylabel("Position [m]")
ax1.legend()

# Subplot 2: Alles auf der Y-Achse über die Zeit
ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
ax2.plot(TIME_VEC, ZMP_REF_VEC[:, 1], 'r--', label='ZMP Ref Y (Füße)', drawstyle='steps-post')
ax2.plot(TIME_VEC, zmp_y_vec, 'tab:orange', label='ZMP Optimiert Y', linewidth=2)
ax2.plot(TIME_VEC, com_y_vec, 'tab:blue', label='CoM Position Y', linewidth=2)
# HIGHLIGHT: Roter transparenter Balken genau über der betroffenen Y-Achse
ax2.axvspan(t_start_push, t_end_push, color='red', alpha=0.2, label='Schubser (Zeitfenster)')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.set_title("Trajektorien auf der Y-Achse")
ax2.set_xlabel("Zeit [s]")
ax2.set_ylabel("Position [m]")
ax2.legend()

# Subplot 3: Die 2D-Bewegung in der XY-Ebene (Vogelperspektive)
ax3 = fig.add_subplot(3, 1, 3)
plot_foot_steps(foot_steps, (foot_length, foot_width), ax3)

ax3.plot(zmp_x_vec, zmp_y_vec, color='tab:orange', label='ZMP Trajektorie', linewidth=2)
ax3.plot(com_x_vec, com_y_vec, color='tab:blue', label='CoM Trajektorie (Schwerpunkt)', linewidth=2.5)

# HIGHLIGHT im 2D-Raum: Finde den Index im Vektor, wo t_push passiert
push_idx = np.argmin(np.abs(TIME_VEC - t_push))
# Zeichne einen roten Punkt/Kreuz an die Stelle, an der der Roboter stand, als er geschubst wurde
ax3.plot(com_x_vec[push_idx], com_y_vec[push_idx], 'rX', markersize=12, label='Ort des Schubsers')

ax3.axis('equal')
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.set_title("2D-Trajektorien in der XY-Ebene (Vogelperspektive)")
ax3.set_xlabel("X-Position [m]")
ax3.set_ylabel("Y-Position [m]")
ax3.legend()

plt.tight_layout()
plt.show()