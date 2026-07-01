"""Task2: Linear inverted pendulum Trajectory planning

The goal of this file is to formulate the optimal control problem (OCP)
in equation 12. 

In this case we will solve the trajectory planning over the entire footstep plan
(= horizon) in one go.

Our state will be the position and velocity of the pendulum in the 2d plane.
x = [cx, vx, cy, vy]
And your control the ZMP position in the 2d plane
u = [px, py]

You will need to fill in the TODO to solve the task.
"""

import numpy as np

from pydrake.all import MathematicalProgram, Solve

import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')

################################################################################
# settings
################################################################################

# Robot Parameters:
# --------------

h           = 0.80   # fixed CoM height (assuming walking on a flat terrain)
g           = 9.81   # norm of the gravity vector
foot_length = 0.10   # foot size in the x-direction
foot_width  = 0.06   # foot size in the y-direciton

# OCP Parameters:
# --------------
T                     = 0.1                                # fixed sampling time interval of computing the ocp in [s]
STEP_TIME             = 0.8                                # fixed time needed for every foot step [s]

NO_SAMPLES_PER_STEP   = int(round(STEP_TIME/T))            # number of ocp samples per step

NO_STEPS              = 10                                 # total number of foot steps in the plan
TOTAL_NO_SAMPLES      = NO_SAMPLES_PER_STEP*NO_STEPS       # total number of ocp samples over the complete plan (= Horizon)

# Cost Parameters:
# ---------------
alpha       = 10**(-1)                                      # ZMP error squared cost weight (= tracking cost)
gamma       = 10**(-3)                                      # CoM velocity error squared cost weight (= smoothing cost)

################################################################################
# helper function for visualization and dynamics
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
    zmp_ref = np.zeros((TOTAL_NO_SAMPLES, 2))

    for i in range(no_steps):
        start = i * no_samples_per_step
        end = start + no_samples_per_step
        # guard against going out of bounds
        end = min(end, TOTAL_NO_SAMPLES)
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
# setup the plan references and system matrices
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

# zmp reference trajecotry
#>>>>TODO: 2. generate the ZMP reference using generate_zmp_reference
ZMP_ref = generate_zmp_reference(foot_steps, NO_SAMPLES_PER_STEP)

#>>>>Note: At this point you can already start plotting things to see if they
# really make sense!

# discrete LIP dynamics
#>>>>TODO: get the static dynamic matrix Ad, Bd
Ad, Bd = discrete_LIP_dynamics(T, g, h)
# continous LIP dynamics
#>>>>TODO: get the static dynamic matrix A, B
A, B = continuous_LIP_dynamics(g, h)
################################################################################
# problem definition
################################################################################

# Define an instance of MathematicalProgram 
prog = MathematicalProgram() 

################################################################################
# variables
# Our state will be the position and velocity of the pendulum in the 2d plane.
# x = [cx, vx, cy, vy]
# And your control the ZMP position in the 2d plane
# u = [px, py]
nx = 4 #>>>>TODO: State dimension = ?
nu = 2 #>>>>TODO: control dimension = ?

state = prog.NewContinuousVariables(TOTAL_NO_SAMPLES, nx, 'state')
control = prog.NewContinuousVariables(TOTAL_NO_SAMPLES, nu, 'control')

# Start: Schwerpunkt über dem ersten Fuß, Geschwindigkeit Null
state_initial = np.array([foot_steps[0][0], 0.0, foot_steps[0][1], 0.0])

# Ziel: Schwerpunkt über dem letzten Fuß, Geschwindigkeit Null
state_terminal = np.array([foot_steps[-1][0], 0.0, foot_steps[-1][1], 0.0])

################################################################################
# constraints

# 1. intial constraint
#>>>>TODO: Add inital state constrain, Hint: prog.AddConstraint
prog.AddBoundingBoxConstraint(state_initial, state_initial, state[0])

# 2. terminal constraint
#>>>>TODO: Add terminal state constrain, Hint: prog.AddConstraint
prog.AddBoundingBoxConstraint(state_terminal, state_terminal, state[-1])

# 3. at each step: respect the LIP descretized dynamics
#>>>>TODO: Enforce the dynamics at every time step
# 1. Hol dir die diskreten Matrizen für eine Achse (Abtastzeit T)
Ad_1d, Bd_1d = discrete_LIP_dynamics(T, g, h)

# 2. Baue die großen Matrizen für das 2D-Problem (X und Y)
# A_big blockdiagonal zusammensetzen
A_big = np.block([
    [Ad_1d, np.zeros((2, 2))],
    [np.zeros((2, 2)), Ad_1d]
])

# B_big blockdiagonal zusammensetzen
B_big = np.block([
    [Bd_1d, np.zeros((2, 1))],
    [np.zeros((2, 1)), Bd_1d]
])

for k in range(TOTAL_NO_SAMPLES - 1):
    # Berechne die Differenz: x_{k+1} - (A*x_k + B*u_k)
    # Wenn die Dynamik erfüllt ist, muss dieser Vektor exakt 0 sein
    dynamics_error = state[k+1] - (A_big @ state[k] + B_big @ control[k])
    
    # Wir zwingen jede Komponente der Differenz dazu, 0 zu sein
    prog.AddLinearEqualityConstraint(dynamics_error, np.zeros(4))

# 4. at each step: keep the ZMP within the foot sole (use the footprint and planned step position)
#>>>>TODO: Add ZMP upper and lower bound to keep the control (ZMP) within each footprints
#Hint: first compute upper and lower bound based on zmp_ref then add constraints.
#Hint: Add constraints at every time step

# Halbe Abmessungen der Fußsohle berechnen
half_l = foot_length / 2.0
half_w = foot_width / 2.0

for k in range(TOTAL_NO_SAMPLES):
    # 1. Aktuelles Zentrum des Fußes aus der Referenz holen
    zmp_center_x = ZMP_ref[k, 0]
    zmp_center_y = ZMP_ref[k, 1]
    
    # 2. Grenzen für die X-Richtung (Länge) berechnen
    zmp_x_min = zmp_center_x - half_l
    zmp_x_max = zmp_center_x + half_l
    
    # 3. Grenzen für die Y-Richtung (Breite) berechnen
    zmp_y_min = zmp_center_y - half_w
    zmp_y_max = zmp_center_y + half_w
    
    # 4. Constraints in Drake hinzufügen (control[k] ist [px, py])
    prog.AddBoundingBoxConstraint(zmp_x_min, zmp_x_max, control[k][0]) # X-Grenze
    prog.AddBoundingBoxConstraint(zmp_y_min, zmp_y_max, control[k][1]) # Y-Grenze

################################################################################
# stepwise cost, note that the cost function is scalar!

# setup our cost: minimize zmp error (tracking), minimize CoM velocity (smoothing)
#>>>>TODO: add the cost at each timestep, hint: prog.AddCost

for k in range(TOTAL_NO_SAMPLES):
    # 1. Tracking cost: alpha * ||p_k - p_ref,k||^2
    zmp_error_x = control[k][0] - ZMP_ref[k][0]
    zmp_error_y = control[k][1] - ZMP_ref[k][1]
    
    prog.AddCost(alpha * (zmp_error_x**2 + zmp_error_y**2))
    
    # 2. Smoothing cost: gamma * ||v_k||^2
    # state[k][1] ist vx, state[k][3] ist vy
    v_x = state[k][1]
    v_y = state[k][3]
    
    prog.AddCost(gamma * (v_x**2 + v_y**2))

################################################################################
# solve

result = Solve(prog)
if not result.is_success:
    print("failure")
print("solved")

# extract the solution
#>>>>TODO: extract your variables from the result object
t = T*np.arange(0, TOTAL_NO_SAMPLES)

result = Solve(prog)
if not result.is_success():  # Hinweis: is_success ist in Drake eine Funktion -> is_success()
    print("failure")
else:
    print("solved")

# Extrahiere die kompletten numerischen Arrays aus dem Result
state_sol = result.GetSolution(state)      # Liefert ein Array der Form (TOTAL_NO_SAMPLES, 4)
control_sol = result.GetSolution(control)  # Liefert ein Array der Form (TOTAL_NO_SAMPLES, 2)

# Zeitachse generieren
t = T * np.arange(0, TOTAL_NO_SAMPLES)

# Fürs spätere Plotten in handliche Variablen aufteilen:
com_x = state_sol[:, 0]
vel_x = state_sol[:, 1]
com_y = state_sol[:, 2]
vel_y = state_sol[:, 3]

zmp_x = control_sol[:, 0]
zmp_y = control_sol[:, 1]

# compute the acceleration
#>>>>TODO: compute the acceleration of the COM

# Die physikalische Konstante (omega^2) berechnen
omega2 = g / h

# Beschleunigung direkt aus der Differenz von CoM und ZMP berechnen
acceleration_x = omega2 * (com_x - zmp_x)
acceleration_y = omega2 * (com_y - zmp_y)

# Optional: Zusammenfassen in ein einziges Array der Form (TOTAL_NO_SAMPLES, 2)
com_ddot = np.column_stack((acceleration_x, acceleration_y))

################################################################################
# plot something

#>>>>TODO: plot everything in x-axis
#>>>>TODO: plot everything in y-axis
#>>>>TODO: plot everything in xy-plane

# Erstelle eine große Figure mit 3 Subplots
fig = plt.figure(figsize=(14, 10))

# ------------------------------------------------------------------------------
# Subplot 1: Alles auf der X-Achse über die Zeit
# ------------------------------------------------------------------------------
ax1 = fig.add_subplot(3, 1, 1)
ax1.plot(t, ZMP_ref[:, 0], 'r--', label='ZMP Ref X (Füße)', drawstyle='steps-post')
ax1.plot(t, zmp_x, 'tab:orange', label='ZMP Optimiert X', linewidth=2)
ax1.plot(t, com_x, 'tab:blue', label='CoM Position X', linewidth=2)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.set_title("Trajektorien auf der X-Achse")
ax1.set_ylabel("Position [m]")
ax1.legend()

# ------------------------------------------------------------------------------
# Subplot 2: Alles auf der Y-Achse über die Zeit
# ------------------------------------------------------------------------------
ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
ax2.plot(t, ZMP_ref[:, 1], 'r--', label='ZMP Ref Y (Füße)', drawstyle='steps-post')
ax2.plot(t, zmp_y, 'tab:orange', label='ZMP Optimiert Y', linewidth=2)
ax2.plot(t, com_y, 'tab:blue', label='CoM Position Y', linewidth=2)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.set_title("Trajektorien auf der Y-Achse")
ax2.set_xlabel("Zeit [s]")
ax2.set_ylabel("Position [m]")
ax2.legend()

# ------------------------------------------------------------------------------
# Subplot 3: Die 2D-Bewegung in der XY-Ebene (Vogelperspektive)
# ------------------------------------------------------------------------------
ax3 = fig.add_subplot(3, 1, 3)
# Zeichne zuerst die physischen Fußstapfen ein
plot_foot_steps(foot_steps, (foot_length, foot_width), ax3)

# Plotte die optimierten Trajektorien darüber
ax3.plot(ZMP_ref[:, 0], ZMP_ref[:, 1], color='tab:orange', label='ZMP Trajektorie', linewidth=2)
ax3.plot(com_x, com_y, color='tab:blue', label='CoM Trajektorie (Schwerpunkt)', linewidth=2.5)

ax3.axis('equal')
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.set_title("2D-Trajektorien in der XY-Ebene (Vogelperspektive)")
ax3.set_xlabel("Y-Position [m]")
ax3.set_ylabel("X-Position [m]")
ax3.legend()

plt.tight_layout()

# HIERMIT WIRD DAS FENSTER DIREKT ANGEZEIGT:
plt.show()