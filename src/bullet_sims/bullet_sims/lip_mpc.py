import numpy as np
from pydrake.all import MathematicalProgram, Solve

################################################################################
# Helper fnc
################################################################################

def continious_LIP_dynamics(g, h):
    """returns the static matrices A,B of the continious LIP dynamics

    Args:
        g (_type_): gravity
        h (_type_): height

    Returns:
        np.array: A, B
    """

    # copy from previous Task 2
    A_axis = np.array([
        [0,     1],
        [g/h,   0],
    ])
    B_axis = np.array([
        [0],
        [-g/h],
    ])
    Z2 = np.zeros((2, 2))
    Z21 = np.zeros((2, 1))

    A = np.block([
        [A_axis, Z2],
        [Z2,     A_axis]
    ])

    B = np.block([
        [B_axis, Z21],
        [Z21,    B_axis]
    ])

    return A, B

def discrete_LIP_dynamics(dt, g, h):
    """returns the matrices static Ad,Bd of the discretized LIP dynamics

    Args:
        dt (_type_): discretization steps
        g (_type_): gravity
        h (_type_): height

    Returns:
        _type_: _description_
    """
    # exact (analytically integrated) discretization of the LIP dynamics,
    # per T6.pdf Eq. 6 not a first-order Euler approximation
    omega = np.sqrt(g / h)
    ch = np.cosh(omega * dt)
    sh = np.sinh(omega * dt)
    Ad_axis = np.array([
        [ch,         sh / omega],
        [omega * sh, ch],
    ])
    Bd_axis = np.array([
        [1 - ch],
        [-omega * sh],
    ])
    Z2 = np.zeros((2, 2))
    Z21 = np.zeros((2, 1))
    Ad = np.block([
        [Ad_axis, Z2],
        [Z2,      Ad_axis]
    ])
    Bd = np.block([
        [Bd_axis, Z21],
        [Z21,     Bd_axis]
    ])
    return Ad, Bd

################################################################################
# LIPInterpolator
################################################################################

class LIPInterpolator:
    """Integrates the linear inverted pendulum model using the 
    continous dynamics. To interpolate the solution to hight 
    """
    def __init__(self, x_inital, conf):
        self.conf = conf
        self.dt = conf.dt
        self.x = x_inital
        self.g = conf.g
        self.h = conf.h
        self.omega = np.sqrt(self.g / self.h)
        self.A, self.B = continious_LIP_dynamics(self.g, self.h)
        self.Ad, self.Bd = discrete_LIP_dynamics(self.dt, self.g, self.h)
        self.u = np.zeros(2)

    def integrate(self, u):
        self.x = self.Ad @ self.x + self.Bd @ u
        self.u = u

    def comState(self):
        # return the center of mass state
        # that is position \in R3, velocity \in R3, acceleration \in R3
        xdot = self.A @ self.x + self.B @ self.u

        c = np.array([self.x[0], self.x[2], self.h])
        c_dot = np.array([self.x[1], self.x[3], 0.0])
        c_ddot = np.array([xdot[1], xdot[3], 0.0])
        return c, c_dot, c_ddot

    def dcm(self):
        # Divergent Component of Motion / Capture Point
        # defined as: dcm = c + c_dot / omega
        c, c_dot, _ = self.comState()
        return c + c_dot / self.omega

    def zmp(self):
        # the LIP control input u is itself the commanded ZMP/CoP location
        return np.array([self.u[0], self.u[1], 0.0])
        
    
################################################################################
# LIPMPC
################################################################################

class LIPMPC:
    def __init__(self, conf):
        self.conf = conf
        self.dt = conf.dt
        self.no_samples = conf.no_mpc_samples_per_horizon

        # discrete LIP dynamics for the MPC.
        self.omega = np.sqrt(conf.g / conf.h)
        self.Ad, self.Bd = discrete_LIP_dynamics(conf.dt_mpc, conf.g, conf.h)

        # solution and references over the horizon
        self.X_k = None
        self.U_k = None
        self.ZMP_ref_k = None

    def buildSolveOCP(self, x_k, ZMP_ref_k, terminal_idx):
        """build and solve ocp

        Args:
            x_k (_type_): inital mpc state
            ZMP_ref_k (_type_): zmp reference over horizon
            terminal_idx (_type_): index within horizon to apply terminal constraint

        Returns:
            _type_: control (first commanded ZMP, length-2 np.array [zmp_x, zmp_y])
        """

        nx = 4
        nu = 2

        ZMP_ref_k = np.atleast_2d(np.asarray(ZMP_ref_k, dtype=float))
        n_ref = ZMP_ref_k.shape[0]
        N = min(self.no_samples, n_ref)
        N = max(N, 2)

        # ZMP / CoP feasibility box. There is no dedicated zmp_limit constant in
        # talos_conf.py, so derive the bound from the foot footprint half-sizes.
        # The commanded ZMP must stay within the foot under the support, which we
        # express as a box around the per-step ZMP reference (the stance foot
        # center). lfxp/lfxn are the fore/aft half-lengths, lfyp/lfyn the lateral.
        lfxp = self.conf.lfxp
        lfxn = self.conf.lfxn
        lfyp = self.conf.lfyp
        lfyn = self.conf.lfyn

        prog = MathematicalProgram()

        state = prog.NewContinuousVariables(N, nx, 'state')
        control = prog.NewContinuousVariables(N, nu, 'control')

        # --- initial state constraint ---
        for j in range(nx):
            prog.AddLinearConstraint(state[0, j] == x_k[j])

        # --- dynamics equality constraints ---
        for i in range(N - 1):
            x_next = self.Ad @ state[i] + self.Bd @ control[i]
            for j in range(nx):
                prog.AddLinearConstraint(state[i + 1, j] == x_next[j])

        # --- ZMP/CoP feasibility constraints (box around the reference) ---
        for i in range(N):
            ref = ZMP_ref_k[i]
            # x (fore/aft): foot extends lfxp forward, lfxn backward of the ankle
            prog.AddLinearConstraint(control[i, 0] <= ref[0] + lfxp)
            prog.AddLinearConstraint(control[i, 0] >= ref[0] - lfxn)
            # y (lateral): symmetric half-width
            prog.AddLinearConstraint(control[i, 1] <= ref[1] + lfyp)
            prog.AddLinearConstraint(control[i, 1] >= ref[1] - lfyn)

        # --- cost ---
        alpha = self.conf.alpha     # ZMP error squared cost weight
        gamma = self.conf.gamma     # CoM velocity smoothing cost weight
        for i in range(N):
            ref = ZMP_ref_k[i]
            # track the ZMP reference with the commanded ZMP (the control)
            zmp_err = control[i] - ref
            prog.AddQuadraticCost(alpha * (zmp_err[0]**2 + zmp_err[1]**2))
            # velocity regularization / smoothing on the CoM velocity states
            # (state layout: [x, xdot, y, ydot]) -> indices 1 and 3
            prog.AddQuadraticCost(gamma * (state[i, 1]**2 + state[i, 3]**2))

        # --- terminal (capture-point) constraint ---
        if terminal_idx is not None and 0 <= terminal_idx < N:
            t_idx = terminal_idx
        else:
            t_idx = N - 1
        ref = ZMP_ref_k[min(t_idx, n_ref - 1)]
        dcm_x = state[t_idx, 0] + state[t_idx, 1] / self.omega
        dcm_y = state[t_idx, 2] + state[t_idx, 3] / self.omega
        prog.AddLinearConstraint(dcm_x == ref[0])
        prog.AddLinearConstraint(dcm_y == ref[1])

        # --- solve ---
        result = Solve(prog)
        if not result.is_success():
            raise RuntimeError(
                "LIPMPC.buildSolveOCP: QP solve failed "
                f"({result.get_solution_result()})"
            )

        self.X_k = result.GetSolution(state)
        self.U_k = result.GetSolution(control)
        self.ZMP_ref_k = ZMP_ref_k

        return self.U_k[0]
    

def generate_zmp_reference(foot_steps, no_samples_per_step):
    """generate the zmp reference given a sequence of footsteps
    """
    zmp_ref = []
    for step in foot_steps:
        pos_xy = step.poseInWorld().translation[:2]
        for _ in range(no_samples_per_step):
            zmp_ref.append(pos_xy)

    return np.array(zmp_ref)
