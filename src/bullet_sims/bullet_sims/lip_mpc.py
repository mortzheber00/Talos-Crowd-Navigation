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

        # weight of the SOFT terminal capture-point penalty. A hard equality
        # makes the QP infeasible once an aggressive maneuver drives the state
        # large (the unstable LIP mode is propagated over the whole horizon);
        # much larger values (~1e3) ill-condition the Hessian.
        self.beta = getattr(conf, 'beta', 100.0)

        # safety margin [m] shrinking the ZMP/CoP feasibility box inside the
        # sole: a commanded ZMP on the exact sole edge means any model error
        # puts the true CoP AT the edge, i.e. the foot starts rolling/tilting
        # about it.
        self.zmp_margin = getattr(conf, 'zmp_margin', 0.02)

        # solution and references over the horizon
        self.X_k = None
        self.U_k = None
        self.ZMP_ref_k = None

    def buildSolveOCP(self, x_k, ZMP_ref_k, terminal_idx, yaw_ref_k=None):
        """build and solve ocp

        Args:
            x_k (_type_): inital mpc state
            ZMP_ref_k (_type_): zmp reference over horizon
            terminal_idx (_type_): index within horizon to apply terminal constraint
            yaw_ref_k (optional): per-sample support-foot yaw [rad] over the
                horizon. When given, the ZMP/CoP feasibility box is rotated into
                the foot frame so it stays aligned with a yawed (turning) foot.
                Defaults to None -> yaw 0 -> world-axis-aligned box (straight walk).

        Returns:
            _type_: control (first commanded ZMP, length-2 np.array [zmp_x, zmp_y])
        """

        nx = 4
        nu = 2

        ZMP_ref_k = np.atleast_2d(np.asarray(ZMP_ref_k, dtype=float))
        # a non-finite state would make Drake assert while BUILDING the initial
        # state constraint (beq.allFinite()), before Solve is even reached
        if not np.all(np.isfinite(np.asarray(x_k, dtype=float))):
            print(f"[lip_mpc] non-finite x_k {x_k} -> holding support-center ZMP")
            return np.asarray(ZMP_ref_k[0], dtype=float).copy()
        n_ref = ZMP_ref_k.shape[0]
        N = min(self.no_samples, n_ref)
        N = max(N, 2)

        # per-sample support-foot yaw (0 -> axis-aligned box, backward compatible)
        if yaw_ref_k is None:
            yaw_arr = np.zeros(n_ref)
        else:
            yaw_arr = np.asarray(yaw_ref_k, dtype=float).ravel()

        # ZMP / CoP feasibility box. There is no dedicated zmp_limit constant in
        # talos_conf.py, so derive the bound from the foot footprint half-sizes.
        # The commanded ZMP must stay within the foot under the support, which we
        # express as a box around the per-step ZMP reference (the stance foot
        # center). lfxp/lfxn are the fore/aft half-lengths, lfyp/lfyn the lateral.
        # shrunk by zmp_margin so the commanded ZMP never sits on the sole edge
        # (>= 5 mm of box always kept, in case of extreme margins)
        m = self.zmp_margin
        lfxp = max(self.conf.lfxp - m, 0.005)
        lfxn = max(self.conf.lfxn - m, 0.005)
        lfyp = max(self.conf.lfyp - m, 0.005)
        lfyn = max(self.conf.lfyn - m, 0.005)

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

        # --- ZMP/CoP feasibility constraints (box in the FOOT frame) ---
        # The commanded ZMP must stay within the support foot. We express the
        # box in the foot's own frame so it rotates with a yawed/turning foot:
        # project the world offset (control - ref) onto the foot axes and bound
        # the fore/aft and lateral components. yaw = 0 recovers the world-aligned
        # box, so straight walking is unchanged.
        for i in range(N):
            ref = ZMP_ref_k[i]
            psi = yaw_arr[min(i, len(yaw_arr) - 1)]
            c, s = np.cos(psi), np.sin(psi)
            dx = control[i, 0] - ref[0]
            dy = control[i, 1] - ref[1]
            local_x = c * dx + s * dy      # fore/aft in the foot frame
            local_y = -s * dx + c * dy     # lateral in the foot frame
            prog.AddLinearConstraint(local_x <= lfxp)
            prog.AddLinearConstraint(local_x >= -lfxn)
            prog.AddLinearConstraint(local_y <= lfyp)
            prog.AddLinearConstraint(local_y >= -lfyn)

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

        # --- terminal (capture-point) penalty ---
        # soft quadratic penalty instead of a hard equality: the equality makes
        # the QP infeasible (and Drake assert-crashes on non-finite x_k) under
        # aggressive maneuvers, since the unstable LIP mode blows the terminal
        # state up over the horizon. The penalty keeps the same capturability
        # bias while the QP stays feasible by construction.
        if terminal_idx is not None and 0 <= terminal_idx < N:
            t_idx = terminal_idx
        else:
            t_idx = N - 1
        ref = ZMP_ref_k[min(t_idx, n_ref - 1)]
        dcm_x = state[t_idx, 0] + state[t_idx, 1] / self.omega
        dcm_y = state[t_idx, 2] + state[t_idx, 3] / self.omega
        prog.AddQuadraticCost(self.beta * ((dcm_x - ref[0])**2 + (dcm_y - ref[1])**2))

        # --- solve ---
        try:
            result = Solve(prog)
            ok = result.is_success()
        except Exception as e:
            print(f"[lip_mpc] QP solve raised {type(e).__name__}: {e} "
                  "-> holding support-center ZMP")
            return np.asarray(ZMP_ref_k[0], dtype=float).copy()
        if not ok:
            print("[lip_mpc] QP solve failed "
                  f"({result.get_solution_result()}) -> holding support-center ZMP")
            return np.asarray(ZMP_ref_k[0], dtype=float).copy()

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


def generate_yaw_reference(foot_steps, no_samples_per_step):
    """generate the support-foot yaw reference [rad] aligned sample-for-sample
    with generate_zmp_reference. Feeds the yaw-aware ZMP box in buildSolveOCP so
    the feasibility box tracks a turning foot. For straight (identity-rotation)
    footsteps this is all zeros.
    """
    yaw_ref = []
    for step in foot_steps:
        R = step.poseInWorld().rotation
        yaw = np.arctan2(R[1, 0], R[0, 0])
        for _ in range(no_samples_per_step):
            yaw_ref.append(yaw)

    return np.array(yaw_ref)
