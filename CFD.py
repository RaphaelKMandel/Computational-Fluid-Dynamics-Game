import numpy as np
from numba import njit
from time import time_ns as tic
from Solvers import convection, diffusion, contourf

time_step = 1e-3
JIT = True
np.set_printoptions(linewidth=150, precision=2)


def my_static_jit(func):
    if JIT:
        return staticmethod(njit(func, cache=True))
    else:
        print('JIT is Disabled!')
        return staticmethod(func)


class Flow:
    def __init__(self, V_in, w_ch, W, H, Nx, Ny, nu=0.00002, dt=time_step, alpha=0.7):
        # Initialize Variables
        self.iter = -1
        self.V_in = V_in
        self.W = W
        self.H = H
        self.Nx = Nx
        self.Ny = Ny
        self.dx = W / Nx
        self.dy = H / Ny
        self.dt = dt
        self.nu = nu
        self.alpha = alpha
        self.outlet = 1.0  # TODO: Find Correct Value
        self.D_x = self.nu * self.dy / self.dx
        self.D_y = self.nu * self.dx / self.dy
        self.f = 12.0 * self.dx * self.dy * self.nu
        self.a_p_t = self.dx * self.dy / self.dt
        self.solid = np.zeros([Ny, Nx], dtype=bool)
        self.porous = np.zeros([Ny, Nx], dtype=bool)
        self.p_x = np.zeros([Ny, Nx + 1], dtype=bool)
        self.p_y = np.zeros([Ny + 1, Nx], dtype=bool)
        self.w_ch = w_ch * np.ones([Ny + 2, Nx + 2])
        # X-Momentum Coefficients
        self.a_w_x = np.zeros([Ny, Nx + 1])
        self.a_e_x = np.zeros([Ny, Nx + 1])
        self.a_s_x = np.zeros([Ny, Nx + 1])
        self.a_n_x = np.zeros([Ny, Nx + 1])
        self.a_p_x = np.zeros([Ny, Nx + 1])
        self.b_x = np.zeros([Ny, Nx + 1])
        # Y-Momentum Coefficients
        self.a_w_y = np.zeros([Ny + 1, Nx])
        self.a_e_y = np.zeros([Ny + 1, Nx])
        self.a_s_y = np.zeros([Ny + 1, Nx])
        self.a_n_y = np.zeros([Ny + 1, Nx])
        self.a_p_y = np.zeros([Ny + 1, Nx])
        self.b_y = np.zeros([Ny + 1, Nx])
        # Pressure Coefficients
        self.A_x = np.zeros([Ny, Nx + 1])
        self.A_y = np.zeros([Ny + 1, Nx])
        self.A_w = np.zeros([Ny, Nx])
        self.A_e = np.zeros([Ny, Nx])
        self.A_s = np.zeros([Ny, Nx])
        self.A_n = np.zeros([Ny, Nx])
        self.A_p = np.zeros([Ny, Nx])
        self.B = np.zeros([Ny, Nx])
        # Initial Guesses
        self.U = np.zeros([Ny + 2, Nx + 3])
        self.V = np.zeros([Ny + 3, Nx + 2])
        self.Uc = np.zeros([Ny + 2, Nx + 2])
        self.Vc = np.zeros([Ny + 2, Nx + 2])
        self.P = np.zeros([Ny + 2, Nx + 2])
        self.P_prime = np.zeros([Ny + 2, Nx + 2])
        self.Fx = np.zeros([Ny + 2, Nx + 3])
        self.Fy = np.zeros([Ny + 3, Nx + 2])
        self.Rx = np.ones([Ny, Nx + 1])
        self.Ry = np.ones([Ny + 1, Nx])
        self.Rx0 = 1
        self.Ry0 = 1
        self.B0 = 1
        # Boundary Conditions - 0: Wall, 1: Symmetry, 2: Inlet, 3: Outlet
        self.left = np.zeros(Ny, dtype='uint8')
        self.right = np.zeros(Ny, dtype='uint8')
        self.bottom = np.zeros(Nx, dtype='uint8')
        self.top = np.zeros(Nx, dtype='uint8')
        self.BCS = np.zeros([Ny + 2, Nx + 2], dtype='int8')
        # Monitors
        self.P_in = 0.0
        self.P_out = 0.0
        self.dP = 0.0

    def initialize(self):
        # Define Boundary Conditions
        self.BCS[1:-1, 0] = -self.outlet * (self.left == 3) + 1.0 * (self.left != 3)
        self.BCS[1:-1, -1] = -self.outlet * (self.right == 3) + 1.0 * (self.right != 3)
        self.BCS[0, 1:-1] = -self.outlet * (self.bottom == 3) + 1.0 * (self.bottom != 3)
        self.BCS[-1, 1:-1] = -self.outlet * (self.top == 3) + 1.0 * (self.top != 3)
        # Apply Boundary Conditions
        self.U = self.apply_boundary_velocity_x(self.U, self.V_in, self.left, self.right, self.bottom, self.top)
        self.V = self.apply_boundary_velocity_y(self.V, self.V_in, self.left, self.right, self.bottom, self.top)
        self.P = self.apply_boundary_pressure(self.P, self.BCS)
        # Face Fluxes
        self.Fx, self.Fy = self.face_fluxes(self.U, self.V, self.dx, self.dy, self.Fx, self.Fy)
        # Apply Solid
        self.set_solid(np.where(self.solid))

    def clear(self):
        self.U = np.zeros([self.Ny + 2, self.Nx + 3])
        self.V = np.zeros([self.Ny + 3, self.Nx + 2])
        self.P = np.zeros([self.Ny + 2, self.Nx + 2])
        self.porous[:, :] = False
        self.p_x[:, :] = False
        self.p_y[:, :] = False

    def reset(self):
        self.clear()
        self.initialize()

    def set_solid(self, solid):
        J, I = solid[0], solid[1]
        for i, j in zip(I, J):
            self.solid[j, i] = True
            self.p_x[j, i:i + 2] = True
            self.p_y[j:j + 2, i] = True

    def set_level(self, solid, boolean):
        J, I = solid[0], solid[1]
        for i, j in zip(I, J):
            if not self.solid[j, i]:
                self.porous[j, i] = boolean
                self.p_x[j, i:i + 2] = boolean  # TODO: Don't Remove Neighbor When Erasing
                self.p_y[j:j + 2, i] = boolean
                if boolean:
                    pass
                    # self.U[j+1, i+1:i+3] = 0.0
                    # self.V[j+1:j+3, i+1] = 0.0
                    #mass_imbalance = self.dy * (self.U[j + 1, i + 1] - self.U[j + 1, i + 2]) + self.dx * (self.V[j + 1, i + 1] - self.V[j + 2, i + 1])
            else:
                print('Cannot Assign Cell. Solid Fixed!')

    def quiver(self):
        self.cell_velocities(self.Uc, self.Vc, self.U, self.V, self.V_in)
        return self.Uc, self.Vc

    @my_static_jit
    def cell_velocities(Uc, Vc, U, V, V_in):
        Ny, Nx = U.shape
        Nx, Ny = Nx - 3, Ny - 2
        K = 0.5 / V_in
        for j in range(1, Ny + 1):
            for i in range(1, Nx + 1):
                Uc[j, i] = K * (U[j, i] + U[j, i + 1])
                Vc[j, i] = K * (V[j, i] + V[j + 1, i])
        return Uc, Vc

    @my_static_jit
    def face_fluxes(U, V, dx, dy, Fx, Fy):
        Ny, Nx = U.shape
        for j in range(Ny):
            for i in range(Nx):
                Fx[j, i] = dy * U[j, i]
        Ny, Nx = V.shape
        for j in range(Ny):
            for i in range(Nx):
                Fy[j, i] = dx * V[j, i]
        return Fx, Fy

    @my_static_jit
    def coefficients_u(U, P, V_in, dy, D_x, D_y, Fx, Fy, a_p_x, a_w_x, a_e_x, a_s_x, a_n_x, b_x, a_p_t, f, p_x, w_ch, left, right, alpha):
        Ny, Nx = a_p_x.shape
        for j in range(Ny):
            for i in range(Nx):
                w = 0.5 * (w_ch[j + 1, i + 1] + w_ch[j + 1, i])
                fx = f / w ** 2
                F_w = (Fx[j + 1, i + 0] + Fx[j + 1, i + 1]) / 2
                F_e = (Fx[j + 1, i + 1] + Fx[j + 1, i + 2]) / 2
                F_s = (Fy[j + 1, i + 0] + Fy[j + 1, i + 1]) / 2
                F_n = (Fy[j + 2, i + 0] + Fy[j + 2, i + 1]) / 2
                C_w = max(0, F_w)
                C_e = max(0, -F_e)
                C_s = max(0, F_s)
                C_n = max(0, -F_n)
                a_w = D_x + C_w
                a_e = D_x + C_e
                a_s = D_y + C_s
                a_n = D_y + C_n
                Up = U[j+1, i+1]
                Uw = U[j+1, i+0]
                Ue = U[j+1, i+2]
                Us = U[j+0, i+1]
                Un = U[j+2, i+1]
                if p_x[j, i]:
                    b0 = -(dy * (P[j + 1, i] - P[j + 1, i + 1]) + a_w * Uw + a_e * Ue + a_s * Us + a_n * Un)
                    px = 20.0 * (a_w + a_e + a_s + a_n + fx)
                else:
                    px = 0.0
                    b0 = 0.5 * (F_w * (Uw + Up) - F_e * (Ue + Up) + F_s * (Us + Up) - F_n * (Un + Up)) - \
                           (C_w * (Uw - Up) + C_e * (Ue - Up) + C_s * (Us - Up) + C_n * (Un - Up))
                a_p = (a_w + a_e + a_s + a_n + a_p_t + fx + px) / alpha
                b = b0 + dy * (P[j + 1, i] - P[j + 1, i + 1]) + (a_p_t + (1 - alpha) * a_p) * Up
                a_p_x[j, i], a_w_x[j, i], a_e_x[j, i], a_s_x[j, i], a_n_x[j, i], b_x[j, i] = a_p, a_w, a_e, a_s, a_n, b
        a_p_x[:, 0][left != 3] = 1e100
        b_x[:, 0][left == 2] = V_in * 1e100
        a_p_x[:, -1][right != 3] = 1e100
        b_x[:, -1][right == 2] = -V_in * 1e100
        return a_p_x, a_w_x, a_e_x, a_s_x, a_n_x, b_x

    @my_static_jit
    def coefficients_v(V, P, V_in, dx, D_x, D_y, Fx, Fy, a_p_y, a_w_y, a_e_y, a_s_y, a_n_y, b_y, a_p_t, f, p_y, w_ch, bottom, top, alpha):
        Ny, Nx = a_p_y.shape
        for j in range(Ny):
            for i in range(Nx):
                w = 0.5 * (w_ch[j + 1, i + 1] + w_ch[j, i + 1])
                fy = f / w ** 2
                F_w = (Fx[j + 0, i + 1] + Fx[j + 1, i + 1]) / 2
                F_e = (Fx[j + 0, i + 2] + Fx[j + 1, i + 2]) / 2
                F_s = (Fy[j + 0, i + 1] + Fy[j + 1, i + 1]) / 2
                F_n = (Fy[j + 2, i + 1] + Fy[j + 1, i + 1]) / 2
                C_w = max(0, F_w)
                C_e = max(0, -F_e)
                C_s = max(0, F_s)
                C_n = max(0, -F_n)
                a_w = D_x + C_w
                a_e = D_x + C_e
                a_s = D_y + C_s
                a_n = D_y + C_n
                Vp = V[j+1, i+1]
                Vw = V[j+1, i+0]
                Ve = V[j+1, i+2]
                Vs = V[j+0, i+1]
                Vn = V[j+2, i+1]
                if p_y[j, i]:
                    b0 = -(dx * (P[j, i + 1] - P[j + 1, i + 1]) + a_w * Vw + a_e * Ve + a_s * Vs + a_n * Vn)
                    py = 20.0 * (a_w + a_e + a_s + a_n + fy)
                else:
                    py = 0.0
                    b0 = 0.5 * (F_w * (Vp+Vw) - F_e * (Vp+Ve) + F_s * (Vp+Vs) - F_n * (Vp+Vn)) - \
                               (C_w * (Vw-Vp) + C_e * (Ve-Vp) + C_s * (Vs-Vp) + C_n * (Vn-Vp))
                a_p = (a_w + a_e + a_s + a_n + a_p_t + fy + py) / alpha
                b = b0 + dx * (P[j, i + 1] - P[j + 1, i + 1]) + (a_p_t + (1 - alpha) * a_p) * Vp
                a_p_y[j, i], a_w_y[j, i], a_e_y[j, i], a_s_y[j, i], a_n_y[j, i], b_y[j, i] = a_p, a_w, a_e, a_s, a_n, b
        a_p_y[0, :][bottom != 3] = 1e100
        b_y[0, :][bottom == 2] = V_in * 1e100
        a_p_y[-1, :][top != 3] = 1e100
        b_y[-1, :][top == 2] = -V_in * 1e100
        return a_p_y, a_w_y, a_e_y, a_s_y, a_n_y, b_y

    @my_static_jit
    def coefficients_p(U, V, dx, dy, a_p_x, a_p_y, A_x, A_y, A_p, B):
        Ny, Nx = A_p.shape
        dx2 = dx * dx
        dy2 = dy * dy
        for j in range(Ny):
            for i in range(Nx + 1):
                A_x[j, i] = dy2 / a_p_x[j, i]
        for j in range(Ny + 1):
            for i in range(Nx):
                A_y[j, i] = dx2 / a_p_y[j, i]
        A_w = A_x[:, :-1]
        A_e = A_x[:, 1:]
        A_s = A_y[:-1, :]
        A_n = A_y[1:, :]
        for j in range(Ny):
            for i in range(Nx):
                A_p[j, i] = A_w[j, i] + A_e[j, i] + A_s[j, i] + A_n[j, i]
                B[j, i] = (dy * (U[j + 1, i + 1] - U[j + 1, i + 2]) +
                           dx * (V[j + 1, i + 1] - V[j + 2, i + 1]))
        return A_x, A_y, A_p, A_w, A_e, A_s, A_n, B

    @my_static_jit
    def apply_boundary_velocity_x(U, V_in, left, right, bottom, top):
        # 0: Wall, 1: Symmetry, 2: Inlet, 3: Outlet
        for j, bc in enumerate(left):
            if bc == 0 or bc == 1:
                U[j + 1, :2] = 0.0
            elif bc == 2:
                U[j + 1, :2] = V_in
            elif bc == 3:
                U[j + 1, 0] = U[j + 1, 1]
        for j, bc in enumerate(right):
            if bc == 0 or bc == 1:
                U[j + 1, -2:] = 0.0
            elif bc == 2:
                U[j + 1, -2:] = -V_in
            elif bc == 3:
                U[j + 1, -1] = U[j + 1, -2]
        for i, bc in enumerate(bottom):
            if bc == 0 or bc == 2 or bc == 3:
                U[0, i + 1] = -U[1, i + 1]
            else:
                U[0, i + 1] = U[1, i + 1]
        for i, bc in enumerate(top):
            if bc == 0 or bc == 2 or bc == 3:
                U[-1, i + 1] = -U[-2, i + 1]
            else:
                U[-1, i + 1] = U[-2, i + 1]
        return U

    @my_static_jit
    def apply_boundary_velocity_y(V, V_in, left, right, bottom, top):
        # 0: Wall, 1: Symmetry, 2: Inlet, 3: Outlet
        for j, bc in enumerate(left):
            if bc == 0 or bc == 2 or bc == 3:
                V[j + 1, 0] = -V[j + 1, 1]
            else:
                V[j + 1, 0] = V[j + 1,  1]
        for j, bc in enumerate(right):
            if bc == 0 or bc == 2 or bc == 3:
                V[j + 1, -1] = -V[j + 1, -2]
            else:
                V[j + 1, -1] = V[j + 1, -2]
        for i, bc in enumerate(bottom):
            if bc == 0 or bc == 1:
                V[:2, i + 1] = 0
            elif bc == 2:
                V[:2, i + 1] = V_in
            elif bc == 3:
                V[0, i + 1] = V[1, i + 1]
        for i, bc in enumerate(top):
            if bc == 0 or bc == 1:
                V[-2:, i + 1] = 0
            elif bc == 2:
                V[-2:, i + 1] = -V_in
            elif bc == 3:
                V[-1, i + 1] = V[-2, i + 1]
        return V

    @my_static_jit
    def apply_boundary_pressure(P, BCS):
        Ny, Nx = BCS.shape
        for j in range(1, Ny):
            P[j, 0] = P[j, 1] * BCS[j, 0]
            P[j, -1] = P[j, -2] * BCS[j, -1]
        for i in range(1, Nx):
            P[0, i] = P[1, i] * BCS[0, i]
            P[-1, i] = P[-2, i] * BCS[-1, i]
        return P

    @my_static_jit
    def correct(U, V, P, P_prime, a_p_x, a_p_y, dx, dy, alpha):
        Ny, Nx = P.shape
        beta = 1 - alpha
        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                Pw = P_prime[j, i - 1]
                Ps = P_prime[j - 1, i]
                Pp = P_prime[j, i]
                P[j, i] += beta * Pp
                U[j, i] += dy * (Pw - Pp) / a_p_x[j - 1, i - 1]
                V[j, i] += dx * (Ps - Pp) / a_p_y[j - 1, i - 1]
        for j in range(1, Ny - 1):
            i = Nx - 1
            Pw = P_prime[j, i - 1]
            Pp = P_prime[j, i]
            U[j, i] += dy * (Pw - Pp) / a_p_x[j - 1, i - 1]
        for i in range(1, Nx - 1):
            j = Ny - 1
            Ps = P_prime[j - 1, i]
            Pp = P_prime[j, i]
            V[j, i] += dx * (Ps - Pp) / a_p_y[j - 1, i - 1]
        return U, V, P

    @my_static_jit
    def residuals(Rx, Ry, U, V, a_p_x, a_w_x, a_e_x, a_s_x, a_n_x, b_x, a_p_y, a_w_y, a_e_y, a_s_y, a_n_y, b_y,
                  left, right, bottom, top):
        Ny, Nx = U.shape
        Nx, Ny = Nx - 2, Ny - 2
        for j in range(Ny):
            for i in range(Nx):
                Rx[j, i] = (b_x[j, i] +
                            a_w_x[j, i] * U[j + 1, i + 0] +
                            a_e_x[j, i] * U[j + 1, i + 2] +
                            a_s_x[j, i] * U[j + 0, i + 1] +
                            a_n_x[j, i] * U[j + 2, i + 1] -
                            a_p_x[j, i] * U[j + 1, i + 1])
        Rx[:, 0][left != 3] = 0.0
        Rx[:, -1][right != 3] = 0.0
        Ny, Nx = V.shape
        Nx, Ny = Nx - 2, Ny - 2
        for j in range(Ny):
            for i in range(Nx):
                Ry[j, i] = (b_y[j, i] +
                            a_w_y[j, i] * V[j + 1, i + 0] +
                            a_e_y[j, i] * V[j + 1, i + 2] +
                            a_s_y[j, i] * V[j + 0, i + 1] +
                            a_n_y[j, i] * V[j + 2, i + 1] -
                            a_p_y[j, i] * V[j + 1, i + 1])
        Ry[0, :][bottom != 3] = 0.0
        Ry[-1, :][top != 3] = 0.0
        return Rx, Ry

    def monitors(self):
        P_in = self.P[1:-1, 0][self.left == 2].sum() + self.P[1:-1, -1][self.right == 2].sum() + \
               self.P[0, 1:-1][self.bottom == 2].sum() + self.P[-1, 1:-1][self.top == 2].sum()
        self.P_in = P_in / ((self.left == 2).sum() + (self.right == 2).sum() +
                       (self.bottom == 2).sum() + (self.top == 2).sum())
        self.P_out = 0.0
        self.dP = self.P_in - self.P_out
        print(f'Pressure Drop: {self.dP}')

    def update_permeability(self):
        j, i = np.where(self.porous)
        self.w_ch[j, i] *= 0
        self.w_ch[self.w_ch < self.w_min] = self.w_min

    def iterate(self):
        # Increment Counter
        self.iter += 1

        # Update Momentum Coefficients
        t0 = tic()
        self.Fx, self.Fy = self.face_fluxes(self.U, self.V, self.dx, self.dy, self.Fx, self.Fy)
        self.a_p_x, self.a_w_x, self.a_e_x, self.a_s_x, self.a_n_x, self.b_x = self.coefficients_u(
            self.U, self.P, self.V_in, self.dy, self.D_x, self.D_y, self.Fx, self.Fy, self.a_p_x, self.a_w_x, self.a_e_x,
            self.a_s_x, self.a_n_x, self.b_x, self.a_p_t, self.f, self.p_x, self.w_ch, self.left, self.right, self.alpha)
        self.a_p_y, self.a_w_y, self.a_e_y, self.a_s_y, self.a_n_y, self.b_y = self.coefficients_v(
            self.V, self.P, self.V_in, self.dx, self.D_x, self.D_y, self.Fx, self.Fy, self.a_p_y, self.a_w_y, self.a_e_y,
            self.a_s_y, self.a_n_y, self.b_y, self.a_p_t, self.f, self.p_y, self.w_ch, self.bottom, self.top, self.alpha)

        # Solve Momentum Equations
        t1 = tic()
        convection(self.U, self.a_p_x, self.a_w_x, self.a_e_x, self.a_s_x, self.a_n_x, self.b_x)
        convection(self.V, self.a_p_y, self.a_w_y, self.a_e_y, self.a_s_y, self.a_n_y, self.b_y)
        self.U = self.apply_boundary_velocity_x(self.U, self.V_in, self.left, self.right, self.bottom, self.top)
        self.V = self.apply_boundary_velocity_y(self.V, self.V_in, self.left, self.right, self.bottom, self.top)

        # Update Pressure Coefficients
        t2 = tic()
        self.A_x, self.A_y, self.A_p, self.A_w, self.A_e, self.A_s, self.A_n, self.B = self.coefficients_p(
            self.U, self.V, self.dx, self.dy, self.a_p_x, self.a_p_y, self.A_x, self.A_y, self.A_p, self.B)

        # Solve Pressure Correction Equation
        t3 = tic()
        self.P_prime = np.zeros([self.Ny + 2, self.Nx + 2])
        diffusion(self.P_prime, self.A_p, self.A_w, self.A_e, self.A_s, self.A_n, self.B, self.BCS)
        self.P_prime = self.apply_boundary_pressure(self.P_prime, self.BCS)

        if np.any(np.abs(self.P_prime) > 1e10):
            print(self.P_prime)

        # Correct Fields
        t4 = tic()
        self.U, self.V, self.P = self.correct(self.U, self.V, self.P, self.P_prime,
                                              self.a_p_x, self.a_p_y, self.dx, self.dy, self.alpha)
        self.P = self.apply_boundary_pressure(self.P, self.BCS)
        self.U = self.apply_boundary_velocity_x(self.U, self.V_in, self.left, self.right, self.bottom, self.top)
        self.V = self.apply_boundary_velocity_y(self.V, self.V_in, self.left, self.right, self.bottom, self.top)

        # Update Monitors
        t5 = tic()
        self.Rx, self.Ry = self.residuals(self.Rx, self.Ry, self.U, self.V,
                                          self.a_p_x, self.a_w_x, self.a_e_x, self.a_s_x, self.a_n_x, self.b_x,
                                          self.a_p_y, self.a_w_y, self.a_e_y, self.a_s_y, self.a_n_y, self.b_y,
                                          self.left, self.right, self.bottom, self.top)
        if self.iter == 0:
            self.Rx0 = np.abs(self.Rx).sum()
            self.Ry0 = np.abs(self.Ry).sum()
            self.B0 = np.abs(self.B).sum()
        t6 = tic()
        print(f'Iteration: {self.iter}')
        print(
            f'Momentum Coefficients: {(t1 - t0) / 1e6}, Solving Momentum: {(t2 - t1) / 1e6}, '
            f'Pressure Coefficients: {(t3 - t2) / 1e6}, Solving Pressure: {(t4 - t3) / 1e6}, '
            f'Correcting Fields: {(t5 - t4) / 1e6}, Updating Metrics: {(t6 - t5) / 1e6}')
        print(f'Residuals: '
              f'{np.abs(self.Rx).sum()/self.Rx0}, '
              f'{np.abs(self.Ry).sum()/self.Ry0}, '
              f'{np.abs(self.B).sum()/self.B0}')
        self.monitors()

        # if self.iter==0:
        #     Ny = self.Ny
        #     Nx = self.Nx
        #     a = np.zeros((Ny, Nx), dtype=bool)
        #     a[3*Ny//8:5*Ny//8, Nx//4] = True
        #     self.set_level(np.where(a), True)


class Convection:
    def __init__(self, flow, nu=0.0000, alpha=0.9):
        # Initialize Variables
        self.iter = -1
        self.T_in = 0
        self.W = flow.W
        self.H = flow.H
        self.Nx = flow.Nx
        self.Ny = flow.Ny
        self.dx = flow.dx
        self.dy = flow.dy
        self.dt = flow.dt
        self.nu = nu
        self.alpha = alpha
        self.D_x = self.nu * self.dy / self.dx
        self.D_y = self.nu * self.dx / self.dy
        self.f = 7.54 * self.dx * self.dy * self.nu / (2*1e-4) * 1000.0  # TODO: Make Separate from w_ch
        self.a_p_0 = self.dx * self.dy / self.dt
        self.solid = flow.solid
        self.porous = flow.porous
        self.source = np.zeros([self.Ny, self.Nx], dtype=bool)
        # Convection Coefficients
        self.a_w = np.zeros([self.Ny, self.Nx])
        self.a_e = np.zeros([self.Ny, self.Nx])
        self.a_s = np.zeros([self.Ny, self.Nx])
        self.a_n = np.zeros([self.Ny, self.Nx])
        self.a_p = np.zeros([self.Ny, self.Nx])
        self.b = np.zeros([self.Ny, self.Nx])
        # Initial Guesses
        self.T = np.zeros([self.Ny + 2, self.Nx + 2])
        self.T_ref = np.zeros([self.Ny + 2, self.Nx + 2])
        self.Fx = flow.Fx
        self.Fy = flow.Fy
        self.R = np.ones([self.Ny, self.Nx])
        self.R0 = 1.0
        # Boundary Conditions - 0: Wall, 1: Symmetry, 2: Inlet, 3: Outlet
        self.left = flow.left
        self.right = flow.right
        self.bottom = flow.bottom
        self.top = flow.top

    def initialize(self):
        # Apply Boundary Conditions
        self.T = self.apply_boundary_conditions(self.T, self.left, self.right, self.bottom, self.top)

    def clear(self):
        self.T[1:-1, 1:-1] = 0.0

    def reset(self):
        self.clear()
        self.initialize()

    @my_static_jit
    def coefficients(T, T_ref, D_x, D_y, Fx, Fy, a_p_t, a_w_t, a_e_t, a_s_t, a_n_t, b_t, a_p_0, f, source,
                     left, right, bottom, top, porous, alpha):
        Ny, Nx = a_p_t.shape
        for j in range(Ny):
            for i in range(Nx):
                Fw = Fx[j + 1, i + 1]
                Fe = Fx[j + 1, i + 2]
                Fs = Fy[j + 1, i + 1]
                Fn = Fy[j + 2, i + 1]
                F_w = max(0, Fw)
                F_e = max(0, -Fe)
                F_s = max(0, Fs)
                F_n = max(0, -Fn)
                a_w = D_x + F_w
                a_e = D_x + F_e
                a_s = D_y + F_s
                a_n = D_y + F_n
                Tp = T[j+1, i+1]
                Tw = T[j+1, i+0]
                Te = T[j+1, i+2]
                Ts = T[j+0, i+1]
                Tn = T[j+2, i+1]
                c = 0.5 * (Fw * (Tp + Tw) - Fe * (Tp + Te) + Fs * (Tp + Ts) - Fn * (Tp + Tn)) - \
                    (F_w * (Tw - Tp) + F_e * (Te - Tp) + F_s * (Ts - Tp) + F_n * (Tn - Tp))
                a_p = (a_w + a_e + a_s + a_n + a_p_0 + f * porous[j, i]) / alpha
                b = c + (a_p_0 + (1 - alpha) * a_p) * Tp + f * porous[j, i] * T_ref[j + 1, i + 1] + f * source[j, i] /15
                a_p_t[j, i], a_w_t[j, i], a_e_t[j, i], a_s_t[j, i], a_n_t[j, i], b_t[j, i] = a_p, a_w, a_e, a_s, a_n, b
        # for j, bc in enumerate(left):
        #     if bc != 2:
        #         a_p_t[j, 0] -= a_w_t[j, 0]
        #         a_w_t[j, 0] = 0.0
        # for j, bc in enumerate(right):
        #     if bc != 2:
        #         a_p_t[j, -1] -= a_e_t[j, -1]
        #         a_e_t[j, -1] = 0.0
        # for i, bc in enumerate(bottom):
        #     if bc != 2:
        #         a_p_t[0, i] -= a_s_t[0, i]
        #         a_s_t[0, i] = 0.0
        # for i, bc in enumerate(top):
        #     if bc != 2:
        #         a_p_t[-1, i] -= a_n_t[-1, i]
        #         a_n_t[-1, i] = 0.0
        return a_p_t, a_w_t, a_e_t, a_s_t, a_n_t, b_t

    @my_static_jit
    def apply_boundary_conditions(T, left, right, bottom, top):
        # 0: Wall, 1: Symmetry, 2: Inlet, 3: Outlet
        for j, bc in enumerate(left):
            if bc != 2:
                T[j + 1, 0] = T[j + 1, 1]
        for j, bc in enumerate(right):
            if bc != 2:
                T[j + 1, -1] = T[j + 1, -2]
        for i, bc in enumerate(bottom):
            if bc != 2:
                T[0, i + 1] = T[1, i + 1]
        for i, bc in enumerate(top):
            if bc != 2:
                T[-1, i + 1] = T[-2, i + 1]
        return T

    @my_static_jit
    def residuals(R, T, a_p, a_w, a_e, a_s, a_n, b):
        Ny, Nx = T.shape
        Nx, Ny = Nx - 2, Ny - 2
        for j in range(Ny):
            for i in range(Nx):
                R[j, i] = (b[j, i] +
                           a_w[j, i] * T[j + 1, i + 0] +
                           a_e[j, i] * T[j + 1, i + 2] +
                           a_s[j, i] * T[j + 0, i + 1] +
                           a_n[j, i] * T[j + 2, i + 1] -
                           a_p[j, i] * T[j + 1, i + 1])
        return R

    def iterate(self):
        self.a_p, self.a_w, self.a_e, self.a_s, self.a_n, self.b = self.coefficients(
            self.T, self.T_ref, self.D_x, self.D_y, self.Fx, self.Fy,
            self.a_p, self.a_w, self.a_e, self.a_s, self.a_n, self.b, self.a_p_0, self.f, self.source,
            self.left, self.right, self.bottom, self.top, self.porous, self.alpha)
        convection(self.T, self.a_p, self.a_w, self.a_e, self.a_s, self.a_n, self.b)
        self.T = self.apply_boundary_conditions(self.T, self.left, self.right, self.bottom, self.top)
        self.R = self.residuals(self.R, self.T, self.a_p, self.a_w, self.a_e, self.a_s, self.a_n, self.b)
        if self.iter == 0:
            self.R0 = np.abs(self.R).sum()
        print(f'Convection Residual: {np.abs(self.R).sum()/self.R0}')


def test2():
    # Parameters
    iters = 3000
    W = 16e-4
    H = 1e-4
    Nx = 16*16
    Ny = 16
    flow = Flow(W, H, Nx, Ny)
    flow.U[1:-1, 1:-1] = flow.V_in
    flow.left[:] = 2
    flow.right[:] = 3
    flow.initialize()
    for _ in range(iters):
        flow.iterate()

    print(f'Theoretical dP: {12*flow.nu*flow.V_in/H**2*W} Pa, Actual dP: {np.average(flow.P[1:-1,0])} Pa')

    plt.figure()
    plt.contourf(flow.P[1:-1, 1:-1])
    plt.colorbar()
    plt.figure()
    plt.contourf(np.sqrt(flow.U[1:-1, 1:-2] ** 2 + flow.V[1:-2, 1:-1] ** 2))
    plt.colorbar()
    return flow


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    # Parameters
    iters = 1000
    V_in = 1
    w_ch = 1e3
    W = 1e-2
    H = 1e-2
    scale = 1
    Nx = 1024 // scale
    Ny = 1024 // scale
    # Create Equations List
    flow = Flow(V_in, w_ch, W, H, Nx, Ny)
    conv = Convection(flow)
    case = 0
    if case == 0:
        flow.U[:, :] = V_in
        flow.left[:] = 2
        flow.right[:] = 3
        conv.T[:, :] = 0.0
        conv.T[:, 0] = 1.0
        rand = np.random.random((Ny, Nx))
        flow.solid[rand < 0.1] = True
        flow.solid[:, 0] = False
        flow.solid[:, -1] = False
        flow.solid[0, :] = False
        flow.solid[-1, :] = False
        #conv.source[Ny // 4:3 * Ny // 4, Nx // 4:3 * Nx // 4] = True
    elif case == 1:
        flow.V[:, :] = V_in
        flow.bottom[:] = 3
        flow.top[:] = 2
        conv.T[:, :] = 0.0
        conv.T[:, 0] = 1.0
    elif case == 2:
        flow.top[:Nx // 4] = 2
        flow.top[3*Nx//4:] = 3
        flow.left[:] = 1
        flow.right[:] = 1
        flow.bottom[:] = 0
        # skip = Ny // 16
        # flow.solid[skip-1:-skip:skip, Nx//4:3*Nx//4] = True
        #flow.solid[Ny//4:3*Ny//4, Nx//4:3*Nx//4] = True
        flow.solid[Ny // 4:3 * Ny // 4, Nx // 2] = True
    elif case == 3:
        flow.left[:] = 2
        flow.bottom[:] = 2
        flow.right[:] = 3
        flow.top[:] = 3
        conv.T[1:-1, 0] = 1.0
        conv.T[0, 1:-1] = 0.0

    flow.initialize()
    conv.initialize()

    for k in range(iters + 1):
        flow.iterate()
        #conv.iterate()
        if k == 0:
            t0 = tic()
    t1 = tic()
    print(f'Average Execution Time: {(t1 - t0) / 1e6 / iters} ms per iteration')

    Uc, Vc = flow.quiver()
    Vmag = np.sqrt(Uc ** 2 + Vc ** 2)
    contourf(flow.P)
    contourf(Vmag)

    print(f'Actual dP: {flow.dP}; Theoretical dP: {12*flow.nu*flow.V_in/flow.H**2*flow.W}')
