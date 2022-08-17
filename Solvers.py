import numpy as np
from numba import njit
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot as plt


JIT = True
implicit = True
np.set_printoptions(linewidth=150, precision=4)


def contourf(field):
    plt.figure()
    plt.contourf(field)
    plt.colorbar()


def my_jit(func):
    if JIT:
        return njit(func, cache=True)
    else:
        print('JIT is Disabled!')
        return func


def convection(x, a_p, a_w, a_e, a_s, a_n, b):
    adi(x, a_p, a_w, a_e, a_s, a_n, b, n_iter=1)
    sip(x, a_p, a_w, a_e, a_s, a_n, b, n_iter=1)


def diffusion(x, a_p, a_w, a_e, a_s, a_n, b, bcs):
    vcycle(x, a_p, a_w, a_e, a_s, a_n, b, bcs, n_iter=1)


def fcycle(v, a_p, a_w, a_e, a_s, a_n, b, bcs, n_iter=1):
    r = residuals(v, a_p, a_w, a_e, a_s, a_n, b)
    v += dofcycle(np.zeros_like(v), a_p, a_w, a_e, a_s, a_n, r, bcs, n_iter)


def vcycle(v, a_p, a_w, a_e, a_s, a_n, b, bcs, n_iter=1):
    r = residuals(v, a_p, a_w, a_e, a_s, a_n, b)
    v += dovcycle(np.zeros_like(v), a_p, a_w, a_e, a_s, a_n, r, bcs, n_iter)


def dofcycle(v, a_p, a_w, a_e, a_s, a_n, r, bcs, n_iter):
    Ny, Nx = a_p.shape
    if Nx % 2 > 0 or Ny % 2 > 0:
        direct_solver(v, a_p, a_w, a_e, a_s, a_n, r, bcs)
    else:
        A_P, A_W, A_E, A_S, A_N, R, BCS = restrict(a_p, a_w, a_e, a_s, a_n, r, bcs)
        NY, NX = A_P.shape
        E = np.zeros([NY + 2, NX + 2])
        E = dofcycle(E, A_P, A_W, A_E, A_S, A_N, R, BCS, n_iter)
        bilinear_interpolate(E, v, bcs)
        vcycle(v, a_p, a_w, a_e, a_s, a_n, r, bcs, n_iter=n_iter)
    return v


def dovcycle(v, a_p, a_w, a_e, a_s, a_n, b, bcs, n_iter):
    Ny, Nx = a_p.shape
    if Nx % 2 > 0 or Ny % 2 > 0:
        direct_solver(v, a_p, a_w, a_e, a_s, a_n, b, bcs)
    else:
        if implicit:
            A_p, A_w, A_e, A_s, A_n, B = implicit_boundary_conditions(a_p, a_w, a_e, a_s, a_n, b, bcs)
        else:
            A_p, A_w, A_e, A_s, A_n, B = a_p, a_w, a_e, a_s, a_n, b
        smoothen(v, A_p, A_w, A_e, A_s, A_n, B, bcs, n_iter)
        if implicit:
            r = residuals(v, A_p, A_w, A_e, A_s, A_n, B)
        else:
            r = residuals(v, a_p, a_w, a_e, a_s, a_n, b)
        A_P, A_W, A_E, A_S, A_N, R, BCS = restrict(a_p, a_w, a_e, a_s, a_n, r, bcs)
        NY, NX = A_P.shape
        E = np.zeros([NY + 2, NX + 2])
        E = dovcycle(E, A_P, A_W, A_E, A_S, A_N, R, BCS, n_iter)
        bilinear_interpolate(E, v, bcs)
        smoothen(v, A_p, A_w, A_e, A_s, A_n, B, bcs, n_iter)
    return v


def smoothen(v, a_p, a_w, a_e, a_s, a_n, b, bcs, n_iter):
    """Smooths the error in a Multigrid V-Cycle. The gs, sgs,
    and rbgs solvers work only when coefficients in x- and y-directions
    are (approximately) equal. When coefficients in x- and y-directions
    are not equal, use either lgs or sip."""
    mode = 4
    if mode == 0:
        gs(v, a_p, a_w, a_e, a_s, a_n, b, n_iter=n_iter)
    elif mode == 1:
        sgs(v, a_p, a_w, a_e, a_s, a_n, b, n_iter=n_iter)
    elif mode == 2:
        rbgs(v, a_p, a_w, a_e, a_s, a_n, b, n_iter=n_iter)
    elif mode == 3:
        lgs(v, a_p, a_w, a_e, a_s, a_n, b, n_iter=n_iter)
    elif mode == 4:
        sip(v, a_p, a_w, a_e, a_s, a_n, b, n_iter=n_iter)
    explicit_boundary_conditions(v, bcs)


@my_jit
def restrict(a_p, a_w, a_e, a_s, a_n, r, bcs):
    Ny, Nx = a_p.shape
    R = np.empty((Ny//2, Nx//2))
    A_w = np.empty((Ny // 2, Nx // 2))
    A_e = np.empty((Ny // 2, Nx // 2))
    A_s = np.empty((Ny // 2, Nx // 2))
    A_n = np.empty((Ny // 2, Nx // 2))
    A_p = np.empty((Ny // 2, Nx // 2))
    for J in range(Ny//2):
        j = 2 * J
        for I in range(Nx//2):
            i = 2 * I
            R[J, I] = (r[j, i] + r[j, i + 1] + r[j + 1, i] + r[j + 1, i + 1])
            a_0 = ((a_p[j, i] + a_p[j, i + 1] + a_p[j + 1, i] + a_p[j + 1, i + 1]) -
                   (a_w[j, i] + a_w[j, i + 1] + a_w[j + 1, i] + a_w[j + 1, i + 1]) -
                   (a_e[j, i] + a_e[j, i + 1] + a_e[j + 1, i] + a_e[j + 1, i + 1]) -
                   (a_s[j, i] + a_s[j, i + 1] + a_s[j + 1, i] + a_s[j + 1, i + 1]) -
                   (a_n[j, i] + a_n[j, i + 1] + a_n[j + 1, i] + a_n[j + 1, i + 1]))

            # if abs(a_0) > 1e-10:
            #     print('a_0 not 0')

            if I == 0:
                a_W = 1.0 / (1.5 / a_w[j, i] + 0.5 / a_w[j, i + 1]) + \
                      1.0 / (1.5 / a_w[j + 1, i] + 0.5 / a_w[j+1, i + 1])
            else:
                a_W = 1.0 / (1.0 / a_w[j, i] + 0.5 / a_w[j, i - 1] + 0.5 / a_w[j, i + 1]) + \
                      1.0 / (1.0 / a_w[j + 1, i] + 0.5 / a_w[j + 1, i - 1] + 0.5 / a_w[j + 1, i + 1])

            if I == Nx//2 - 1:
                a_E = 1.0 / (1.5 / a_e[j, i + 1] + 0.5 / a_e[j, i]) + \
                      1.0 / (1.5 / a_e[j + 1, i + 1] + 0.5 / a_e[j+1, i])
            else:
                a_E = 1.0 / (1.0 / a_e[j, i + 1] + 0.5 / a_e[j, i] + 0.5 / a_e[j, i + 2]) + \
                      1.0 / (1.0 / a_e[j + 1, i + 1] + 0.5 / a_e[j + 1, i] + 0.5 / a_e[j + 1, i + 2])

            if J == 0:
                a_S = 1.0 / (1.5 / a_s[j, i] + 0.5 / a_s[j + 1, i]) + \
                      1.0 / (1.5 / a_s[j, i + 1] + 0.5 / a_s[j + 1, i + 1])
            else:
                a_S = 1.0 / (1.0 / a_s[j, i] + 0.5 / a_s[j + 1, i] + 0.5 / a_s[j - 1, i]) + \
                      1.0 / (1.0 / a_s[j, i + 1] + 0.5 / a_s[j + 1, i + 1] + 0.5 / a_s[j - 1, i + 1])

            if J == Ny//2 - 1:
                a_N = 1.0 / (1.5 / a_n[j + 1, i] + 0.5 / a_n[j, i]) + \
                      1.0 / (1.5 / a_n[j + 1, i + 1] + 0.5 / a_n[j, i + 1])
            else:
                a_N = 1.0 / (1.0 / a_n[j + 1, i] + 0.5 / a_n[j + 2, i] + 0.5 / a_n[j, i]) + \
                      1.0 / (1.0 / a_n[j + 1, i + 1] + 0.5 / a_n[j + 2, i + 1] + 0.5 / a_n[j, i + 1])

            A_w[J, I] = a_W
            A_e[J, I] = a_E
            A_s[J, I] = a_S
            A_n[J, I] = a_N
            A_p[J, I] = a_0 + a_W + a_E + a_S + a_N
            #A_p[J, I] = a_W + a_E + a_S + a_N

    NY, NX = A_p.shape
    BCS = np.zeros((NY + 2, NX + 2))
    for J in range(1, NY + 1):
        j = 2 * J - 1
        BCS[J, 0] = 0.5 * (bcs[j, 0] + bcs[j + 1, 0])
        BCS[J, -1] = 0.5 * (bcs[j, -1] + bcs[j + 1, -1])
    for I in range(1, NX + 1):
        i = 2 * I - 1
        BCS[0, I] = 0.5 * (bcs[0, i] + bcs[0, i + 1])
        BCS[-1, I] = 0.5 * (bcs[-1, i] + bcs[-1, i + 1])

    return A_p, A_w, A_e, A_s, A_n, R, BCS


@my_jit
def inject(coarse, fine, bcs):
    Ny, Nx = coarse.shape
    Nx, Ny = Nx - 2, Ny - 2
    for J in range(Ny):
        j = 2 * J + 1
        for I in range(Nx):
            i = 2 * I + 1
            c = coarse[J + 1, I + 1]
            fine[j, i] += c
            fine[j+1, i] += c
            fine[j, i+1] += c
            fine[j+1, i+1] += c
    return fine


@my_jit
def bilinear_interpolate(coarse, fine, bcs):
    Ny, Nx = fine.shape
    Nx, Ny = Nx - 2, Ny - 2
    for j in range(1, Ny//2 + 1):
        J = 2 * j - 1
        for i in range(1, Nx//2 + 1):
            I = 2 * i - 1
            fine[J, I] += (9 * coarse[j, i] + coarse[j-1, i-1] + 3 * coarse[j-1, i] + 3 * coarse[j, i-1]) / 16
            fine[J+1, I] += (9 * coarse[j, i] + coarse[j+1, i-1] + 3 * coarse[j+1, i] + 3 * coarse[j, i-1]) / 16
            fine[J, I+1] += (9 * coarse[j, i] + coarse[j-1, i+1] + 3 * coarse[j-1, i] + 3 * coarse[j, i+1]) / 16
            fine[J+1, I+1] += (9 * coarse[j, i] + coarse[j+1, i+1] + 3 * coarse[j+1, i] + 3 * coarse[j, i+1]) / 16
    explicit_boundary_conditions(fine, bcs)


@my_jit
def gs(x, A_p, A_w, A_e, A_s, A_n, B, n_iter=1):
    Ny, Nx = A_p.shape
    for n_it in range(n_iter):
        for j in range(1, Ny + 1):
            for i in range(1, Nx + 1):
                x[j, i] = (A_w[j - 1, i - 1] * x[j, i - 1] +
                           A_e[j - 1, i - 1] * x[j, i + 1] +
                           A_s[j - 1, i - 1] * x[j - 1, i] +
                           A_n[j - 1, i - 1] * x[j + 1, i] +
                           B[j - 1, i - 1]) / A_p[j - 1, i - 1]


@my_jit
def sgs(x, A_p, A_w, A_e, A_s, A_n, B, n_iter=1):
    Ny, Nx = A_p.shape
    for n_it in range(n_iter):
        for j in range(1, Ny + 1):
            for i in range(1, Nx + 1):
                x[j, i] = (A_w[j - 1, i - 1] * x[j, i - 1] +
                           A_e[j - 1, i - 1] * x[j, i + 1] +
                           A_s[j - 1, i - 1] * x[j - 1, i] +
                           A_n[j - 1, i - 1] * x[j + 1, i] +
                           B[j - 1, i - 1]) / A_p[j - 1, i - 1]
        for j in range(Ny, 0, -1):
            for i in range(Nx, 0, -1):
                x[j, i] = (A_w[j - 1, i - 1] * x[j, i - 1] +
                           A_e[j - 1, i - 1] * x[j, i + 1] +
                           A_s[j - 1, i - 1] * x[j - 1, i] +
                           A_n[j - 1, i - 1] * x[j + 1, i] +
                           B[j - 1, i - 1]) / A_p[j - 1, i - 1]


@my_jit
def rbgs(x, A_p, A_w, A_e, A_s, A_n, B, n_iter=1):
    Ny, Nx = A_p.shape
    for n_it in range(n_iter):
        for i0, j0 in zip((0, 1, 0, 1), (0, 1, 1, 0)):
            for j in range(1 + j0, Ny + j0, 2):
                for i in range(1 + i0, Nx + i0, 2):
                    x[j, i] = (A_w[j - 1, i - 1] * x[j, i - 1] +
                               A_e[j - 1, i - 1] * x[j, i + 1] +
                               A_s[j - 1, i - 1] * x[j - 1, i] +
                               A_n[j - 1, i - 1] * x[j + 1, i] +
                               B[j - 1, i - 1]) / A_p[j - 1, i - 1]


@my_jit
def lgs(x, a_p, a_w, a_e, a_s, a_n, b, n_iter=1):
    Ny, Nx = a_p.shape
    for n_it in range(n_iter):
        for i in range(Nx):
            A_p = a_p[:, i]
            A_w = a_s[:, i]
            A_e = a_n[:, i]
            B = b[:, i] + a_w[:, i] * x[1:-1, i] + a_e[:, i] * x[1:-1, i+2]
            tdma(x[1:-1, i + 1], A_p, A_w, A_e, B)
        for j in range(Ny):
            A_p = a_p[j, :]
            A_w = a_w[j, :]
            A_e = a_e[j, :]
            B = b[j, :] + a_s[j, :] * x[j, 1:-1] + a_n[j, :] * x[j+2, 1:-1]
            tdma(x[j+1, 1:-1], A_p, A_w, A_e, B)


@my_jit
def adi(x, a_p, a_w, a_e, a_s, a_n, b, n_iter=1):
    Ny, Nx = a_p.shape
    for n_it in range(n_iter):
        r = residuals(x, a_p, a_w, a_e, a_s, a_n, b)
        A_p = np.zeros(Nx)
        A_w = np.zeros(Nx)
        A_e = np.zeros(Nx)
        R = np.zeros(Nx)
        E = np.zeros(Nx)
        for j in range(Ny):
            for i in range(Nx):
                R[i] += r[j, i]
                A_p[i] += a_p[j, i] - a_s[j, i] - a_n[j, i]
                A_w[i] += a_w[j, i]
                A_e[i] += a_e[j, i]
        for i in range(Nx):
            A_p[i] += a_s[0, i]
            A_p[i] += a_n[-1, i]
        tdma(E, A_p, A_w, A_e, R)
        for j in range(Ny):
            for i in range(Nx):
                x[j+1, i+1] += E[i]

        r = residuals(x, a_p, a_w, a_e, a_s, a_n, b)
        A_p = np.zeros(Ny)
        A_w = np.zeros(Ny)
        A_e = np.zeros(Ny)
        R = np.zeros(Ny)
        E = np.zeros(Ny)
        for j in range(Ny):
            for i in range(Nx):
                R[j] += r[j, i]
                A_p[j] += a_p[j, i] - a_w[j, i] - a_e[j, i]
                A_w[j] += a_s[j, i]
                A_e[j] += a_n[j, i]
        for j in range(Ny):
            A_p[j] += a_w[j, 0]
            A_p[j] += a_e[j, -1]
        tdma(E, A_p, A_w, A_e, R)
        for j in range(Ny):
            for i in range(Nx):
                x[j+1, i+1] += E[j]


@my_jit
def sip(x, a_p, a_w, a_e, a_s, a_n, b, n_iter=1):
    Ny, Nx = a_p.shape
    alpha = 0.92
    L_p = np.zeros((Ny + 2, Nx + 2))
    L_w = np.zeros((Ny + 2, Nx + 2))
    L_s = np.zeros((Ny + 2, Nx + 2))
    U_e = np.zeros((Ny + 2, Nx + 2))
    U_n = np.zeros((Ny + 2, Nx + 2))
    for i in range(1, Nx + 1):
        for j in range(1, Ny + 1):
            L_w[j, i] = -a_w[j-1, i-1] / (1.0 + alpha * U_n[j, i - 1])
            L_s[j, i] = -a_s[j-1, i-1] / (1.0 + alpha * U_e[j - 1, i])
            L_p[j, i] = a_p[j-1, i-1] - L_w[j, i] * U_e[j, i - 1] - L_s[j, i] * U_n[j-1, i] + \
                         alpha * (L_w[j, i] * U_n[j, i - 1] + L_s[j, i] * U_e[j-1, i])
            U_n[j, i] = (-a_n[j-1, i-1] - alpha * L_w[j, i] * U_n[j, i-1]) / L_p[j, i]
            U_e[j, i] = (-a_e[j-1, i-1] - alpha * L_s[j, i] * U_e[j-1, i]) / L_p[j, i]

    r = np.zeros((Ny + 2, Nx + 2))
    d = np.zeros((Ny + 2, Nx + 2))
    R = np.zeros((Ny + 2, Nx + 2))
    for it in range(n_iter):
        r[1:-1, 1:-1] = residuals(x, a_p, a_w, a_e, a_s, a_n, b)
        for i in range(1, Nx + 1):
            for j in range(1, Ny + 1):
                R[j, i] = (r[j, i] - L_s[j, i] * R[j-1, i] - L_w[j, i] * R[j, i-1]) / L_p[j, i]

        for i in range(Nx, 0, -1):
            for j in range(Ny, 0, -1):
                d[j, i] = R[j, i] - U_n[j, i] * d[j+1, i] - U_e[j, i] * d[j, i+1]
                x[j, i] += d[j, i]


@my_jit
def tdma(x, a_p, a_w, a_e, b):
    N = a_p.shape[0]
    c = np.empty(N)
    d = np.empty(N)
    c[0] = -a_e[0] / a_p[0]
    d[0] = b[0] / a_p[0]
    for i in range(1, N):
        c[i] = -a_e[i] / (a_p[i] + a_w[i] * c[i - 1])
        d[i] = (b[i] + a_w[i] * d[i - 1]) / (a_p[i] + a_w[i] * c[i - 1])

    x[N - 1] = d[N - 1]
    for i in range(N - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]


def direct_solver(x, a_p, a_w, a_e, a_s, a_n, b, bcs):
    # Array Sizes
    Ny, Nx = a_p.shape
    N = Nx * Ny
    # Implicit Boundary Conditions
    A_p, A_w, A_e, A_s, A_n, B = implicit_boundary_conditions(a_p, a_w, a_e, a_s, a_n, b, bcs)
    # Explicit Boundary Conditions
    B[:, 0] += A_w[:, 0] * x[1:-1, 0]
    A_w[:, 0] = 0.0
    B[:, -1] += A_e[:, -1] * x[1:-1, -1]
    A_e[:, -1] = 0.0
    B[0, :] += A_s[0, :] * x[0, 1:-1]
    A_s[0, :] = 0.0
    B[-1, :] += A_n[-1, :] * x[-1, 1:-1]
    A_n[-1, :] = 0.0
    # Assemble Sparse Matrix
    if N == 1:
        x[1:-1, 1:-1] = B / A_p
    else:
        # Assemble Sparse Matrix
        if Nx > 1 and Ny > 1:
            i = np.hstack((range(0, N), range(0, N-1), range(1, N), range(0, N - Nx), range(Nx, N)))
            j = np.hstack((range(0, N), range(1, N), range(0, N-1), range(Nx, N), range(0, N - Nx)))
            v = np.hstack((A_p.flatten(),
                           -A_w.flatten()[1:], -A_e.flatten()[:-1],
                           -A_s.flatten()[Nx:], -A_n.flatten()[:-Nx]))
        elif Nx > 1:
            i = np.hstack((range(0, N), range(0, N - 1), range(1, N)))
            j = np.hstack((range(0, N), range(1, N), range(0, N - 1)))
            v = np.hstack((A_p.flatten(), -A_w.flatten()[1:], -A_e.flatten()[:-1]))
        elif Ny > 1:
            i = np.hstack((range(0, N), range(0, N - Nx), range(Nx, N)))
            j = np.hstack((range(0, N), range(Nx, N), range(0, N - Nx)))
            v = np.hstack((A_p.flatten(), -A_s.flatten()[Nx:], -A_n.flatten()[:-Nx]))
        A = sp.coo_matrix((v, (j, i)), shape=(N, N))
        # Convert to Compatible Sparse Format
        A = A.tocsc()
        # Sparse Solve
        x[1:-1, 1:-1] = spsolve(A, B.flatten()).reshape(Ny, Nx)
    explicit_boundary_conditions(x, bcs)


@my_jit
def explicit_boundary_conditions(x, bcs):
    Ny, Nx = x.shape
    # Left/Right
    for j in range(1, Ny - 1):
        bc = bcs[j, 0]
        if not bc == 0:
            x[j, 0] = x[j, 1] * bc
        bc = bcs[j, -1]
        if not bc == 0:
            x[j, -1] = x[j, -2] * bc
    # Top/Bottom
    for i in range(1, Nx - 1):
        bc = bcs[0, i]
        if not bc == 0:
            x[0, i] = x[1, i] * bc
        bc = bcs[-1, i]
        if not bc == 0:
            x[-1, i] = x[-2, i] * bc
    # Corners
    x[0, 0] = (x[0, 1] + x[1, 0]) / 2
    x[-1, 0] = (x[-1, 1] + x[-2, 0]) / 2
    x[0, -1] = (x[0, -2] + x[1, -1]) / 2
    x[-1, -1] = (x[-1, -2] + x[-2, -1]) / 2


@my_jit
def implicit_boundary_conditions(a_p, a_w, a_e, a_s, a_n, b, bcs):
    A_p = a_p.copy()
    A_w = a_w.copy()
    A_e = a_e.copy()
    A_s = a_s.copy()
    A_n = a_n.copy()
    B = b.copy()
    Ny, Nx = a_p.shape
    # Left/Right
    for j in range(Ny):
        bc = bcs[j + 1, 0]
        if not bc == 0:
            A_p[j, 0] -= A_w[j, 0] * bc
            A_w[j, 0] = 0.0

        bc = bcs[j + 1, -1]
        if not bc == 0:
            A_p[j, -1] -= A_e[j, -1] * bc
            A_e[j, -1] = 0.0
    # Top/Bottom
    for i in range(Nx):
        bc = bcs[0, i + 1]
        if not bc == 0:
            A_p[0, i] -= A_s[0, i] * bc
            A_s[0, i] = 0.0
        bc = bcs[-1, i + 1]
        if not bc == 0:
            A_p[-1, i] -= A_n[-1, i] * bc
            A_n[-1, i] = 0.0
    return A_p, A_w, A_e, A_s, A_n, B


@my_jit
def residuals(x, a_p, a_w, a_e, a_s, a_n, b):
    Ny, Nx = a_p.shape
    r = np.empty((Ny, Nx))
    for j in range(Ny):
        for i in range(Nx):
            r[j, i] = a_w[j, i] * x[j+1, i] + \
                      a_e[j, i] * x[j+1, i+2] + \
                      a_s[j, i] * x[j, i+1] + \
                      a_n[j, i] * x[j+2, i+1] + \
                      b[j, i] - a_p[j, i] * x[j+1, i+1]
    return r
