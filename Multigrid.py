import numpy as np
from numba import njit, prange
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot as plt
from time import time_ns as tic


JIT = True
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


def vcycle(v, a_p, a_w, a_e, a_s, a_n, b, bcs, n_iter=1):
    for _ in range(1):
        r = deferred_correction(v, a_p, a_w, a_e, a_s, a_n, b)
        v += dovcycle(np.zeros_like(v), r, bcs, n_iter)


def dovcycle(v, b, bcs, n_iter):
    Ny, Nx = b.shape
    if Nx % 2 > 0 or Ny % 2 > 0:
        #smoothen(v, b, bcs, n_iter)
        #gs(v, b, bcs, 10)
        smooth_boundaries(v, b, bcs, 10)
    else:
        smoothen(v, b, bcs, n_iter)
        r = residuals(v, b)
        R, BCS = restrict(r, bcs)
        NY, NX = R.shape
        E = np.zeros([NY + 2, NX + 2])
        E = dovcycle(E, R, BCS, n_iter)
        prolong(E, v, bcs)
        smoothen(v, b, bcs, n_iter)
    return v


def smoothen(x, b, bcs, n_iter):
    rbgs(x, b, bcs, n_iter)


def restrict(r, bcs):
    R, BCS = restrict_residuals(r, bcs)
    return R, BCS


def prolong(coarse, fine, bcs):
    mode = 1
    if mode == 0:
        inject(coarse, fine, bcs)
    elif mode == 1:
        bilinear_interpolate(coarse, fine, bcs)


@my_jit
def deferred_correction(v, a_p, a_w, a_e, a_s , a_n, b):
    Ny, Nx = b.shape
    r = np.empty((Ny, Nx))
    for j in range(Ny):
        for i in range(Nx):
            r[j, i] = (a_w[j, i] * v[j+1, i] +
                       a_e[j, i] * v[j + 1, i + 2] +
                       a_s[j, i] * v[j, i + 1] +
                       a_n[j, i] * v[j + 2, i + 1] + b[j, i]) / a_p[j, i] - v[j + 1, i + 1]
    return r


@njit(parallel=True, cache=True)
def jb(x, b, bcs, n_iter):
    Ny, Nx = b.shape
    xn = x.copy()
    for n_it in range(n_iter):
        for j in prange(1, Ny + 1):
            for i in range(1, Nx + 1):
                xn[j, i] = 0.25 * (x[j, i - 1] + x[j, i + 1] + x[j - 1, i] + x[j + 1, i]) + b[j - 1, i - 1]
        x, xn = xn, x
        explicit_boundary_conditions(x, bcs)
    return xn


@my_jit
def gs(x, b, bcs, n_iter):
    Ny, Nx = b.shape
    for n_it in range(n_iter):
        for j in range(1, Ny + 1):
            for i in range(1, Nx + 1):
                x[j, i] = 0.25 * (x[j, i - 1] + x[j, i + 1] + x[j - 1, i] + x[j + 1, i]) + b[j - 1, i - 1]
        explicit_boundary_conditions(x, bcs)


@my_jit
def rbgs(x, b, bcs, n_iter):
    Ny, Nx = b.shape
    for n_it in range(n_iter):
        for i0, j0 in zip((0, 1, 0, 1), (0, 1, 1, 0)):
            for j in range(1 + j0, Ny + j0, 2):
                for i in range(1 + i0, Nx + i0, 2):
                    x[j, i] = 0.25 * (x[j, i - 1] + x[j, i + 1] + x[j - 1, i] + x[j + 1, i]) + b[j - 1, i - 1]
        smooth_boundaries(x, b, bcs, 5)
        #explicit_boundary_conditions(x, bcs)


@my_jit
def smooth_boundaries(x, b, bcs, n_iter):
    Ny, Nx = b.shape
    for n_it in range(n_iter):
        for j in range(1, Ny + 1):
            i = 1
            x0 = 0.25 * (x[j, i - 1] + x[j, i + 1] + x[j - 1, i] + x[j + 1, i]) + b[j - 1, i - 1]
            x[j, i] = x0
            x[j, i-1] = bcs[j, i-1] * x0
            i = Nx
            x0 = 0.25 * (x[j, i - 1] + x[j, i + 1] + x[j - 1, i] + x[j + 1, i]) + b[j - 1, i - 1]
            x[j, i] = x0
            x[j, i+1] = bcs[j, i+1] * x0
        for i in range(1, Nx + 1):
            j = 1
            x0 = 0.25 * (x[j, i - 1] + x[j, i + 1] + x[j - 1, i] + x[j + 1, i]) + b[j - 1, i - 1]
            x[j, i] = x0
            x[j-1, i] = bcs[j-1, i] * x0
            j = Ny
            x0 = 0.25 * (x[j, i - 1] + x[j, i + 1] + x[j - 1, i] + x[j + 1, i]) + b[j - 1, i - 1]
            x[j, i] = x0
            x[j+1, i] = bcs[j+1, i] * x0


@my_jit
def restrict_residuals(r, bcs):
        Ny, Nx = r.shape
        NX, NY = Nx//2, Ny//2
        R = np.empty((NY, NX))
        for J in range(Ny//2):
            j = 2 * J
            for I in range(Nx//2):
                i = 2 * I
                R[J, I] = r[j, i] + r[j, i + 1] + r[j + 1, i] + r[j + 1, i + 1]
        BCS = np.zeros((NY + 2, NX + 2))
        for J in range(1, NY + 1):
            j = 2 * J - 1
            BCS[J, 0] = 0.5 * (bcs[j, 0] + bcs[j + 1, 0])
            BCS[J, -1] = 0.5 * (bcs[j, -1] + bcs[j + 1, -1])
        for I in range(1, NX + 1):
            i = 2 * I - 1
            BCS[0, I] = 0.5 * (bcs[0, i] + bcs[0, i + 1])
            BCS[-1, I] = 0.5 * (bcs[-1, i] + bcs[-1, i + 1])
        return R, BCS


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
            fine[j + 1, i] += c
            fine[j, i + 1] += c
            fine[j + 1, i + 1] += c
    explicit_boundary_conditions(fine, bcs)
    return fine


@my_jit
def bilinear_interpolate(coarse, fine, bcs):
    Ny, Nx = fine.shape
    Nx, Ny = Nx - 2, Ny - 2
    for j in range(1, Ny // 2 + 1):
        J = 2 * j - 1
        for i in range(1, Nx // 2 + 1):
            I = 2 * i - 1
            fine[J, I] += (9 * coarse[j, i] + coarse[j - 1, i - 1] + 3 * coarse[j - 1, i] + 3 * coarse[
                j, i - 1]) / 16
            fine[J + 1, I] += (9 * coarse[j, i] + coarse[j + 1, i - 1] + 3 * coarse[j + 1, i] + 3 * coarse[
                j, i - 1]) / 16
            fine[J, I + 1] += (9 * coarse[j, i] + coarse[j - 1, i + 1] + 3 * coarse[j - 1, i] + 3 * coarse[
                j, i + 1]) / 16
            fine[J + 1, I + 1] += (9 * coarse[j, i] + coarse[j + 1, i + 1] + 3 * coarse[j + 1, i] + 3 * coarse[
                j, i + 1]) / 16
    explicit_boundary_conditions(fine, bcs)


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
def residuals(x, b):
    Ny, Nx = b.shape
    r = np.empty((Ny, Nx))
    for j in range(Ny):
        for i in range(Nx):
            r[j, i] = 0.25 * (
                      x[j+1, i] +
                      x[j+1, i+2] +
                      x[j, i+1] +
                      x[j+2, i+1]) + \
                      b[j, i] - x[j+1, i+1]
    return r


@my_jit
def res(v, a_p, a_w, a_e, a_s, a_n, b):
    Ny, Nx = b.shape
    r = np.empty((Ny, Nx))
    R = 0.0
    for j in range(1, Ny + 1):
        for i in range(1, Nx + 1):
            r0 = r[j - 1, i - 1] = b[j - 1, i - 1] + \
                                   a_w[j - 1, i - 1] * v[j, i - 1] + \
                                   a_e[j - 1, i - 1] * v[j, i + 1] + \
                                   a_s[j - 1, i - 1] * v[j - 1, i] + \
                                   a_n[j - 1, i - 1] * v[j + 1, i] - \
                                   a_p[j - 1, i - 1] * v[j, i]
            R += abs(r0)
    print(R)
    return r, R


if __name__ == '__main__':
    Nx = 512
    Ny = Nx
    a = (0.1+np.arange(Nx+2)/Nx).repeat(Ny+2).reshape((Nx+2, Ny+2)).T
    a[Ny//4:3*Ny//4, Nx//2] = 0.000000001
    a_w = 0.5 * (a[1:-1, :-2] + a[1:-1, 1:-1])
    a_e = 0.5 * (a[1:-1, 2:] + a[1:-1, 1:-1])
    a_s = 0.5 * (a[:-2, 1:-1] + a[1:-1, 1:-1])
    a_n = 0.5 * (a[2:, 1:-1] + a[1:-1, 1:-1])
    b = np.zeros((Ny, Nx))

    b[Ny//4:3*Ny//4, Nx//2-1] = 1.0 / Ny

    a_p = a_w + a_e + a_s + a_n

    v = np.zeros((Ny + 2, Nx + 2))
    bcs = np.ones((Ny + 2, Nx + 2))
    bcs[1:-1, -1] = -1

    vcycle(v.copy(), a_p, a_w, a_e, a_s, a_n, b, bcs)

    iters = 500
    Rs = []
    r, R = res(v, a_p, a_w, a_e, a_s, a_n, b)
    Rs.append(R)
    t0 = tic()
    for k in range(iters):
        vcycle(v, a_p, a_w, a_e, a_s, a_n, b, bcs, n_iter=1)
        r, R = res(v, a_p, a_w, a_e, a_s, a_n, b)
        Rs.append(R)
    t1 = tic()
    print(f'Time: {(t1-t0)/1e6/iters} ms per iteration; Convergence Factor: {(Rs[-1]/Rs[0])**(1/iters)} per iteration')
    contourf(v)
