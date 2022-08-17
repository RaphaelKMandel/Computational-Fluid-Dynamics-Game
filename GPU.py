import numpy as np
import cupy as cp
from time import time_ns as tic
from numba import vectorize, cuda
from matplotlib import pyplot as plt

Nx = 1024
Ny = 512
iters = 1000


def jacobi(x, a_p, a_w, a_e, a_s, a_n, b, n_iter):
    for k in range(n_iter):
        x[1:-1, 1:-1] = (b + a_w * x[1:-1, :-2] + a_e * x[1:-1, 2:] + a_s * x[:-2, 1:-1] + a_n * x[2:, 1:-1]) / a_p


def maincupy():
    x = cp.zeros([Ny+2, Nx+2], dtype=cp.float64)
    a_w = cp.ones([Ny, Nx], dtype=cp.float64)
    a_e = cp.ones([Ny, Nx], dtype=cp.float64)
    a_s = cp.ones([Ny, Nx], dtype=cp.float64)
    a_n = cp.ones([Ny, Nx], dtype=cp.float64)
    #a_w[:, 0] = 0.0
    #a_e[:, -1] = 0.0
    a_s[0, :] = 0.0
    a_n[-1, :] = 0.0
    a_p = a_w + a_e + a_s + a_n
    b = cp.zeros([Ny, Nx], dtype=cp.float64)
    b[:, :] = 0.0 / (Nx * Ny)
    x[:, 0] = 1.0

    jacobi(x.copy(), a_p, a_w, a_e, a_s, a_n, b, 1)

    t0 = tic()
    for k in range(iters):
        jacobi(x, a_p, a_w, a_e, a_s, a_n, b, 5)
        y = x.get()
    t1 = tic()

    print((t1-t0)/1e6/iters)

    return x


def mainnumpy():
    x = np.zeros([Ny+2, Nx+2], dtype=np.float64)
    a_w = np.ones([Ny, Nx], dtype=np.float64)
    a_e = np.ones([Ny, Nx], dtype=np.float64)
    a_s = np.ones([Ny, Nx], dtype=np.float64)
    a_n = np.ones([Ny, Nx], dtype=np.float64)
    #a_w[:, 0] = 0.0
    #a_e[:, -1] = 0.0
    a_s[0, :] = 0.0
    a_n[-1, :] = 0.0
    a_p = a_w + a_e + a_s + a_n
    b = np.zeros([Ny, Nx], dtype=np.float64)
    b[:, :] = 0.0 / (Nx * Ny)
    x[:, 0] = 1.0

    jacobi(x.copy(), a_p, a_w, a_e, a_s, a_n, b, 1)

    t0 = tic()
    for k in range(iters):
        jacobi(x, a_p, a_w, a_e, a_s, a_n, b, 1)
    t1 = tic()

    print((t1-t0)/1e6/iters)
    return x


if __name__ == '__main__':
    xg = maincupy()
    xc = mainnumpy()

    plt.figure()
    plt.contourf(xg.get())
    plt.colorbar()

    plt.figure()
    plt.contourf(xc)
    plt.colorbar()
