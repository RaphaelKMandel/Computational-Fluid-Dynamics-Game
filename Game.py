import pygame
import numpy as np
from Constants import *
from numba import njit

# Screen and Mesh
SCREEN_SCALE = (6 * 1) // 1
ASPECT_RATIO = 2
BORDER_THICKNESS = SCREEN_SCALE
Ny = 256 // 2
Nx = Ny * ASPECT_RATIO
SCREEN_WIDTH = Nx * SCREEN_SCALE + 2 * BORDER_THICKNESS
SCREEN_HEIGHT = Ny * SCREEN_SCALE + 2 * BORDER_THICKNESS

nx = min(32, Nx)
ny = min(32, Ny)
skip = min(Ny, Nx) // min(nx, ny)

rgb = np.zeros([Nx + 2, Ny + 2, 3], dtype="uint8")
RGB = np.zeros([SCREEN_WIDTH, SCREEN_HEIGHT, 3], dtype="uint8")

time_step = 1e-3
V_in = 1.0
#w_ch = 4e-5
#w_ch = 1e-6
H = 1e-2
W = H * ASPECT_RATIO

# TODO: Organize Inputs


def get_rgb(z, rgb, solid, porous):
    # z_min = 0.0
    # z_max = 1.0
    # print(f'z_max: {z.max()}')
    # z[z < z_min] = z_min
    # z[z > z_max] = z_max

    z_min = z.min()
    z_max = z.max()
    if z_min > 0.0:
        z_min = 0.0
    if z_max < 1.0:
        z_max = 1.0

    # TODO: Make Switching Dynamic

    z = (z - z_min) / (z_max - z_min)

    z[1:-1, 1:-1][solid] = None
    z[1:-1, 1:-1][porous] = None
    z = z.transpose()
    rgb[:, :, 0] = 255.0 * z
    rgb[:, :, 1] = 0
    rgb[:, :, 2] = 255.0 * (1.0 - z)
    return rgb


def get_mouse():
    x, y = pygame.mouse.get_pos()
    if x < BORDER_THICKNESS or x > SCREEN_WIDTH - BORDER_THICKNESS:
        i = None
    else:
        i = int((x - BORDER_THICKNESS) / (SCREEN_WIDTH - 2 * BORDER_THICKNESS) * Nx)
    if y < BORDER_THICKNESS or y > SCREEN_HEIGHT - BORDER_THICKNESS:
        j = None
    else:
        j = int((y - BORDER_THICKNESS) / (SCREEN_HEIGHT - 2 * BORDER_THICKNESS) * Ny)
    print(f'{(x, y, i, j)}')
    return i, j


def draw_display(t, rgb, RGB, solid, porous, display):
    rgb = get_rgb(t, rgb, solid, porous)
    inject(rgb, RGB, SCREEN_SCALE)
    RGB[:BORDER_THICKNESS, :, :] = BLACK
    RGB[-BORDER_THICKNESS:, :, :] = BLACK
    RGB[:, :BORDER_THICKNESS, :] = BLACK
    RGB[:, -BORDER_THICKNESS:, :] = BLACK
    #RGB[OUT] = 0
    display.fill(BLACK)
    surf = pygame.surfarray.make_surface(RGB)
    display.blit(surf, (0, 0))


def my_quiver_plot(i, j, display):
    J, I = i[1:-1, 1:-1].shape
    dx = i[1:-1:skip, 1:-1:skip] * SCREEN_SCALE * 0.75
    dy = j[1:-1:skip, 1:-1:skip] * SCREEN_SCALE * 0.75
    x_n = np.arange(0.5, I + 0.5) / I
    y_n = np.arange(0.5, J + 0.5) / J
    for col in range(I//skip):
        x0 = SCREEN_WIDTH * x_n[col*skip] + SCREEN_SCALE
        for row in range(J//skip):
            y0 = SCREEN_HEIGHT * y_n[row*skip] + SCREEN_SCALE
            x1 = x0 + dx[row, col]
            y1 = y0 + dy[row, col]
            pygame.draw.line(display, WHITE, (x0, y0), (x1, y1))


@njit(cache=True)
def bilinear_interpolation(f, F, scale):
    ny, nx, nz = f.shape
    for j in range(ny-1):
        for i in range(nx - 1):
            f1 = f[j, i, :]
            f2 = f[j, i + 1, :]
            f3 = f[j + 1, i, :]
            f4 = f[j + 1, i + 1, :]
            for n in range(scale):
                y = (n + 1 / 2) / scale
                for m in range(scale):
                    x = (m + 1 / 2) / scale
                    for k in range(nz):
                        F[j * scale + n, i * scale + m, k] = (1-x)*(1-y)*f1[k] + \
                                                             x*(1-y)*f2[k] + \
                                                             (1-x)*y*f3[k] + \
                                                             x*y*f4[k]


@njit(cache=True)
def inject(f, F, scale):
    ny, nx, nz = f.shape
    for j in range(ny-1):
        J = j * scale
        for i in range(nx - 1):
            I = i * scale
            F[J:J+scale, I:I+scale, :] = f[j, i, :]
