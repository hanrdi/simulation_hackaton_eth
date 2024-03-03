import firedrake as fd
from visualizer import plot_approx

from perlin_noise import PerlinNoise

from ufl.conditional import MaxValue

Nx = 100
Ny = 100
Lx = 1.0  # non-dim
Ly = 1.0  # non-dim

initial_rho_value = 0.5

mesh = fd.RectangleMesh(Nx, Ny, Lx, Ly)

RHO = fd.FunctionSpace(mesh, "DG", 0)
# RHOF = fd.FunctionSpace(mesh, "CG", 1)

rho = fd.Function(RHO, name="Volume Fraction")

x, y = fd.SpatialCoordinate(mesh)


def gaussian(x, mean=0.5, stddev=0.5):
    # 1 / (stddev * fd.sqrt(2 * fd.pi)) *
    return fd.exp(-0.5 * ((x - mean) / stddev) ** 2)


def circle(x, y, c_x=0.5, c_y=0.5):
    return 1 - (x - c_x) ** 2 - (y - c_y) ** 2


def normalize(x):
    t = x**2
    return t


rho.interpolate(normalize(fd.sin(y * 100) * gaussian(y) * circle(x, y)))


plot_approx(rho)
