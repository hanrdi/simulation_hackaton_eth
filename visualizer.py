import matplotlib.pyplot as plt
import h5py

from firedrake.checkpointing import CheckpointFile
from firedrake import *
from firedrake.pyplot import triplot
from firedrake.pyplot import tricontourf
from firedrake.pyplot import quiver

import numpy as np

import os


# data_path = "./data/output_seed1/"
data_path = "./output_seed0/"


def plot_mesh(mesh):
    fig, axes = plt.subplots()
    triplot(mesh, axes=axes)
    axes.legend()
    plt.show()


def plot_function(function):
    fig, axes = plt.subplots()
    contours = tricontourf(function, axes=axes, cmap="inferno")
    axes.set_aspect("equal")
    fig.colorbar(contours)
    plt.show()


def plot_approx(function, mesh):
    fig, axes = plt.subplots()

    num_x, num_y = 50, 50
    x = np.arange(0 + 0.5 / num_x, 1.0, 1.0 / num_x)
    y = np.arange(0 + 0.5 / num_y, 1.0, 1.0 / num_y)
    X, Y = np.meshgrid(x, y)
    XY = np.array([X.flatten(), Y.flatten()]).T

    print(XY.shape)

    # print("Mesh dat", mesh.dat)

    mesh_eval = VertexOnlyMesh(mesh, XY)
    p0dg = FunctionSpace(mesh_eval, "DG", 0)

    f = assemble(Interpolate(function, p0dg)).dat.data_ro

    cmesh = plt.pcolormesh(np.reshape(f, (num_x, num_y)), cmap="inferno")
    axes.set_aspect("equal")
    fig.colorbar(cmesh)
    plt.show()


def main():
    filenames = [
        data_path + file for file in os.listdir(data_path) if file.endswith("h5")
    ]
    # Sort the files by creation date so we get temporally sorted datapoints
    filenames.sort(key=os.path.getctime)

    # for file_name in [data_path + "results-90.h5"]:
    for file_name in filenames:
        num = int(file_name[len(data_path) + len("results-") :].split(".")[0])

        if num % 20 != 19:
            continue

        print("Looking at", file_name, ":")

        with CheckpointFile(file_name, "r") as afile:
            mesh = afile.load_mesh()
            rho = afile.load_function(mesh, "rho")
            rhof = afile.load_function(mesh, "rhof")
            up = afile.load_function(mesh, "up")
            t_total = afile.load_function(mesh, "t_total")

            vel, press = up.sub(0), up.sub(1)
            channel_t, substrate_t = t_total.sub(0), t_total.sub(1)

            # function space dim

            # .data

            # plot_mesh(mesh)
            # plot_function(press)
            plot_approx(press, mesh)


if __name__ == "__main__":
    main()
