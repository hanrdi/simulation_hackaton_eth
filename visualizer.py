import matplotlib.pyplot as plt
import h5py

from firedrake.checkpointing import CheckpointFile
from firedrake import *
from firedrake.pyplot import triplot
from firedrake.pyplot import tricontourf
from firedrake.pyplot import quiver

import numpy as np

import os


data_path = "./data/output_seed1/"


def plot_mesh(mesh):
    fig, axes = plt.subplots()
    triplot(mesh, axes=axes)
    axes.legend()
    plt.show()


def plot_function(function, mesh):
    # x, y = SpatialCoordinate(mesh)
    # function.interpolate(x + y)
    fig, axes = plt.subplots()
    # levels = np.linspace(0, 1, 51)
    contours = tricontourf(function, axes=axes, cmap="inferno")
    axes.set_aspect("equal")
    fig.colorbar(contours)
    plt.show()


def main():
    filenames = [
        data_path + file for file in os.listdir(data_path) if file.endswith("h5")
    ]
    # Sort the files by creation date so we get temporally sorted datapoints
    filenames.sort(key=os.path.getctime)

    # for file_name in [data_path + "results-90.h5"]:
    for file_name in filenames:

        print("Looking at", file_name, ":")

        with CheckpointFile(file_name, "r") as afile:
            mesh = afile.load_mesh()
            rho = afile.load_function(mesh, "rho")
            rhof = afile.load_function(mesh, "rhof")
            up = afile.load_function(mesh, "up")

            print(" Num subfunctions", len(up.subfunctions))
            vel, press = up.sub(0), up.sub(1)

            # plot_mesh(mesh)
            plot_function(press, mesh)


if __name__ == "__main__":
    main()
