import matplotlib.pyplot as plt
import h5py

from firedrake.checkpointing import CheckpointFile
from firedrake import *
from firedrake.pyplot import triplot, tricontourf, quiver, FunctionPlotter, tripcolor

import numpy as np

import os

seed = 99
# data_path = "./data/output_seed1/"
data_path = "./output_seed" + str(seed) + "/"
format_output_path = "./formatted_data" + str(seed) + "/"


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


def plot_vom(function, mesh):

    fig, axes = plt.subplots()

    num_x, num_y = 100, 100
    x = np.arange(0 + 0.5 / num_x, 1.0, 1.0 / num_x)
    y = np.arange(0 + 0.5 / num_y, 1.0, 1.0 / num_y)
    X, Y = np.meshgrid(x, y)
    XY = np.array([X.flatten(), Y.flatten()]).T

    mesh_eval = VertexOnlyMesh(mesh, XY)
    p0dg = FunctionSpace(mesh_eval, "DG", 0)

    f = assemble(Interpolate(function, p0dg)).dat.data_ro

    f_at_points = np.reshape(f, (num_x, num_y))

    cmesh = plt.pcolormesh(f_at_points, cmap="inferno")
    axes.set_aspect("equal")
    fig.colorbar(cmesh)
    plt.show()


def plot_tria(function, mesh):
    fig, axes = plt.subplots()

    function_plotter = FunctionPlotter(mesh, 1)
    triangulation = function_plotter.triangulation
    f_at_points = function_plotter(function)

    tripc = plt.tripcolor(triangulation, f_at_points)

    axes.set_aspect("equal")
    fig.colorbar(tripc)
    plt.show()


def eval_function(function, num_x=100, num_y=100):

    x = np.arange(0 + 0.5 / num_x, 1.0, 1.0 / num_x)
    y = np.arange(0 + 0.5 / num_y, 1.0, 1.0 / num_y)
    X, Y = np.meshgrid(x, y)
    points = np.array([X.flatten(), Y.flatten()]).T

    out = [function.at(p.tolist()) for p in points]

    target_shape = (num_x, num_y) if out[0].shape == () else (num_x, num_y, 2)

    f_at_points = np.reshape(out, target_shape)

    return points, f_at_points


def plot_approx(function):
    fig, axes = plt.subplots()

    _p, f_at_points = eval_function(function)

    cmesh = plt.pcolormesh(f_at_points, cmap="inferno")
    axes.set_aspect("equal")
    fig.colorbar(cmesh)
    plt.show()


def main():
    filenames = [
        data_path + file for file in os.listdir(data_path) if file.endswith("h5")
    ]
    # Sort the files by creation date so we get temporally sorted datapoints
    filenames.sort(key=os.path.getctime)

    for file_name in filenames:
        # num = int(file_name[len(data_path) + len("results-") :].split(".")[0])

        print(file_name)

        with CheckpointFile(file_name, "r") as afile:
            mesh = afile.load_mesh()

            function_names = ["rho", "rhof"]
            composite_functions = {
                "up": {"vel": 0, "press": 1},
                "t_total": {"channel_t": 0, "substrate_t": 1},
            }

            def plot(out, title=""):
                fig, axes = plt.subplots()
                cmesh = plt.pcolormesh(out, cmap="inferno")
                axes.set_aspect("equal")
                fig.colorbar(cmesh)

                plt.title(title)
                # plt.show()

            for f in function_names:
                func = afile.load_function(mesh, f)
                x, y = eval_function(func)
                filename = (
                    format_output_path + file_name[len(data_path) :] + "_" + f + ".out"
                )
                print("Saving", filename)
                np.savetxt(filename, y)

                plot(y, title=f)
                plt.savefig(
                    format_output_path + file_name[len(data_path) :] + "_" + f + ".png"
                )

            for cf_name, cf in composite_functions.items():
                for sf_name, subfunc in cf.items():
                    func = afile.load_function(mesh, cf_name).sub(subfunc)
                    x, y = eval_function(func)

                    # Introduce dim to handle vector valued fields like velocity
                    dim_quant = 1 if len(y.shape) == 2 else y.shape[-1]
                    for dim in range(dim_quant):

                        filename = (
                            format_output_path
                            + file_name[len(data_path) :]
                            + "_"
                            + sf_name
                            + "_"
                            + str(dim)
                            + ".out"
                        )

                        arr = y if dim_quant == 1 else y[:, :, dim]
                        np.savetxt(filename, arr)

                        plot(
                            arr,
                            title=sf_name + str(dim),
                        )
                        plt.savefig(
                            format_output_path
                            + file_name[len(data_path) :]
                            + "_"
                            + sf_name
                            + "_"
                            + str(dim)
                            + ".png"
                        )


if __name__ == "__main__":
    main()
