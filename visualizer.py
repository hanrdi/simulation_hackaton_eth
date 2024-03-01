import matplotlib.pyplot as plt
import h5py

import os


data_path = "./data/output_seed1/"


def main():
    filenames = [file for file in os.listdir(data_path) if file.endswith("h5")]

    for file_name in filenames:
        print("Looking at", file_name, ":")

        file = h5py.File(data_path + file_name, "r")

        def printname(name, object):
            print("name:", name)
            print("object", object)

        file.visititems(printname)

        # print(file["topologies"].keys())

        break


if __name__ == "__main__":
    main()
