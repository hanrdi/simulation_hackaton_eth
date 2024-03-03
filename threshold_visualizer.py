import numpy as np
import matplotlib.pyplot as plt


example_file = "/home/steinraf/Coding/simulation_hackaton_eth/formatted_data99/results-73.h5_rhof.out"

data = np.loadtxt(example_file)


num_x, num_y = 4, 4
t = np.linspace(1.0 / (num_x * num_y), 1.0 - 1.0 / (num_x * num_y), num_x * num_y)
fig, axs = plt.subplots(num_x, num_y, layout=None)


def thresh(d, t):
    d[d > t] = 1.0
    d[d < 1.0] = 0.0


def binary_search(data, target_fract, tol=0.001):
    min = 0.0
    max = 1.0
    s = data.shape[0] * data.shape[1]
    print("Size", s)
    c = 0
    while min < max:
        d = np.copy(data)
        t = (max + min) / 2
        thresh(d, t)
        fraction = np.sum(d) / s
        if np.abs(target_fract - fraction) < tol or c > 100:
            return t

        if fraction < target_fract:
            min = t
        else:
            max = t
        print(f"{target_fract}:{t} ->({min}/{max}) -> {fraction}")
        c += 1


for idx, ax in enumerate(axs.flat):

    d = np.copy(data)

    threshold = binary_search(data, t[idx])
    thresh(d, threshold)

    ax.set_title(threshold)
    ax.pcolormesh(threshold)

# for x in range(num_x):
#     for y in range(num_y):
#         axes[x * num_y + x].pcolormesh(data)

plt.show()
