import numpy as np

N = 7540
x_min = 0.0
x_max = 741_745_090 * 5e-8

values = np.linspace(x_min, x_max, N)

with open("output.txt", "w") as f:
    for v in values:
        f.write(f"{v}\n")