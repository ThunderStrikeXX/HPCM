import numpy as np
import matplotlib.pyplot as plt

def f(Re):
    t = 0.79 * np.log(Re) - 1.64
    return 1.0 / (t*t)

def Nu(Re, Pr):
    Nu_lam = 4.36
    if Re <= 1000.0:
        return Nu_lam
    ff = f(Re)
    fp8 = ff / 8.0
    num = fp8 * (Re - 1000.0) * Pr
    den = 1.0 + 12.7 * np.sqrt(fp8) * (Pr**(2/3) - 1.0)
    return Nu_lam + num/den

Pr = 1.0
Re_vals = np.logspace(2, 5, 400)
Nu_vals = np.array([Nu(Re, Pr) for Re in Re_vals])

plt.figure()
plt.loglog(Re_vals, Nu_vals)
plt.xlabel("Re")
plt.ylabel("Nu")
plt.grid(True, which="both")
plt.show()
