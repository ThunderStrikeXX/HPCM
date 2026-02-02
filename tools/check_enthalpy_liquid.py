import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Functions (as in C++)
# -----------------------------
def cp(T):
    dXT = T - 273.15
    return 1436.72 - 0.58 * dXT + 4.627e-4 * dXT**2   # J/(kg K)

def h_liquid(T):
    T = np.clip(T, 300.0, 2500.0)
    return (
        -365.77
        + 1.6582e0 * T
        - 4.2395e-4 * T**2
        + 1.4847e-7 * T**3
        + 2992.6 / T
    ) * 1e3   # J/kg

# -----------------------------
# Temperature range
# -----------------------------
T = np.linspace(300.0, 2400.0, 2000)

# -----------------------------
# Integral of cp
# -----------------------------
Tref = 300.0
href = h_liquid(Tref)

cp_vals = cp(T)
h_int = np.zeros_like(T)

for i in range(1, len(T)):
    h_int[i] = h_int[i-1] + 0.5 * (cp_vals[i] + cp_vals[i-1]) * (T[i] - T[i-1])

h_cp = href + h_int

# -----------------------------
# Plot
# -----------------------------
plt.figure()
plt.plot(T, h_liquid(T), label="h_liquido (CODATA)")
plt.plot(T, h_cp, "--", label="∫cp dT + h_ref")
plt.xlabel("T [K]")
plt.ylabel("h [J/kg]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
