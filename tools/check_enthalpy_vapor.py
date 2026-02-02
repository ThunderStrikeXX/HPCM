import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Data from C++ implementation
# -----------------------------
Tgrid = np.array([
    400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,
    1600,1700,1800,1900,2000,2100,2200,2300,2400
], dtype=float)

Cpgrid = np.array([
    860,1250,1800,2280,2590,2720,2700,2620,2510,2430,2390,2360,
    2340,2410,2460,2530,2660,2910,3400,4470,8030
], dtype=float)

Tcrit = 2503.0

# -----------------------------
# Cp formulations
# -----------------------------

# (1) Tabulated / interpolated Cp (original)
def cp_interp(T):
    T = np.asarray(T)
    out = np.interp(T, Tgrid, Cpgrid)
    out[T >= 2500.0] = 417030.0
    return out  # [J/kg/K]

# (2) Polynomial Cp correlation
def cp_poly(T):
    T = np.asarray(T)
    return (
        1.658e3
        - 4.239e-1 * T
        + 4.452e-4 * T**2
        - 1.484e-7 * T**3
    )  # [J/kg/K]

# -----------------------------
# Enthalpy correlations
# -----------------------------
def h_liquid(T):
    T = np.clip(T, 300.0, 2500.0)
    return (
        -365.77
        + 1.6582e0 * T
        - 4.2395e-4 * T**2
        + 1.4847e-7 * T**3
        + 2992.6 / T
    ) * 1e3

def h_vap(T):
    T = np.clip(T, 300.0, 2500.0)
    theta = 1.0 - T / Tcrit
    return (393.37 * theta + 4398.6 * theta**0.29302) * 1e3

def h_vapor(T):
    return h_liquid(T) + h_vap(T)

# -----------------------------
# Cp integration
# -----------------------------
Tref = 400.0
href = h_vapor(Tref)

T = np.linspace(400.0, 2400.0, 2000)

def integrate_cp(cp_fun):
    cp_vals = cp_fun(T)
    h_int = np.zeros_like(T)
    for i in range(1, len(T)):
        h_int[i] = h_int[i-1] + 0.5 * (cp_vals[i] + cp_vals[i-1]) * (T[i] - T[i-1])
    return href + h_int

h_cp_interp = integrate_cp(cp_interp)
h_cp_poly   = integrate_cp(cp_poly)

# -----------------------------
# Plot
# -----------------------------
plt.figure()
plt.plot(T, h_vapor(T), label="h_vapore (CODATA + Δh_vap)")
plt.plot(T, h_cp_interp, "--", label="∫cp_interp dT + h_ref")
plt.plot(T, h_cp_poly, ":", label="∫cp_poly dT + h_ref")
plt.xlabel("T [K]")
plt.ylabel("h [J/kg]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
