import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Sodium correlations (from C++)
# -------------------------------
def h_liquid_sodium(T):
    T = np.clip(T, 300.0, 2500.0)
    return (
        -365.77
        + 1.6582e0 * T
        - 4.2395e-4 * T**2
        + 1.4847e-7 * T**3
        + 2992.6 / T
    ) * 1e3

def h_vap_sodium(T):
    T = np.clip(T, 300.0, 2500.0)
    Tcrit = 2503.0
    theta = 1.0 - T / Tcrit
    return (393.37 * theta + 4398.6 * theta**0.29302) * 1e3

def h_vapor_sodium(T):
    return h_liquid_sodium(T) + h_vap_sodium(T)

# -------------------------------
# Fit interval
# -------------------------------
T_fit = np.linspace(800, 1200, 300)

bg, ag = np.polyfit(T_fit, h_vapor_sodium(T_fit), 1)
bl, al = np.polyfit(T_fit, h_liquid_sodium(T_fit), 1)

# -------------------------------
# Print coefficients
# -------------------------------
print("=== Linear coefficients (fit 800–1200 K) ===")
print(f"h_g(T) = {ag:.6e} + {bg:.6e} * T   [J/kg]")
print(f"h_l(T) = {al:.6e} + {bl:.6e} * T   [J/kg]")

# -------------------------------
# Reference and linearized curves
# -------------------------------
h_g_corr = h_vapor_sodium(T_fit)
h_l_corr = h_liquid_sodium(T_fit)

h_g_lin = ag + bg * T_fit
h_l_lin = al + bl * T_fit

# -------------------------------
# Percentage errors
# -------------------------------
err_g_pct = 100.0 * (h_g_lin - h_g_corr) / h_g_corr
err_l_pct = 100.0 * (h_l_lin - h_l_corr) / h_l_corr

# -------------------------------
# Plot 1: Vapor enthalpy
# -------------------------------
plt.figure()
plt.plot(T_fit, h_g_corr, label="h_g correlation")
plt.plot(T_fit, h_g_lin, "--", label="h_g linear")
plt.xlabel("Temperature [K]")
plt.ylabel("Enthalpy [J/kg]")
plt.legend()
plt.grid(True)

# -------------------------------
# Plot 2: Liquid enthalpy
# -------------------------------
plt.figure()
plt.plot(T_fit, h_l_corr, label="h_l correlation")
plt.plot(T_fit, h_l_lin, "--", label="h_l linear")
plt.xlabel("Temperature [K]")
plt.ylabel("Enthalpy [J/kg]")
plt.legend()
plt.grid(True)

# -------------------------------
# Plot 3: Vapor relative error
# -------------------------------
plt.figure()
plt.plot(T_fit, err_g_pct, label="h_g relative error [%]")
plt.axhline(0.0)
plt.xlabel("Temperature [K]")
plt.ylabel("Relative error [%]")
plt.legend()
plt.grid(True)

# -------------------------------
# Plot 4: Liquid relative error
# -------------------------------
plt.figure()
plt.plot(T_fit, err_l_pct, label="h_l relative error [%]")
plt.axhline(0.0)
plt.xlabel("Temperature [K]")
plt.ylabel("Relative error [%]")
plt.legend()
plt.grid(True)

plt.show()
