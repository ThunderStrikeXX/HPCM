import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Correlazioni sodio (dal C++)
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
# Intervallo e fit
# -------------------------------
T_fit = np.linspace(800, 1200, 300)

bg, ag = np.polyfit(T_fit, h_vapor_sodium(T_fit), 1)
bl, al = np.polyfit(T_fit, h_liquid_sodium(T_fit), 1)

# -------------------------------
# Stampa coefficienti
# -------------------------------
print("=== Coefficienti lineari (fit 800–1200 K) ===")
print(f"h_g(T) = {ag:.6e} + {bg:.6e} * T   [J/kg]")
print(f"h_l(T) = {al:.6e} + {bl:.6e} * T   [J/kg]")

# -------------------------------
# Rette
# -------------------------------
h_g_lin = ag + bg * T_fit
h_l_lin = al + bl * T_fit

# -------------------------------
# Plot
# -------------------------------
plt.figure()
plt.plot(T_fit, h_vapor_sodium(T_fit), label="h_g correlazione")
plt.plot(T_fit, h_g_lin, "--", label="h_g lineare")
plt.plot(T_fit, h_liquid_sodium(T_fit), label="h_l correlazione")
plt.plot(T_fit, h_l_lin, "--", label="h_l lineare")
plt.xlabel("Temperature [K]")
plt.ylabel("Enthalpy [J/kg]")
plt.legend()
plt.grid(True)
plt.show()
