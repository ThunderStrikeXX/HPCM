import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Reference temperature
# ============================================================
Tref = 298.15  # K

# ============================================================
# Liquid sodium: cp and enthalpy
# ============================================================
a_l = 1658.2
b_l = -0.8478
c_l = 1.484e-3

def cp_liquid(T):
    return a_l + b_l*T + c_l*T**2

def h_liquid(T):
    return (
        a_l*(T - Tref)
        + 0.5*b_l*(T**2 - Tref**2)
        + (c_l/3.0)*(T**3 - Tref**3)
    )

# ============================================================
# Sodium vapor
# ============================================================
R_Na = 361.6          # J/(kg K)
cp_v = 2.5 * R_Na    # constant cp

def delta_h_lv(T):
    return 4.26e6 - 1.15e3*T  # J/kg

def h_vapor(T):
    return h_liquid(T) + delta_h_lv(T)

# ============================================================
# Temperature range
# ============================================================
T = np.linspace(400, 1200, 500)

# ============================================================
# Plot cp
# ============================================================
plt.figure()
plt.plot(T, cp_liquid(T), label="cp liquid Na")
plt.plot(T, np.full_like(T, cp_v), label="cp vapor Na")
plt.xlabel("Temperature [K]")
plt.ylabel("cp [J/kg/K]")
plt.legend()
plt.grid(True)

# ============================================================
# Plot enthalpy
# ============================================================
plt.figure()
plt.plot(T, h_liquid(T)/1e6, label="h liquid Na")
plt.plot(T, h_vapor(T)/1e6, label="h vapor Na")
plt.xlabel("Temperature [K]")
plt.ylabel("Enthalpy [MJ/kg]")
plt.legend()
plt.grid(True)

plt.show()
