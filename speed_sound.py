import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect_left

# -----------------------------
# Linear interpolation function
# -----------------------------
def interp_T(Tgrid, Ygrid, T):
    i = bisect_left(Tgrid, T)
    if i == 0:
        return Ygrid[0]
    if i >= len(Tgrid):
        return Ygrid[-1]
    T0, T1 = Tgrid[i-1], Tgrid[i]
    Y0, Y1 = Ygrid[i-1], Ygrid[i]
    return Y0 + (Y1 - Y0) * (T - T0) / (T1 - T0)

# -----------------------------
# cp(T) and cv(T)
# -----------------------------
Tgrid_cp = np.array([400,500,600,700,800,900,1000,1100,1200,1300,
                     1400,1500,1600,1700,1800,1900,2000,2100,2200,
                     2300,2400,2500])

Cpgrid = np.array([860,1250,1800,2280,2590,2720,2700,2620,2510,2430,
                   2390,2360,2340,2410,2460,2530,2660,2910,3400,
                   4470,8030,417030])

Tgrid_cv = np.array(Tgrid_cp)
Cvgrid = np.array([490,840,1310,1710,1930,1980,1920,1810,1680,1580,
                   1510,1440,1390,1380,1360,1300,1300,1300,1340,
                   1760,17030])

def cp(T):
    if T >= 2500: return 417030.0
    return interp_T(Tgrid_cp, Cpgrid, T)

def cv(T):
    if T >= 2500: return 17030.0
    return interp_T(Tgrid_cv, Cvgrid, T)

def gamma(T):
    return cp(T) / cv(T)

# -----------------------------
# Gas constant for sodium vapor
# -----------------------------
R = 8.314 / 0.023  # J/(kg K)  (M = 23 g/mol)

# -----------------------------
# Speed of sound: a = sqrt(gamma R T)
# -----------------------------
def sound_speed(T):
    return np.sqrt(gamma(T) * R * T)

# -----------------------------
# Generate curve
# -----------------------------
T = np.linspace(400, 2400, 500)
a = np.array([sound_speed(t) for t in T])

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(7,5))
plt.plot(T, a)
plt.xlabel("Temperature [K]")
plt.ylabel("Speed of sound [m/s]")
plt.title("Sodium vapor: speed of sound vs temperature")
plt.grid(True)
plt.tight_layout()
plt.show()
