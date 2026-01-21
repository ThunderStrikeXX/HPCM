import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
k  = 0.01      # [W/m/K]
Dh = 0.01      # [m]
Pr = 0.01      # [-] (tipico sodio vapore, modifica se necessario)

Nu_lam = 4.36
Re1 = 2000.0
Re2 = 3000.0

# -----------------------------
# Friction factor (Gnielinski)
# Blasius (turbulent smooth pipe)
# -----------------------------
def f_blasius(Re):
    return 0.3164 * Re**(-0.25)

# -----------------------------
# Gnielinski Nusselt
# -----------------------------
def Nu_gnielinski(Re):
    f = f_blasius(Re)
    fp8 = f / 8.0
    num = fp8 * (Re - 1000.0) * Pr
    den = 1.0 + 12.7 * np.sqrt(fp8) * (Pr**(2.0/3.0) - 1.0)
    return num / den

# -----------------------------
# Convective coefficient
# -----------------------------
def h_conv(Re):
    Re = np.asarray(Re)
    Nu = np.zeros_like(Re)

    lam = Re <= Re1
    turb = Re >= Re2
    blend = (~lam) & (~turb)

    Nu[lam] = Nu_lam
    Nu[turb] = Nu_gnielinski(Re[turb])

    chi = (Re[blend] - Re1) / (Re2 - Re1)
    Nu[blend] = (1.0 - chi) * Nu_lam + chi * Nu_gnielinski(Re[blend])

    return Nu * k / Dh

# -----------------------------
# Plot
# -----------------------------
Re = np.logspace(2, 5, 1000)
h  = h_conv(Re)

plt.figure()
plt.semilogx(Re, h)
plt.axvline(Re1, linestyle='--')
plt.axvline(Re2, linestyle='--')
plt.xlabel("Re")
plt.ylabel("h [W/m²/K]")
plt.title("Convective heat transfer coefficient")
plt.grid(True)
plt.show()
