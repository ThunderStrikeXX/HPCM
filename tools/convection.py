import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Correlations (Python equivalents of C++ code)
# -------------------------------------------------

def friction_factor(Re):
    Re = np.asarray(Re)
    if np.any(Re <= 0.0):
        raise ValueError("Re <= 0 in friction_factor")
    return (0.79 * np.log(Re) - 1.64) ** -2.0


def Nusselt(Re, Pr):
    Re = np.asarray(Re)
    if np.any(Re <= 0.0) or Pr <= 0.0:
        raise ValueError("Re or Pr <= 0 in Nusselt")

    Nu_lam = 4.36
    Re_lam = 2300.0
    Re_turb = 4000.0

    def Nu_turb(Re_loc):
        f = friction_factor(Re_loc)
        num = (f / 8.0) * (Re_loc - 1000.0) * Pr
        den = 1.0 + 12.7 * np.sqrt(f / 8.0) * (Pr ** (2.0 / 3.0) - 1.0)
        return num / den

    Nu = np.zeros_like(Re)

    mask_lam = Re <= Re_lam
    mask_turb = Re >= Re_turb
    mask_trans = (~mask_lam) & (~mask_turb)

    Nu[mask_lam] = Nu_lam
    Nu[mask_turb] = Nu_turb(Re[mask_turb])

    chi = (np.log(Re[mask_trans]) - np.log(Re_lam)) / \
          (np.log(Re_turb) - np.log(Re_lam))
    Nu[mask_trans] = (1.0 - chi) * Nu_lam + chi * Nu_turb(Re[mask_trans])

    return Nu


def h_conv(Re, Pr, k, Dh):
    if k <= 0.0 or Dh <= 0.0:
        raise ValueError("k or Dh <= 0 in h_conv")
    return Nusselt(Re, Pr) * k / Dh


# -------------------------------------------------
# Parameters (example values for sodium vapor)
# -------------------------------------------------

Pr = 0.03          # typical order of magnitude for Na vapor
k = 0.06           # W/m/K (example)
Dh = 5e-3          # m

Re = np.logspace(2, 6, 500)

# -------------------------------------------------
# Compute quantities
# -------------------------------------------------

f = friction_factor(Re)
Nu = Nusselt(Re, Pr)
h = h_conv(Re, Pr, k, Dh)

# -------------------------------------------------
# Plots
# -------------------------------------------------

plt.figure()
plt.loglog(Re, f)
plt.xlabel("Reynolds number [-]")
plt.ylabel("Darcy friction factor [-]")
plt.grid(True, which="both")
plt.tight_layout()

plt.figure()
plt.loglog(Re, Nu)
plt.xlabel("Reynolds number [-]")
plt.ylabel("Nusselt number [-]")
plt.grid(True, which="both")
plt.tight_layout()

plt.figure()
plt.loglog(Re, h)
plt.xlabel("Reynolds number [-]")
plt.ylabel("Convective heat transfer coefficient h [W/m$^2$/K]")
plt.grid(True, which="both")
plt.tight_layout()

plt.show()
