import numpy as np
import matplotlib.pyplot as plt

# Saturation pressure function [Pa]
def P_sat(T):
    val_MPa = np.exp(11.9463 - 12633.7 / T - 0.4672 * np.log(T))
    return val_MPa * 1e6

# Temperature range [K]
T = np.linspace(500.0, 1200.0, 500)

# Compute p_sat / T
y = P_sat(T) / T

# Plot
plt.figure()
plt.plot(T, y)
plt.xlabel("Temperature T [K]")
plt.ylabel("p_sat / T [Pa/K]")
plt.grid(True)
plt.tight_layout()
plt.show()
