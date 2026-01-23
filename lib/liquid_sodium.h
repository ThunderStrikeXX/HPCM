#pragma once

using data_type = float;

/**
 * @brief Provides thermophysical properties for Liquid Sodium (Na).
 *
 * This namespace contains constant data and functions to calculate key
 * temperature-dependent properties of liquid sodium.
 * All functions accept temperature T in Kelvin [K] and return values
 * in standard SI units unless otherwise specified.
 * The function give warnings if the input temperature is below the
 * (constant) solidification temperature.
 */
namespace liquid_sodium {

    /// Critical temperature [K]
    constexpr data_type Tcrit = 2509.46;

    /// Solidification temperature [K]
    constexpr data_type Tsolid = 370.87;

    /**
    * @brief Density [kg/m3] as a function of temperature T
    *   Keenan–Keyes / Vargaftik
    */
    inline data_type rho(data_type T) {

        return 219.0 + 275.32 * (1.0 - T / Tcrit) + 511.58 * pow(1.0 - T / Tcrit, 0.5);
    }

    /**
    * @brief Thermal conductivity [W/(m*K)] as a function of temperature T
    *   Vargaftik
    */
    inline data_type k(data_type T) {

        return 124.67 - 0.11381 * T + 5.5226e-5 * T * T - 1.1842e-8 * T * T * T;
    }

    /**
    * @brief Specific heat at constant pressure [J/(kg·K)] as a function of temperature
    *   Vargaftik / Fink & Leibowitz
    */
    inline data_type cp(data_type T) {

        data_type dXT = T - 273.15;
        return 1436.72 - 0.58 * dXT + 4.627e-4 * dXT * dXT;
    }

    /**
    * @brief Dynamic viscosity [Pa·s] using Shpilrain et al. correlation, valid for 371 K < T < 2500 K
    *   Shpilrain et al
    */
    inline data_type mu(data_type T) {

        return std::exp(-6.4406 - 0.3958 * std::log(T) + 556.835 / T);
    }

    /// Enthalpy of liquid sodium [J/kg] (CODATA correlation)
    inline data_type h(data_type T) {
        // Numerical safety only
        if (T < 300.0)  T = 300.0;
        if (T > 2500.0) T = 2500.0;

        return (
            -365.77
            + 1.6582e0 * T
            - 4.2395e-4 * T * T
            + 1.4847e-7 * T * T * T
            + 2992.6 / T
            ) * 1e3;   // J/kg
    }
}