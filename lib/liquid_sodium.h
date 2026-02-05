#pragma once

#include "numeric_types.h"

namespace liquid_sodium {

    // Critical temperature [K]
    constexpr data_type Tcrit = 2509.46;

    // Solidification temperature [K]
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
    * @brief Dynamic viscosity [Pa·s] using Shpilrain et al. correlation, valid for 371 K < T < 2500 K
    *   Shpilrain et al
    */
    inline data_type mu(data_type T) {

        return std::exp(-6.4406 - 0.3958 * std::log(T) + 556.835 / T);
    }

    // From h_l(T) = al + bl * T
    inline data_type cp_l_linear() {

        return 1.256230e3;   // J/(kg·K)
    }

    inline data_type h_l_linear(data_type T) {

        constexpr data_type al = -2.359582e5;   // J/kg
        constexpr data_type bl = 1.256230e3;    // J/(kg·K)

        return al + bl * T;                     // J/kg
    }

    inline data_type T_from_h_l_linear(data_type h) {

        constexpr data_type al = -2.359582e5;   // J/kg
        constexpr data_type bl = 1.256230e3;    // J/(kg·K)

        return (h - al) / bl;                   // K
    }
}