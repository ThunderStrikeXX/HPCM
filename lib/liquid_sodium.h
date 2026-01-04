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
    constexpr double Tcrit = 2509.46;

    /// Solidification temperature [K]
    constexpr double Tsolid = 370.87;

    /**
    * @brief Density [kg/m3] as a function of temperature T
    *   Keenan–Keyes / Vargaftik
    */
    inline double rho(double T) {

        if (T < Tsolid && warnings == true) std::cout << "Warning, temperature " << T << " is below solidification temperature (" << Tsolid << ")!";
        return 219.0 + 275.32 * (1.0 - T / Tcrit) + 511.58 * pow(1.0 - T / Tcrit, 0.5);
    }

    /**
    * @brief Thermal conductivity [W/(m*K)] as a function of temperature T
    *   Vargaftik
    */
    inline double k(double T) {

        if (T < Tsolid && warnings == true) std::cout << "Warning, temperature " << T << " is below solidification temperature!";
        return 124.67 - 0.11381 * T + 5.5226e-5 * T * T - 1.1842e-8 * T * T * T;
    }

    /**
    * @brief Specific heat at constant pressure [J/(kg·K)] as a function of temperature
    *   Vargaftik / Fink & Leibowitz
    */
    inline double cp(double T) {

        if (T < Tsolid && warnings == true) std::cout << "Warning, temperature " << T << " is below solidification temperature!";
        double dXT = T - 273.15;
        return 1436.72 - 0.58 * dXT + 4.627e-4 * dXT * dXT;
    }

    /**
    * @brief Dynamic viscosity [Pa·s] using Shpilrain et al. correlation, valid for 371 K < T < 2500 K
    *   Shpilrain et al
    */
    inline double mu(double T) {

        if (T < Tsolid && warnings == true) std::cout << "Warning, temperature " << T << " is below solidification temperature!";
        return std::exp(-6.4406 - 0.3958 * std::log(T) + 556.835 / T);
    }

    /**
      * @brief Liquid sodium enthalpy [J/kg]
      *     NIST Shomate coefficients for Na(l), 370.98–1170.525 K
      *
      *     Cp° = A + B*t + C*t^2 + D*t^3 + E/t^2  [J/mol/K]
      *     H° - H°298.15 = A*t + B*t^2/2 + C*t^3/3 + D*t^4/4 - E/t + F - H  [kJ/mol]
      *     with t = T/1000
      */
    inline double h(double T) {

        const double T_min = 370.98, T_max = 1170.525;
        if (T < T_min) T = T_min;
        if (T > T_max) T = T_max;

        const double A = 40.25707;
        const double B = -28.23849;
        const double C = 20.69402;
        const double D = -3.641872;
        const double E = -0.079874;
        const double F = -8.782300;
        const double H = 2.406001; // NIST “H” coeff (not temperature)

        const double t = T / 1000.0;

        // kJ/mol relative to 298.15 K
        const double H_kJ_per_mol =
            A * t + B * t * t / 2.0 + C * t * t * t / 3.0 + D * t * t * t * t / 4.0 - E / t + F - H;

        // Convert to J/kg
        const double M_kg_per_mol = 22.98976928e-3; // Molar mass Na
        return (H_kJ_per_mol * 1000.0) / M_kg_per_mol;
    }
}