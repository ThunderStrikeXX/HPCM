#pragma once
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <omp.h>
#include <array>
#include <algorithm>
#include <cstddef>
#include <filesystem>

bool warnings = false;

// =======================================================================
//
//                       [MATERIAL PROPERTIES]
//
// =======================================================================

#pragma region steel_properties

/**
 * @brief Provides material properties for steel.
 *
 * This namespace contains lookup tables and helper functions to retrieve
 * temperature-dependent thermodynamic properties of steel, specifically:
 * - Specific Heat Capacity (Cp)
 * - Density (rho)
 * - Thermal Conductivity (k)
 *
 * All functions accept temperature in Kelvin [K] and return values in
 * standard SI units unless otherwise specified.
 */
namespace steel {

    /// Temperature values of the Cp table [K]
    constexpr std::array<double, 15> T = { 300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700 };

    /// Specific heat values of the Cp table [J kg^-1 K^-1]
    constexpr std::array<double, 15> Cp_J_kgK = { 510.0296,523.4184,536.8072,550.1960,564.0032,577.3920,590.7808,604.1696,617.5584,631.3656,644.7544,658.1432,671.5320,685.3392,698.7280 };

    /**
    * @brief Specific heat interpolation in temperature with complexity O(1)
    * @param Tquery Temperature for which the Cp is needed.
    */
    inline double cp(double Tquery) {

        if (Tquery <= T.front()) return Cp_J_kgK.front();
        if (Tquery >= T.back())  return Cp_J_kgK.back();

        int i = static_cast<int>((Tquery - 300.0) / 100.0);

        if (i < 0) i = 0;

        int iMax = static_cast<int>(T.size()) - 2;

        if (i > iMax) i = iMax;

        double x0 = 300.0 + 100.0 * i, x1 = x0 + 100.0;
        double y0 = Cp_J_kgK[static_cast<std::size_t>(i)];
        double y1 = Cp_J_kgK[static_cast<std::size_t>(i + 1)];
        double t = (Tquery - x0) / (x1 - x0);

        return y0 + t * (y1 - y0);
    }

    /**
    * @brief Density [kg/m3] as a function of temperature T 
    */
    inline double rho(double T) { return (7.9841 - 2.6560e-4 * T - 1.158e-7 * T * T) * 1e3; }

    /**
    * @brief Thermal conductivity [W/(m*K)] as a function of temperature T 
    */
    inline double k(double T) { return (3.116e-2 + 1.618e-4 * T) * 100.0; }
}

#pragma endregion

#pragma region liquid_sodium_properties

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
    */
    inline double rho(double T) {

        if (T < Tsolid && warnings == true) std::cout << "Warning, temperature " << T << " is below solidification temperature (" << Tsolid << ")!";
        return 219.0 + 275.32 * (1.0 - T / Tcrit) + 511.58 * pow(1.0 - T / Tcrit, 0.5);
    }

    /**
    * @brief Thermal conductivity [W/(m*K)] as a function of temperature T
    */
    inline double k(double T) {

        if (T < Tsolid && warnings == true) std::cout << "Warning, temperature " << T << " is below solidification temperature!";
        return 124.67 - 0.11381 * T + 5.5226e-5 * T * T - 1.1842e-8 * T * T * T;
    }

    /**
    * @brief Specific heat at constant pressure [J/(kg·K)] as a function of temperature 
    */
    inline double cp(double T) {

        if (T < Tsolid && warnings == true) std::cout << "Warning, temperature " << T << " is below solidification temperature!";
        double dXT = T - 273.15;
        return 1436.72 - 0.58 * dXT + 4.627e-4 * dXT * dXT;
    }

    /**
    * @brief Dynamic viscosity [Pa·s] using Shpilrain et al. correlation, valid for 371 K < T < 2500 K
    */
    inline double mu(double T) {

        if (T < Tsolid && warnings == true) std::cout << "Warning, temperature " << T << " is below solidification temperature!";
        return std::exp(-6.4406 - 0.3958 * std::log(T) + 556.835 / T);
    }

    /**
      * @brief Liquid sodium enthalpy [J/kg]NIST Shomate coefficients for Na(l), 370.98–1170.525 K
      * Cp° = A + B*t + C*t^2 + D*t^3 + E/t^2  [J/mol/K]
      * H° - H°298.15 = A*t + B*t^2/2 + C*t^3/3 + D*t^4/4 - E/t + F - H  [kJ/mol]
      * with t = T/1000
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

#pragma endregion

#pragma region vapor_sodium_properties

/**
 * @brief Provides thermophysical and transport properties for Sodium Vapor.
 *
 * This namespace contains constant data and functions to calculate key properties
 * of sodium vapor.
 * All functions primarily accept temperature T in Kelvin [K] and return values
 * in standard SI units unless otherwise noted.
 */
namespace vapor_sodium {

    constexpr double Tcrit_Na = 2509.46;           ///< Critical temperature [K]
    constexpr double Ad_Na = 3.46;                 ///< Adiabatic factor [-]
    constexpr double m_g_Na = 23e-3;               ///< Molar mass [kg/mol]

    /**
    * @brief Functions that clamps a value x to the range [a, b]
    */
    constexpr inline double clamp(double x, double a, double b) { return std::max(a, std::min(x, b)); }

    /**
    * @brief 1D table interpolation in T over monotone grid
    */
    template<size_t N>
    double interp_T(const std::array<double, N>& Tgrid, const std::array<double, N>& Ygrid, double T) {

        if (T <= Tgrid.front()) return Ygrid.front();
        if (T >= Tgrid.back())  return Ygrid.back();

        // locate interval
        size_t i = 0;
        while (i + 1 < N && !(Tgrid[i] <= T && T <= Tgrid[i + 1])) ++i;

        return Ygrid[i] + (T - Tgrid[i]) / (Tgrid[i + 1] - Tgrid[i]) * (Ygrid[i + 1] - Ygrid[i]);
    }

    /**
      * @brief Enthalpy of sodium vapor [J/kg] from NIST Shomate equation.
      * Valid for 1170.525 K ≤ T ≤ 6000 K.
      * Reference state: H(298.15 K) = 0 (per NIST convention).
      *
      * @param T Temperature [K]
      * @return Enthalpy of sodium vapor [J/kg]
      */
    inline double h(double T) {
        constexpr double T_min = 1170.525;
        constexpr double T_max = 6000.0;
        if (T < T_min) T = T_min;
        if (T > T_max) T = T_max;

        const double A = 20.80573;
        const double B = 0.277206;
        const double C = -0.392086;
        const double D = 0.119634;
        const double E = -0.008879;
        const double F = 101.0386;
        const double H = 107.2999;

        double t = T / 1000.0;

        double H_kJ_per_mol = A * t
            + B * t * t / 2.0
            + C * t * t * t / 3.0
            + D * t * t * t * t / 4.0
            - E / t
            + F
            - H;

        const double M_kg_per_mol = 22.98976928e-3;
        return (H_kJ_per_mol * 1000.0) / M_kg_per_mol; // J/kg
    }

    /**
    * @brief Saturation pressure [Pa] as a function of temperature T
    */
    inline double P_sat(double T) {

        const double val_MPa = std::exp(11.9463 - 12633.7 / T - 0.4672 * std::log(T));
        return val_MPa * 1e6;
    }

    /**
    * @brief Derivative of saturation pressure with respect to temperature [Pa/K] as a function of temperature T
    */
    inline double dP_sat_dVT(double T) {

        const double val_MPa_per_K =
            (12633.73 / (T * T) - 0.4672 / T) * std::exp(11.9463 - 12633.73 / T - 0.4672 * std::log(T));
        return val_MPa_per_K * 1e6;
    }

    /**
    * @brief Density of saturated vapor [kg/m^3] as a function of temperature T
    */
    inline double rho(double T) {

        const double hv = vapor_sodium::h(T) - liquid_sodium::h(T);     // [J/kg]
        const double dPdVT = dP_sat_dVT(T);                             // [Pa/K]
        const double rhol = liquid_sodium::rho(T);                      // [kg/m^3]
        const double denom = hv / (T * dPdVT) + 1.0 / rhol;
        return 1.0 / denom;                                             // [kg/m^3]
    }

    /**
    * @brief Specific heat at constant pressure from table interpolation [J/(kg*K)] as a function of temperature T
    */
    inline double cp(double T) {

        static const std::array<double, 21> Tgrid = { 400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400 };
        static const std::array<double, 21> Cpgrid = { 860,1250,1800,2280,2590,2720,2700,2620,2510,2430,2390,2360,2340,2410,2460,2530,2660,2910,3400,4470,8030 };

        // Table also lists 2500 K = 417030; extreme near critical. If needed, extend:
        if (T >= 2500.0) return 417030.0;

        return interp_T(Tgrid, Cpgrid, T);
    }

    /**
    * @brief Specific heat at constant volume from table interpolation [J/(kg*K)] as a function of temperature T
    */
    inline double cv(double T) {

        static const std::array<double, 21> Tgrid = { 400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400 };
        static const std::array<double, 21> Cvgrid = { 490, 840,1310,1710,1930,1980,1920,1810,1680,1580,1510,1440,1390,1380,1360,1330,1300,1300,1340,1440,1760 };

        // Table also lists 2500 K = 17030; extreme near critical. If needed, extend:
        if (T >= 2500.0) return 17030.0;

        return interp_T(Tgrid, Cvgrid, T);
    }

    /**
    * @brief Dynamic viscosity of sodium vapor [Pa·s] as a function of temperature T
    */
    inline double mu(double T) { return 6.083e-9 * T + 1.2606e-5; }

    /**
     * @brief Thermal conductivity [W/(m*K)] of sodium vapor over an extended range.
     *
     * Performs bilinear interpolation inside the experimental grid.
     * Outside 900–1500 K or 981–98066 Pa, it extrapolates using kinetic-gas scaling (k ~ sqrt(T))
     * referenced to the nearest boundary. Prints warnings when extrapolating outside of the boundaries.
     *
     * @param T Temperature [K]
     * @param P Pressure [Pa]
     */
    inline double k(double T, double P) {

        static const std::array<double, 7> Tgrid = { 900,1000,1100,1200,1300,1400,1500 };
        static const std::array<double, 5> Pgrid = { 981,4903,9807,49033,98066 };

        static const double Ktbl[7][5] = {
            // P = 981,   4903,    9807,    49033,   98066  [Pa]
            {0.035796, 0.0379,  0.0392,  0.0415,  0.0422},   // 900 K
            {0.034053, 0.043583,0.049627,0.0511,  0.0520},   // 1000 K
            {0.036029, 0.039399,0.043002,0.060900,0.0620},   // 1100 K
            {0.039051, 0.040445,0.042189,0.052881,0.061133}, // 1200 K
            {0.042189, 0.042886,0.043816,0.049859,0.055554}, // 1300 K
            {0.045443, 0.045908,0.046373,0.049859,0.054508}, // 1400 K
            {0.048930, 0.049162,0.049511,0.051603,0.054043}  // 1500 K
        };

        // Clamping function
        auto clamp_val = [](double x, double minv, double maxv) {
            return (x < minv) ? minv : ((x > maxv) ? maxv : x);
            };

        auto idz = [](double x, const auto& grid) {
            size_t i = 0;
            while (i + 1 < grid.size() && x > grid[i + 1]) ++i;
            return i;
            };

        const double Tmin = Tgrid.front(), Tmax = Tgrid.back();
        const double Pmin = Pgrid.front(), Pmax = Pgrid.back();

        bool Tlow = (T < Tmin);
        bool Thigh = (T > Tmax);
        bool Plow = (P < Pmin);
        bool Phigh = (P > Pmax);

        double Tc = clamp_val(T, Tmin, Tmax);
        double Pc = clamp_val(P, Pmin, Pmax);

        const size_t iT = idz(Tc, Tgrid);
        const size_t iP = idz(Pc, Pgrid);

        const double T0 = Tgrid[iT], T1 = Tgrid[std::min(iT + 1ul, Tgrid.size() - 1)];
        const double P0 = Pgrid[iP], P1 = Pgrid[std::min(iP + 1ul, Pgrid.size() - 1)];

        const double q11 = Ktbl[iT][iP];
        const double q21 = Ktbl[std::min(iT + 1ul, Tgrid.size() - 1)][iP];
        const double q12 = Ktbl[iT][std::min(iP + 1ul, Pgrid.size() - 1)];
        const double q22 = Ktbl[std::min(iT + 1ul, Tgrid.size() - 1)][std::min(iP + 1ul, Pgrid.size() - 1)];

        double k_interp = 0.0;

        // Bilinear interpolation
        if ((T1 != T0) && (P1 != P0)) {
            const double t = (Tc - T0) / (T1 - T0);
            const double u = (Pc - P0) / (P1 - P0);
            k_interp = (1 - t) * (1 - u) * q11 + t * (1 - u) * q21 + (1 - t) * u * q12 + t * u * q22;
        }
        else if (T1 != T0) {
            const double t = (Tc - T0) / (T1 - T0);
            k_interp = q11 + t * (q21 - q11);
        }
        else if (P1 != P0) {
            const double u = (Pc - P0) / (P1 - P0);
            k_interp = q11 + u * (q12 - q11);
        }
        else {
            k_interp = q11;
        }

        // Extrapolation handling
        if (Tlow || Thigh || Plow || Phigh) {
            if (Tlow && warnings == true)
                std::cerr << "[Warning] Sodium vapor k: T=" << T << " < " << Tmin << " K. Using sqrt(T) extrapolation.\n";
            if (Thigh && warnings == true)
                std::cerr << "[Warning] Sodium vapor k: T=" << T << " > " << Tmax << " K. Using sqrt(T) extrapolation.\n";
            if ((Plow || Phigh) && warnings == true)
                std::cerr << "[Warning] Sodium vapor k: P outside ["
                << Pmin << "," << Pmax << "] Pa. Using constant-P approximation.\n";

            double Tref = (Tlow ? Tmin : (Thigh ? Tmax : Tc));
            double k_ref = k_interp;
            double k_extrap = k_ref * std::sqrt(T / Tref);
            return k_extrap;
        }

        return k_interp;
    }

    /**
    * @brief Friction factor [-] (Gnielinski correlation) as a function of Reynolds number. 
    * Retrieves an error if Re < 0.
    */
    inline double f(double Re) {

        if (Re <= 0.0) throw std::invalid_argument("Error: Re < 0");

        const double t = 0.79 * std::log(Re) - 1.64;
        return 1.0 / (t * t);
    }

    /**
    * @brief Nusselt number [-] (Gnielinski correlation) as a function of Reynolds number
    * Retrieves an error if Re < 0 or Nu < 0.
    */
    inline double Nu(double Re, double Pr) {

        // If laminar, Nu is constant
        if (Re < 1000) return 4.36;

        if (Re <= 0.0 || Pr <= 0.0) throw std::invalid_argument("Error: Re or Pr < 0");

        const double f = vapor_sodium::f(Re);
        const double fp8 = f / 8.0;
        const double num = fp8 * (Re - 1000.0) * Pr;
        const double den = 1.0 + 12.7 * std::sqrt(fp8) * (std::cbrt(Pr * Pr) - 1.0); // Pr^(2/3)
        return num / den;
    }

    /**
    * @brief Convective heat transfer coefficient [W/m^2/K] (Gnielinski correlation) as a function of Reynolds number
    * Retrieves an error if Re < 0 or Nu < 0.
    */ 
    inline double h_conv(double Re, double Pr, double k, double Dh) {

        const double Nu = vapor_sodium::Nu(Re, Pr);
        return Nu * k / Dh;
    }
}

#pragma endregion

// =======================================================================
//
//                        [VARIOUS ALGORITHMS]
//
// =======================================================================

#pragma region various_algorithms
/**
 * @brief Solves a tridiagonal system of linear equations A*x = d using the Thomas Algorithm (TDMA).
 *
 * The method consists of two main phases: forward elimination and back substitution,
 * which is optimized for the sparse tridiagonal structure.
 *
 * @param a The sub-diagonal vector (size N, with a[0] being zero/unused).
 * @param b The main diagonal vector (size N). Must contain non-zero elements.
 * @param c The super-diagonal vector (size N, with c[N-1] being zero/unused).
 * @param d The right-hand side vector (size N).
 * @return std::vector<double> The solution vector 'x' (size N).
 * * @note This implementation assumes the system is diagonally dominant or otherwise
 * stable, as it does not include pivoting. The vectors 'a', 'b', 'c', and 'd' must
 * all have the same size N, corresponding to the size of the system.
 */
std::vector<double> solveTridiagonal(const std::vector<double>& a,
    const std::vector<double>& b,
    const std::vector<double>& c,
    const std::vector<double>& d) {

    int n = b.size();
    std::vector<double> c_star(n, 0.0);
    std::vector<double> d_star(n, 0.0);
    std::vector<double> x(n, 0.0);

    c_star[0] = c[0] / b[0];
    d_star[0] = d[0] / b[0];

    for (int i = 1; i < n; i++) {

        double m = b[i] - a[i] * c_star[i - 1];
        c_star[i] = c[i] / m;
        d_star[i] = (d[i] - a[i] * d_star[i - 1]) / m;
    }

    x[n - 1] = d_star[n - 1];

    for (int i = n - 2; i >= 0; i--)
        x[i] = d_star[i] - c_star[i] * x[i + 1];

    return x;
}

/**
 * @brief Generate N linearly spaced values from T_min to T_max.
 * @param T_min start value.
 * @param T_max end value.
 * @param N number of points.
 * @return Vector of uniformly spaced values.
 */
std::vector<double> linspace(double T_min, double T_max, int N) {
    std::vector<double> T(N);
    double dT = (T_max - T_min) / (N - 1);
    for (int i = 0; i < N; i++) T[i] = T_min + i * dT;
    return T;
}

/**
 * @brief Compute adaptive time step for the wall region based on Fourier and source limits.
 * @param dz spatial step size.
 * @param dt_old previous time step.
 * @param T wall temperature field.
 * @param St volumetric source term field.
 * @return New time step satisfying diffusion and source stability constraints.
 */
double new_dt_w(double dz, double dt_old,
    const std::vector<double>& T,
    const std::vector<double>& St) {

    const double CFo = 0.5, CS = 0.5;           ///< Limit coefficients
    const double epsS = 1e-12;                  ///< This is to prevent divisions by zero (e.g. if the source is zero)
    const double theta = 0.9;                   ///< Adjusting coefficient for the timestep candidate 
    const double rdown = 0.2;                   ///< Coefficient for damping the timestep correction
    const double dt_min = 1e-12, dt_max = 1e-3; ///< Timestep boundaries [s]

    int N = St.size();

    double dt_cand = dt_max;
    for (int i = 0; i < N; ++i) {

        const double alpha = steel::k(T[i]) / (steel::cp(T[i]) * steel::rho(T[i]));

        /// Minimum time step due to CFL
        const double dt_c = CFo * dz * dz / alpha;

        /// Minimum time step due to CS limit
        double dt_s = CS * steel::rho(T[i]) * steel::cp(T[i]) / (std::abs(St[i]) + epsS);

        double dti = std::min(dt_c, dt_s);
        dt_cand = std::min(dt_cand, dti);  // Loop on each node to find the minimum timestep necessary
    }

    // Timestep lower boundary overrall and damping the correction
    double dt_new = std::min(dt_max, std::max(dt_min, std::max(theta * dt_cand, rdown * dt_old)));
    return dt_new;
}


/**
 * @brief Compute adaptive time step for the wick region based on CFL and source limits.
 * @param dz spatial step size.
 * @param dt_old previous time step.
 * @param u wick velocity field.
 * @param T wick bulk temperature field.
 * @param Sm wick mass source term field.
 * @return New time step satisfying advection and source stability constraints.
 */
double new_dt_x(double dz, double dt_old,
    const std::vector<double>& u,
    const std::vector<double>& T,
    const std::vector<double>& Sm) {

    const double CFL = 0.5, CS = 0.5;                  ///< Limit coefficients
    const double epsS = 1e-12;                         ///< This is to prevent divisions by zero (e.g. if the source is zero)
    const double theta = 0.9;                          ///< Adjusting coefficient for the timestep candidate 
    const double rdown = 0.2;                          ///< Coefficient for damping the timestep correction
    const double dt_min = 1e-12, dt_max = 1e-3;        ///< Timestep boundaries [s]

    int N = u.size();

    double dt_cand = dt_max;
    for (int i = 0; i < N; ++i) {

        // Minimum time step due to CFL
        double dt_c = CFL * dz / std::abs(u[i]);

        // Minimum time step due to CS limit
        double dt_s = CS * liquid_sodium::rho(T[i]) / (std::abs(Sm[i]) + epsS);

        double dti = std::min(dt_c, dt_s);
        dt_cand = std::min(dt_cand, dti);     // Loop on each node to find the minimum timestep necessary
    }

    // Timestep lower boundary overrall and damping the correction
    double dt_new = std::min(dt_max, std::max(dt_min, std::max(theta * dt_cand, rdown * dt_old)));
    return dt_new;
}

/**
 * @brief Compute adaptive time step for the vapor region from CFL, source, and pressure limits.
 * @param dz spatial step size.
 * @param dt_old previous time step.
 * @param u velocity field.
 * @param T temperature field.
 * @param rho density field.
 * @param Sm mass source term field.
 * @param bVU compressibility-related coefficient field.
 * @return New time step satisfying acoustic, source, and pressure stability constraints.
 */
double new_dt_v(double dz, double dt_old,
    const std::vector<double>& u,
    const std::vector<double>& T,
    const std::vector<double>& rho,
    const std::vector<double>& Sm,
    const std::vector<double>& bVU) {

    const double gamma = 1.32;                          ///< Vapor sodium gas constant [-]
    const double Rv = 361.8;                            ///< Gas constant for the sodium vapor [J/(kg K)]
    const double CFL = 0.5, CS = 0.5, CP = 0.5;         ///< Limit coefficients
    const double epsS = 1e-12;                          ///< This is to prevent divisions by zero (e.g. if the source is zero)
    const double theta = 0.9;                           ///< Adjusting coefficient for the timestep candidate 
    const double rdown = 0.2;                           ///< Coefficient for damping the timestep correction
    const double dt_min = 1e-12, dt_max = 1e3;          ///< Timestep boundaries [s]

    int N = u.size();
    auto invb = [&](int i) { return 1.0 / std::max(bVU[i], 1e-30); };

    double dt_cand = dt_max;
    for (int i = 0; i < N; ++i) {

        // Minimum time step due to CFL
        double a = std::sqrt(gamma * Rv * T[i]);
        double dt_c = CFL * dz / (std::abs(u[i]) + a);

        // Minimum time step due to CS limit
        double dt_s = CS * rho[i] / (std::abs(Sm[i]) + epsS);

        // Minimum time step due to CP limit
        double dt_p = 1e99;
        if (i > 0 && i < N - 1) {
            double invbL = 0.5 * (invb(i - 1) + invb(i));
            double invbR = 0.5 * (invb(i) + invb(i + 1));
            double rhoL = 0.5 * (rho[i - 1] + rho[i]);
            double rhoR = 0.5 * (rho[i] + rho[i + 1]);
            double El = rhoL * invbL / dz;
            double Er = rhoR * invbR / dz;
            double psi = 1.0 / (Rv * T[i]);
            dt_p = CP * psi * dz / (El + Er + 1e-30);
        }

        double dti = std::min(std::min(dt_c, dt_s), std::min(dt_s, dt_p));
        dt_cand = std::min(dt_cand, dti);   // Loop on each node to find the minimum timestep necessary
    }

    // Timestep lower boundary overrall and damping the correction
    double dt_new = std::min(dt_max, std::max(dt_min, std::max(theta * dt_cand, rdown * dt_old)));
    return dt_new;
}

#pragma endregion

/**
 * @brief Simplified 1D transient coupling between wall, wick, and vapor.
 *
 * Entry point for the solver performing sequential time-integration
 * of the solid wall, porous wick, and vapor core.
 */
int main() {

    // =======================================================================
    //
    //                       [CONSTANTS AND VARIABLES]
    //
    // =======================================================================

    #pragma region constants_and_variables

    // Mathematical constants
    const double M_PI = 3.14159265358979323846;

    // Physical properties
    const double emissivity = 0.9;          ///< Wall emissivity [-]
    const double sigma = 5.67e-8;           ///< Stefan-Boltzmann constant [W/m^2/K^4]
    const double Rv = 361.8;                ///< Gas constant for the sodium vapor [J/(kg K)]
    const double Pr_t = 0.01;               ///< Prandtl turbulent number for sodium vapor [-]
    const double gamma = 1.66;              ///< TODO: MAKE THIS PROPERTY TEMPERATURE DEPENDENT Ratio between constant pressure specific heat and constant volume specific heat [-] 

    // Environmental boundary conditions
    const double h_conv = 10;               ///< Convective heat transfer coefficient for external heat removal [W/m^2/K]
    const double power = 1e3;               ///< Power at the evaporator side [W]
    const double T_env = 280.0;             ///< External environmental temperature [K]

    // Evaporation and condensation parameters
    const double eps_s = 1.0;                                           ///< Surface fraction of the wick available for phasic interface [-]
    const double sigma_e = 1.0;                                         ///< Evaporation accomodation coefficient [-]. 1 means optimal evaporation
    const double sigma_c = 1.0;                                         ///< Condensation accomodation coefficient [-]. 1 means optimal condensation
    double Omega = 1.0;

    // Geometric parameters
    const int N = 200;                                                         ///< Number of axial nodes [-]
    const double L = 0.982; 			                                        ///< Length of the heat pipe [m]
    const double dz = L / N;                                                    ///< Axial discretization step [m]
    const double evaporator_length = 0.502;                                     ///< Evaporator length [m]
    const double adiabatic_length = 0.188;                                      ///< Adiabatic length [m]
    const double condenser_length = 0.292;                                      ///< Condenser length [m]
    const double evaporator_nodes = std::floor(evaporator_length / dz);         ///< Number of evaporator nodes
    const double condenser_nodes = std::ceil(condenser_length / dz);            ///< Number of condenser nodes
    const double adiabatic_nodes = N - (evaporator_nodes + condenser_nodes);    ///< Number of adiabatic nodes
    const double r_o = 0.01335;                                             ///< Outer wall radius [m]
    const double r_w = 0.0112;                                          ///< Wall-wick interface radius [m]
    const double r_inner = 0.01075;                                             ///< Vapor-wick interface radius [m]

    // Surfaces 
    const double A_w_outer = 2 * M_PI * r_o * dz;                                       ///< Wall radial area (at r_o) [m^2]
    const double A_w_cross = M_PI * (r_o * r_o - r_w * r_w);        ///< Wall cross-sectional area [m^2]
    const double A_x_interface = 2 * M_PI * r_w * dz;                               ///< Wick radial area (at r_w) [m^2]
    const double A_x_cross = M_PI * (r_w * r_w - r_inner * r_inner);        ///< Wick cross-sectional area [m^2]
    const double A_v_inner = 2 * M_PI * r_inner * dz;                                       ///< Vapor radial area (at r_inner) [m^2]
    const double A_v_cross = M_PI * r_inner * r_inner;                                      ///< Vapor cross-sectional area [m^2]

    // Time-stepping parameters
    double dt = 5e-8;                               ///< Initial time step [s] (then it is updated according to the limits)
    const int nSteps = 1000000000;                          ///< Number of timesteps
    const double time_total = nSteps * dt;          ///< Total simulation time [s]

    // Wick permeability parameters
    const double K = 1e-10;                          ///< Permeability [m^2]
    const double CF = 1e5;                          ///< Forchheimer coefficient [1/m]

    // PISO Wick parameters
    const int tot_outer_iter_x = 2000;             ///< Outer iterations per time-step [-]
    const int tot_inner_iter_x = 100;               ///< Inner iterations per outer iteration [-]
    const double outer_tol_x = 1e-6;                ///< Tolerance for the outer iterations (velocity) [-]
    const double inner_tol_x = 1e-6;                ///< Tolerance for the inner iterations (pressure) [-]

    // PISO Vapor parameters
    const int tot_outer_iter_v = 1000;             ///< Outer iterations per time-step [-]
    const int tot_inner_iter_v = 100;               ///< Inner iterations per outer iteration [-]
    const double outer_tol_v = 1e-4;                ///< Tolerance for the outer iterations (velocity) [-]
    const double inner_tol_v = 1e-4;                ///< Tolerance for the inner iterations (pressure) [-]

    // Mesh z positions
    std::vector<double> mesh(N, 0.0);
    for (int i = 0; i < N; ++i) mesh[i] = i * dz;

    // Node partition
    const int N_e = static_cast<int>(std::floor(evaporator_length / dz));   ///< Number of nodes of the evaporator region [-]
    const int N_c = static_cast<int>(std::ceil(condenser_length / dz));     ///< Number of nodes of the condenser region [-]
    const int N_a = N - (N_e + N_c);                                        ///< Number of nodes of the adiabadic region [-]

    const double T_full = 800.0;

    // Initialization of the initial temperatures using the extremes in a linear distribution
    std::vector<double> T_o_w(N, T_full);
    std::vector<double> T_w_bulk(N, T_full);
    std::vector<double> T_w_x(N, T_full);
    std::vector<double> T_x_bulk(N, T_full);
    std::vector<double> T_x_v(N, T_full);
    std::vector<double> T_v_bulk(N, T_full);

    //// Initial values for the temperatures distributions
    //double T_o_w_min = 875.0, T_o_w_max = 1100.0;               ///< Extremes for the outer wall initial temperature [K]
    //double T_w_min = 880.0, T_w_max = 1095.0;                   ///< Extremes for the wall bulk initial temperature [K]
    //double T_w_x_min = 885.0, T_w_x_max = 1090.0;               ///< Extremes for the wall wick interface initial temperature [K]
    //double T_x_min = 890.0, T_x_max = 1085.0;                   ///< Extremes for the wick bulk initial temperature [K]
    //double T_x_v_min = 895.0, T_x_v_max = 1080.0;               ///< Extremes for the wick vapor initial temperature [K]
    //double T_v_min = 900.0, T_v_max = 1075.0;                   ///< Extremes for the vapor bulk initial temperature [K]

    //// Initialization of the initial temperatures using the extremes in a linear distribution
    //std::vector<double> T_o_w = linspace(T_o_w_max, T_o_w_min, N);
    //std::vector<double> T_w_bulk = linspace(T_w_max, T_w_min, N);
    //std::vector<double> T_w_x = linspace(T_w_x_max, T_w_x_min, N);
    //std::vector<double> T_x_bulk = linspace(T_x_max, T_x_min, N);
    //std::vector<double> T_x_v = linspace(T_x_v_max, T_x_v_min, N);
    //std::vector<double> T_v_bulk = linspace(T_v_max, T_v_min, N);

    // Old temperature variables
    std::vector<double> T_x_v_old = T_x_v;
    std::vector<double> T_old_x = T_x_bulk;
    std::vector<double> T_old_v = T_v_bulk;

    // Wick fields
    std::vector<double> u_x(N, -0.000001);                                              ///< Wick velocity field [m/s]
    std::vector<double> p_x(N, vapor_sodium::P_sat(T_v_bulk[N - 1]));               ///< Wick pressure field [Pa]
    std::vector<double> p_prime_x(N, 0.0);                                          ///< Wick correction pressure field [Pa]
    std::vector<double> p_old_x(N, vapor_sodium::P_sat(T_v_bulk[N - 1]));           ///< Wick old pressure field [Pa]
    std::vector<double> p_storage_x(N + 2, vapor_sodium::P_sat(T_x_v[N - 1]));      ///< Wick padded pressure vector for R&C correction [Pa]
    double* p_padded_x = &p_storage_x[1];                                           ///< Poìnter to work on the wick pressure padded storage with the same indes

    // Vapor fields
    std::vector<double> u_v(N, 0.1);                                                ///< Vapor velocity field [m/s]
    std::vector<double> p_v(N, vapor_sodium::P_sat(T_v_bulk[N - 1]));               ///< Vapor pressure field [Pa]
    std::vector<double> rho_v(N, 8e-3);                                             ///< Vapor density field [Pa]
    std::vector<double> p_prime_v(N, 0.0);                                          ///< Vapor correction pressure field [Pa]
    std::vector<double> p_old_v(N, vapor_sodium::P_sat(T_v_bulk[N - 1]));           ///< Vapor old pressure field [Pa]
    std::vector<double> rho_old_v(N, 8e-3);                                         ///< Vapor old density field [Pa]
    std::vector<double> u_old_v(N, 0.1);                                            ///< Vapor old velocity field [Pa]
    std::vector<double> p_storage_v(N + 2, vapor_sodium::P_sat(T_v_bulk[N - 1]));   ///< Vapor padded pressure vector for R&C correction [Pa]
    double* p_padded_v = &p_storage_v[1];                                           ///< Poìnter to work on the storage with the same indes

    for (int i = 0; i < N; ++i) p_v[i] = vapor_sodium::P_sat(T_x_v[i]);

    // Wick BCs
    const double u_inlet_x = 0.0;                                   ///< Wick inlet velocity [m/s]
    const double u_outlet_x = 0.0;                                  ///< Wick outlet velocity [m/s]
    double p_outlet_x = vapor_sodium::P_sat(T_x_v[N - 1]);          ///< Wick outlet pressure [Pa]

    // Vapor BCs
    const double u_inlet_v = 0.0;                                   ///< Vapor inlet velocity [m/s]
    const double u_outlet_v = 0.0;                                  ///< Vapor outlet velocity [m/s]
    double p_outlet_v = vapor_sodium::P_sat(T_v_bulk[N - 1]);       ///< Vapor outlet pressure [Pa]

    // Turbulence constants for sodium vapor (SST model)
    const double I = 0.05;                                          ///< Turbulence intensity [-]
    const double L_t = 0.07 * L;                                    ///< Turbulence length scale [m]
    const double k0 = 1.5 * pow(I * u_inlet_v, 2);                  ///< Initial turbulent kinetic energy [m^2/s^2]
    const double omega0 = sqrt(k0) / (0.09 * L_t);                  ///< Initial specific dissipation rate [1/s]
    const double sigma_k = 0.85;                                    ///< k-equation turbulent Prandtl number [-]
    const double sigma_omega = 0.5;                                 ///< ω-equation turbulent Prandtl number [-]
    const double beta_star = 0.09;                                  ///< β* constant for SST model [-]
    const double beta = 0.075;                                      ///< β constant for turbulence model [-]
    const double alpha = 5.0 / 9.0;                                 ///< α blending coefficient for SST model [-]

    // Turbulence fields for sodium vapor
    std::vector<double> k_turb(N, k0);
    std::vector<double> omega_turb(N, omega0);
    std::vector<double> mu_t(N, 0.0);

    /**
    * @brief Vapor Equation of State update function. Updates density
    */
    auto eos_update = [&](std::vector<double>& rho_, const std::vector<double>& p_, const std::vector<double>& T_) {

        for (int i = 0; i < N; i++) { rho_[i] = std::max(1e-6, p_[i] / (Rv * T_v_bulk[i])); }

        }; eos_update(rho_v, p_v, T_v_bulk);

    const double q_pp_evaporator = power / (2 * M_PI * evaporator_length * r_o);     ///< Heat flux at evaporator from given power [W/m^2]

    // Heat fluxes at the interfaces [W/m^2]
    std::vector<double> q_o_w(N, 0.0);           ///< Heat flux [W/m^2] across outer wall in wall region (positive if directed to wall)
    std::vector<double> q_w_x_wall(N, 0.0);      ///< Heat flux [W/m^2] across wall-wick interface in the wall region (positive if directed to wick)
    std::vector<double> q_w_x_wick(N, 0.0);      ///< Heat flux [W/m^2] across wall-wick interface in the wick region (positive if directed to wick)
    std::vector<double> q_x_v_wick(N, 0.0);      ///< Heat flux [W/m^2] across wick-vapor interface in the wick region (positive if directed to vapor)
    std::vector<double> q_x_v_vapor(N, 0.0);     ///< Heat flux [W/m^2] across wick-vapor interface in the vapor region (positive if directed to vapor)
    
    std::vector<double> Q_w(N, 0.0);             ///< Wall heat source [W/m3]
    std::vector<double> Q_x(N, 0.0);             ///< Wick heat source [W/m3]
    std::vector<double> Q_v(N, 0.0);             ///< Vapor heat source [W/m3]

    std::vector<double> Q_mass_vapor(N, 0.0);    ///< Heat volumetric source [W/m3] due to evaporation condensation. To be summed to the vapor
    std::vector<double> Q_mass_wick(N, 0.0);     ///< Heat volumetric source [W/m3] due to evaporation condensation. To be summed to the wick

    // Models
    const int rhie_chow_on_off_x = 1;             ///< 0: no wick RC correction, 1: wick with RC correction
    const int rhie_chow_on_off_v = 1;             ///< 0: no vapor RC correction, 1: vapor with RC correction
    const int SST_model_turbulence_on_off = 0;    ///< 0: no vapor turbulence, 1: vapor with turbulence

    // Mass sources/fluxes
    std::vector<double> phi_x_v(N, 0.0);            ///< Mass flux [kg/m2/s] at the wick-vapor interface (positive if evaporation)
    std::vector<double> Gamma_xv_vapor(N, 0.0);     ///< Volumetric mass source [kg / (m^3 s)] (positive if evaporation)
    std::vector<double> Gamma_xv_wick(N, 0.0);      ///< Volumetric mass source [kg / (m^3 s)] (positive if evaporation)

    /// The coefficient bU is needed in momentum predictor loop and pressure correction to estimate the velocities at the faces using the Rhie and Chow correction
    std::vector<double> aXU(N, 0.0),                                                    ///< Lower tridiagonal coefficient for wick velocity
        bXU(N, liquid_sodium::rho(1000)* dz / dt + 2 * liquid_sodium::mu(1000) / dz),  ///< Central tridiagonal coefficient for wick velocity
        cXU(N, 0.0),                                                                    ///< Upper tridiagonal coefficient for wick velocity
        dXU(N, 0.0);                                                                    ///< Known vector coefficient for wick velocity

    /// The coefficient bVU is needed in momentum predictor loop and pressure correction to estimate the velocities aVT the faces using the Rhie and Chow correction
    std::vector<double> aVU(N, 0.0),                                                    ///< Lower tridiagonal coefficient for vapor velocity
        bVU(N, 2 * (4.0 / 3.0 * vapor_sodium::mu(1000) / dz) + dz / dt * rho_v[0]),     ///< Central tridiagonal coefficient for vapor velocity
        cVU(N, 0.0),                                                                    ///< Upper tridiagonal coefficient for vapor velocity
        dVU(N, 0.0);                                                                    ///< Known vector for vapor velocity

    double total_mass = 0;

    // Create result folder
    std::filesystem::create_directories("results");

    // Print results in file
    std::ofstream mesh_io("mesh.txt", std::ios::trunc);

    std::ofstream v_velocity_output("results/vapor_velocity.txt", std::ios::trunc);
    std::ofstream v_pressure_output("results/vapor_pressure.txt", std::ios::trunc);
    std::ofstream v_bulk_temperature_output("results/vapor_bulk_temperature.txt", std::ios::trunc);

    std::ofstream x_velocity_output("results/wick_velocity.txt", std::ios::trunc);
    std::ofstream x_pressure_output("results/wick_pressure.txt", std::ios::trunc);
    std::ofstream x_bulk_temperature_output("results/wick_bulk_temperature.txt", std::ios::trunc);

    std::ofstream x_v_temperature_output("results/wick_vapor_interface_temperature.txt", std::ios::trunc);
    std::ofstream w_x_temperature_output("results/wall_wick_interface_temperature.txt", std::ios::trunc);
    std::ofstream o_w_temperature_output("results/outer_wall_temperature.txt", std::ios::trunc);
    std::ofstream w_bulk_temperature_output("results/wall_bulk_temperature.txt", std::ios::trunc);

    std::ofstream x_v_mass_flux_output("results/wick_vapor_mass_source.txt", std::ios::trunc);

    std::ofstream o_w_heat_flux_output("results/outer_wall_heat_flux.txt", std::ios::trunc);
    std::ofstream w_x_heat_flux_output("results/wall_wick_heat_flux.txt", std::ios::trunc);
    std::ofstream x_v_heat_flux_output("results/wick_vapor_heat_flux.txt", std::ios::trunc);

    std::ofstream rho_output("results/rho_vapor.txt", std::ios::trunc);

    mesh_io << std::setprecision(8);

    v_velocity_output << std::setprecision(8);
    v_pressure_output << std::setprecision(8);
    v_bulk_temperature_output << std::setprecision(8);

    x_velocity_output << std::setprecision(8);
    x_pressure_output << std::setprecision(8);
    x_bulk_temperature_output << std::setprecision(8);

    x_v_temperature_output << std::setprecision(8);
    w_x_temperature_output << std::setprecision(8);
    o_w_temperature_output << std::setprecision(8);
    w_bulk_temperature_output << std::setprecision(8);

    x_v_mass_flux_output << std::setprecision(8);

    o_w_heat_flux_output << std::setprecision(8);
    w_x_heat_flux_output << std::setprecision(8);
    x_v_heat_flux_output << std::setprecision(8);

    rho_output << std::setprecision(8);

    for (int i = 0; i < N; ++i) {
        mesh_io << i * dz << " ";
    }

    mesh_io.flush();

    #pragma endregion

    /// Print number of working threads
    std::cout << "Threads: " << omp_get_max_threads() << "\n";

    double start = omp_get_wtime();

    int n = 0;

    /**
     * @brief Time-stepping loop. The timestep is calculated at the beginning of each loop.
     */
    /*for (int n = 0; n < nSteps; ++n)*/ 
    while(true) {

        n += 1; 

        /*dt = std::min(std::min(new_dt_w(dz, dt, T_w_bulk, Q_w),
                               new_dt_x(dz, dt, u_x, T_x_bulk, Gamma_xv)), 
                      std::min(new_dt_x(dz, dt, u_x, T_x_bulk, Gamma_xv), 
                               new_dt_v(dz, dt, u_v, T_v_bulk, rho_v, Gamma_xv, bVU)));*/

        // =======================================================================
        //
        //                           [1. SOLVE WALL]
        //
        // =======================================================================

        #pragma region wall

        std::vector<double> aTW(N, 0.0),        ///< Lower tridiagonal coefficient for wall temperature
            bTW(N, 0.0),                        ///< Central tridiagonal coefficient for wall temperature
            cTW(N, 0.0),                        ///< Upper tridiagonal coefficient for wall temperature
            dTW(N, 0.0);                        ///< Known vector coefficient for wall temperature

        /// BCs for the first node: zero gradient, adiabatic face
        aTW[0] = 0.0; 
        bTW[0] = 1.0; 
        cTW[0] = -1.0; 
        dTW[0] = 0.0;
        
        /// BCs for the last node: zero gradient, adiabatic face
        aTW[N - 1] = -1.0; 
        bTW[N - 1] = 1.0; 
        cTW[N - 1] = 0.0; 
        dTW[N - 1] = 0.0;

        /**
         * @brief Loop on nodes. Assembles the coefficients for the tridiagonal system of wall temperature
         */
        for (int i = 1; i < N - 1; ++i) {

            /// Physical properties of the i-th wall node
            const double cp = steel::cp(T_w_bulk[i]);
            const double k = steel::k(T_w_bulk[i]);
            const double rho = steel::rho(T_w_bulk[i]);

            /// Useful constant
            const double r = (k * dt) / (cp * dz * dz);

            /// Volumetric heat source in the wall due to heat flux at the outer wall [W/m^3]
            const double volum_heat_source_o_w = q_o_w[i] * 2 * r_o / (r_o * r_o - r_w * r_w);

            /// Volumetric heat source in the wall due to heat flux at the wall-wick interface [W/m^3]
            const double volum_heat_source_w_x = q_w_x_wall[i] * 2 * r_w / (r_o * r_o - r_w * r_w);

            Q_w[i] = volum_heat_source_o_w - volum_heat_source_w_x;     ///< Wall volumetric heat source [W/m^3]

            aTW[i] = -r;
            bTW[i] = 1 + 2 * r;
            cTW[i] = -r;
            dTW[i] = T_w_bulk[i] + Q_w[i] * dt / (cp * rho);
        }

        T_w_bulk = solveTridiagonal(aTW, bTW, cTW, dTW);        ///< Vector of final wall bulk temperatures

        #pragma endregion

        // =======================================================================
        //
        //                           [2. SOLVE WICK]
        //
        // =======================================================================

        #pragma region wick

        /// Evaluates maximum absolute value of the wick velocity
        const double max_abs_u =
            std::abs(*std::max_element(u_x.begin(), u_x.end(),
                [](double a, double b) { return std::abs(a) < std::abs(b); }
            ));

        /// Evaluates minimum value of the wick temperature
        const double min_T_wick = *std::min_element(T_x_bulk.begin(), T_x_bulk.end());

        /*std::cout << "Solving wick! Time elapsed:" << dt * n << "/" << time_total
            << ", max courant number: " << max_abs_u * dt / dz
            << ", max reynolds number: " << max_abs_u * r_w *liquid_sodium::rho(min_T_wick) / liquid_sodium::mu(min_T_wick) << "\n";*/

        // Backup old variables
        T_old_x = T_x_bulk;
        p_old_x = p_x;

        /**
         * Pressure coupling hypotheses: the meniscus in the last cell of the domain is 
         * considered flat, so the pressure of the wick is equal to the pressure of the vapor
         */
        p_outlet_x = p_v[N - 1];

        /// Outer iterations variables reset
        double u_error_x = 1.0;
        int outer_iter_x = 0;

        /// Inner iterations variables initialization
        double p_error_x;
        int inner_iter_x;

        /**
         * @brief Outer iterations. Stop when maximum number of iterations or velocity stops updating.
         */
        while (outer_iter_x < tot_outer_iter_x && u_error_x > outer_tol_x) {

            // =======================================================================
            //
            //                      [MOMENTUM PREDICTOR]
            //
            // =======================================================================

            #pragma region momentum_predictor

            // Parallelizing here does not save time
            for (int i = 1; i < N - 1; i++) {

                const double rho_P = liquid_sodium::rho(T_x_bulk[i]);       ///< Density of the central cell
                const double rho_L = liquid_sodium::rho(T_x_bulk[i - 1]);   ///< Density of the left cell
                const double rho_R = liquid_sodium::rho(T_x_bulk[i + 1]);   ///< Density of the right cell

                const double mu_P = liquid_sodium::mu(T_x_bulk[i]);         ///< Dynamic viscosity of the central cell
                const double mu_L = liquid_sodium::mu(T_x_bulk[i - 1]);     ///< Dynamic viscosity of the left cell
                const double mu_R = liquid_sodium::mu(T_x_bulk[i + 1]);     ///< Dynamic viscosity of the right cell

                const double D_l = 0.5 * (mu_P + mu_L) / dz;                ///< Diffusion coefficient [kg/(m2 s)] of the left face (average)
                const double D_r = 0.5 * (mu_P + mu_R) / dz;                ///< Diffusion coefficient [kg/(m2 s)] of the right face (average)

                /// RC correction for the left face velocity
                const double rhie_chow_l = -(1.0 / bXU[i - 1] + 1.0 / bXU[i]) / (8 * dz) * (p_padded_x[i - 2] - 3 * p_padded_x[i - 1] + 3 * p_padded_x[i] - p_padded_x[i + 1]);
                
                /// RC correction for the right face velocity 
                const double rhie_chow_r = -(1.0 / bXU[i + 1] + 1.0 / bXU[i]) / (8 * dz) * (p_padded_x[i - 1] - 3 * p_padded_x[i] + 3 * p_padded_x[i + 1] - p_padded_x[i + 2]);

                /// Left face velocity [m/s] (average + RC correction)
                const double u_l_face = 0.5 * (u_x[i - 1] + u_x[i]) + rhie_chow_on_off_x * rhie_chow_l;

                /// Right face velocity [m/s] (average + RC correction)
                const double u_r_face = 0.5 * (u_x[i] + u_x[i + 1]) + rhie_chow_on_off_x * rhie_chow_r;

                const double rho_l = (u_l_face >= 0) ? rho_L : rho_P;       ///< Density of the left face (average)
                const double rho_r = (u_r_face >= 0) ? rho_P : rho_R;       ///< Density of the right face (average)

                const double F_l = rho_l * u_l_face;                        ///< Mass flux [kg/(m2 s)] of the left face 
                const double F_r = rho_r * u_r_face;                        ///< Mass flux [kg/(m2 s)] of the right face 

                aXU[i] = -std::max(F_l, 0.0) - D_l;
                cXU[i] = -std::max(-F_r, 0.0) - D_r;
                bXU[i] = (std::max(F_r, 0.0) + std::max(-F_l, 0.0)) + rho_P * dz / dt + D_l + D_r + mu_P / K * dz + CF * mu_P * dz / sqrt(K) * abs(u_x[i]);
                dXU[i] = -0.5 * (p_x[i + 1] - p_x[i - 1]) + rho_P * u_x[i] * dz / dt /* + Su_x[i] * dz */;
            }

            /// Diffusion coefficients for the first and last node to define BCs
            const double D_first = liquid_sodium::mu(T_x_bulk[0]) / dz;
            const double D_last = liquid_sodium::mu(T_x_bulk[N - 1]) / dz;

            /// Velocity BCs: zero velocity on the first node
            bXU[0] = (liquid_sodium::rho(T_x_bulk[0]) * dz / dt + 2 * D_first); cXU[0] = 0.0; dXU[0] = (liquid_sodium::rho(T_x_bulk[0]) * dz / dt + 2 * D_first) * u_inlet_x;
            
            /// Velocity BCs: zero velocity on the last node
            aXU[N - 1] = 0.0; bXU[N - 1] = (liquid_sodium::rho(T_x_bulk[N - 1]) * dz / dt + 2 * D_last); dXU[N - 1] = (liquid_sodium::rho(T_x_bulk[N - 1]) * dz / dt + 2 * D_last) * u_outlet_x;

            u_x = solveTridiagonal(aXU, bXU, cXU, dXU);

            #pragma endregion

            /// Inner iterations variables reset
            p_error_x = 1.0;
            inner_iter_x = 0;

            /**
             * @brief Inner (PISO) iterations. Stop when maximum number of iterations or pressure stops updating.
             */
            while (inner_iter_x < tot_inner_iter_x && p_error_x > inner_tol_x) {

                // =======================================================================
                //
                //                       [CONTINUITY SATISFACTOR]
                //
                // =======================================================================

                #pragma region continuity_satisfactor

                /// Tridiagonal coefficients for the pressure correction
                std::vector<double> aXP(N, 0.0), 
                                        bXP(N, 0.0), 
                                        cXP(N, 0.0), 
                                        dXP(N, 0.0);

                /**
                 * @brief Calculates the coefficients for the tridiagonal system of the pressure correction p'
                 */
                for (int i = 1; i < N - 1; i++) {

                    const double rho_P = liquid_sodium::rho(T_x_bulk[i]);       ///< Density [kg/m3] of the central cell
                    const double rho_L = liquid_sodium::rho(T_x_bulk[i - 1]);   ///< Density [kg/m3] of the left cell
                    const double rho_R = liquid_sodium::rho(T_x_bulk[i + 1]);   ///< Density [kg/m3] of the right cell

                    const double d_l_face = 0.5 * (1.0 / bXU[i - 1] + 1.0 / bXU[i]) / dz;    ///< 1/Ap [(m2 s)/kg] average on left face
                    const double d_r_face = 0.5 * (1.0 / bXU[i] + 1.0 / bXU[i + 1]) / dz;    ///< 1/Ap [(m2 s)/kg] average on right face

                    /// RC correction for the left face velocity
                    const double rhie_chow_l = -d_l_face / 4 * (p_padded_x[i - 2] - 3 * p_padded_x[i - 1] + 3 * p_padded_x[i] - p_padded_x[i + 1]);

                    /// RC correction for the right face velocity 
                    const double rhie_chow_r = -d_r_face / 4 * (p_padded_x[i - 1] - 3 * p_padded_x[i] + 3 * p_padded_x[i + 1] - p_padded_x[i + 2]);

                    const double rho_l = 0.5 * (rho_L + rho_P);                         ///< Density [kg/m3] on the left face
                    const double E_l = rho_l * d_l_face;                                ///< Diffusion coefficient [s/m] on the left face

                    const double rho_r = 0.5 * (rho_P + rho_R);                         ///< Density [kg/m3] on the right face
                    const double E_r = rho_r * d_r_face;                                ///< Diffusion coefficient [s/m] on the right face

                    const double u_l_star = 0.5 * (u_x[i - 1] + u_x[i]) + rhie_chow_on_off_x * rhie_chow_l;                   ///< Left face velocity [m/s] (average + RC correction)
                    const double u_r_star = 0.5 * (u_x[i] + u_x[i + 1]) + rhie_chow_on_off_x * rhie_chow_r;                   ///< Right face velocity [m/s] (average + RC correction)

                    const double phi_l_star = (u_l_star > 0.0) ? rho_L * u_l_star : rho_P * u_l_star;  ///< Left face mass flux [kg/(m^2 s)] (upwind discretization)
                    const double phi_r_star = (u_r_star > 0.0) ? rho_P * u_r_star : rho_R * u_r_star;  ///< Right face mass flux [kg/(m^2 s)] (upwind discretization)

                    const double mass_imbalance = (phi_r_star - phi_l_star);  ///< Mass difference of the fluxes across faces [kg/(m^2 s)]
                    const double mass_flux = Gamma_xv_wick[i] * dz;                  ///< Mass flux [kg/(m^2 s)] due to external source (positive if out of the wick)

                    aXP[i] = -E_l;                              ///< [s/m]
                    cXP[i] = -E_r;                              ///< [s/m]
                    bXP[i] = E_l + E_r;                         ///< [s/m]
                    dXP[i] = -mass_flux - mass_imbalance;       ///< [kg/(m^2 s)]
                }

                /// BCs for the correction of pressure: zero gradient at first node
                bXP[0] = 1.0; 
                cXP[0] = -1.0; 
                dXP[0] = 0.0;

                /// BCs for the correction of pressure: zero at first node
                aXP[N - 1] = 0.0;
                bXP[N - 1] = 1.0; 
                dXP[N - 1] = 0.0;

                p_prime_x = solveTridiagonal(aXP, bXP, cXP, dXP);

                #pragma endregion

                // =======================================================================
                //
                //                        [PRESSURE CORRECTOR]
                //
                // =======================================================================

                #pragma region pressure_corrector

                /**
                  * @brief Corrects the pressure with p'
                  */
                p_error_x = 0.0;

                for (int i = 0; i < N; i++) {

                    double p_prev_x = p_x[i];
                    p_x[i] += p_prime_x[i];         // Note that PISO does not require an under-relaxation factor
                    p_storage_x[i + 1] = p_x[i];

                    p_error_x = std::max(p_error_x, std::fabs(p_x[i] - p_prev_x));
                }

                p_storage_x[0] = p_storage_x[1];
                p_storage_x[N + 1] = p_outlet_x;

                #pragma endregion

                // =======================================================================
                //
                //                        [VELOCITY CORRECTOR]
                //
                // =======================================================================

                #pragma region velocity_corrector

                /**
                  * @brief Corrects the velocity with p'
                  */
                u_error_x = 0.0;

                for (int i = 1; i < N - 1; i++) {

                    double u_prev = u_x[i];
                    u_x[i] = u_x[i] - (p_prime_x[i + 1] - p_prime_x[i - 1]) / (2.0 * dz * bXU[i]);

                    u_error_x = std::max(u_error_x, std::fabs(u_x[i] - u_prev));
                }

                #pragma endregion

                inner_iter_x++;
            }

            outer_iter_x++;  
        }

        // =======================================================================
        //
        //                        [TEMPERATURE CALCULATOR]
        //
        // =======================================================================

        #pragma region temperature_calculator

        /// Tridiagonal coefficients for the wick temperature
        std::vector<double> aXT(N, 0.0), 
                                bXT(N, 0.0), 
                                cXT(N, 0.0), 
                                dXT(N, 0.0);

        /**
         * @brief Calculates the coefficients for the tridiagonal system of the wick temperature T
         */
        for (int i = 1; i < N - 1; i++) {

            const double rho_P = liquid_sodium::rho(T_x_bulk[i]);       ///< Density [kg/m3] of the central cell
            const double rho_L = liquid_sodium::rho(T_x_bulk[i - 1]);   ///< Density [kg/m3]of the left cell
            const double rho_R = liquid_sodium::rho(T_x_bulk[i + 1]);   ///< Density [kg/m3]of the right cell

            const double k_cond_P = liquid_sodium::k(T_x_bulk[i]);      ///< Thermal conductivity [W/(m*K)] of the central cell
            const double k_cond_L = liquid_sodium::k(T_x_bulk[i - 1]);  ///< Thermal conductivity [W/(m*K)] of the left cell
            const double k_cond_R = liquid_sodium::k(T_x_bulk[i + 1]);  ///< Thermal conductivity [W/(m*K)] of the right cell

            const double cp_P = liquid_sodium::cp(T_x_bulk[i]);         ///< S. h. at constant pressure [J/(kg K)] of the central cell
            const double cp_L = liquid_sodium::cp(T_x_bulk[i - 1]);     ///< S. h. at constant pressure [J/(kg K)] of the left cell
            const double cp_R = liquid_sodium::cp(T_x_bulk[i + 1]);     ///< S. h. at constant pressure [J/(kg K)] of the right cell

            const double rhoCp_dzdt = rho_P * cp_P * dz / dt;           ///< Useful constant [W/(m^2 K)]

            const double D_l = 0.5 * (k_cond_P + k_cond_L) / dz;        ///< Diffusion coefficient [W/(m^2 K)] of the left face (average)
            const double D_r = 0.5 * (k_cond_P + k_cond_R) / dz;        ///< Diffusion coefficient [W/(m^2 K)] of the right face (average)

            /// RC correction for the left face velocity
            const double rhie_chow_l = -(1.0 / bXU[i - 1] + 1.0 / bXU[i]) / (8 * dz) * (p_padded_x[i - 2] - 3 * p_padded_x[i - 1] + 3 * p_padded_x[i] - p_padded_x[i + 1]);

            /// RC correction for the right face velocity 
            const double rhie_chow_r = -(1.0 / bXU[i + 1] + 1.0 / bXU[i]) / (8 * dz) * (p_padded_x[i - 1] - 3 * p_padded_x[i] + 3 * p_padded_x[i + 1] - p_padded_x[i + 2]);

            /// Left face velocity [m/s] (average + RC correction)
            const double u_l_face = 0.5 * (u_x[i - 1] + u_x[i]) + rhie_chow_on_off_x * rhie_chow_l;

            /// Right face velocity [m/s] (average + RC correction)
            const double u_r_face = 0.5 * (u_x[i] + u_x[i + 1]) + rhie_chow_on_off_x * rhie_chow_r;

            const double rho_l = (u_l_face >= 0) ? rho_L : rho_P;       ///< Density [kg/m3] of the left face (upwind)
            const double rho_r = (u_r_face >= 0) ? rho_P : rho_R;       ///< Density [kg/m3] of the right face (upwind)

            // Upwind specific heat
            const double cp_l = (u_l_face >= 0) ? cp_L : cp_P;          ///< S. h. at constant pressure [J/(kg K)] of the left face (upwind)
            const double cp_r = (u_r_face >= 0) ? cp_P : cp_R;          ///< S. h. at constant pressure [J/(kg K)] of the right face (upwind)

            const double Fl = rho_l * u_l_face;                         ///< Mass flux [kg/(m2 s)] of the left face (upwind)
            const double Fr = rho_r * u_r_face;                         ///< Mass flux [kg/(m2 s)] of the left face (upwind)

            const double C_l = (Fl * cp_l);                             ///< Mass flux [W/(m^2 K)] of the left face (upwind)
            const double C_r = (Fr * cp_r);                             ///< Mass flux [W/(m^2 K)] of the left face (upwind)

            /// Volumetric heat source due to heat flux at the wall-wick interface [W/m^3]
            double volum_heat_source_w_x = q_w_x_wick[i] * 2 * r_w / (r_w * r_w - r_inner * r_inner);

            /// Volumetric heat source due to heat flux at the wick-vapor interface [W/m^3]
            double volum_heat_source_x_v = q_x_v_wick[i] * 2 * r_inner / (r_w * r_w - r_inner * r_inner);

            /**
              * Wick volumetric heat source [W/m^3] due to phase change and interface heat flux
              */ 
            Q_x[i] = volum_heat_source_w_x*dz - volum_heat_source_x_v*dz + Q_mass_wick[i];  

            aXT[i] = -D_l - std::max(C_l, 0.0);                                                 ///< [W/(m2 K)]
            cXT[i] = -D_r - std::max(-C_r, 0.0);                                                ///< [W/(m2 K)]
            bXT[i] = (std::max(C_r, 0.0) + std::max(-C_l, 0.0)) + D_l + D_r + rhoCp_dzdt;       ///< [W/(m2 K)]
            dXT[i] = rhoCp_dzdt * T_old_x[i] + Q_x[i];                                          ///< [W/m2]
        }

        /// Temperature BCs: zero gradient on the first node
        bXT[0] = 1.0; 
        cXT[0] = -1.0; 
        dXT[0] = 0.0;

        /// Temperature BCs: zero gradient on the last node
        aXT[N - 1] = -1.0; 
        bXT[N - 1] = 1.0; 
        dXT[N - 1] = 0.0;

        T_x_bulk = solveTridiagonal(aXT, bXT, cXT, dXT);

        #pragma endregion

        #pragma endregion

        // =======================================================================
        //
        //                   [3. SOLVE INTERFACES AND FLUXES]
        //
        // =======================================================================

        #pragma region parabolic_profiles 

        /**
         * Temperature distribution coefficients (six coefficients per node, two parabolas)
         * First three coefficients are a_w, b_w, c_w
         * Last three coefficients are a_x, b_x, c_x
         */
        std::array<std::array<double, 6>, N> ABC{};

        /**
         * @brief Calculates the parabolas coefficients for each node
         */

        total_mass = 0.0;

        for (int i = 0; i < N; ++i) {

            std::array<std::array<double, 6>, 6> A{};           ///< Linear system matrix A
            std::array<double, 6> B{};                          ///< Linear system vector B

            /**
             * Mass flux from the wick to the vapor [kg/(m2 s)]. 
             * Calculated using the kinetic gas theory.
             */
            phi_x_v[i] = (sigma_e * vapor_sodium::P_sat(T_x_v[i]) / std::sqrt(T_x_v[i]) - 
                            sigma_c * Omega * p_v[i] / std::sqrt(T_v_bulk[i])) /
                            (std::sqrt(2 * M_PI * Rv));

            Gamma_xv_vapor[i] = phi_x_v[i] * 2.0 * eps_s / r_inner;
            Gamma_xv_wick[i] = phi_x_v[i] * (2.0 * r_inner * eps_s) / (r_w * r_w - r_inner * r_inner);

            /**
             * Variable b [-], used to calculate omega. 
             * Ratio of the overrall speed to the most probable velocity of the vapor.
             */
            const double b = std::abs(-phi_x_v[i] / (p_v[i] * std::sqrt(2.0 / (Rv * T_v_bulk[i]))));        

            /**
              * Linearization of the omega [-] function to correct the net evaporation/condensation mass flux
              */
            if (b < 0.1192) Omega = 1.0 + b * std::sqrt(M_PI);
            else if (b <= 0.9962) Omega = 0.8959 + 2.6457 * b;
            else Omega = 2.0 * b * std::sqrt(M_PI);

            const double k_w = steel::k(T_w_bulk[i]);                                           ///< Wall thermal conductivity [W/(m K)]
            const double k_x = liquid_sodium::k(T_x_bulk[i]);                                   ///< Liquid thermal conductivity [W/(m K)]
            const double cp_v = vapor_sodium::cp(T_v_bulk[i]);                                  ///< Vapor specific heat [J/(kg K)]
            const double k_v_cond = vapor_sodium::k(T_v_bulk[i], p_v[i]);                       ///< Vapor thermal conductivity [W/(m K)]
            const double k_v_eff = k_v_cond + mu_t[i] * cp_v / Pr_t;                            ///< Effective vapor thermal conductivity [W/(m K)]
            const double mu_v = vapor_sodium::mu(T_v_bulk[i]);                                  ///< Vapor dynamic viscosity [Pa*s]
            const double Dh_v = 2.0 * r_inner;                                                  ///< Hydraulic diameter of the vapor core [m]
            const double Re_v = rho_v[i] * std::fabs(u_v[i]) * Dh_v / mu_v;                     ///< Reynolds number [-]
            const double Pr_v = cp_v * mu_v / k_v_cond;                                         ///< Prandtl number [-]
            const double H_xm = vapor_sodium::h_conv(Re_v, Pr_v, k_v_cond, Dh_v);               ///< Convective heat transfer coefficient at the vapor-wick interface [W/m^2/K]
            const double Psat = vapor_sodium::P_sat(T_x_v[i]);                                  ///< Saturation pressure [Pa]        
            const double dPsat_dT = Psat * std::log(10.0) * (7740.0 / (T_x_v[i] * T_x_v[i]));   ///< Derivative of the saturation pressure wrt T [Pa/K]   

            const double fac = (2.0 * r_inner * eps_s * beta) / (r_w * r_w);    ///< Useful factor in the coefficients calculation [s / m^2]

            double h_xv_v;          ///< Specific enthalpy [J/kg] of vapor upon phase change between wick and vapor
            double h_vx_x;          ///< Specific enthalpy [J/kg] of wick upon phase change between vapor and wick

            if (phi_x_v[i] > 0.0) {

                // Evaporation case
                h_xv_v = vapor_sodium::h(T_x_v[i]);
                h_vx_x = liquid_sodium::h(T_x_v[i]);

            } else {

                // Condensation case
                h_xv_v = vapor_sodium::h(T_v_bulk[i]);
                h_vx_x = liquid_sodium::h(T_x_v[i])
                    + (vapor_sodium::h(T_v_bulk[i]) - vapor_sodium::h(T_x_v[i]));
            }

            /**
             * Variation of the vapor pressure due to density and temperature changes [Pa]
             * Calculating using THROHPUT linearization approach (function of density and temperature)
             */
            /*const double dPg = (p_v[i] / rho_v[i]) * (rho_v[i] - rho_old_v[i])
                + (p_v[i] / T_v_bulk[i]) * (T_v_bulk[i] - T_old_v[i]);   */

            // Coefficients for the linearization of the new mass transfer rate
            /*bGamma[i] = -(Gamma_xv[i] / (2.0 * T_x_v[i])) + fac * sigma_e * dPsat_dT;   ///< b coefficient [kg/(m3 s K)] 
            aGamma[i] = 0.5 * Gamma_xv[i] + fac * sigma_e * dPsat_dT * T_x_v[i];        ///< a coefficient [kg/(m3 s)]
            cGamma[i] = -fac * sigma_c * Omega;                                         ///< c coefficient [s/m2]*/

            /// Coefficients for the parabolic temperature profiles in wall and wick (check equations)
            const double E1w = 2.0 / 3.0 * (r_o + r_w - 1 / (1 / r_o + 1 / r_w));
            const double E2w = 0.5 * (r_o * r_o + r_w * r_w);
            const double E1x = 2.0 / 3.0 * (r_w + r_inner - 1 / (1 / r_w + 1 / r_inner));
            const double E2x = 0.5 * (r_w * r_w + r_inner * r_inner);

            const double E3 = H_xm;

            const double E4 = -k_x + H_xm * r_inner;

            const double E5 = -2.0 * r_inner * k_x + H_xm * r_inner * r_inner;

            const double E6 = H_xm * T_v_bulk[i] - (h_xv_v - h_vx_x) * phi_x_v[i];

            const double k_int_w = steel::k(T_w_x[i]);
            const double k_int_x = liquid_sodium::k(T_w_x[i]);
            const double k_bulk_w = steel::k(T_w_bulk[i]);

            const double alpha = 1.0 / (2 * r_o * (E1w - r_w) + r_w * r_w - E2w);
            const double delta = T_x_bulk[i] - T_w_bulk[i] + q_o_w[i] / k_bulk_w * (E1w - r_w) - (E1x - r_w) * (E6 - E3 * T_x_bulk[i]) / (E4 - E1x * E3);
            const double gamma = r_w * r_w + ((E5 - E2x * E3) * (E1x - r_w)) / (E4 - E1x * E3) - E2x;

            ABC[i][5] = (-q_o_w[i] * k_int_w / k_bulk_w +
                2 * k_int_w * (r_o - r_w) * alpha * delta +
                k_int_x * (E6 - E3 * T_x_bulk[i]) / (E4 - E1x * E3))
                /
                (2 * (r_w - r_o) * k_int_w * alpha * gamma +
                    (E5 - E2x * E3) / (E4 - E1x * E3) * k_int_x -
                    2 * r_w * k_int_x);

            ABC[i][2] = alpha * (delta + gamma * ABC[i][5]);

            ABC[i][1] = q_o_w[i] / k_bulk_w - 2 * r_o * ABC[i][2];

            ABC[i][0] = T_w_bulk[i] - E1w * q_o_w[i] / k_bulk_w + (2 * r_o * E1w - E2w) * ABC[i][2]; 

            ABC[i][4] = (E6 - E3 * T_x_bulk[i] - (E5 - E2x * E3) * ABC[i][5]) / (E4 - E1x * E3); 

            ABC[i][3] = T_x_bulk[i] - E1x * ABC[i][4] - E2x * ABC[i][5];

            // Update temperatures at the interfaces
            T_o_w[i] = ABC[i][0] + ABC[i][1] * r_o + ABC[i][2] * r_o * r_o;                         ///< Temperature at the outer wall
            T_w_x[i] = ABC[i][0] + ABC[i][1] * r_w + ABC[i][2] * r_w * r_w;                         ///< Temperature at the wall wick interface
            T_x_v_old[i] = T_x_v[i];                                                                ///< Temperature backup
            T_x_v[i] = ABC[i][3] + ABC[i][4] * r_inner + ABC[i][5] * r_inner * r_inner;             ///< Temperature at the wick vapor interface

            // Update heat fluxes at the interfaces
            if (i <= evaporator_nodes) q_o_w[i] = q_pp_evaporator;                                  ///< Evaporator imposed heat flux
            else if (i >= (N - condenser_nodes)) {

                double conv = h_conv * (T_o_w[i] - T_env);                                          ///< Condenser convective heat flux
                double irr = emissivity * sigma * (std::pow(T_o_w[i], 4) - std::pow(T_env, 4));     ///< Condenser irradiation heat flux

                q_o_w[i] = -(conv + irr);                                                           ///< Heat flux at the outer wall (positive if to the wall)
            }
                                                                                       
            q_w_x_wall[i] = steel::k(T_w_x[i]) * (ABC[i][1] + 2.0 * ABC[i][2] * r_w);               // Heat flux across wall-wick interface (positive if to wick)
            q_w_x_wick[i] = liquid_sodium::k(T_w_x[i]) * (ABC[i][1] + 2.0 * ABC[i][2] * r_w);       // Heat flux across wall-wick interface (positive if to wick)
            q_x_v_wick[i] = liquid_sodium::k(T_x_v[i]) * (ABC[i][4] + 2.0 * ABC[i][5] * r_inner);           // Heat flux across wick-vapor interface (positive if to vapor)
            q_x_v_vapor[i] = vapor_sodium::k(T_x_v[i], p_v[i]) * (ABC[i][4] + 2.0 * ABC[i][5] * r_inner);   // Heat flux across wick-vapor interface (positive if to vapor)

            Q_mass_vapor[i] = Gamma_xv_vapor[i] * h_xv_v;  ///< Volumetric heat source [W/m3] due to evaporation/condensation (to be summed to the vapor)
            Q_mass_wick[i] = -Gamma_xv_wick[i] * h_vx_x;  ///< Volumetric heat source [W/m3] due to evaporation/condensation (to be summed to the wick)
        }

        // Coupling hypotheses: temperature is transferred to the pressure of the sodium vapor
        p_outlet_v = vapor_sodium::P_sat(T_x_v[N - 1]);

        #pragma endregion

        // =======================================================================
        //
        //                           [4. SOLVE VAPOR]
        //
        // =======================================================================

        #pragma region vapor

        /// Evaluates maximum absolute value of the vapor velocity
        const double max_abs_u_v =
            std::abs(*std::max_element(u_x.begin(), u_x.end(),
                [](double a, double b) { return std::abs(a) < std::abs(b); }
            ));

        /// Evaluates maximum density value of the vapor
        const double max_rho_v = *std::max_element(rho_v.begin(), rho_v.end());

        /// Evaluates minimum temperature value of the vapor
        const double min_T_v = *std::min_element(T_v_bulk.begin(), T_v_bulk.end());

        /*std::cout << "Solving vapor! Time elapsed:" << dt * n << "/" << time_total
            << ", max courant number: " << max_abs_u_v * dt / dz
            << ", max reynolds number: " << max_abs_u_v * r_w * max_rho_v / vapor_sodium::mu(min_T_v) << "\n";*/

        /// Backup old variables
        T_old_v = T_v_bulk;
        rho_old_v = rho_v;
        p_old_v = p_v;

        /**
          * Wick vapor coupling hypotheses: the pressure in the last cell of the domain is considered 
          * the saturation pressure at the temperature of the interface.
          */
        p_v[N - 1] = p_outlet_v;

        /// Outer iterations variables reset
        double u_error_v = 1.0;
        int outer_iter_v = 0;

        /// Inner iterations variables initialization
        double p_error_v;
        int inner_iter_v;

        /**
          * @brief Outer iterations. Stop when maximum number of iterations or velocity stops updating.
          */
        while (outer_iter_v < tot_outer_iter_v && u_error_v > outer_tol_v) {

            // =======================================================================
            //
            //                      [MOMENTUM PREDICTOR]
            //
            // =======================================================================

            #pragma region momentum_predictor

            
            for (int i = 1; i < N - 1; i++) {

                const double rho_P = rho_v[i];                              ///< Density [kg/m3] of the central cell
                const double rho_L = rho_v[i - 1];                          ///< Density [kg/m3] of the left cell
                const double rho_R = rho_v[i + 1];                          ///< Density [kg/m3] of the right cell

                const double mu_P = vapor_sodium::mu(T_v_bulk[i]);          ///< Dynamic viscosity [kg/(m s)] of the central cell
                const double mu_L = vapor_sodium::mu(T_v_bulk[i - 1]);      ///< Dynamic viscosity [kg/(m s)] of the left cell
                const double mu_R = vapor_sodium::mu(T_v_bulk[i + 1]);      ///< Dynamic viscosity [kg/(m s)] of the right cell

                const double D_l = 4.0 / 3.0 * 0.5 * (mu_P + mu_L) / dz;    ///< Diffusion coefficient [kg/(m2 s)] of the left face (average)
                const double D_r = 4.0 / 3.0 * 0.5 * (mu_P + mu_R) / dz;    ///< Diffusion coefficient [kg/(m2 s)] of the right face (average)

                /// RC correction for the left face velocity
                const double rhie_chow_l = -(1.0 / bVU[i - 1] + 1.0 / bVU[i]) / (8 * dz) * (p_padded_v[i - 2] - 3 * p_padded_v[i - 1] + 3 * p_padded_v[i] - p_padded_v[i + 1]);
                
                /// RC correction for the right face velocity
                const double rhie_chow_r = -(1.0 / bVU[i + 1] + 1.0 / bVU[i]) / (8 * dz) * (p_padded_v[i - 1] - 3 * p_padded_v[i] + 3 * p_padded_v[i + 1] - p_padded_v[i + 2]);

                const double u_l_face = 0.5 * (u_v[i - 1] + u_v[i]) 
                    + rhie_chow_on_off_v * rhie_chow_l;                     ///< Left face velocity [m/s] (average + RC correction)

                const double u_r_face = 0.5 * (u_v[i] + u_v[i + 1]) 
                    + rhie_chow_on_off_v * rhie_chow_r;                     ///< Right face velocity [m/s] (average + RC correction)

                const double rho_l = (u_l_face >= 0) ? rho_L : rho_P;       ///< Density [kg/m3] of the left face (upwind)
                const double rho_r = (u_r_face >= 0) ? rho_P : rho_R;       ///< Density [kg/m3] of the right face (upwind)

                const double F_l = rho_l * u_l_face;                        ///< Mass flux [kg/(m2 s)] of the left face (upwind)
                const double F_r = rho_r * u_r_face;                        ///< Mass flux [kg/(m2 s)] of the right face (upwind)

                const double Re = u_v[i] * (2 * r_inner) * rho_P / mu_P;    ///< Reynolds number [-]

                const double f = (Re < 1187.4) ? 64 / Re : 0.3164 * std::pow(Re, -0.25);    ///< Friction factor [-] (according to THROHPUT)
                const double F = 0.25 * f * rho_P * std::abs(u_v[i]) / r_inner;             ///< Friction force [kg/(m2 s)]

                aVU[i] = -std::max(F_l, 0.0) - D_l;                                                             ///< [kg/(m2 s)]
                cVU[i] = -std::max(-F_r, 0.0) - D_r;                                                            ///< [kg/(m2 s)]
                bVU[i] = (std::max(F_r, 0.0) + std::max(-F_l, 0.0)) + rho_P * dz / dt + D_l + D_r + F * dz;     ///< [kg/(m2 s)]
                dVU[i] = -0.5 * (p_v[i + 1] - p_v[i - 1]) + rho_P * u_v[i] * dz / dt /* + Su[i] * dz */;        ///< [kg/(m s2)]
            }

            /// Diffusion coefficients for the first and last node to define BCs
            const double D_first = 4.0 / 3.0 * 0.5 * (vapor_sodium::mu(T_v_bulk[0]) + vapor_sodium::mu(T_v_bulk[1])) / dz;
            const double D_last = 4.0 / 3.0 * 0.5 * (vapor_sodium::mu(T_v_bulk[N - 1]) + vapor_sodium::mu(T_v_bulk[N - 2])) / dz;

            /// Velocity BCs needed variables for the first node
            const double u_r_face_first = 0.5 * (u_v[1]);
            const double rho_r_first = (u_r_face_first >= 0) ? rho_v[0] : rho_v[1];
            const double F_r_first = rho_r_first * u_r_face_first;

            /// Velocity BCs needed variables for the last node
            const double u_l_face_last = 0.5 * (u_v[N - 2]);
            const double rho_l_last = (u_l_face_last >= 0) ? rho_v[N - 2] : rho_v[N - 1];
            const double F_l_last = rho_l_last * u_l_face_last;
            
            /// Velocity BCs: zero velocity on the first node
            bVU[0] = std::max(F_r_first, 0.0) + rho_v[0] * dz / dt + 2 * D_first;
            cVU[0] = 0.0; 
            dVU[0] = bVU[0] * u_inlet_v;

            /// Velocity BCs: zero velocity on the last node
            aVU[N - 1] = 0.0;
            bVU[N - 1] = -std::max(-F_l_last, 0.0) + rho_v[N - 1] * dz / dt + 2 * D_last;
            dVU[N - 1] = bVU[N - 1] * u_outlet_v;

            u_v = solveTridiagonal(aVU, bVU, cVU, dVU);

            #pragma endregion

            /// Inner iterations variables reset
            p_error_v = 1.0;
            inner_iter_v = 0;

            /**
              * @brief Inner (PISO) iterations. Stop when maximum number of iterations or velocity stops updating.
              */
            while (inner_iter_v < tot_inner_iter_v && p_error_v > inner_tol_v) {

                // =======================================================================
                //
                //                      [CONTINUITY SATISFACTOR]
                //
                // =======================================================================

                #pragma region continuity_satisfactor

                /// Tridiagonal coefficients for the pressure correction
                std::vector<double> aVP(N, 0.0), 
                                        bVP(N, 0.0), 
                                        cVP(N, 0.0), 
                                        dVP(N, 0.0);

                /**
                 * @brief Calculates the coefficients for the tridiagonal system of the pressure correction p'
                 */

                
                for (int i = 1; i < N - 1; i++) {

                    const double rho_P = rho_v[i];              ///< Density [kg/m3] of the central cell
                    const double rho_L = rho_v[i - 1];          ///< Density [kg/m3] of the left cell
                    const double rho_R = rho_v[i + 1];          ///< Density [kg/m3] of the right cell

                    const double d_l_face = 0.5 * (1.0 / bVU[i - 1] + 1.0 / bVU[i]) / dz;    ///< 1/Ap [(m2 s)/kg] average on left face
                    const double d_r_face = 0.5 * (1.0 / bVU[i] + 1.0 / bVU[i + 1]) / dz;    ///< 1/Ap [(m2 s)/kg] average on right face

                    /// RC correction for the left face velocity
                    const double rhie_chow_l = -d_l_face / 4 * (p_padded_v[i - 2] - 3 * p_padded_v[i - 1] + 3 * p_padded_v[i] - p_padded_v[i + 1]);
                    
                    /// RC correction for the right face velocity
                    const double rhie_chow_r = -d_r_face / 4 * (p_padded_v[i - 1] - 3 * p_padded_v[i] + 3 * p_padded_v[i + 1] - p_padded_v[i + 2]);

                    const double rho_l = 0.5 * (rho_v[i - 1] + rho_v[i]);               ///< Density [kg/m3] on the left face
                    const double E_l = rho_l * d_l_face;                                ///< Diffusion coefficient [s/m] on the left face

                    const double rho_r = 0.5 * (rho_v[i] + rho_v[i + 1]);               ///< Density [kg/m3] on the right face
                    const double E_r = rho_r * d_r_face;                                ///< Diffusion coefficient [s/m] on the right face

                    const double psi_i = 1.0 / (Rv * T_v_bulk[i]);                      ///< Compressibility [kg/J] assuming ideal gas

                    const double u_l_star = 0.5 * (u_v[i - 1] + u_v[i]) +
                        rhie_chow_on_off_v * rhie_chow_l;                   ///< Left face velocity [m/s] (average + RC correction)
                    const double u_r_star = 0.5 * (u_v[i] + u_v[i + 1]) +
                        rhie_chow_on_off_v * rhie_chow_r;                   ///< Right face velocity [m/s] (average + RC correction)

                    const double phi_l_star = (u_l_star > 0.0) ? rho_L * u_l_star : rho_P * u_l_star;  ///< Left face mass flux [kg/(m^2 s)] (upwind discretization)
                    const double phi_r_star = (u_r_star > 0.0) ? rho_P * u_r_star : rho_R * u_r_star;  ///< Right face mass flux [kg/(m^2 s)] (upwind discretization)

                    const double mass_imbalance = (phi_r_star - phi_l_star);        ///< Mass difference of the fluxes across faces [kg/(m2 s)]
                    const double mass_flux_source = Gamma_xv_vapor[i] * dz;               ///< Mass flux [kg/(m2 s)] due to external source (positive if out of the wick)
                    const double change_density = (rho_P - rho_old_v[i]) * dz / dt; ///< Term [kg/(m2 s)] related to the change in density

                    aVP[i] = -E_l;                                                  ///< [s/m]
                    cVP[i] = -E_r;                                                  ///< [s/m]
                    bVP[i] = E_l + E_r + psi_i * dz / dt;                           ///< [s/m]
                    dVP[i] = +mass_flux_source - mass_imbalance - change_density;          ///< [kg/(m^2 s)]

                    
                }

                /// BCs for the correction of pressure: zero gradient at first node
                bVP[0] = 1.0; 
                cVP[0] = -1.0; 
                dVP[0] = 0.0;

                /// BCs for the correction of pressure: zero at first node
                aVP[N - 1] = 0.0;
                bVP[N - 1] = 1.0;  
                dVP[N - 1] = 0.0;

                p_prime_v = solveTridiagonal(aVP, bVP, cVP, dVP);

                #pragma endregion

                // =======================================================================
                //
                //                        [PRESSURE CORRECTOR]
                //
                // =======================================================================

                #pragma region pressure_corrector

                /**
                  * @brief Corrects the pressure with p'
                  */
                p_error_v = 0.0;

                
                for (int i = 0; i < N; i++) {

                    double p_prev = p_v[i];
                    p_v[i] += p_prime_v[i]; // Note that PISO does not require an under-relaxation factor
                    p_storage_v[i + 1] = p_v[i];

                    p_error_v = std::max(p_error_v, std::fabs(p_v[i] - p_prev));
                }

                p_storage_v[0] = p_storage_v[1];
                p_storage_v[N + 1] = p_outlet_v;

                

                #pragma endregion

                // Save density connected to the pressure field of timestep n
                rho_old_v = rho_v;

                // Update density with new p,T, to get the density connected to the pressure field of timestep n+1
                eos_update(rho_v, p_v, T_v_bulk);

                // =======================================================================
                //
                //                        [VELOCITY CORRECTOR]
                //
                // =======================================================================

                #pragma region velocity_corrector

                /**
                  * @brief Corrects the velocity with p' and checks the sonic limit
                  */
                u_error_v = 0.0;

                
                for (int i = 1; i < N - 1; i++) {

                    const double u_prev_v = u_v[i];
                    const double sonic_limit = std::sqrt(gamma * Rv * T_v_bulk[i]);

                    const double calc_velocity = u_v[i] - (p_prime_v[i + 1] - p_prime_v[i - 1]) / (2.0 * dz * bVU[i]);

                    if (calc_velocity < sonic_limit) {

                        u_v[i] = calc_velocity;

                    } else {

                        //std::cout << "Sonic limit reached, limiting velocity" << "\n";
                        u_v[i] = sonic_limit;

                    }

                    u_error_v = std::max(u_error_v, std::fabs(u_v[i] - u_prev_v));
                }

                #pragma endregion

                inner_iter_v++;
            }

            outer_iter_v++;
        }

        // =======================================================================
        //
        //                        [TURBULENCE MODELIZATION]
        //
        // =======================================================================

        #pragma region turbulence_SST

        // TODO: check discretization scheme.

        /**
          * @brief Models the effects of turbulence on thermal conductivity and dynamic viscosity
          */
        if (SST_model_turbulence_on_off == 1) {

            const double sigma_k = 0.85;        ///< Diffusion coefficient for k [-]
            const double sigma_omega = 0.5;     ///< Diffusion coefficient for ω [-]
            const double beta_star = 0.09;      ///< Production limiter coefficient [-]
            const double beta = 0.075;          ///< Dissipation coefficient for ω [-]
            const double alpha = 5.0 / 9.0;     ///< Blending coefficient [-]

            /// Tridiagonal coefficients for k
            std::vector<double> aK(N, 0.0), 
                                    bK(N, 0.0), 
                                    cK(N, 0.0), 
                                    dK(N, 0.0);

            /// Tridiagonal coefficients for omega
            std::vector<double> aW(N, 0.0), 
                                    bW(N, 0.0), 
                                    cW(N, 0.0), 
                                    dW(N, 0.0);

            std::vector<double> dudz(N, 0.0);       ///< Velocity gradient du/dz [1/s]
            std::vector<double> Pk(N, 0.0);         ///< Turbulence production rate term [m2/s3]

            /// Compute velocity gradient and turbulence production term
            for (int i = 1; i < N - 1; i++) {
                dudz[i] = (u_v[i + 1] - u_v[i - 1]) / (2.0 * dz);
                Pk[i] = mu_t[i] * pow(dudz[i], 2.0);
            }

            /**
             * @brief Assemble coefficients for the k-equation (turbulent kinetic energy) in 1D.
             */
            for (int i = 1; i < N - 1; i++) {

                double mu = vapor_sodium::mu(T_v_bulk[i]);
                double mu_eff = mu + mu_t[i];
                double Dw = mu_eff / (sigma_k * dz * dz);
                double De = mu_eff / (sigma_k * dz * dz);

                aK[i] = -Dw;
                cK[i] = -De;
                bK[i] = rho_v[i] / dt + Dw + De + beta_star * rho_v[i] * omega_turb[i];
                dK[i] = rho_v[i] / dt * k_turb[i] + Pk[i];
            }

            /// k BCs, constant value at the inlet
            bK[0] = 1.0; 
            cK[0] = 0.0; 
            dK[0] = k_turb[0];

            /// k BCs, constant value at the outlet
            aK[N - 1] = 0.0; 
            bK[N - 1] = 1.0; 
            dK[N - 1] = k_turb[N - 1];

            k_turb = solveTridiagonal(aK, bK, cK, dK);

            /**
             * @brief Assemble coefficients for the ω-equation (specific dissipation rate) in 1D.
             */
            for (int i = 1; i < N - 1; i++) {

                double mu = vapor_sodium::mu(T_v_bulk[i]);
                double mu_eff = mu + mu_t[i];
                double Dw = mu_eff / (sigma_omega * dz * dz);
                double De = mu_eff / (sigma_omega * dz * dz);

                aW[i] = -Dw;
                cW[i] = -De;
                bW[i] = rho_v[i] / dt + Dw + De + beta * rho_v[i] * omega_turb[i];
                dW[i] = rho_v[i] / dt * omega_turb[i] + alpha * (omega_turb[i] / k_turb[i]) * Pk[i];
            }

            /// w BCs, constant value at the inlet
            bW[0] = 1.0;  
            cW[0] = 0.0;
            dW[0] = omega_turb[0];

            /// w BCs, constant value at the outlet
            aW[N - 1] = 0.0; 
            bW[N - 1] = 1.0; 
            dW[N - 1] = omega_turb[N - 1];

            omega_turb = solveTridiagonal(aW, bW, cW, dW);

            /**
              * @brief Update turbulent viscosity using k/ω and apply limiter.
              */
            for (int i = 0; i < N; i++) {

                double mu = vapor_sodium::mu(T_v_bulk[i]);
                double denom = std::max(omega_turb[i], 1e-6);
                mu_t[i] = rho_v[i] * k_turb[i] / denom;
                mu_t[i] = std::min(mu_t[i], 1000.0 * mu); // Update with limiter
            }
        }

        #pragma endregion

        // =======================================================================
        //
        //                        [TEMPERATURE CALCULATOR]
        //
        // =======================================================================

        #pragma region temperature_calculator

        // Energy equation for T (implicit), upwind convection, central diffusion
        std::vector<double> aVT(N, 0.0), bVT(N, 0.0), cVT(N, 0.0), dVT(N, 0.0);

        
        for (int i = 1; i < N - 1; i++) {

            const double rho_P = rho_v[i];                                          ///< Density [kg/m3] of the central cell
            const double rho_L = rho_v[i - 1];                                      ///< Density [kg/m3] of the left cell
            const double rho_R = rho_v[i + 1];                                      ///< Density [kg/m3] of the right cell

            const double k_cond_P = vapor_sodium::k(T_v_bulk[i], p_v[i]);           ///< Thermal conductivity [W/(m*K)] of the central cell       
            const double k_cond_L = vapor_sodium::k(T_v_bulk[i - 1], p_v[i - 1]);   ///< Thermal conductivity [W/(m*K)] of the left cell
            const double k_cond_R = vapor_sodium::k(T_v_bulk[i + 1], p_v[i + 1]);   ///< Thermal conductivity [W/(m*K)] of the right cell

            const double cp_P = vapor_sodium::cp(T_v_bulk[i]);                      ///< S. h. at constant pressure [J/(kg K)] of the central cell
            const double cp_L = vapor_sodium::cp(T_v_bulk[i - 1]);                  ///< S. h. at constant pressure [J/(kg K)] of the left cell
            const double cp_R = vapor_sodium::cp(T_v_bulk[i + 1]);                  ///< S. h. at constant pressure [J/(kg K)] of the right cell

            const double mu_P = vapor_sodium::mu(T_v_bulk[i]);                      ///< Dynamic viscosity [J/(kg K)] of the central cell

            const double rhoCp_dzdt = rho_old_v[i] * cp_P * dz / dt;                ///< Useful constant [W/(m^2 K)]

            const double keff_P = k_cond_P + SST_model_turbulence_on_off * (mu_t[i] * cp_P / Pr_t);     ///< Effective t. c. [J/(kg K)] of the central cell
            const double keff_L = k_cond_L + SST_model_turbulence_on_off * (mu_t[i - 1] * cp_L / Pr_t); ///< Effective t. c. [J/(kg K)] of the left cell
            const double keff_R = k_cond_R + SST_model_turbulence_on_off * (mu_t[i + 1] * cp_R / Pr_t); ///< Effective t. c. [J/(kg K)] of the right cell

            const double D_l = 0.5 * (keff_P + keff_L) / dz;                        ///< Diffusion coefficient [W/(m^2 K)] of the left face (average)
            const double D_r = 0.5 * (keff_P + keff_R) / dz;                        ///< Diffusion coefficient [W/(m^2 K)] of the right face (average)

            /// RC correction for the left face velocity
            const double rhie_chow_l = -(1.0 / bVU[i - 1] + 1.0 / bVU[i]) / (8 * dz) * (p_padded_v[i - 2] - 3 * p_padded_v[i - 1] + 3 * p_padded_v[i] - p_padded_v[i + 1]);
            
            /// RC correction for the right face velocity 
            const double rhie_chow_r = -(1.0 / bVU[i + 1] + 1.0 / bVU[i]) / (8 * dz) * (p_padded_v[i - 1] - 3 * p_padded_v[i] + 3 * p_padded_v[i + 1] - p_padded_v[i + 2]);

            /// Left face velocity [m/s] (average + RC correction)
            const double u_l_face = 0.5 * (u_v[i - 1] + u_v[i]) + rhie_chow_on_off_v * rhie_chow_l;

            /// Right face velocity [m/s] (average + RC correction)
            const double u_r_face = 0.5 * (u_v[i] + u_v[i + 1]) + rhie_chow_on_off_v * rhie_chow_r;

            const double rho_l = (u_l_face >= 0) ? rho_L : rho_P;   ///< Density [kg/m3] of the left face (upwind)
            const double rho_r = (u_r_face >= 0) ? rho_P : rho_R;   ///< Density [kg/m3] of the right face (upwind)

            // Upwind specific heat
            const double cp_l = (u_l_face >= 0) ? cp_L : cp_P;      ///< S. h. at constant pressure [J/(kg K)] of the left face (upwind)
            const double cp_r = (u_r_face >= 0) ? cp_P : cp_R;      ///< S. h. at constant pressure [J/(kg K)] of the right face (upwind)

            const double Fl = rho_l * u_l_face;                     ///< Mass flux [kg/(m2 s)] of the left face (upwind)       
            const double Fr = rho_r * u_r_face;                     ///< Mass flux [kg/(m2 s)] of the left face (upwind)

            const double C_l = (Fl * cp_l);                         ///< Mass flux [W/(m^2 K)] of the left face (upwind)
            const double C_r = (Fr * cp_r);                         ///< Mass flux [W/(m^2 K)] of the left face (upwind)

            Q_v[i] = 2 * q_x_v_vapor[i] / r_inner + Q_mass_vapor[i];      ///< Vapor volumetric heat source [W/m^3] due to phase change and heat flux at the interface

            const double dpdz_up = (u_v[i] >= 0.0)
                ? u_v[i] * (p_v[i] - p_v[i - 1]) / dz
                : u_v[i] * (p_v[i + 1] - p_v[i]) / dz;

            aVT[i] = -D_l - std::max(C_l, 0.0);                     ///< [W/(m2 K)]
            cVT[i] = -D_r - std::max(-C_r, 0.0);                    ///< [W/(m2 K)]
            bVT[i] = (std::max(C_r, 0.0) + std::max(-C_l, 0.0)) +   
                        D_l + D_r + rhoCp_dzdt;                                     ///< [W/(m2 K)]

            const double pressure_work = (p_v[i] - p_old_v[i]) / dt + dpdz_up;

            const double viscous_dissipation = 0.5 * mu_P * ((u_v[i + 1] - u_v[i]) * (u_v[i + 1] - u_v[i]) + (u_v[i] + u_v[i - 1]) * (u_v[i] + u_v[i - 1])) / (dz * dz);

            dVT[i] = rhoCp_dzdt * T_old_v[i] + 
                        pressure_work * dz +
                        viscous_dissipation * dz +
                        Q_v[i] * dz;                                ///< [W/m2]
        }

        /// Temperature BCs: zero gradient on the first node
        bVT[0] = 1.0; 
        cVT[0] = -1.0; 
        dVT[0] = 0.0;

        /// Temperature BCs: zero gradient on the last node
        aVT[N - 1] = -1.0; 
        bVT[N - 1] = 1.0; 
        dVT[N - 1] = 0.0;

        T_v_bulk = solveTridiagonal(aVT, bVT, cVT, dVT);

        // Update density with new p,T
        eos_update(rho_v, p_v, T_v_bulk);

        #pragma endregion

        #pragma endregion

        // =======================================================================
        //
        //                          [4. DRY-OUT LIMIT]
        //
        // =======================================================================

        // TODO Check if the capillary limit is satisfied or not

        // =======================================================================
        //
        //                             [5. OUTPUT]
        //
        // =======================================================================

        #pragma region output

        // Output in file
        //if (true) {

        //    std::cout << "Temperature outer wall: \n";
        //    for (int ix = 0; ix < N; ++ix) {
        //        std::cout << T_o_w[ix] << ", ";
        //    } std::cout << "\n";

        //    std::cout << "Temperature bulk wall: \n";
        //    for (int ix = 0; ix < N; ++ix) {
        //        std::cout << T_w_bulk[ix] << ", ";
        //    } std::cout << "\n";

        //    std::cout << "Temperature wall-wick: \n";
        //    for (int ix = 0; ix < N; ++ix) {
        //        std::cout << T_w_x[ix] << ", ";
        //    } std::cout << "\n";

        //    std::cout << "Temperature wick bulk: \n";
        //    for (int ix = 0; ix < N; ++ix) {
        //        std::cout << T_x_bulk[ix] << ", ";
        //    } std::cout << "\n";

        //    std::cout << "Temperature wick vapor: \n";
        //    for (int ix = 0; ix < N; ++ix) {
        //        std::cout << T_x_v[ix] << ", ";
        //    } std::cout << "\n";

        //    std::cout << "Temperature vapor bulk: \n";
        //    for (int ix = 0; ix < N; ++ix) {
        //        std::cout << T_v_bulk[ix] << ", ";
        //    } std::cout << "\n";

        //    std::cout << "Pressure liquid: \n";
        //    for (int ix = 0; ix < N; ++ix) {
        //        std::cout << p_x[ix] << ", ";
        //    } std::cout << "\n";

        //    std::cout << "Velocity liquid: \n";
        //    for (int ix = 0; ix < N; ++ix) {
        //        std::cout << u_x[ix] << ", ";
        //    } std::cout << "\n";

        //    std::cout << "Pressure vapor: \n";
        //    for (int ix = 0; ix < N; ++ix) {
        //        std::cout << p_v[ix] << ", ";
        //    } std::cout << "\n";

        //    std::cout << "Velocity vapor: \n";
        //    for (int ix = 0; ix < N; ++ix) {
        //        std::cout << u_v[ix] << ", ";
        //    } std::cout << "\n\n";

        //    std::cout << "Mass flux: \n";
        //    for (int ix = 0; ix < N; ++ix) {
        //        std::cout << Gamma_xv[ix] << ", ";
        //    } std::cout << "\n\n";
        //}

        if(n % 10000 == 0){
            for (int i = 0; i < N; ++i) {

                v_velocity_output << u_v[i] << " ";
                v_pressure_output << p_v[i] << " ";
                v_bulk_temperature_output << T_v_bulk[i] << " ";

                x_velocity_output << u_x[i] << " ";
                x_pressure_output << p_x[i] << " ";
                x_bulk_temperature_output << T_x_bulk[i] << " ";

                x_v_temperature_output << T_x_v[i] << " ";
                w_x_temperature_output << T_w_x[i] << " ";
                o_w_temperature_output << T_o_w[i] << " ";
                w_bulk_temperature_output << T_w_bulk[i] << " ";

                x_v_mass_flux_output << phi_x_v[i] << " ";

                o_w_heat_flux_output << q_o_w[i] << " ";
                w_x_heat_flux_output << q_w_x_wall[i] << " ";
                x_v_heat_flux_output << q_w_x_wick[i] << " ";

                rho_output << rho_v[i] << " ";
            }

            v_velocity_output << "\n";
            v_pressure_output << "\n";
            v_bulk_temperature_output << "\n";

            x_velocity_output << "\n";
            x_pressure_output << "\n";
            x_bulk_temperature_output << "\n";

            x_v_temperature_output << "\n";
            w_x_temperature_output << "\n";
            o_w_temperature_output << "\n";
            w_bulk_temperature_output << "\n";

            x_v_mass_flux_output << "\n";

            o_w_heat_flux_output << "\n";
            w_x_heat_flux_output << "\n";
            x_v_heat_flux_output << "\n";

            rho_output << "\n";

            v_velocity_output.flush();
            v_pressure_output.flush();
            v_bulk_temperature_output.flush();

            x_velocity_output.flush();
            x_pressure_output.flush();
            x_bulk_temperature_output.flush();

            x_v_temperature_output.flush();
            w_x_temperature_output.flush();
            o_w_temperature_output.flush();
            w_bulk_temperature_output.flush();

            x_v_mass_flux_output.flush();

            o_w_heat_flux_output.flush();
            w_x_heat_flux_output.flush();
            x_v_heat_flux_output.flush();

            rho_output.flush();
        }

        /*system("python plot_data.py mesh.txt "
            "results/vapor_velocity.txt "
            "results/vapor_bulk_temperature.txt "
            "results/vapor_pressure.txt "
            "results/wick_velocity.txt "
            "results/wick_bulk_temperature.txt "
            "results/wick_pressure.txt "
            "results/wall_bulk_temperature.txt "
            "results/outer_wall_temperature.txt "
            "results/wall_wick_interface_temperature.txt "
            "results/wick_vapor_interface_temperature.txt "
            "results/outer_wall_heat_flux.txt "
            "results/wall_wick_heat_flux.txt "
            "results/wick_vapor_heat_flux.txt "
            "results/wick_vapor_mass_source.txt "
            "results/rho_vapor.txt");*/

        #pragma endregion
    }

    mesh_io.close();

    v_velocity_output.close();
    v_pressure_output.close();
    v_bulk_temperature_output.close();

    x_velocity_output.close();
    x_pressure_output.close();
    x_bulk_temperature_output.close();

    x_v_temperature_output.close();
    w_x_temperature_output.close();
    o_w_temperature_output.close();
    w_bulk_temperature_output.close();

    x_v_mass_flux_output.close();

    o_w_heat_flux_output.close();
    w_x_heat_flux_output.close();
    x_v_heat_flux_output.close();

    rho_output.close();

    double end = omp_get_wtime();
    printf("Execution time: %.6f s\n", end - start);

    return 0;
}