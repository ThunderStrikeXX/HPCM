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

// =======================================================================
//
//                        [SOLVING ALGORITHMS]
//
// =======================================================================

#pragma region solving_algorithms
/**
 * @brief Solves a tridiagonal system of linear equations A*x = d using the Thomas Algorithm (TDMA).
 *
 * This function efficiently solves a system where the coefficient matrix A is tridiagonal,
 * meaning it only has non-zero elements on the main diagonal, the sub-diagonal, and the super-diagonal.
 * The system is defined by:
 * - 'a': The sub-diagonal (below the main diagonal). a[0] is typically unused.
 * - 'b': The main diagonal.
 * - 'c': The super-diagonal (above the main diagonal). c[N-1] is typically unused.
 * - 'd': The right-hand side vector.
 *
 * The method consists of two main phases: forward elimination and back substitution,
 * which is optimized for the sparse tridiagonal structure.
 *
 * @param a The sub-diagonal vector (size N, with a[0] often being zero/unused).
 * @param b The main diagonal vector (size N). Must contain non-zero elements.
 * @param c The super-diagonal vector (size N, with c[N-1] often being zero/unused).
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

using Vec6 = std::array<double, 6>;
using Mat6x6 = std::array<std::array<double, 6>, 6>;

/**
 * @brief Solves a 6x6 system of linear equations A*x = b using Gaussian elimination.
 *
 * This function applies Gaussian elimination with partial pivoting to transform the
 * augmented matrix [A|b] into an upper triangular form. It then solves for the
 * unknown vector 'x' using back substitution. The function works only for 6x6
 * systems due to its fixed size implementation.
 *
 * @param A The 6x6 coefficient matrix. This matrix is modified in-place
 * (transformed into an upper triangular matrix).
 * @param b The 6-element right-hand side vector. This vector is also modified
 * in-place to reflect the operations on A.
 * @return Vec6 The solution vector 'x' for the system A*x = b.
 * @throws std::runtime_error If the matrix A is singular (or near-singular,
 * meaning a pivot element is zero), preventing a unique solution.
 *
 * @note This implementation uses **partial pivoting** to ensure numerical stability
 * by selecting the largest absolute element in the current column below the
 * pivot as the pivot element.
 */
static Vec6 solve6(Mat6x6 A, Vec6 b) {

    const int N = 6;

    for (int k = 0; k < N; ++k) {

        int piv = k;
        double mx = std::abs(A[k][k]);

        for (int i = k + 1; i < N; ++i) {
            double v = std::abs(A[i][k]);
            if (v > mx) { mx = v; piv = i; }
        }

        if (mx == 0.0) { throw std::runtime_error("Singular matrix"); }

        if (piv != k) {
            std::swap(A[piv], A[k]);
            std::swap(b[piv], b[k]);
        }

        double diag = A[k][k];

        for (int j = k; j < N; ++j) A[k][j] /= diag;

        b[k] /= diag;

        for (int i = k + 1; i < N; ++i) {
            double f = A[i][k];
            if (f == 0.0) continue;
            for (int j = k; j < N; ++j) A[i][j] -= f * A[k][j];
            b[i] -= f * b[k];
        }
    }

    Vec6 x{};
    for (int i = N - 1; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < N; ++j) s -= A[i][j] * x[j];
        x[i] = s;
    }

    return x;
}

#pragma endregion

// =======================================================================
//
//                       [MATERIAL PROPERTIES]
//
// =======================================================================

#pragma region steel_properties

/**
 * @brief Provides material properties for a specific type of steel.
 *
 * This namespace contains constant lookup tables and helper functions to retrieve
 * temperature-dependent thermodynamic properties of steel, specifically:
 * - Specific Heat Capacity (Cp)
 * - Density (rho)
 * - Thermal Conductivity (k)
 *
 * All functions accept temperature in **Kelvin [K]** and return values in
 * standard SI units.
 */
namespace steel {

    // Temperature [K]
    constexpr std::array<double, 15> T = { 300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700 };

    // Specific heat [J kg^-1 K^-1]
    constexpr std::array<double, 15> Cp_J_kgK = { 510.0296,523.4184,536.8072,550.1960,564.0032,577.3920,590.7808,604.1696,617.5584,631.3656,644.7544,658.1432,671.5320,685.3392,698.7280 };

    // Specific heat interpolation in temperature with complexity O(1)
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

    // Density [kg/m^3]
    double rho(double T) { return (7.9841 - 2.6560e-4 * T - 1.158e-7 * T * T) * 1e3; }

    // Thermal conductivity [W/(m·K)]
    double k(double T) { return (3.116e-2 + 1.618e-4 * T) * 100.0; }
}

#pragma endregion

#pragma region liquid_sodium_properties

#pragma region liquid_sodium_properties

/**
 * @brief Provides thermophysical properties for Liquid Sodium (Na).
 *
 * This namespace contains constant data and functions to calculate key
 * temperature-dependent properties of liquid sodium, which is commonly used
 * as a coolant in fast breeder reactors.
 * * All functions accept temperature T in **Kelvin [K]** and return values
 * in standard SI units.
 */
namespace liquid_sodium {

    // Critical temperature [K]
    constexpr double Tcrit = 2509.46;

    // Solidification temperature, gives warning if below
    constexpr double Tsolid = 370.87;

    // Density [kg/m^3]
    double rho(double T) {

        if (T < Tsolid) std::cout << "Warning, temperature " << T << " is below solidification temperature!";
        return 219.0 + 275.32 * (1.0 - T / Tcrit) + 511.58 * pow(1.0 - T / Tcrit, 0.5);
    }

    // Thermal conductivity [W/(m·K)]
    double k(double T) {

        if (T < Tsolid) std::cout << "Warning, temperature " << T << " is below solidification temperature!";
        return 124.67 - 0.11381 * T + 5.5226e-5 * T * T - 1.1842e-8 * T * T * T;
    }

    // Specific heat [J/(kg·K)]
    double cp(double T) {

        if (T < Tsolid) std::cout << "Warning, temperature " << T << " is below solidification temperature!";
        double dXT = T - 273.15;
        return 1436.72 - 0.58 * dXT + 4.627e-4 * dXT * dXT;
    }

    // Dynamic viscosity [Pa·s] using Shpilrain et al. correlation, valid for 371 K < T < 2500 K
    double mu(double T) {

        if (T < Tsolid) std::cout << "Warning, temperature " << T << " is below solidification temperature!";
        return std::exp(-6.4406 - 0.3958 * std::log(T) + 556.835 / T);
    }
}

#pragma endregion

#pragma region vapor_sodium_properties

/**
 * @brief Provides thermophysical and transport properties for Saturated Sodium Vapor.
 *
 * This namespace contains constant data and functions to calculate key properties
 * of sodium vapor, particularly focusing on its behavior near the saturation curve
 * and critical region. It includes functions for thermodynamic properties and
 * flow/heat transfer correlations.
 *
 * All functions primarily accept temperature T in **Kelvin [K]** and return values
 * in standard SI units unless otherwise noted.
 */

namespace vapor_sodium {

    constexpr double Tcrit_Na = 2509.46;           // Critical temperature [K]
    constexpr double Ad_Na = 3.46;                 // Adiabatic factor [-]
    constexpr double m_g_Na = 23e-3;               // Molar mass [kg/mol]

    // Functions that clamps a value x to the range [a, b]
    inline double clamp(double x, double a, double b) { return std::max(a, std::min(x, b)); }

    // 1D table interpolation in T over monotone grid
    template<size_t N>
    double interp_T(const std::array<double, N>& Tgrid, const std::array<double, N>& Ygrid, double T) {

        if (T <= Tgrid.front()) return Ygrid.front();
        if (T >= Tgrid.back())  return Ygrid.back();

        // locate interval
        size_t i = 0;
        while (i + 1 < N && !(Tgrid[i] <= T && T <= Tgrid[i + 1])) ++i;

        return Ygrid[i] + (T - Tgrid[i]) / (Tgrid[i + 1] - Tgrid[i]) * (Ygrid[i + 1] - Ygrid[i]);
    }

    // Enthalpy of vaporization [J/kg]
    inline double h_vap(double T) {

        const double r = 1.0 - T / Tcrit_Na;
        return (393.37 * r + 4398.6 * std::pow(r, 0.29302)) * 1e3;
    }

    // Saturation pressure [Pa]
    inline double P_sat(double T) {

        const double val_MPa = std::exp(11.9463 - 12633.7 / T - 0.4672 * std::log(T));
        return val_MPa * 1e6;
    }

    // Derivative of saturation pressure with respect to temperature [Pa/K]
    inline double dP_sat_dVT(double T) {

        const double val_MPa_per_K =
            (12633.73 / (T * T) - 0.4672 / T) * std::exp(11.9463 - 12633.73 / T - 0.4672 * std::log(T));
        return val_MPa_per_K * 1e6;
    }

    // Density of saturated vapor [kg/m^3]
    inline double rho(double T) {

        const double hv = h_vap(T);                         // [J/kg]
        const double dPdVT = dP_sat_dVT(T);                   // [Pa/K]
        const double rhol = liquid_sodium::rho(T);          // [kg/m^3]
        const double denom = hv / (T * dPdVT) + 1.0 / rhol;
        return 1.0 / denom;                                 // [kg/m^3]
    }

    // Specific heats from table interpolation [J/kg/K]
    inline double cp(double T) {

        static const std::array<double, 21> Tgrid = { 400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400 };
        static const std::array<double, 21> Cpgrid = { 860,1250,1800,2280,2590,2720,2700,2620,2510,2430,2390,2360,2340,2410,2460,2530,2660,2910,3400,4470,8030 };

        // Table also lists 2500 K = 417030; extreme near critical. If needed, extend:
        if (T >= 2500.0) return 417030.0;

        return interp_T(Tgrid, Cpgrid, T);
    }

    inline double cv(double T) {

        static const std::array<double, 21> Tgrid = { 400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400 };

        static const std::array<double, 21> Cvgrid = { 490, 840,1310,1710,1930,1980,1920,1810,1680,1580,1510,1440,1390,1380,1360,1330,1300,1300,1340,1440,1760 };

        // Table also lists 2500 K = 17030; extreme near critical. If needed, extend:
        if (T >= 2500.0) return 17030.0;

        return interp_T(Tgrid, Cvgrid, T);
    }

    // Dynamic viscosity of sodium vapor [Pa·s]
    inline double mu(double T) {
        return 6.083e-9 * T + 1.2606e-5;

    }

    /**
     * @brief Calculates the Thermal Conductivity (k) of sodium vapor over an extended range.
     *
     * Performs bilinear interpolation inside the experimental grid.
     * Outside 900–1500 K or 981–98066 Pa, it extrapolates using kinetic-gas scaling (k ∝ sqrt(T))
     * referenced to the nearest boundary. Prints warnings when extrapolating.
     *
     * @param T Temperature [K]
     * @param P Pressure [Pa]
     * @return double Thermal conductivity [W/(m·K)]
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

        // local clamp
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

        // bilinear interpolation
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

        // extrapolation handling
        if (Tlow || Thigh || Plow || Phigh) {
            if (Tlow)
                std::cerr << "[Warning] Sodium vapor k(): T=" << T << " < " << Tmin << " K. Using sqrt(T) extrapolation.\n";
            if (Thigh)
                std::cerr << "[Warning] Sodium vapor k(): T=" << T << " > " << Tmax << " K. Using sqrt(T) extrapolation.\n";
            if (Plow || Phigh)
                std::cerr << "[Warning] Sodium vapor k(): P outside ["
                << Pmin << "," << Pmax << "] Pa. Using constant-P approximation.\n";

            double Tref = (Tlow ? Tmin : (Thigh ? Tmax : Tc));
            double k_ref = k_interp;
            double k_extrap = k_ref * std::sqrt(T / Tref);
            return k_extrap;
        }

        return k_interp;
    }


    // Friction factor [-] (Gnielinski correlation)
    inline double f(double Re) {

        if (Re <= 0.0) throw std::invalid_argument("Error: Re < 0");

        const double t = 0.79 * std::log(Re) - 1.64;
        return 1.0 / (t * t);
    }

    // Nusselt number [-] (Gnielinski correlation)
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

    // Convective heat transfer coefficient [W/m^2/K] (Gnielinski correlation)
    inline double h_conv(double Re, double Pr, double k, double Dh) {
        if (k <= 0.0 || Dh <= 0.0) throw std::invalid_argument("k, Dh > 0");
        const double Nu = vapor_sodium::Nu(Re, Pr);
        return Nu * k / Dh;
    }
}

#pragma endregion

//-------------------------------------------------------------
// Simplified 1D transient coupling between wall, wick, and vapor
//-------------------------------------------------------------

int main() {

    // =======================================================================
    //
    //               [CONSTANTS AND VARIABLES DEFINITIONS]
    //
    // =======================================================================

    #pragma region constants_and_variables

    // Mathematical constants
    const double M_PI = 3.14159265358979323846;

    // Physical properties
    const double emissivity = 0.9;          // Wall emissivity [-]
    const double sigma = 5.67e-8;           // Stefan-Boltzmann constant [W/m^2/K^4]
    const double Rv = 361.8;                // Gas constant for the sodium vapor [J/(kg K)]
    const double Pr_t = 0.01;                     // Prandtl turbulent number for sodium vapor [-]

    // Environmental boundary conditions
    const double h_conv = 1000;             // Convective heat transfer coefficient for external heat removal [W/m^2/K]
    const double q_pp_evaporator = 1e6;     // Heat flux at evaporator [W/m^2]
    const double T_env = 280.0;             // External environmental temperature [K]

    // Geometric parameters
    const int N = 5;                                                            // Number of axial nodes [-]
    const double L = 0.982; 			                                        // Length of the heat pipe [m]
    const double dz = L / N;                                                    // Axial discretization step [m]
    const double evaporator_length = 0.502;                                     // Evaporator length [m]
    const double adiabatic_length = 0.188;                                      // Adiabatic length [m]
    const double condenser_length = 0.292;                                      // Condenser length [m]
    const double evaporator_nodes = std::floor(evaporator_length / dz);         // Number of evaporator nodes
    const double condenser_nodes = std::ceil(condenser_length / dz);            // Number of condenser nodes
    const double adiabatic_nodes = N - (evaporator_nodes + condenser_nodes);    // Number of adiabatic nodes
    const double r_outer = 0.01335;                                             // Outer wall radius [m]
    const double r_interface = 0.0112;                                          // Wall-wick interface radius [m]
    const double r_inner = 0.01075;                                             //  Vapor-wick interface radius [m]

    // Surfaces 
    const double A_w_outer = 2 * M_PI * r_outer * dz;                                       // Wall radial area (at r_outer) [m^2]
    const double A_w_cross = M_PI * (r_outer * r_outer - r_interface * r_interface);        // Wall cross-sectional area [m^2]
    const double A_x_interface = 2 * M_PI * r_interface * dz;                               // Wick radial area (at r_interface) [m^2]
    const double A_x_cross = M_PI * (r_interface * r_interface - r_inner * r_inner);        // Wick cross-sectional area [m^2]
    const double A_v_inner = 2 * M_PI * r_inner * dz;                                       // Vapor radial area (at r_inner) [m^2]
    const double A_v_cross = M_PI * r_inner * r_inner;                                      // Vapor cross-sectional area [m^2]

    // Time-stepping parameters
    const double dt = 1e-5;                         // Time step [s]
    const int nSteps = 50;                         // Number of timesteps
    const double time_total = nSteps * dt;          // Total simulation time [s]

    // Wick permeability parameters
    const double K = 1e-4;                          // Permeability [m^2]
    const double CF = 0.0;                          // Forchheimer coefficient [1/m]

    // SIMPLE parameters for the wick
    const int tot_x_iter = 1000;                   // Maximum number of SIMPLE iterations
    const double tol_x = 1e-8;                 // Convergence tolerance on the velocity field

    // Numerical and relaxation parameters for the wick
    const double p_x_relax = 0.3;                     // Pressure relaxation factor
    const double v_x_relax = 1.0;                     // Velocity relaxation factor

    // PISO parameters for the vapor
    const int tot_v_iter = 1000;       // Inner iterations per step [-]
    const int corr_v_iter = 2;        // PISO correctors per iteration [-]
    const double tol_v = 1e-8;        // Tolerance for the inner iterations [-]

    // --- Initial temperature fields ---
    // Node partition
    const int N_e = static_cast<int>(std::floor(evaporator_length / dz));
    const int N_c = static_cast<int>(std::ceil(condenser_length / dz));
    const int N_a = N - (N_e + N_c);

    // Representative boundary temperatures (K)
    const double T_evap_hot = 1200.0;
    const double T_cond_cold = 650.0;
    const double T_init = 1000;

    // Derived temperature drops
    const double dT_total = T_evap_hot - T_cond_cold;
    const double dT_e = 0.05 * dT_total;   // evaporator axial drop
    const double dT_a = 0.02 * dT_total;   // adiabatic axial drop

    // Offsets between media (K)
    const double d_wall_wick = 10.0;   // wall–wick
    const double d_wick_vap = 5.0;    // wick–vapor
    const double d_wall_outer = 30.0;  // wall bulk–outer

    // Axial temperature profile generator
    auto T_axial = [&](int i) {
        if (i < N_e)
            return T_evap_hot - dT_e * (double(i) / N_e);
        else if (i < N_e + N_a)
            return T_evap_hot - dT_e - dT_a * (double(i - N_e) / N_a);
        else
            return T_evap_hot - dT_e - dT_a -
            ((T_evap_hot - dT_e - dT_a - T_cond_cold) *
                (double(i - N_e - N_a) / N_c));
        };

    // Allocate vectors
    std::vector<double> T_w_outer(N);
    std::vector<double> T_w_bulk(N);
    std::vector<double> T_w_x(N);
    std::vector<double> T_x_bulk(N);
    std::vector<double> T_x_v(N);
    std::vector<double> T_v_bulk(N);
    std::vector<double> T_x_v_old(N);

    // Initialize temperatures
    for (int i = 0; i < N; ++i) {
        double Tbase = T_axial(i);       // wick–vapor interface baseline
        T_x_v[i] = Tbase;
        T_x_v_old[i] = T_x_v[i];
        T_x_bulk[i] = Tbase + 0.5 * d_wick_vap;
        T_w_x[i] = Tbase + d_wick_vap;
        T_w_bulk[i] = T_w_x[i] + 0.5 * d_wall_wick;
        T_w_outer[i] = T_w_bulk[i] - d_wall_outer;
        T_v_bulk[i] = T_x_v[i] - 0.5 * d_wick_vap;
    }

    // Liquid initial conditions
    std::vector<double> u_x(N, -0.01), p_x(N, vapor_sodium::P_sat(T_x_v[N - 1])), p_prime_x(N, 0.0);
    std::vector<double> T_old_x(N, T_init - 200), p_old_x(N, vapor_sodium::P_sat(T_x_v[N - 1]));        // Backup values
    std::vector<double> p_storage_x(N + 2, vapor_sodium::P_sat(T_x_v[N - 1]));                          // Storage for ghost nodes at the boundaries
    double* p_padded_x = &p_storage_x[1];                                                               // Poìnter to work on the storage with the same indes
    

    // Vapor inital conditions
    std::vector<double> u_v(N, 0.1), p_v(N, vapor_sodium::P_sat(T_x_v[N - 1])), rho_v(N, 8e-3), p_prime_v(N, 0.0);
    std::vector<double> p_old_v(N, vapor_sodium::P_sat(T_x_v[N - 1])), T_old_v(N, T_init), rho_old_v(N, 8e-3), u_old_v(N, 0.1);
    std::vector<double> p_storage_v(N + 2, vapor_sodium::P_sat(T_x_v[N - 1]));                          // Storage for ghost nodes aVT the boundaries
    double* p_padded_v = &p_storage_v[1];                                                               // Poìnter to work on the storage with the same indes


    // Liquid boundary conditions (Dirichlet u at inlet, p at outlet, T at both ends)
    const double u_inlet_x = 0.01;                              // Inlet velocity [m/s]
    const double u_outlet_x = 0.0;                              // Outlet velocity [m/s]
    double p_outlet_x = vapor_sodium::P_sat(T_x_v[N - 1]);      // Outlet pressure [Pa]

    // Boundary conditions for the vapor phase
    const double u_inlet_v = 0.0;      // Inlet velocity [m/s]
    const double u_outlet_v = 0.0;      // Inlet velocity [m/s]
    double p_outlet_v = vapor_sodium::P_sat(T_x_v[N - 1]);  // Outlet pressure [Pa]

    // Turbulence constants for sodium vapor (SST model)
    const double I = 0.05;                          // Turbulence intensity (5%)
    const double L_t = 0.07 * L;                    // Turbulence length scale
    const double k0 = 1.5 * pow(I * u_inlet_v, 2);        // Initial turbulent kinetic energy
    const double omega0 = sqrt(k0) / (0.09 * L_t);        // Initial specific dissipation
    const double sigma_k = 0.85;
    const double sigma_omega = 0.5;
    const double beta_star = 0.09;
    const double beta = 0.075;
    const double alpha = 5.0 / 9.0;

    // Turbulence fields for sodium vapor initialization
    std::vector<double> k_turb(N, k0);
    std::vector<double> omega_turb(N, omega0);
    std::vector<double> mu_t(N, 0.0);

    // Vapor Equation of State update function
    auto eos_update = [&](std::vector<double>& rho_, const std::vector<double>& p_, const std::vector<double>& T_) {

        // #pragma omp parallel
        for (int i = 0; i < N; i++) { rho_[i] = std::max(1e-6, p_[i] / (Rv * T_v_bulk[i])); }

        }; eos_update(rho_v, p_v, T_v_bulk);

    // Mass transfer parameters (three coefficients per node)
    std::vector<double> aGamma(N, 0.0), bGamma(N, 0.0), cGamma(N, 0.0);
    std::vector<double> aGammaOld(N, 0.0), bGammaOld(N, 0.0), cGammaOld(N, 0.0);

    // Heat fluxes at the interfaces [W/m^2]
    std::vector<double> q_o_w(N, 0.0);           // Heat flux across outer wall (directed to wall)
    std::vector<double> q_w_x_wall(N, 0.0);      // Heat flux across wall-wick interface (directed to wick)
    std::vector<double> q_w_x_wick(N, 0.0);      // Heat flux across wall-wick interface (directed to wick)
    std::vector<double> q_x_v_wick(N, 0.0);      // Heat flux across wick-vapor interface (directed to vapor)
    std::vector<double> q_x_v_vapor(N, 0.0);     // Heat flux across wick-vapor interface (directed to vapor)

    // Evaporation mass flux at the wick-vapor interface [kg/m2/s]
    std::vector<double> m_dot_x_v(N, 0.0);

    // Models
    const int rhie_chow_on_off_x = 1;                 // 0: no RC correction, 1: with RC correction
    const int rhie_chow_on_off_v = 1;                 // 0: no RC correction, 1: with RC correction
    const int SST_model_turbulence_on_off = 0;        // 0: no turbulence, 1: with turbulence

    // Print results in file
    std::ofstream fout("solution.txt");
    fout << std::setprecision(8);

    // Print number of working threads
    std::cout << "Threads: " << omp_get_max_threads() << "\n";

    std::vector<double> Gamma_xg_new(N, 0.0);

    // The coefficient bVU is needed in momentum predictor loop and pressure correction to estimate the velocities aVT the faces using the Rhie and Chow correction
    std::vector<double> aVU(N, 0.0), bVU(N, 2 * (4.0 / 3.0 * vapor_sodium::mu(T_init) / dz) + dz / dt * rho_v[0]), cVU(N, 0.0), dVU(N, 0.0);

    // The coefficient bU is needed in momentum predictor loop and pressure correction to estimate the velocities at the faces using the Rhie and Chow correction
    std::vector<double> aXU(N, 0.0), bXU(N, liquid_sodium::rho(T_init) * dz / dt + 2 * liquid_sodium::mu(T_init) / dz), cXU(N, 0.0), dXU(N, 0.0);

    #pragma endregion

    // Time-stepping loop
    for (int n = 0; n < nSteps; ++n) {

        // =======================================================================
        //
        //                        [1. SOLVE WALL TEMPERATURE]
        //
        // =======================================================================

        #pragma region wall_conduction

        // Initialization of the coefficients for tridiagonal solver
        std::vector<double> aTW(N, 0.0), bTW(N, 0.0), cTW(N, 0.0), dTW(N, 0.0);

        // Boundary conditions for tridiagonal solver (Neumann BCs)
        aTW[0] = 0.0; bTW[0] = 1.0; cTW[0] = -1.0; dTW[0] = 0.0;
        aTW[N - 1] = -1.0; bTW[N - 1] = 1.0; cTW[N - 1] = 0.0; dTW[N - 1] = 0.0;

        for (int ix = 1; ix < N - 1; ++ix) {

            const double Cp_wall_node = steel::cp(T_w_bulk[ix]);
            const double k_wall_node = steel::k(T_w_bulk[ix]);
            const double rho_wall_node = steel::rho(T_w_bulk[ix]);

            const double r = (k_wall_node * dt) / (Cp_wall_node * dz * dz);

            // Volumetric heat source in the wall due to heat flux at the outer wall [W/m^3]
            const double volum_heat_source_o_w = q_o_w[ix] * 2 * r_outer / (r_outer * r_outer - r_interface * r_interface);

            // Volumetric heat source in the wall due to heat flux at the wall-wick interface [W/m^3]
            const double volum_heat_source_w_x = q_w_x_wall[ix] * 2 * r_interface / (r_outer * r_outer - r_interface * r_interface);

            aTW[ix] = -r;
            bTW[ix] = 1 + 2 * r;
            cTW[ix] = -r;
            dTW[ix] = T_w_bulk[ix] +
                volum_heat_source_o_w * dt / (rho_wall_node * Cp_wall_node) -
                volum_heat_source_w_x * dt / (rho_wall_node * Cp_wall_node);
        }

        T_w_bulk = solveTridiagonal(aTW, bTW, cTW, dTW);

        #pragma endregion

        // =======================================================================
        //
        //                   [2. SOLVE WICK CONDUCTION/CONVECTION]
        //
        // =======================================================================

        #pragma region wick_conduction_convection

        std::cout << "Solving wick, time:" << dt * n << "/" << time_total << ", courant number: " << u_inlet_x * dt / dz << "\n";

        // Loop on SIMPLE iterations
        int iter_x = 0;
        double maxErr_x = 1.0;     // This value is set to enter the loop, then it updates

        while (iter_x < tot_x_iter && maxErr_x > tol_x) {

            // === 1. Momentum predictor ===

            std::vector<double> aUX(N, 0.0), bUX(N, 0.0), cUX(N, 0.0), dUX(N, 0.0), b_SIMPLEC(N, 0.0);;

            // #pragma omp parallel
            for (int ix = 1; ix < N - 1; ix++) {

                const double rho_node = liquid_sodium::rho(T_x_bulk[ix]);
                const double mu_node = liquid_sodium::mu(T_x_bulk[ix]);

                aUX[ix] = -mu_node / (dz * dz);
                cUX[ix] = -mu_node / (dz * dz);
                bUX[ix] = rho_node / dt + 2 * mu_node / (dz * dz) + rho_node * (u_x[ix] - u_x[ix - 1]) / dz + mu_node / K + CF * rho_node / sqrt(K) * abs(u_x[ix]);
                dUX[ix] = rho_node * u_x[ix] / dt - (p_x[ix + 1] - p_x[ix - 1]) / (2 * dz);
                b_SIMPLEC[ix] = bUX[ix] - (cUX[ix] + aUX[ix]);
            }

            // Velocity boundary conditions in the evaporator side
            bUX[0] = 1;   cUX[0] = 0;   dUX[0] = u_inlet_x;
            double b_0_RC = liquid_sodium::rho(T_x_bulk[0]) / dt + 2 * liquid_sodium::mu(T_x_bulk[0]) / (dz * dz) + liquid_sodium::mu(T_x_bulk[0]) / K + CF * liquid_sodium::rho(T_x_bulk[0]) / sqrt(K) * abs(u_x[0]);
            b_SIMPLEC[0] = liquid_sodium::rho(T_x_bulk[0]) / dt + 2 * liquid_sodium::mu(T_x_bulk[0]) / (dz * dz) + liquid_sodium::mu(T_x_bulk[0]) / K + CF * liquid_sodium::rho(T_x_bulk[0]) / sqrt(K) * abs(u_x[0]);

            // Velocity boundary conditions in the condenser side
            aUX[N - 1] = -1;  bUX[N - 1] = 1;   dUX[N - 1] = 0;
            double b_N_1_RC = liquid_sodium::rho(T_x_bulk[N - 1]) / dt + 3 * liquid_sodium::mu(T_x_bulk[N - 1]) / (dz * dz) + liquid_sodium::mu(T_x_bulk[N - 1]) / K + CF * liquid_sodium::rho(T_x_bulk[N - 1]) / sqrt(K) * abs(u_x[N - 1]);
            b_SIMPLEC[N - 1] = liquid_sodium::rho(T_x_bulk[N - 1]) / dt + 3 * liquid_sodium::mu(T_x_bulk[N - 1]) / (dz * dz) + liquid_sodium::mu(T_x_bulk[N - 1]) / K + CF * liquid_sodium::rho(T_x_bulk[N - 1]) / sqrt(K) * abs(u_x[N - 1]);

            u_x = solveTridiagonal(aUX, bUX, cUX, dUX);

            // === 2. Pressure corrector ===

            std::vector<double> aPX(N, 0.0), bPX(N, 0.0), cPX(N, 0.0), dPX(N, 0.0);

            double rhiechow = 0.0;
            double dudz = 0.0;

            // #pragma omp parallel
            for (int ix = 2; ix < N - 2; ix++) {

                rhiechow = -1 / (8 * dz) * (1 / bUX[ix + 1] - 1 / bUX[ix - 1]) *
                    (-p_x[ix - 2] + 4 * p_x[ix - 1] - 6 * p_x[ix] + 4 * p_x[ix + 1] - p_x[ix + 2]);

                dudz = (u_x[ix + 1] - u_x[ix - 1]) / (2 * dz)
                    + rhiechow_coeff_x * rhiechow;

                aPX[ix] = 1.0 / dz / dz;
                cPX[ix] = 1.0 / dz / dz;
                bPX[ix] = -2.0 / dz / dz;
                dPX[ix] = liquid_sodium::rho(T_x_bulk[ix]) / dt * dudz;
            }

            // Pressure boundary condition in the evaporator side
            cPX[0] = -1;   bPX[0] = 1;    dPX[0] = 0;

            // Pressure boundary condition in the condenser side
            aPX[N - 1] = 0;    bPX[N - 1] = 1;    dPX[N - 1] = 0;

            // Pressure correction without Rhie and Chow for the second node
            aPX[1] = 1.0 / (dz * dz);  cPX[1] = 1.0 / (dz * dz);  bPX[1] = -2.0 / (dz * dz);

            double rhiechow1 = -1 / (8 * dz) * (1 / bUX[2] - 1 / b_0_RC) *
                (-3 * p_x[0] + 4 * p_x[2] - p_x[3]);
            dPX[1] = liquid_sodium::rho(T_x_bulk[1]) / dt * ((u_x[2] - u_x[0]) / (2 * dz) + rhiechow_coeff_x * rhiechow1);

            // Pressure correction with Rhie and Chow for the second-to-last node
            aPX[N - 2] = 1.0 / dz / dz;    cPX[N - 2] = 1.0 / dz / dz;    bPX[N - 2] = -2.0 / dz / dz;

            double rhiechow2 = -1 / (8 * dz) * (1 / b_N_1_RC - 1 / bUX[N - 3]) *
                (-p_x[N - 4] + 4 * p_x[N - 3] - 6 * p_x[N - 2]);
            dPX[N - 2] = liquid_sodium::rho(T_x_bulk[N - 2]) / dt * ((u_x[N - 1] - u_x[N - 3]) / (2 * dz) + rhiechow_coeff_x * rhiechow2);

            p_prime_x = solveTridiagonal(aPX, bPX, cPX, dPX);

            // === 3. Pressure updater ===

            for (int ix = 0; ix < N; ix++) p_x[ix] += p_x_relax * p_prime_x[ix]; // With relaxing factor

            // Error on the velocity field
            maxErr_x = 0.0;

            // === 4. Velocity updater ===

            for (int ix = 1; ix < N - 1; ix++) {

                double u_old = u_x[ix];
                u_x[ix] = u_x[ix] - v_x_relax / b_SIMPLEC[ix] * (p_prime_x[ix + 1] - p_prime_x[ix - 1]) / (2 * dz);

                maxErr_x = std::max(maxErr_x, std::fabs(u_x[ix] - u_old));
            }

            // Boundary condition, zero gradient on right side velocity
            u_x[N - 1] = u_x[N - 2];

            iter_x++;
        }

        // Enforce boundary conditions on pressure
        p_x[N - 1] = p_outlet_x; p_x[0] = p_x[1];

        // === 5. Temperature calculator ===

        std::vector<double> aTX(N, 0.0), bTX(N, 0.0), cTX(N, 0.0), dTX(N, 0.0);

        // #pragma omp parallel
        for (int ix = 1; ix < N - 1; ix++) {

            const double rho_x = liquid_sodium::rho(T_x_bulk[ix]);
            const double cp_x = liquid_sodium::cp(T_x_bulk[ix]);
            const double k_x = liquid_sodium::k(T_x_bulk[ix]);
            const double k_w = steel::k(T_w_bulk[ix]);
            const double alpha_x = k_x / (rho_x * cp_x);

            // Volumetric heat source due to heat flux at the wall-wick interface [W/m^3]
            double volum_heat_source_w_x = q_w_x_wick[ix] * 2 * r_interface / (r_interface * r_interface - r_inner * r_inner);

            // Volumetric heat source due to heat flux at the wick-vapor interface [W/m^3]
            double volum_heat_source_x_v = q_x_v_wick[ix] * 2 * r_inner / (r_interface * r_interface - r_inner * r_inner);

            // Fully implicit discretization 
            aTX[ix] = -alpha_x * dt / (dz * dz) - u_x[ix] * dt / dz;
            cTX[ix] = -alpha_x * dt / (dz * dz);
            bTX[ix] = 1 + u_x[ix] * dt / dz + 2 * dt * alpha_x / (dz * dz);
            dTX[ix] = T_x_bulk[ix] +
                volum_heat_source_w_x * dt / (rho_x * cp_x) -
                volum_heat_source_x_v * dt / (rho_x * cp_x);
        }

        // Boundary conditions at the evaporator side (Neumann BC)
        bTX[0] = 1.0;    cTX[0] = -1.0;   dTX[0] = 0.0;

        // Boundary conditions at the condenser side (Neumann BC)
        aTX[N - 1] = -1.0;   bTX[N - 1] = 1.0;    dTX[N - 1] = 0.0;

        T_x_bulk = solveTridiagonal(aTX, bTX, cTX, dTX);

        #pragma endregion

        // =======================================================================
        //
        //               [0. SOLVE PARABOLIC DISTRIBUTION IN WALL AND WICK]
        //
        // =======================================================================

        #pragma region parabolic_profiles 

        // Temperature distribution coefficients (six coefficients per node)
        std::array<std::array<double, 6>, N> ABC{};

        for (int ix = 0; ix < N; ++ix) {

            Mat6x6 A{};
            Vec6 B{};

            const double eps_s = 0.4;                                             // Surface fraction of the wick available for phasic interface [-]
            const double beta = 1.0 / std::sqrt(2 * M_PI * Rv * T_x_v[ix]);       // Mass transfer parameter beta [sqrt(kg/J)]
            const double sigma_e = 1.0;                                           // Evaporation accomodation coefficient [-]. 1 means optimal evaporation
            const double sigma_c = 1.0;                                           // Condensation accomodation coefficient [-]. 1 means optimal condensation

            m_dot_x_v[ix] = beta * (vapor_sodium::P_sat(T_x_v[ix]) - p_v[ix]);                      // Mass flux from the wick to the vapor [kg/m2/s]

            // const double b = -(vapor_sodium::P_sat(T_x_v[ix]) - p_v[ix]) / p_v[ix];
            const double b = std::abs(-m_dot_x_v[ix] / (p_v[ix] * std::sqrt(2.0 / (Rv * T_v_bulk[ix]))));     // Ratio of the overrall speed to the most probable velocity of the vapor

            // Linearization of the omega function to correct the net evaporation/condensation mass flux
            double Omega;
            if (b < 0.1192) Omega = 1.0 + b * std::sqrt(M_PI);
            else if (b <= 0.9962) Omega = 0.8959 + 2.6457 * b;
            else Omega = 2.0 * b * std::sqrt(M_PI);

            const double k_w = steel::k(T_w_bulk[ix]);                                          // Wall thermal conductivity [W/m/K]
            const double k_x = liquid_sodium::k(T_x_bulk[ix]);                                  // Liquid thermal conductivity [W/m/K]
            const double cp_v = vapor_sodium::cp(T_v_bulk[ix]);                                 // Vapor specific heat [J/kg/K]
            const double k_v_cond = vapor_sodium::k(T_v_bulk[ix], p_v[ix]);                     // Vapor thermal conductivity [W/m/K]
            const double k_v_eff = k_v_cond + mu_t[ix] * cp_v / Pr_t;                           // Effective vapor thermal conductivity [W/m/K]
            const double mu_v = vapor_sodium::mu(T_v_bulk[ix]);                                 // Vapor dynamic viscosity [Pa s]
            const double Dh_v = 2.0 * r_inner;                                                  // Hydraulic diameter of the vapor core [m]
            const double Re_v = rho_v[ix] * std::fabs(u_v[ix]) * Dh_v / mu_v;                   // Reynolds number [-]
            const double Pr_v = cp_v * mu_v / k_v_cond;                                         // Prandtl number [-]
            const double H_xm = vapor_sodium::h_conv(Re_v, Pr_v, k_v_cond, Dh_v);               // Convective heat transfer coefficient at the vapor-wick interface [W/m^2/K]
            const double Psat = vapor_sodium::P_sat(T_w_x[ix]);                                 // Saturation pressure [Pa]         

            const double dPsat_dT = Psat * std::log(10.0) * (7740.0 / (T_w_x[ix] * T_w_x[ix]));

            const double fac = (2.0 * r_inner * eps_s * beta) / (r_interface * r_interface);    // Useful factor in the coefficients calculation
            const double h_xg_x = vapor_sodium::h_vap(T_x_v[ix]);                               // Enthalpy of vaporization at T_x_v [J/kg]

            const double dPg = (p_v[ix] / rho_v[ix]) * (rho_v[ix] - rho_old_v[ix])
                + (p_v[ix] / T_v_bulk[ix]) * (T_v_bulk[ix] - T_old_v[ix]);      // Variation of the vapor pressure due to density and temperature changes [Pa]

            // Calculate old mass transfer rate
            Gamma_xg_new[ix] = fac * (sigma_e * Psat - sigma_c * Omega * p_v[ix]);

            // Coefficients for the linearization of the new mass transfer rate
            bGamma[ix] = -(Gamma_xg_new[ix] / (2.0 * T_x_v[ix])) + fac * sigma_e * dPsat_dT;
            aGamma[ix] = 0.5 * Gamma_xg_new[ix] + fac * sigma_e * dPsat_dT * T_x_v[ix];
            cGamma[ix] = -fac * sigma_c * Omega;

            // Coefficients for the parabolic temperature profiles in wall and wick
            const double Eio1 = 2.0 / 3.0 * (r_outer + r_interface - 1 / (1 / r_outer + 1 / r_interface));
            const double Eio2 = 0.5 * (r_outer * r_outer + r_interface * r_interface);
            const double Evi1 = 2.0 / 3.0 * (r_interface + r_inner - 1 / (1 / r_interface + 1 / r_inner));
            const double Evi2 = 0.5 * (r_interface * r_interface + r_inner * r_inner);

            const double Ex3 = H_xm + (h_xg_x * r_interface * r_interface) / (2.0 * r_inner) * bGamma[ix];

            const double Ex4 =
                -k_x +
                H_xm * r_inner +
                h_xg_x * r_interface * r_interface / 2.0 * bGamma[ix];

            const double Ex5 =
                -2.0 * r_inner * k_x +
                H_xm * r_inner * r_inner +
                h_xg_x * r_interface * r_interface / 2.0 * bGamma[ix] * r_inner;

            const double Ex6 = -H_xm;

            const double Ex7 = (h_xg_x * r_interface * r_interface) / (2.0 * r_inner) * cGamma[ix];

            const double Ex8 = (h_xg_x * r_interface * r_interface) / (2.0 * r_inner) * aGamma[ix];

            // LHS matrix A 6x6
            A[0] = { 1.0, Eio1, Eio2, 0.0, 0.0, 0.0 };
            A[1] = { 0.0, 0.0,    0.0,    1.0, Evi1, Evi2 };
            A[2] = { 1.0, r_interface,  r_interface * r_interface, -1.0, -r_interface, -r_interface * r_interface };
            A[3] = { 0.0, steel::k(T_w_x[ix]),  2.0 * r_interface * steel::k(T_w_x[ix]), 0.0, -liquid_sodium::k(T_w_x[ix]), -2.0 * r_interface * liquid_sodium::k(T_w_x[ix]) };
            A[4] = { 0.0, 1.0,    2.0 * r_outer, 0.0, 0.0, 0.0 };
            A[5] = { 0.0, 0.0,    0.0,       Ex3, Ex4, Ex5 };

            // RHS vector B
            B[0] = T_w_bulk[ix];                                    // row1: [0 1 0 0 0]
            B[1] = T_x_bulk[ix];                                    // row2: [0 0 1 0 0]
            B[2] = 0.0;                                             // row3: [0 0 0 0 0]
            B[3] = 0.0;                                             // row4: [0 0 0 0 0]
            B[4] = q_o_w[ix] / steel::k(T_w_bulk[ix]);              // row5: [q''/kw 0 0 0 0]
            B[5] = Ex6 * T_v_bulk[ix] + Ex7 * dPg + Ex8;                  // row6: [Ex8 0 0 Ex6 Ex7]

            ABC[ix] = solve6(A, B); // Returns [a_w, b_w, c_w, a_x, b_x, c_x]

            // Update temperatures at the interfaces
            T_w_outer[ix] = ABC[ix][0] + ABC[ix][1] * r_outer + ABC[ix][2] * r_outer * r_outer;
            T_w_x[ix] = ABC[ix][0] + ABC[ix][1] * r_interface + ABC[ix][2] * r_interface * r_interface;
            T_x_v_old[ix] = T_x_v[ix];
            T_x_v[ix] = ABC[ix][3] + ABC[ix][4] * r_inner + ABC[ix][5] * r_inner * r_inner;

            // Update heat fluxes at the interfaces
            if (ix <= evaporator_nodes) q_o_w[ix] = q_pp_evaporator;
            else if (ix >= (N - condenser_nodes)) {

                double conv = h_conv * (T_w_outer[ix] - T_env);
                double irr = emissivity * sigma * (std::pow(T_w_outer[ix], 4) - std::pow(T_env, 4));

                q_o_w[ix] = -(conv + irr);
            }                                                                                       // Heat flux at the outer wall (positive if to wall)
            q_w_x_wall[ix] = steel::k(T_w_x[ix]) * (ABC[ix][1] + 2.0 * ABC[ix][2] * r_interface);                        // Heat flux across wall-wick interface (positive if to wick)
            q_w_x_wick[ix] = liquid_sodium::k(T_w_x[ix]) * (ABC[ix][1] + 2.0 * ABC[ix][2] * r_interface);                        // Heat flux across wall-wick interface (positive if to wick)
            q_x_v_wick[ix] = liquid_sodium::k(T_x_v[ix]) * (ABC[ix][4] + 2.0 * ABC[ix][5] * r_inner);                        // Heat flux across wick-vapor interface (positive if to vapor)
            q_x_v_vapor[ix] = vapor_sodium::k(T_x_v[ix], p_v[ix]) * (ABC[ix][4] + 2.0 * ABC[ix][5] * r_inner);                        // Heat flux across wick-vapor interface (positive if to vapor)

            p_outlet_v = vapor_sodium::P_sat(T_x_v[N - 1]);
            p_outlet_x = vapor_sodium::P_sat(T_x_v[N - 1]);

            printf(" ");
        }

        #pragma endregion

        // =======================================================================
        //
        //                   [3. SOLVE VAPOR CONDUCTION/CONVECTION]
        //
        // =======================================================================

        #pragma region vapor_conduction_convection

        const double max_u = *std::max_element(u_v.begin(), u_v.end());
        const double max_rho = *std::max_element(rho_v.begin(), rho_v.end());
        const double min_T = *std::min_element(T_v_bulk.begin(), T_v_bulk.end());

        std::cout << "Solving! Time elapsed:" << dt * n << "/" << time_total
            << ", max courant number: " << max_u * dt / dz
            << ", max reynolds number: " << max_u * 2 * r_inner * max_rho / vapor_sodium::mu(min_T) << "\n";

        // Backup variables
        T_old_v = T_v_bulk;
        rho_old_v = rho_v;
        p_old_v = p_v;

        // PISO iterations
        double maxErr = 1.0;
        int iter = 0;

        while (iter < tot_v_iter && maxErr > tol_v) {

            // =======================================================================
            //
            //                      [MOMENTUM PREDICTOR]
            //
            // =======================================================================

            #pragma region momentum_predictor

            #pragma omp parallel
            for (int i = 1; i < N - 1; i++) {

                const double rho_P = rho_v[i];
                const double rho_L = rho_v[i - 1];
                const double rho_R = rho_v[i + 1];

                const double mu_P = vapor_sodium::mu(T_v_bulk[i]);
                const double mu_L = vapor_sodium::mu(T_v_bulk[i - 1]);
                const double mu_R = vapor_sodium::mu(T_v_bulk[i + 1]);

                const double D_l = 4.0 / 3.0 * 0.5 * (mu_P + mu_L) / dz;
                const double D_r = 4.0 / 3.0 * 0.5 * (mu_P + mu_R) / dz;

                const double rhie_chow_l = -(1.0 / bVU[i - 1] + 1.0 / bVU[i]) / (8 * dz) * (p_padded_v[i - 2] - 3 * p_padded_v[i - 1] + 3 * p_padded_v[i] - p_padded_v[i + 1]);
                const double rhie_chow_r = -(1.0 / bVU[i + 1] + 1.0 / bVU[i]) / (8 * dz) * (p_padded_v[i - 1] - 3 * p_padded_v[i] + 3 * p_padded_v[i + 1] - p_padded_v[i + 2]);

                const double u_l_face = 0.5 * (u_v[i - 1] + u_v[i]) + rhie_chow_on_off_v * rhie_chow_l;
                const double u_r_face = 0.5 * (u_v[i] + u_v[i + 1]) + rhie_chow_on_off_v * rhie_chow_r;

                const double rho_l = (u_l_face >= 0) ? rho_L : rho_P;
                const double rho_r = (u_r_face >= 0) ? rho_P : rho_R;

                const double F_l = rho_l * u_l_face;
                const double F_r = rho_r * u_r_face;

                // Estimate of the friction factor
                const double r_inner = 0.01075;
                const double Re = u_v[i] * (2 * r_inner) * rho_P / mu_P;

                const double f = (Re < 1187.4) ? 64 / Re : 0.3164 * std::pow(Re, -0.25);
                const double F = 0.25 * f * rho_P * std::abs(u_v[i]) / r_inner;

                aVU[i] = -std::max(F_l, 0.0) - D_l;
                cVU[i] = std::max(-F_r, 0.0) - D_r;
                bVU[i] = (std::max(F_r, 0.0) - std::max(-F_l, 0.0)) + rho_P * dz / dt + D_l + D_r + F;
                dVU[i] = -0.5 * (p_v[i + 1] - p_v[i - 1]) + rho_P * u_v[i] * dz / dt + Su[i] * dz;

                printf("");
            }

            // Velocity BC: Dirichlet aVT l, dirichlet aVT r
            const double D_first = 4.0 / 3.0 * vapor_sodium::mu(T_v_bulk[0]) / dz;
            const double D_last = 4.0 / 3.0 * vapor_sodium::mu(T_v_bulk[N - 1]) / dz;

            bVU[0] = rho_v[0] * dz / dt + 2 * D_first; cVU[0] = 0.0; dVU[0] = (rho_v[0] * dz / dt + 2 * D_first) * u_inlet_v;
            aVU[N - 1] = 0.0; bVU[N - 1] = rho_v[N - 1] * dz / dt + 2 * D_last; dVU[N - 1] = (rho_v[N - 1] * dz / dt + 2 * D_last) * u_outlet_v;

            u_v = solveTridiagonal(aVU, bVU, cVU, dVU);

            #pragma endregion

            for (int piso = 0; piso < corr_v_iter; piso++) {

                // =======================================================================
                //
                //                      [CONTINUITY SATISFACTOR]
                //
                // =======================================================================

                #pragma region continuity_satisfactor

                std::vector<double> aP(N, 0.0), bP(N, 0.0), cP(N, 0.0), dP(N, 0.0);

                #pragma omp parallel
                for (int i = 1; i < N - 1; i++) {

                    const double rho_P = rho_v[i];
                    const double rho_L = rho_v[i - 1];
                    const double rho_R = rho_v[i + 1];

                    const double rhie_chow_l = -(1.0 / bVU[i - 1] + 1.0 / bVU[i]) / (8 * dz) * (p_padded_v[i - 2] - 3 * p_padded_v[i - 1] + 3 * p_padded_v[i] - p_padded_v[i + 1]);
                    const double rhie_chow_r = -(1.0 / bVU[i + 1] + 1.0 / bVU[i]) / (8 * dz) * (p_padded_v[i - 1] - 3 * p_padded_v[i] + 3 * p_padded_v[i + 1] - p_padded_v[i + 2]);

                    const double rho_l = 0.5 * (rho_v[i - 1] + rho_v[i]);
                    const double d_l_face = 0.5 * (1.0 / bVU[i - 1] + 1.0 / bVU[i]); // 1/Ap average on west face
                    const double E_l = rho_l * d_l_face / dz;

                    const double rho_r = 0.5 * (rho_v[i] + rho_v[i + 1]);
                    const double d_r_face = 0.5 * (1.0 / bVU[i] + 1.0 / bVU[i + 1]);  // 1/Ap average on east face
                    const double E_r = rho_r * d_r_face / dz;

                    const double psi_i = 1.0 / (Rv * T_v_bulk[i]); // Compressibility assuming ideal gas

                    const double u_l_star = 0.5 * (u_v[i - 1] + u_v[i]) + rhie_chow_on_off_v * rhie_chow_l;
                    const double mdot_l_star = (u_l_star > 0.0) ? rho_L * u_l_star : rho_P * u_l_star;

                    const double u_r_star = 0.5 * (u_v[i] + u_v[i + 1]) + rhie_chow_on_off_v * rhie_chow_r;
                    const double mdot_r_star = (u_r_star > 0.0) ? rho_P * u_r_star : rho_R * u_r_star;

                    const double mass_imbalance = (rho_P - rho_old_v[i]) * dz / dt + (mdot_r_star - mdot_l_star);

                    aP[i] = -E_l;
                    cP[i] = -E_r;
                    bP[i] = E_l + E_r + psi_i * dz / dt;
                    dP[i] = Sm[i] * dz - mass_imbalance;

                    printf("");
                }

                // BCs for p': zero gradient aVT inlet and zero correction aVT outlet
                bP[0] = 1.0; cP[0] = -1.0; dP[0] = 0.0;
                bP[N - 1] = 1.0; aP[N - 1] = 0.0; dP[N - 1] = 0.0;

                p_prime_v = solveTridiagonal(aP, bP, cP, dP);

                #pragma endregion

                // =======================================================================
                //
                //                        [PRESSURE CORRECTOR]
                //
                // =======================================================================

                #pragma region pressure_corrector

                for (int i = 0; i < N; i++) {

                    p_v[i] += p_prime_v[i]; // Note that PISO does not require an under-relaxation factor
                    p_storage_v[i + 1] = p_v[i];

                }

                p_storage_v[0] = p_storage_v[1];
                p_storage[N + 1] = p_storage_v;

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

                maxErr = 0.0;
                for (int i = 1; i < N - 1; i++) {

                    const double u_prev_v = u_v[i];
                    u_v[i] = u_v[i] - (p_prime_v[i + 1] - p_prime_v[i - 1]) / (2.0 * dz * bVU[i]);
                    maxErr = std::max(maxErr, std::fabs(u_v[i] - u_prev_v));
                }

                #pragma endregion

            }

            iter++;
        }

        #pragma endregion

        // =======================================================================
        //
        //                        [TURBULENCE MODELIZATION]
        //
        // =======================================================================

        #pragma region turbulence_SST

        // TODO: check discretization scheme

        if (SST_model_turbulence_on_off == 1) {

            // --- Turbulence transport equations (1D implicit form) ---
            const double sigma_k = 0.85;
            const double sigma_omega = 0.5;
            const double beta_star = 0.09;
            const double beta = 0.075;
            const double alpha = 5.0 / 9.0;

            std::vector<double> aK(N, 0.0), bK(N, 0.0), cK(N, 0.0), dK(N, 0.0);
            std::vector<double> aW(N, 0.0), bW(N, 0.0), cW(N, 0.0), dW(N, 0.0);

            // --- Compute strain rate and production ---
            std::vector<double> dudz(N, 0.0);
            std::vector<double> Pk(N, 0.0);

            for (int i = 1; i < N - 1; i++) {
                dudz[i] = (u_v[i + 1] - u_v[i - 1]) / (2.0 * dz);
                Pk[i] = mu_t[i] * pow(dudz[i], 2.0);
            }

            // --- k-equation ---
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

            // k BCs: constant initial values aVT the boundaries
            bK[0] = 1.0; dK[0] = k_turb[0]; cK[0] = 0.0;
            aK[N - 1] = 0.0; bK[N - 1] = 1.0; dK[N - 1] = k_turb[N - 1];

            k_turb = solveTridiagonal(aK, bK, cK, dK);

            // --- omega-equation ---
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
            bW[0] = 1.0; dW[0] = omega_turb[0]; cW[0] = 0.0;
            aW[N - 1] = 0.0; bW[N - 1] = 1.0; dW[N - 1] = omega_turb[N - 1];

            omega_turb = solveTridiagonal(aW, bW, cW, dW);

            // --- Update turbulent viscosity ---
            for (int i = 0; i < N; i++) {

                double mu = vapor_sodium::mu(T_v_bulk[i]);
                double denom = std::max(omega_turb[i], 1e-6);
                mu_t[i] = rho_v[i] * k_turb[i] / denom;
                mu_t[i] = std::min(mu_t[i], 1000.0 * mu); // limiter
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

        #pragma omp parallel
        for (int i = 1; i < N - 1; i++) {

            const double rho_P = rho_v[i];
            const double rho_L = rho_v[i - 1];
            const double rho_R = rho_v[i + 1];

            const double k_cond_P = vapor_sodium::k(T_v_bulk[i], p_v[i]);
            const double k_cond_L = vapor_sodium::k(T_v_bulk[i - 1], p_v[i - 1]);
            const double k_cond_R = vapor_sodium::k(T_v_bulk[i + 1], p_v[i + 1]);

            const double cp_P = vapor_sodium::cp(T_v_bulk[i]);
            const double cp_L = vapor_sodium::cp(T_v_bulk[i - 1]);
            const double cp_R = vapor_sodium::cp(T_v_bulk[i + 1]);

            const double rhoCp_dzdt = rho_old_v[i] * cp_P * dz / dt;

            const double keff_P = k_cond_P + SST_model_turbulence_on_off * (mu_t[i] * cp_P / Pr_t);
            const double keff_L = k_cond_L + SST_model_turbulence_on_off * (mu_t[i - 1] * cp_L / Pr_t);
            const double keff_R = k_cond_R + SST_model_turbulence_on_off * (mu_t[i + 1] * cp_R / Pr_t);

            // Linear interpolation diffusion coefficient
            const double D_l = 0.5 * (keff_P + keff_L) / dz;
            const double D_r = 0.5 * (keff_P + keff_R) / dz;

            const double rhie_chow_l = -(1.0 / bVU[i - 1] + 1.0 / bVU[i]) / (8 * dz) * (p_padded_v[i - 2] - 3 * p_padded_v[i - 1] + 3 * p_padded_v[i] - p_padded_v[i + 1]);
            const double rhie_chow_r = -(1.0 / bVU[i + 1] + 1.0 / bVU[i]) / (8 * dz) * (p_padded_v[i - 1] - 3 * p_padded_v[i] + 3 * p_padded_v[i + 1] - p_padded_v[i + 2]);

            const double u_l_face = 0.5 * (u_v[i - 1] + u_v[i]) + rhie_chow_on_off_v * rhie_chow_l;
            const double u_r_face = 0.5 * (u_v[i] + u_v[i + 1]) + rhie_chow_on_off_v * rhie_chow_r;

            // Upwind density
            const double rho_l = (u_l_face >= 0) ? rho_L : rho_P;
            const double rho_r = (u_r_face >= 0) ? rho_P : rho_R;

            // Upwind specific heat
            const double cp_l = (u_l_face >= 0) ? cp_L : cp_P;
            const double cp_r = (u_r_face >= 0) ? cp_P : cp_R;

            const double Fl = rho_l * u_l_face;
            const double Fr = rho_r * u_r_face;

            const double C_l = (Fl * cp_l);
            const double C_r = (Fr * cp_r);

            aVT[i] = -D_l - std::max(C_l, 0.0);
            cVT[i] = -D_r + std::max(-C_r, 0.0);
            bVT[i] = (std::max(C_r, 0.0) - std::max(-C_l, 0.0)) + D_l + D_r + rhoCp_dzdt;

            const double pressure_work = (p_v[i] - p_old_v[i]) / dt;
            dVT[i] = rhoCp_dzdt * T_old_v[i] + pressure_work * dz + St[i] * dz;

        }

        // Temperature BCs
        bVT[0] = 1.0; cVT[0] = -1.0; dVT[0] = 0.0;
        aVT[N - 1] = -1.0; bVT[N - 1] = 1.0; dVT[N - 1] = 0.0;

        T_v_bulk = solveTridiagonal(aVT, bVT, cVT, dVT);

        // Update density with new p,T
        eos_update(rho_v, p_v, T_v_bulk);

        #pragma endregion

        // =======================================================================
        //
        //                               [4. OUTPUT]
        //
        // =======================================================================

        #pragma region output

        // Output in file
        if (true) {

            std::cout << "Temperature outer wall: \n";
            for (int ix = 0; ix < N; ++ix) {
                std::cout << T_w_outer[ix] << ", ";
            } std::cout << "\n";

            std::cout << "Temperature bulk wall: \n";
            for (int ix = 0; ix < N; ++ix) {
                std::cout << T_w_bulk[ix] << ", ";
            } std::cout << "\n";

            std::cout << "Temperature wall-wick: \n";
            for (int ix = 0; ix < N; ++ix) {
                std::cout << T_w_x[ix] << ", ";
            } std::cout << "\n";

            std::cout << "Temperature wick bulk: \n";
            for (int ix = 0; ix < N; ++ix) {
                std::cout << T_x_bulk[ix] << ", ";
            } std::cout << "\n";

            std::cout << "Temperature wick vapor: \n";
            for (int ix = 0; ix < N; ++ix) {
                std::cout << T_x_v[ix] << ", ";
            } std::cout << "\n";

            std::cout << "Temperature vapor bulk: \n";
            for (int ix = 0; ix < N; ++ix) {
                std::cout << T_v_bulk[ix] << ", ";
            } std::cout << "\n";

            std::cout << "Pressure liquid: \n";
            for (int ix = 0; ix < N; ++ix) {
                std::cout << p_x[ix] << ", ";
            } std::cout << "\n";

            std::cout << "Velocity liquid: \n";
            for (int ix = 0; ix < N; ++ix) {
                std::cout << u_x[ix] << ", ";
            } std::cout << "\n";

            std::cout << "Pressure vapor: \n";
            for (int ix = 0; ix < N; ++ix) {
                std::cout << p_v[ix] << ", ";
            } std::cout << "\n";

            std::cout << "Velocity vapor: \n";
            for (int ix = 0; ix < N; ++ix) {
                std::cout << u_v[ix] << ", ";
            } std::cout << "\n\n";
        }

        // Output in file
        /*if (true) {
            for (int ix = 0; ix < N; ++ix) {
                fout << T_w_outer[ix] << ", ";
            } fout << "\n";

            for (int ix = 0; ix < N; ++ix) {
                fout << T_w_bulk[ix] << ", ";
            } fout << "\n";

            for (int ix = 0; ix < N; ++ix) {
                fout << T_w_x[ix] << ", ";
            } fout << "\n";

            for (int ix = 0; ix < N; ++ix) {
                fout << T_x_bulk[ix] << ", ";
            } fout << "\n";

            for (int ix = 0; ix < N; ++ix) {
                fout << T_x_v[ix] << ", ";
            } fout << "\n";

            for (int ix = 0; ix < N; ++ix) {
                fout << T_v_bulk[ix] << ", ";
            } fout << "\n";

            for (int ix = 0; ix < N; ++ix) {
                fout << p_x[ix] << ", ";
            } fout << "\n";

            for (int ix = 0; ix < N; ++ix) {
                fout << u_x[ix] << ", ";
            } fout << "\n";

            for (int ix = 0; ix < N; ++ix) {
                fout << p_v[ix] << ", ";
            } fout << "\n";

            for (int ix = 0; ix < N; ++ix) {
                fout << u_v[ix] << ", ";
            } fout << "\n\n";
        }*/

#pragma endregion
    }

    return 0;
}