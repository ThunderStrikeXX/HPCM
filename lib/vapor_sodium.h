#pragma once

#include "numeric_types.h"

/**
 * @brief Provides thermophysical and transport properties for Sodium Vapor.
 *
 * This namespace contains constant data and functions to calculate key properties
 * of sodium vapor.
 * All functions primarily accept temperature T in Kelvin [K] and return values
 * in standard SI units unless otherwise noted.
 */
namespace vapor_sodium {

    /**
    * @brief 1D table interpolation in T over monotone grid
    */
    template<size_t N>
    data_type interp_T(const std::array<data_type, N>& Tgrid, const std::array<data_type, N>& Ygrid, data_type T) {

        if (T <= Tgrid.front()) return Ygrid.front();
        if (T >= Tgrid.back())  return Ygrid.back();

        size_t i = 0;
        while (i + 1 < N && Tgrid[i + 1] < T) ++i;

        if (i + 1 >= N) return Ygrid[N - 1];

        // interpolazione
        data_type T0 = Tgrid[i];
        data_type T1 = Tgrid[i + 1];
        data_type Y0 = Ygrid[i];
        data_type Y1 = Ygrid[i + 1];

        return Y0 + (T - T0) / (T1 - T0) * (Y1 - Y0);
    }

    /// Enthalpy of liquid sodium [J/kg] (CODATA correlation)
    inline data_type h_liquid_sodium(data_type T) {
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

    /// Enthalpy of vaporization of sodium Δh_vap(T) [J/kg]
    /// Critical-scaling correlation (vanishes at Tcrit)
    inline data_type h_vap_sodium(data_type T) {

        // Numerical safety
        if (T < 300.0)  T = 300.0;
        if (T > 2500.0) T = 2500.0;

        // Sodium critical temperature [K]
        constexpr data_type Tcrit = 2503.0;

        const data_type theta = 1.0 - T / Tcrit;

        return (
            393.37 * theta
            + 4398.6 * std::pow(theta, 0.29302)
            ) * 1e3;   // J/kg
    }

    /// Enthalpy of sodium vapor [J/kg]
    /// h_v = h_l,CODATA + h_vap,Fink95
    inline data_type h(data_type T) {
        return h_liquid_sodium(T) + h_vap_sodium(T);
    }

    // From h_g(T) = ag + bg * T
    inline data_type cp_g_linear() {

        return 3.589755e2;   // J/(kg·K)
    }

    inline data_type h_g_linear(data_type T) {

        constexpr data_type ag = 4.683166e6;   // J/kg
        constexpr data_type bg = 3.589755e2;   // J/(kg·K)

        return ag + bg * T;   // J/kg
    }

    inline data_type T_from_h_g_linear(data_type h) {

        constexpr data_type ag = 4.683166e6;   // J/kg
        constexpr data_type bg = 3.589755e2;   // J/(kg·K)

        return (h - ag) / bg; // K
    }

    /*

    inline data_type T_from_h_vapor(data_type h_target) {

        constexpr data_type T_min = 300.0;
        constexpr data_type T_max = 2500.0;
        constexpr int max_iter = 80;

        // tolleranze su entalpia [J/kg]
        constexpr data_type rel_tol = 1e-8;
        constexpr data_type abs_tol = 1e-3;   // 1 mJ/kg: praticamente zero

        data_type T_lo = T_min;
        data_type T_hi = T_max;

        const data_type h_lo = vapor_sodium::h(T_lo);
        const data_type h_hi = vapor_sodium::h(T_hi);

        // Se la tua h(T) è monotona crescente, questo deve valere.
        // Se per qualche ragione non lo è, l'inversione non è definita.
        if (h_hi <= h_lo) return T_min;

        // Clamp su range coperto
        if (h_target <= h_lo) return T_lo;
        if (h_target >= h_hi) return T_hi;

        for (int it = 0; it < max_iter; ++it) {

            const data_type T_mid = 0.5 * (T_lo + T_hi);
            const data_type h_mid = vapor_sodium::h(T_mid);

            const data_type err = h_mid - h_target;
            const data_type tol_h = abs_tol + rel_tol * std::max(std::abs(h_target), std::abs(h_mid));

            if (std::abs(err) <= tol_h)
                return T_mid;

            // monotona crescente: se h_mid troppo grande, abbassa T_hi
            if (err > 0.0) T_hi = T_mid;
            else           T_lo = T_mid;
        }

        return 0.5 * (T_lo + T_hi);
    }

    */

    /**
    * @brief Saturation pressure [Pa] as a function of temperature T
    *   Satou-Moriyama
    */
    inline data_type P_sat(data_type T) {

        const data_type val_MPa = std::exp(11.9463 - 12633.7 / T - 0.4672 * std::log(T));
        return val_MPa * 1e6;
    }

    /**
    * @brief Derivative of saturation pressure with respect to temperature [Pa/K] as a function of temperature T
    *   Satou-Moriyama
    */
    inline data_type dP_sat_dT(data_type T) {

        const data_type val_MPa_per_K =
            (12633.73 / (T * T) - 0.4672 / T) * std::exp(11.9463 - 12633.73 / T - 0.4672 * std::log(T));
        return val_MPa_per_K * 1e6;
    }

    /**
    * @brief Specific heat at constant pressure from table interpolation [J/(kg*K)] as a function of temperature T
    *   Fink & Leibowitz
    */
    inline data_type cp(data_type T) {

        static const std::array<data_type, 21> Tgrid = { 400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400 };
        static const std::array<data_type, 21> Cpgrid = { 860,1250,1800,2280,2590,2720,2700,2620,2510,2430,2390,2360,2340,2410,2460,2530,2660,2910,3400,4470,8030 };

        // Table also lists 2500 K = 417030; extreme near critical. If needed, extend:
        if (T >= 2500.0) return 417030.0;

        return interp_T(Tgrid, Cpgrid, T);
    }

    /**
    * @brief Dynamic viscosity of sodium vapor [Pa·s] as a function of temperature T
    *   Linear fit ANL
    */
    inline data_type mu(data_type T) { return 6.083e-9 * T + 1.2606e-5; }

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
    inline data_type k(data_type T, data_type P) {

        static const std::array<data_type, 7> Tgrid = { 900,1000,1100,1200,1300,1400,1500 };
        static const std::array<data_type, 5> Pgrid = { 981,4903,9807,49033,98066 };

        static const data_type Ktbl[7][5] = {
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
        auto clamp_val = [](data_type x, data_type minv, data_type maxv) {
            return (x < minv) ? minv : ((x > maxv) ? maxv : x);
            };

        auto idz = [](data_type x, const auto& grid) {
            size_t i = 0;
            while (i + 1 < grid.size() && x > grid[i + 1]) ++i;
            return i;
            };

        const data_type Tmin = Tgrid.front(), Tmax = Tgrid.back();
        const data_type Pmin = Pgrid.front(), Pmax = Pgrid.back();

        bool Tlow = (T < Tmin);
        bool Thigh = (T > Tmax);
        bool Plow = (P < Pmin);
        bool Phigh = (P > Pmax);

        data_type Tc = clamp_val(T, Tmin, Tmax);
        data_type Pc = clamp_val(P, Pmin, Pmax);

        const size_t iT = idz(Tc, Tgrid);
        const size_t iP = idz(Pc, Pgrid);

        const data_type T0 = Tgrid[iT], T1 = Tgrid[std::min(iT + 1ul, Tgrid.size() - 1)];
        const data_type P0 = Pgrid[iP], P1 = Pgrid[std::min(iP + 1ul, Pgrid.size() - 1)];

        const data_type q11 = Ktbl[iT][iP];
        const data_type q21 = Ktbl[std::min(iT + 1ul, Tgrid.size() - 1)][iP];
        const data_type q12 = Ktbl[iT][std::min(iP + 1ul, Pgrid.size() - 1)];
        const data_type q22 = Ktbl[std::min(iT + 1ul, Tgrid.size() - 1)][std::min(iP + 1ul, Pgrid.size() - 1)];

        data_type k_interp = 0.0;

        // Bilinear interpolation
        if ((T1 != T0) && (P1 != P0)) {
            const data_type t = (Tc - T0) / (T1 - T0);
            const data_type u = (Pc - P0) / (P1 - P0);
            k_interp = (1 - t) * (1 - u) * q11 + t * (1 - u) * q21 + (1 - t) * u * q12 + t * u * q22;
        }
        else if (T1 != T0) {
            const data_type t = (Tc - T0) / (T1 - T0);
            k_interp = q11 + t * (q21 - q11);
        }
        else if (P1 != P0) {
            const data_type u = (Pc - P0) / (P1 - P0);
            k_interp = q11 + u * (q12 - q11);
        }
        else {
            k_interp = q11;
        }

        // Extrapolation handling
        if (Tlow || Thigh || Plow || Phigh) {
            
            data_type Tref = (Tlow ? Tmin : (Thigh ? Tmax : Tc));
            data_type k_ref = k_interp;
            data_type k_extrap = k_ref * std::sqrt(T / Tref);
            return k_extrap;
        }

        return k_interp;
    }

    /**
     * @brief Darcy friction factor (Petukhov correlation, smooth pipe)
     *        Valid for 3e3 < Re < 5e6
     */
    inline data_type friction_factor(data_type Re) {
        if (Re <= 0.0)
            throw std::invalid_argument("Re <= 0 in friction_factor");

        return std::pow(0.79 * std::log(Re) - 1.64, -2.0);
    }

    /**
     * @brief Nusselt number for internal flow
     *        Laminar + Petukhov–Gnielinski turbulent
     *        Smooth logarithmic blending
     */
    inline data_type Nusselt(
        data_type Re,
        data_type Pr
    ) {
        if (Re < 0.0 || Pr < 0.0)
            throw std::invalid_argument("Re or Pr <= 0 in Nusselt");

        // -----------------------------
        // Laminar fully developed
        // -----------------------------
        constexpr data_type Nu_lam = 4.36;

        // -----------------------------
        // Turbulent (Petukhov–Gnielinski)
        // -----------------------------
        auto Nu_turb = [&](data_type Re_loc) {
            const data_type f = friction_factor(Re_loc);
            const data_type num = (f / 8.0) * (Re_loc - 1000.0) * Pr;
            const data_type den = 1.0 + 12.7 * std::sqrt(f / 8.0)
                * (std::pow(Pr, 2.0 / 3.0) - 1.0);
            return num / den;
            };

        // -----------------------------
        // Transition limits
        // -----------------------------
        constexpr data_type Re_lam = 2300.0;
        constexpr data_type Re_turb = 4000.0;

        // -----------------------------
        // Regime selection
        // -----------------------------
        if (Re <= Re_lam)
            return Nu_lam;

        if (Re >= Re_turb)
            return Nu_turb(Re);

        // -----------------------------
        // Logarithmic blending
        // -----------------------------
        const data_type chi =
            (std::log(Re) - std::log(Re_lam)) /
            (std::log(Re_turb) - std::log(Re_lam));

        return (1.0 - chi) * Nu_lam + chi * Nu_turb(Re);
    }

    /**
     * @brief Convective heat transfer coefficient [W/m2/K]
     *        Sodium vapor – internal flow in heat pipe
     */
    inline data_type h_conv(
        data_type Re,
        data_type Pr,
        data_type k,
        data_type Dh
    ) {
        if (Dh <= 0.0 || k <= 0.0)
            throw std::invalid_argument("Dh or k <= 0 in h_conv");

        return Nusselt(Re, Pr) * k / Dh;
    }

    inline data_type surf_ten(data_type T) {
        constexpr data_type Tm = 371.0;
        data_type val = 0.196 - 2.48e-4 * (T - Tm);
        return val > 0.0 ? val : 0.0;
    }

    /**
    * @brief Specific heat at constant volume from table interpolation [J/(kg*K)]
    *        Fink & Leibowitz
    */
    inline data_type cv(data_type T) {

        static const std::array<data_type, 22> Tgrid =
        { 400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,
          1600,1700,1800,1900,2000,2100,2200,2300,2400, 2500 };

        // valori convertiti in J/kgK (kJ/kgK * 1000)
        static const std::array<data_type, 22> Cvgrid =
        { 490, 840, 1310, 1710, 1930, 1980, 1920, 1810, 1680, 1580, 1510, 1440, 1390, 1380, 1360, 1300, 1300, 1300, 1340, 1760, 17030 };

        // valore tabellato a 2500 K = 17.03 kJ/kgK
        if (T >= 2500.0) return 17030.0;

        return interp_T(Tgrid, Cvgrid, T);
    }

    inline data_type gamma(data_type T) {
        data_type cp_val = cp(T);
        data_type cv_val = cv(T);
        return cp_val / cv_val;
    }


    /**
     * @brief Vapor enthalpy obtained by numerical integration of cp(T)
     *        using the same reference enthalpy as liquid sodium.
     *
     * h_v(T) = h_l(T_ref) + ∫_{T_ref}^{T} cp(T) dT
     *
     * Reference:
     *  - cp(T): Fink & Leibowitz (tabulated, interpolated)
     *  - h_l(T): CODATA liquid sodium enthalpy

    inline data_type h(data_type T) {
        // --- reference temperature ---
        constexpr data_type Tref = 400.0;

        // clamp for numerical safety
        if (T < Tref)  T = Tref;
        if (T > 2500.0) T = 2500.0;

        // reference enthalpy: liquid sodium
        const data_type href = h(Tref);   // J/kg

        // --- numerical integration (trapezoidal rule) ---
        constexpr int N = 200;             // fixed quadrature resolution
        const data_type dT = (T - Tref) / N;

        data_type integral = 0.0;
        data_type Ti = Tref;

        for (int i = 0; i < N; ++i) {
            const data_type Tj = Ti + dT;
            integral += 0.5 * (cp(Ti) + cp(Tj)) * dT;
            Ti = Tj;
        }

        return href + integral;            // J/kg
    }
         */

}
