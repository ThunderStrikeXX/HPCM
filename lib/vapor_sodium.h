#pragma once

namespace vapor_sodium {

    // From h_g(T) = ag + bg * T
    inline double cp_g_linear() {

        return 3.589755e2;   // J/(kg·K)
    }

    inline double h_g_linear(double T) {

        constexpr double ag = 4.683166e6;   // J/kg
        constexpr double bg = 3.589755e2;   // J/(kg·K)

        return ag + bg * T;   // J/kg
    }

    inline double T_from_h_g_linear(double h) {

        constexpr double ag = 4.683166e6;   // J/kg
        constexpr double bg = 3.589755e2;   // J/(kg·K)

        return (h - ag) / bg; // K
    }

    /**
    * @brief Saturation pressure [Pa] as a function of temperature T
    *   Satou-Moriyama
    */
    inline double P_sat(double T) {

        const double val_MPa = std::exp(11.9463 - 12633.7 / T - 0.4672 * std::log(T));
        return val_MPa * 1e6;
    }

    /**
    * @brief Derivative of saturation pressure with respect to temperature [Pa/K] as a function of temperature T
    *   Satou-Moriyama
    */
    inline double dP_sat_dT(double T) {

        const double val_MPa_per_K =
            (12633.73 / (T * T) - 0.4672 / T) * std::exp(11.9463 - 12633.73 / T - 0.4672 * std::log(T));
        return val_MPa_per_K * 1e6;
    }

    /**
    * @brief Dynamic viscosity of sodium vapor [Pa·s] as a function of temperature T
    *   Linear fit ANL
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
            
            double Tref = (Tlow ? Tmin : (Thigh ? Tmax : Tc));
            double k_ref = k_interp;
            double k_extrap = k_ref * std::sqrt(T / Tref);
            return k_extrap;
        }

        return k_interp;
    }

    /**
     * @brief Darcy friction factor (Petukhov correlation, smooth pipe)
     *        Valid for 3e3 < Re < 5e6
     */
    inline double friction_factor(double Re) {
        if (Re <= 0.0)
            throw std::invalid_argument("Re <= 0 in friction_factor");

        return std::pow(0.79 * std::log(Re) - 1.64, -2.0);
    }

    /**
     * @brief Nusselt number for internal flow
     *        Laminar + Petukhov–Gnielinski turbulent
     *        Smooth logarithmic blending
     */
    inline double Nusselt(
        double Re,
        double Pr
    ) {
        if (Re < 0.0 || Pr < 0.0)
            throw std::invalid_argument("Re or Pr <= 0 in Nusselt");

        // -----------------------------
        // Laminar fully developed
        // -----------------------------
        constexpr double Nu_lam = 4.36;

        // -----------------------------
        // Turbulent (Petukhov–Gnielinski)
        // -----------------------------
        auto Nu_turb = [&](double Re_loc) {
            const double f = friction_factor(Re_loc);
            const double num = (f / 8.0) * (Re_loc - 1000.0) * Pr;
            const double den = 1.0 + 12.7 * std::sqrt(f / 8.0)
                * (std::pow(Pr, 2.0 / 3.0) - 1.0);
            return num / den;
            };

        // -----------------------------
        // Transition limits
        // -----------------------------
        constexpr double Re_lam = 2300.0;
        constexpr double Re_turb = 4000.0;

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
        const double chi =
            (std::log(Re) - std::log(Re_lam)) /
            (std::log(Re_turb) - std::log(Re_lam));

        return (1.0 - chi) * Nu_lam + chi * Nu_turb(Re);
    }

    /**
     * @brief Convective heat transfer coefficient [W/m2/K]
     *        Sodium vapor – internal flow in heat pipe
     */
    inline double h_conv(
        double Re,
        double Pr,
        double k,
        double Dh
    ) {
        if (Dh <= 0.0 || k <= 0.0)
            throw std::invalid_argument("Dh or k <= 0 in h_conv");

        return Nusselt(Re, Pr) * k / Dh;
    }

    inline double surf_ten(double T) {
        constexpr double Tm = 371.0;
        double val = 0.196 - 2.48e-4 * (T - Tm);
        return val > 0.0 ? val : 0.0;
    }

    inline double gamma(double T) {
        double cp_val = cp_g_linear();
        double cv_val = cp_g_linear() - 361.5;
        return cp_val / cv_val;
    }
}
