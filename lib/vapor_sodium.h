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
    * @brief Function that clamps a value x to the range [a, b]
    */
    constexpr inline double clamp(double x, double a, double b) { return std::max(a, std::min(x, b)); }

    /**
    * @brief 1D table interpolation in T over monotone grid
    */
    template<size_t N>
    double interp_T(const std::array<double, N>& Tgrid, const std::array<double, N>& Ygrid, double T) {

        if (T <= Tgrid.front()) return Ygrid.front();
        if (T >= Tgrid.back())  return Ygrid.back();

        size_t i = 0;
        while (i + 1 < N && Tgrid[i + 1] < T) ++i;

        if (i + 1 >= N) return Ygrid[N - 1];  // fallback assoluto

        // interpolazione
        double T0 = Tgrid[i];
        double T1 = Tgrid[i + 1];
        double Y0 = Ygrid[i];
        double Y1 = Ygrid[i + 1];

        return Y0 + (T - T0) / (T1 - T0) * (Y1 - Y0);
    }

    /// Enthalpy of liquid sodium [J/kg] (CODATA correlation)
    inline double h_liquid_sodium(double T) {
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

    /// Enthalpy of vaporization of sodium [J/kg]
    /// Fink (1995)
    inline double h_vap_sodium(double T) {
        // Numerical safety
        if (T < 300.0)  T = 300.0;
        if (T > 2500.0) T = 2500.0;

        return (2128.4 + 0.86496 * T) * 1e3; // J/kg
    }

    /// Enthalpy of sodium vapor [J/kg]
    /// h_v = h_l,CODATA + h_vap,Fink95
    inline double h(double T) {
        return h_liquid_sodium(T) + h_vap_sodium(T);
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
    * @brief Specific heat at constant pressure from table interpolation [J/(kg*K)] as a function of temperature T
    *   Fink & Leibowitz
    */
    inline double cp(double T) {

        static const std::array<double, 21> Tgrid = { 400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400 };
        static const std::array<double, 21> Cpgrid = { 860,1250,1800,2280,2590,2720,2700,2620,2510,2430,2390,2360,2340,2410,2460,2530,2660,2910,3400,4470,8030 };

        // Table also lists 2500 K = 417030; extreme near critical. If needed, extend:
        if (T >= 2500.0) return 417030.0;

        return interp_T(Tgrid, Cpgrid, T);
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
    * @brief Friction factor [-] (Prandtl–von Kármán smooth pipe law) as a function of Reynolds number.
    * Retrieves an error if Re < 0.*/

    inline double f(double Re) {

        if (Re <= 0.0) throw std::invalid_argument("Error: Re < 0");

        const double t = 0.79 * std::log(Re) - 1.64;
        return 1.0 / (t * t);
    }    

    /**
     * @brief Convective heat transfer coefficient [W/m^2/K]
     *        with smooth blending between laminar and Gnielinski turbulent regimes.
     *
     * Regimes:
     *  - Re <= Re1      : purely laminar
     *  - Re >= Re2      : purely Gnielinski
     *  - Re1 < Re < Re2: linear blending
     */
    inline double h_conv(
        double Re,
        double Pr,
        double k,
        double Dh
    ) {
        if (Re <= 0.0 || Pr <= 0.0)
            throw std::invalid_argument("Error: Re or Pr <= 0");

        // -----------------------------
        // Parameters
        // -----------------------------
        constexpr double Nu_lam = 4.36;
        constexpr double Re1 = 2000.0;
        constexpr double Re2 = 3000.0;

        // -----------------------------
        // Laminar
        // -----------------------------
        const double Nu_laminar = Nu_lam;

        // -----------------------------
        // Turbulent (Gnielinski standard)
        // -----------------------------
        auto Nu_gnielinski = [&](double Re_loc) {
            const double f = vapor_sodium::f(Re_loc);
            const double fp8 = f / 8.0;

            const double num = fp8 * (Re_loc - 1000.0) * Pr;
            const double den = 1.0 + 12.7 * std::sqrt(fp8)
                * (std::pow(Pr, 2.0 / 3.0) - 1.0);

            return num / den;
            };

        // -----------------------------
        // Regime selection
        // -----------------------------
        double Nu;

        if (Re <= Re1) {
            Nu = Nu_laminar;
        }
        else if (Re >= Re2) {
            Nu = Nu_gnielinski(Re);
        }
        else {
            const double chi = (Re - Re1) / (Re2 - Re1);  // 0 → 1
            Nu = (1.0 - chi) * Nu_laminar + chi * Nu_gnielinski(Re);
        }

        // -----------------------------
        // Convective coefficient
        // -----------------------------
        return Nu * k / Dh;
    }

    inline double surf_ten(double T) {
        constexpr double Tm = 371.0;
        double val = 0.196 - 2.48e-4 * (T - Tm);
        return val > 0.0 ? val : 0.0;
    }

    /**
    * @brief Specific heat at constant volume from table interpolation [J/(kg*K)]
    *        Fink & Leibowitz
    */
    inline double cv(double T) {

        static const std::array<double, 22> Tgrid =
        { 400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,
          1600,1700,1800,1900,2000,2100,2200,2300,2400, 2500 };

        // valori convertiti in J/kgK (kJ/kgK * 1000)
        static const std::array<double, 22> Cvgrid =
        { 490, 840, 1310, 1710, 1930, 1980, 1920, 1810, 1680, 1580, 1510, 1440, 1390, 1380, 1360, 1300, 1300, 1300, 1340, 1760, 17030 };

        // valore tabellato a 2500 K = 17.03 kJ/kgK
        if (T >= 2500.0) return 17030.0;

        return interp_T(Tgrid, Cvgrid, T);
    }

    inline double gamma(double T) {
        double cp_val = cp(T);
        double cv_val = cv(T);
        return cp_val / cv_val;
    }
}
