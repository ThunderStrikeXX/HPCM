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
    *   ASME / ITER Handbook / FEA material model
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
    *   ASME / ITER Handbook / FEA material model
    */
    inline double rho(double T) { return (7.9841 - 2.6560e-4 * T - 1.158e-7 * T * T) * 1e3; }

    /**
    * @brief Thermal conductivity [W/(m*K)] as a function of temperature T
    *   ASME / ITER Handbook / FEA material model
    */
    inline double k(double T) { return (8.116e-2 + 1.618e-4 * T) * 100.0; }
}