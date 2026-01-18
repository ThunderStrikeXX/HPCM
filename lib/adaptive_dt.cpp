#include "adaptive_dt.h"

#include <algorithm>
#include <cmath>

// ============================================================================
// WALL REGION
// ============================================================================

// Calculates maximum timestep to prevent DT > CSW in the wall
double new_dt_w(
    double dz,
    double dt_old,
    const std::vector<double>& T,
    const std::vector<double>& St)
{
    const double CSW   = 0.5;           // Maximum DT tolerated per timestep [K]
    const double epsS  = 1e-12;         // Lower heat source boundary [W/m3]
    const double theta = 0.9;           // Factor to apply a safety margin [-] 
    const double rdown = 0.2;           // Factor to prevent excessive timestep reduction [-]
    const double dt_min = 1e-8;         // Minimum timestep tolerated [s]
    const double dt_max = 1;            // Maximum timestep tolerated [s]

    const int N = static_cast<int>(St.size());  // Number of nodes [-]

    double dt_cand = dt_max;        

    for (int i = 0; i < N; ++i) {
        double dt_s =
            CSW * steel::rho(T[i]) * steel::cp(T[i]) /
            (std::abs(St[i]) + epsS);

        dt_cand = std::min(dt_cand, dt_s);
    }

    return std::min(
        dt_max,
        std::max(
            dt_min,
            std::max(theta * dt_cand, rdown * dt_old)
        )
    );
}

// ============================================================================
// WICK REGION
// ============================================================================

// Calculates maximum timestep to prevent Drho > CSX_mass and DT > CSX_flux in the wick
double new_dt_x(
    double dz,
    double dt_old,
    const std::vector<double>& u,
    const std::vector<double>& T,
    const std::vector<double>& Sm,
    const std::vector<double>& Qf)
{
    const double CSX_mass = 0.5;        // Maximum Drho tolerated per timestep [kg/m3]
    const double CSX_flux = 0.5;        // Maximum DT tolerated per timestep [K]
    const double epsS  = 1e-12;         // Lower mass source boundary [kg/m3s]
    const double epsT  = 1e-12;         // Lower heat source boundary [W/m3]
    const double theta = 0.9;           // Factor to apply a safety margin [-] 
    const double rdown = 0.2;           // Factor to prevent excessive timestep reduction [-]
    const double dt_min = 1e-8;         // Minimum timestep tolerated [s]
    const double dt_max = 1;            // Maximum timestep tolerated [s]

    const int N = static_cast<int>(u.size());   // Number of nodes [-]

    double dt_cand = dt_max;

    for (int i = 0; i < N; ++i) {

        double dt_mass =
            CSX_mass * liquid_sodium::rho(T[i]) /
            (std::abs(Sm[i]) + epsS);

        double dt_flux =
            CSX_flux * liquid_sodium::rho(T[i]) *
            liquid_sodium::cp(T[i]) /
            (std::abs(Qf[i]) + epsT);

        double dti = std::min(dt_mass, dt_flux);
        dt_cand = std::min(dt_cand, dti);
    }

    return std::min(
        dt_max,
        std::max(
            dt_min,
            std::max(theta * dt_cand, rdown * dt_old)
        )
    );
}

// ============================================================================
// VAPOR REGION
// ============================================================================

// Calculates maximum timestep to prevent Drho > CSV_mass and DT > CSV_flux 
// and the compressibility limit in the vapor
double new_dt_v(
    double dz,
    double dt_old,
    const std::vector<double>& u,
    const std::vector<double>& T,
    const std::vector<double>& rho,
    const std::vector<double>& Sm,
    const std::vector<double>& Qf,
    const std::vector<double>& bVU)
{
	const double Rv = 361.5;                // Gas constant for the sodium vapor [J/(kgK)]

    const double CSV_mass = 500;            // Maximum Drho tolerated per timestep [kg/m3]  
    const double CSV_flux = 500;            // Maximum DT tolerated per timestep [K]
    const double CP  = 100;                // Maximum DP tolerated per timestep [Pa]

    const double epsS  = 1e-12;             // Lower mass source boundary [kg/m3s]
    const double epsT  = 1e-12;             // Lower heat source boundary [W/m3]
    const double theta = 0.9;               // Factor to apply a safety margin [-] 
    const double rdown = 0.2;               // Factor to prevent excessive timestep reduction [-]

    const double dt_min = 1e-8;             // Minimum timestep tolerated [s]
    const double dt_max = 1;                // Maximum timestep tolerated [s]

	const int N = static_cast<int>(u.size());   // Number of nodes [-]

    auto invb = [&](int i) {
		return 1.0 / std::max(bVU[i], 1e-30);   // Momentum equation coefficient inverse [m2s/kg]
    };

    double dt_cand = dt_max;

    for (int i = 0; i < N; ++i) {

        double dt_mass =
            CSV_mass * rho[i] /
            (std::abs(Sm[i]) + epsS);

        double dt_flux =
            CSV_flux * rho[i] *
            vapor_sodium::cp(T[i]) /
            (std::abs(Qf[i]) + epsT);

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

        double dti = std::min(
            std::min(dt_mass, dt_flux),
            std::min(dt_mass, dt_p)
        );

        dt_cand = std::min(dt_cand, dti);
    }

    return std::min(
        dt_max,
        std::max(
            dt_min,
            std::max(theta * dt_cand, rdown * dt_old)
        )
    );
}