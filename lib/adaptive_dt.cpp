#include "adaptive_dt.h"

#include <algorithm>
#include <cmath>

using data_type = double;

// ============================================================================
// WALL REGION
// ============================================================================

// Calculates maximum timestep to prevent DT > CSW in the wall
data_type new_dt_w(
    data_type dt_old,
    const std::vector<data_type>& T,
    const std::vector<data_type>& St)
{
    const data_type CSW   = static_cast<data_type>(0.5);           // Maximum DT tolerated per timestep [K]
    const data_type epsS  = static_cast<data_type>(1e-12);         // Lower heat source boundary [W/m3]
    const data_type theta = static_cast<data_type>(0.9);           // Factor to apply a safety margin [-] 
    const data_type rdown = static_cast<data_type>(0.2);           // Factor to prevent excessive timestep reduction [-]
    const data_type dt_min = static_cast<data_type>(1e-8);         // Minimum timestep tolerated [s]
    const data_type dt_max = static_cast<data_type>(1);            // Maximum timestep tolerated [s]

    const std::size_t N = static_cast<std::size_t>(St.size());  // Number of nodes [-]

    data_type dt_cand = dt_max;        

    for (std::size_t i = 0; i < N; ++i) {
        data_type dt_s =
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
data_type new_dt_x(
    data_type dt_old,
    const std::vector<data_type>& u,
    const std::vector<data_type>& T,
    const std::vector<data_type>& Sm,
    const std::vector<data_type>& Qf)
{
    const data_type CSX_mass = static_cast<data_type>(0.5);        // Maximum Drho tolerated per timestep [kg/m3]
    const data_type CSX_flux = static_cast<data_type>(0.5);        // Maximum DT tolerated per timestep [K]
    const data_type epsS  = static_cast<data_type>(1e-12);         // Lower mass source boundary [kg/m3s]
    const data_type epsT  = static_cast<data_type>(1e-12);         // Lower heat source boundary [W/m3]
    const data_type theta = static_cast<data_type>(0.9);           // Factor to apply a safety margin [-] 
    const data_type rdown = static_cast<data_type>(0.2);           // Factor to prevent excessive timestep reduction [-]
    const data_type dt_min = static_cast<data_type>(1e-8);         // Minimum timestep tolerated [s]
    const data_type dt_max = static_cast<data_type>(1);            // Maximum timestep tolerated [s]

    const std::size_t N = static_cast<std::size_t>(u.size());   // Number of nodes [-]

    data_type dt_cand = dt_max;

    for (std::size_t i = 0; i < N; ++i) {

        data_type dt_mass =
            CSX_mass * liquid_sodium::rho(T[i]) /
            (std::abs(Sm[i]) + epsS);

        data_type dt_flux =
            CSX_flux * liquid_sodium::rho(T[i]) *
            liquid_sodium::cp(T[i]) /
            (std::abs(Qf[i]) + epsT);

        data_type dti = std::min(dt_mass, dt_flux);
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
data_type new_dt_v(
    data_type dz,
    data_type dt_old,
    const std::vector<data_type>& u,
    const std::vector<data_type>& T,
    const std::vector<data_type>& rho,
    const std::vector<data_type>& Sm,
    const std::vector<data_type>& Qf,
    const std::vector<data_type>& bVU)
{
	const data_type Rv = static_cast<data_type>(361.5);                // Gas constant for the sodium vapor [J/(kgK)]

    const data_type CSV_mass = static_cast<data_type>(500);            // Maximum Drho tolerated per timestep [kg/m3]  
    const data_type CSV_flux = static_cast<data_type>(500);            // Maximum DT tolerated per timestep [K]
    const data_type CP  = static_cast<data_type>(100);                // Maximum DP tolerated per timestep [Pa]

    const data_type epsS  = static_cast<data_type>(1e-12);             // Lower mass source boundary [kg/m3s]
    const data_type epsT  = static_cast<data_type>(1e-12);             // Lower heat source boundary [W/m3]
    const data_type theta = static_cast<data_type>(0.9);               // Factor to apply a safety margin [-] 
    const data_type rdown = static_cast<data_type>(0.2);               // Factor to prevent excessive timestep reduction [-]

    const data_type dt_min = static_cast<data_type>(1e-8);             // Minimum timestep tolerated [s]
    const data_type dt_max = static_cast<data_type>(1);                // Maximum timestep tolerated [s]

	const std::size_t N = static_cast<std::size_t>(u.size());   // Number of nodes [-]

    auto invb = [&](std::size_t i) {
		return 1.0 / std::max(bVU[i], static_cast<data_type>(1e-12));   // Momentum equation coefficient inverse [m2s/kg]
    };

    data_type dt_cand = dt_max;

    for (std::size_t i = 0; i < N; ++i) {

        data_type dt_mass =
            CSV_mass * rho[i] /
            (std::abs(Sm[i]) + epsS);

        data_type dt_flux =
            CSV_flux * rho[i] *
            vapor_sodium::cp(T[i]) /
            (std::abs(Qf[i]) + epsT);

        data_type dt_p = 1e99;

        if (i > 0 && i < N - 1) {
            data_type invbL = static_cast<data_type>(0.5) * (invb(i - 1) + invb(i));
            data_type invbR = static_cast<data_type>(0.5) * (invb(i) + invb(i + 1));

            data_type rhoL = static_cast<data_type>(0.5) * (rho[i - 1] + rho[i]);
            data_type rhoR = static_cast<data_type>(0.5) * (rho[i] + rho[i + 1]);

            data_type El = rhoL * invbL / dz;
            data_type Er = rhoR * invbR / dz;

            data_type psi = 1.0 / (Rv * T[i]);

            dt_p = CP * psi * dz / (El + Er + 1e-30);
        }

        data_type dti = std::min(
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