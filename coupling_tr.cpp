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
#include <string>

bool warnings = false;

#include "steel.h"
#include "liquid_sodium.h"
#include "vapor_sodium.h"

// =======================================================================
//
//                        [VARIOUS ALGORITHMS]
//
// =======================================================================

#pragma region solving_algorithms
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

#pragma endregion

#pragma region initialization_algorithms

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

#pragma endregion

#pragma region time_step_algorithms

/**
 * @brief Compute adaptive time step for the wall region based on source limits.
 * @param dz spatial step size.
 * @param dt_old previous time step.
 * @param T wall temperature field.
 * @param St volumetric source term field.
 * @return New time step satisfying diffusion and source stability constraints.
 */
double new_dt_w(double dz, double dt_old,
    const std::vector<double>& T,
    const std::vector<double>& St) {

    const double CSW = 0.5;                      /// Limit coefficient: the maximum change in temperature for each node in [K] for each timestep
    const double epsS = 1e-12;                   /// This is to prevent divisions by zero (e.g. if the source is zero)
    const double theta = 0.9;                    /// Adjusting coefficient for the timestep candidate 
    const double rdown = 0.2;                    /// Coefficient for damping the timestep correction
    const double dt_min = 1e-12, dt_max = 1e-3;  /// Timestep boundaries [s]

    int N = St.size();

    double dt_cand = dt_max;
    for (int i = 0; i < N; ++i) {

        const double alpha = steel::k(T[i]) / (steel::cp(T[i]) * steel::rho(T[i]));

        /// Minimum time step due to CS limit
        double dt_s = CSW * steel::rho(T[i]) * steel::cp(T[i]) / (std::abs(St[i]) + epsS);

        dt_cand = std::min(dt_cand, dt_s);  // Loop on each node to find the minimum timestep necessary
    }

    // Timestep lower boundary overrall and damping the correction
    double dt_new = std::min(dt_max, std::max(dt_min, std::max(theta * dt_cand, rdown * dt_old)));
    return dt_new;
}


/**
 * @brief Compute adaptive time step for the wick region based on mass and heat source limits.
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
    const std::vector<double>& Sm,
    const std::vector<double>& Qf) {

    const double CSX_mass = 0.5;                    /// Limit coefficients: mass fraction allowed to change per time step
    const double CSX_flux = 0.5;                    /// Limit coefficients: mass fraction allowed to change per time step
    const double epsS = 1e-12;                      /// This is to prevent divisions by zero (e.g. if the source is zero)
	const double epsT = 1e-12;					    /// This is to prevent divisions by zero (e.g. if the source is zero)    
    const double theta = 0.9;                       /// Adjusting coefficient for the timestep candidate 
    const double rdown = 0.2;                       /// Coefficient for damping the timestep correction
    const double dt_min = 1e-12, dt_max = 1e-3;     /// Timestep boundaries [s]

    int N = u.size();

    double dt_cand = dt_max;
    for (int i = 0; i < N; ++i) {

        // Minimum time step due to mass source limit
        double dt_mass = CSX_mass * liquid_sodium::rho(T[i]) / (std::abs(Sm[i]) + epsS);

        /// Minimum time step due to heat source limit
        double dt_flux = CSX_flux * liquid_sodium::rho(T[i]) * liquid_sodium::cp(T[i]) / (std::abs(Qf[i]) + epsT);

        double dti = std::min(dt_mass, dt_flux);
        dt_cand = std::min(dt_cand, dti);
    }

    // Timestep lower boundary overrall and damping the correction
    double dt_new = std::min(dt_max, std::max(dt_min, std::max(theta * dt_cand, rdown * dt_old)));
    return dt_new;
}

/**
 * @brief Compute adaptive time step for the vapor region mass source, heat source, and pressure limits.
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
	const std::vector<double>& Qf,
    const std::vector<double>& bVU) {

    const double gamma = 1.32;                          /// Vapor sodium gas constant [-]
    const double Rv = 361.8;                            /// Gas constant for the sodium vapor [J/(kg K)]
    const double CSV_mass = 0.5;                        /// Limit coefficient: mass fraction allowed to change per time step
    const double CSV_flux = 0.5;                        /// Limit coefficient: mass fraction allowed to change per time step
    const double CP = 10;                              /// Limit coefficient:
    const double epsS = 1e-12;                          /// This is to prevent divisions by zero (e.g. if the source is zero)
	const double epsT = 1e-12;                          /// This is to prevent divisions by zero (e.g. if the source is zero)
    const double theta = 0.9;                           /// Adjusting coefficient for the timestep candidate 
    const double rdown = 0.2;                           /// Coefficient for damping the timestep correction
    const double dt_min = 1e-12, dt_max = 1e3;          /// Timestep boundaries [s]

    int N = u.size();
    auto invb = [&](int i) { return 1.0 / std::max(bVU[i], 1e-30); };

    double dt_cand = dt_max;
    for (int i = 0; i < N; ++i) {

        // Minimum time step due to mass source limit
        double dt_mass = CSV_mass * vapor_sodium::rho(T[i]) / (std::abs(Sm[i]) + epsS);

        /// Minimum time step due to heat source limit
        double dt_flux = CSV_flux * vapor_sodium::rho(T[i]) * vapor_sodium::cp(T[i]) / (std::abs(Qf[i]) + epsT);

        // Minimum time step due to compressibility limit
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

        double dti = std::min(std::min(dt_mass, dt_flux), std::min(dt_mass, dt_p));
        dt_cand = std::min(dt_cand, dti);   // Loop on each node to find the minimum timestep necessary
    }

    // Timestep lower boundary overrall and damping the correction
    double dt_new = std::min(dt_max, std::max(dt_min, std::max(theta * dt_cand, rdown * dt_old)));
    return dt_new;
}

#pragma endregion

/**
 * @brief Simplified 1D transient coupling between wall, wick, and vapor.
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
    const double emissivity = 0.5;          /// Wall emissivity [-]
    const double sigma = 5.67e-8;           /// Stefan-Boltzmann constant [W/m^2/K^4]
    const double Rv = 361.8;                /// Gas constant for the sodium vapor [J/(kg K)]
    const double gamma = 1.32;              /// TODO: MAKE THIS PROPERTY TEMPERATURE DEPENDENT Ratio between constant pressure specific heat and constant volume specific heat [-] 

    // Environmental boundary conditions
    const double h_conv = 10;               /// Convective heat transfer coefficient for external heat removal [W/m^2/K]
    const double power = 1e3;               /// Power at the evaporator side [W]
    const double T_env = 280.0;             /// External environmental temperature [K]

    // Evaporation and condensation parameters
    const double eps_s = 1.0;               /// Surface fraction of the wick available for phasic interface [-]
    const double sigma_e = 0.05;            /// Evaporation accomodation coefficient [-]. 1 means optimal evaporation
    const double sigma_c = 0.05;            /// Condensation accomodation coefficient [-]. 1 means optimal condensation
	double Omega = 1.0;                     /// Initialization of Omega parameter for evaporation/condensation model [-]
            
    // Geometric parameters
    const int N = 100;                                                          /// Number of axial nodes [-]
    const double L = 0.982; 			                                        /// Length of the heat pipe [m]
    const double dz = L / N;                                                    /// Axial discretization step [m]
    const double evaporator_length = 0.502;                                     /// Evaporator length [m]
    const double adiabatic_length = 0.188;                                      /// Adiabatic length [m]
    const double condenser_length = 0.292;                                      /// Condenser length [m]
    const double evaporator_nodes = std::floor(evaporator_length / dz);         /// Number of evaporator nodes
    const double condenser_nodes = std::ceil(condenser_length / dz);            /// Number of condenser nodes
    const double adiabatic_nodes = N - (evaporator_nodes + condenser_nodes);    /// Number of adiabatic nodes
    const double r_o = 0.01335;                                                 /// Outer wall radius [m]
    const double r_i = 0.0112;                                                  /// Wall-wick interface radius [m]
    const double r_v = 0.01075;                                                 /// Vapor-wick interface radius [m]

    // Surfaces 
    const double A_w_outer = 2 * M_PI * r_o * dz;                           /// Wall radial area (at r_o) [m^2]
    const double A_w_cross = M_PI * (r_o * r_o - r_i * r_i);                /// Wall cross-sectional area [m^2]
    const double A_x_interface = 2 * M_PI * r_i * dz;                       /// Wick radial area (at r_i) [m^2]
    const double A_x_cross = M_PI * (r_i * r_i - r_v * r_v);                /// Wick cross-sectional area [m^2]
    const double A_v_inner = 2 * M_PI * r_v * dz;                           /// Vapor radial area (at r_v) [m^2]
    const double A_v_cross = M_PI * r_v * r_v;                              /// Vapor cross-sectional area [m^2]

    // Time-stepping parameters
    double dt_user = 1e-4;                  /// Initial time step [s] (then it is updated according to the limits)
    const int nSteps = 1e20;                /// Number of timesteps [-]
    double time_total = 0.0;                /// Total simulation time [s]

	// Picard iteration parameters
	const double maxPicard = 1000;          /// Maximum number of Picard iterations per time step [-]
	const double picTolerance = 1e-4;   	/// Tolerance for Picard iterations [-]   

    // Wick permeability parameters
    const double K = 1e-10;                 /// Permeability [m^2]
    const double CF = 1e5;                  /// Forchheimer coefficient [1/m]

    // PISO Wick parameters
    const int tot_outer_iter_x = 2000;      /// Outer iterations per time-step [-]
    const int tot_inner_iter_x = 100;       /// Inner iterations per outer iteration [-]
    const double outer_tol_x = 1e-6;        /// Tolerance for the outer iterations (velocity) [-]
    const double inner_tol_x = 1e-6;        /// Tolerance for the inner iterations (pressure) [-]

    // PISO Vapor parameters
    const int tot_outer_iter_v = 2000;      /// Outer iterations per time-step [-]
    const int tot_inner_iter_v = 100;       /// Inner iterations per outer iteration [-]
    const double outer_tol_v = 1e-6;        /// Tolerance for the outer iterations (velocity) [-]
    const double inner_tol_v = 1e-6;        /// Tolerance for the inner iterations (pressure) [-]

    // Mesh z positions
    std::vector<double> mesh(N, 0.0);
    for (int i = 0; i < N; ++i) mesh[i] = i * dz;

    // Node partition
    const int N_e = static_cast<int>(std::floor(evaporator_length / dz));   /// Number of nodes of the evaporator region [-]
    const int N_c = static_cast<int>(std::ceil(condenser_length / dz));     /// Number of nodes of the condenser region [-]
    const int N_a = N - (N_e + N_c);                                        /// Number of nodes of the adiabadic region [-]

	/// Constant temperature for initialization
    const double T_init = 800.0;

    // Initialization of the temperatures 
    std::vector<double> T_o_w(N, T_init);
    std::vector<double> T_w_bulk(N, T_init);
    std::vector<double> T_w_x(N, T_init);
    std::vector<double> T_x_bulk(N, T_init);
    std::vector<double> T_x_v(N, T_init);
    std::vector<double> T_v_bulk(N, T_init);

    // Wick fields
    std::vector<double> u_x(N, -0.0001);                                            /// Wick velocity field [m/s]
    std::vector<double> p_x(N, vapor_sodium::P_sat(T_v_bulk[N - 1]));               /// Wick pressure field [Pa]
    std::vector<double> p_storage_x(N + 2, vapor_sodium::P_sat(T_x_v[N - 1]));      /// Wick padded pressure vector for R&C correction [Pa]
    double* p_padded_x = &p_storage_x[1];                                           /// Poìnter to work on the wick pressure padded storage with the same indes

    std::vector<double> u_x_old = u_x;
    std::vector<double> p_x_old = p_x;

    // Vapor fields
    std::vector<double> u_v(N, 0.1);                                                /// Vapor velocity field [m/s]
    std::vector<double> p_v(N, 0.0);                                                /// Vapor pressure field [Pa]
    std::vector<double> rho_v(N, 8e-3);                                             /// Vapor density field [Pa]
    std::vector<double> p_storage_v(N + 2, vapor_sodium::P_sat(T_v_bulk[N - 1]));   /// Vapor padded pressure vector for R&C correction [Pa]
    double* p_padded_v = &p_storage_v[1];                                           /// Poìnter to work on the storage with the same indes

    std::vector<double> u_v_old = u_v;
    std::vector<double> p_v_old = p_v;
    std::vector<double> rho_v_old = rho_v;

    /// Initialization of the vapor pressure
    for (int i = 0; i < N; ++i) p_v[i] = vapor_sodium::P_sat(T_x_v[i]);

    // Wick BCs
    const double u_inlet_x = 0.0;                               /// Wick inlet velocity [m/s]
    const double u_outlet_x = 0.0;                              /// Wick outlet velocity [m/s]
    double p_outlet_x = vapor_sodium::P_sat(T_x_v[N - 1]);      /// Wick outlet pressure [Pa]

    // Vapor BCs
    const double u_inlet_v = 0.0;                               /// Vapor inlet velocity [m/s]
    const double u_outlet_v = 0.0;                              /// Vapor outlet velocity [m/s]
    double p_outlet_v = vapor_sodium::P_sat(T_v_bulk[N - 1]);   /// Vapor outlet pressure [Pa]

    /// Heat flux at evaporator from given power [W/m^2]
    const double q_pp_evaporator = power / (2 * M_PI * evaporator_length * r_o);     

    // Heat fluxes at the interfaces [W/m^2]
    std::vector<double> q_o_w(N, 0.0);           /// Heat flux [W/m^2] across outer wall in wall region (positive if directed to wall)
    std::vector<double> q_w_x_wall(N, 0.0);      /// Heat flux [W/m^2] across wall-wick interface in the wall region (positive if directed to wick)
    std::vector<double> q_w_x_wick(N, 0.0);      /// Heat flux [W/m^2] across wall-wick interface in the wick region (positive if directed to wick)
    std::vector<double> q_x_v_wick(N, 0.0);      /// Heat flux [W/m^2] across wick-vapor interface in the wick region (positive if directed to vapor)
    std::vector<double> q_x_v_vapor(N, 0.0);     /// Heat flux [W/m^2] across wick-vapor interface in the vapor region (positive if directed to vapor)

    std::vector<double> Q_flux_wall(N, 0.0);     /// Wall heat source due to fluxes [W/m3]
    std::vector<double> Q_flux_wick(N, 0.0);     /// Wick heat source due to fluxes [W/m3]
    std::vector<double> Q_flux_vapor(N, 0.0);    /// Vapor heat source due to fluxes[W/m3]

    std::vector<double> Q_mass_vapor(N, 0.0);    /// Heat volumetric source [W/m3] due to evaporation condensation. To be summed to the vapor
    std::vector<double> Q_mass_wick(N, 0.0);     /// Heat volumetric source [W/m3] due to evaporation condensation. To be summed to the wick

    // Mass sources/fluxes
    std::vector<double> phi_x_v(N, 0.0);            /// Mass flux [kg/m2/s] at the wick-vapor interface (positive if evaporation)
    std::vector<double> Gamma_xv_vapor(N, 0.0);     /// Volumetric mass source [kg / (m^3 s)] (positive if evaporation)
    std::vector<double> Gamma_xv_wick(N, 0.0);      /// Volumetric mass source [kg / (m^3 s)] (positive if evaporation)

    // Old values
    std::vector<double> T_o_w_old = T_o_w;
    std::vector<double> T_w_bulk_old = T_w_bulk;
    std::vector<double> T_w_x_old = T_w_x;
    std::vector<double> T_x_bulk_old = T_x_bulk;
    std::vector<double> T_x_v_old = T_x_v;
    std::vector<double> T_v_bulk_old = T_v_bulk;

    // Iter values
    std::vector<double> T_o_w_iter(N, 0.0);
    std::vector<double> T_w_bulk_iter(N, 0.0);
    std::vector<double> T_w_x_iter(N, 0.0);
    std::vector<double> T_x_bulk_iter(N, 0.0);
    std::vector<double> T_x_v_iter(N, 0.0);
    std::vector<double> T_v_bulk_iter(N, 0.0);

    // Turbulence constants for sodium vapor (SST model)
    const double Pr_t = 0.01;                                   /// Prandtl turbulent number for sodium vapor [-]
    const double I = 0.05;                                      /// Turbulence intensity [-]
    const double L_t = 0.07 * L;                                /// Turbulence length scale [m]
    const double k0 = 1.5 * pow(I * u_inlet_v, 2);              /// Initial turbulent kinetic energy [m^2/s^2]
    const double omega0 = sqrt(k0) / (0.09 * L_t);              /// Initial specific dissipation rate [1/s]
    const double sigma_k = 0.85;                                /// k-equation turbulent Prandtl number [-]
    const double sigma_omega = 0.5;                             /// ω-equation turbulent Prandtl number [-]
    const double beta_star = 0.09;                              /// β* constant for SST model [-]
    const double beta = 0.075;                                  /// β constant for turbulence model [-]
    const double alpha = 5.0 / 9.0;                             /// α blending coefficient for SST model [-]

    // Turbulence fields for sodium vapor
    std::vector<double> k_turb(N, k0);
    std::vector<double> omega_turb(N, omega0);
    std::vector<double> mu_t(N, 0.0);
    
    // Vapor Equation of State update function. Updates density
    auto eos_update = [&](std::vector<double>& rho_, const std::vector<double>& p_, const std::vector<double>& T_) {

        for (int i = 0; i < N; i++) { rho_[i] = std::max(1e-6, p_[i] / (Rv * T_[i])); }

    }; eos_update(rho_v, p_v, T_v_bulk);

    // Models
    const int rhie_chow_on_off_x = 1;             /// 0: no wick RC correction, 1: wick with RC correction
    const int rhie_chow_on_off_v = 1;             /// 0: no vapor RC correction, 1: vapor with RC correction
    const int SST_model_turbulence_on_off = 0;    /// 0: no vapor turbulence, 1: vapor with turbulence


    /// The coefficient bVU is needed in momentum predictor loop and 
    /// pressure correction to estimate the velocities at the faces using the Rhie and Chow correction
	/// as well as to estimate the dt limit due to compressibility
    std::vector<double> 
        aVU(N, 0.0),                                                /// Lower tridiagonal coefficient for vapor velocity
        bVU(N, 2 * (4.0 / 3.0 * vapor_sodium::mu(T_init) / dz) 
            + dz / dt_user * rho_v[0]),                                  /// Central tridiagonal coefficient for vapor velocity
        cVU(N, 0.0),                                                /// Upper tridiagonal coefficient for vapor velocity
        dVU(N, 0.0);                                                /// Known vector for vapor velocity

    // Create result folder
    int new_case = 0;
    std::string name = "case_0";
    while (true) {
        name = "case_" + std::to_string(new_case);
        if (!std::filesystem::exists(name)) {
            std::filesystem::create_directory(name);
            break;
        }
        new_case++;
    }

    // Print results in file
    std::ofstream mesh_output(name + "/mesh.txt", std::ios::trunc);
    std::ofstream time_output(name + "/time.txt", std::ios::trunc);

    std::ofstream v_velocity_output(name + "/vapor_velocity.txt", std::ios::trunc);
    std::ofstream v_pressure_output(name + "/vapor_pressure.txt", std::ios::trunc);
    std::ofstream v_bulk_temperature_output(name + "/vapor_bulk_temperature.txt", std::ios::trunc);

    std::ofstream x_velocity_output(name + "/wick_velocity.txt", std::ios::trunc);
    std::ofstream x_pressure_output(name + "/wick_pressure.txt", std::ios::trunc);
    std::ofstream x_bulk_temperature_output(name + "/wick_bulk_temperature.txt", std::ios::trunc);

    std::ofstream x_v_temperature_output(name + "/wick_vapor_interface_temperature.txt", std::ios::trunc);
    std::ofstream w_x_temperature_output(name + "/wall_wick_interface_temperature.txt", std::ios::trunc);
    std::ofstream o_w_temperature_output(name + "/outer_wall_temperature.txt", std::ios::trunc);
    std::ofstream w_bulk_temperature_output(name + "/wall_bulk_temperature.txt", std::ios::trunc);

    std::ofstream x_v_mass_flux_output(name + "/wick_vapor_mass_source.txt", std::ios::trunc);

    std::ofstream o_w_heat_flux_output(name + "/outer_wall_heat_flux.txt", std::ios::trunc);
    std::ofstream w_x_heat_flux_output(name + "/wall_wick_heat_flux.txt", std::ios::trunc);
    std::ofstream x_v_heat_flux_output(name + "/wick_vapor_heat_flux.txt", std::ios::trunc);

    std::ofstream rho_output(name + "/rho_vapor.txt", std::ios::trunc);

    const int global_precision = 8;

    mesh_output << std::setprecision(global_precision);
    time_output << std::setprecision(global_precision);

    v_velocity_output << std::setprecision(global_precision);
    v_pressure_output << std::setprecision(global_precision);
    v_bulk_temperature_output << std::setprecision(global_precision);

    x_velocity_output << std::setprecision(global_precision);
    x_pressure_output << std::setprecision(global_precision);
    x_bulk_temperature_output << std::setprecision(global_precision);

    x_v_temperature_output << std::setprecision(global_precision);
    w_x_temperature_output << std::setprecision(global_precision);
    o_w_temperature_output << std::setprecision(global_precision);
    w_bulk_temperature_output << std::setprecision(global_precision);

    x_v_mass_flux_output << std::setprecision(global_precision);

    o_w_heat_flux_output << std::setprecision(global_precision);
    w_x_heat_flux_output << std::setprecision(global_precision);
    x_v_heat_flux_output << std::setprecision(global_precision);

    rho_output << std::setprecision(global_precision);

    for (int i = 0; i < N; ++i) mesh_output << i * dz << " ";

    mesh_output.flush();
    mesh_output.close();

    #pragma endregion

    /// Print number of working threads
    std::cout << "Threads: " << omp_get_max_threads() << "\n";

    double start = omp_get_wtime();

    int n = 0;

    double dt_code = dt_user;
	double dt = dt_user;

    /**
     * @brief Time-stepping loop. The timestep is calculated at the beginning of each loop.
     */
    for (int n = 0; n < nSteps; ++n) { 

        double dt_cand_w = new_dt_w(dz, dt, T_w_bulk, Q_flux_wall);
        double dt_cand_x = new_dt_x(dz, dt, u_x, T_x_bulk, Gamma_xv_wick, Q_flux_wick);
        double dt_cand_v = new_dt_v(dz, dt, u_v, T_v_bulk, rho_v, Gamma_xv_vapor, Q_flux_vapor, bVU);

        dt_code = std::min(std::min(dt_cand_w, dt_cand_x), std::min(dt_cand_x, dt_cand_v));

        dt = std::min(dt_user, dt_code);

        time_total += dt;

		// Picard: iter values = old values
        for (int i = 0; i < N; ++i) {

             T_o_w_iter = T_o_w_old;
             T_w_bulk_iter = T_w_bulk_old;
             T_w_x_iter = T_w_x_old;
             T_x_bulk_iter = T_x_bulk_old;
             T_x_v_iter = T_x_v_old;
             T_v_bulk_iter = T_v_bulk_old;
        }

		// Store old values for time derivative
        for (int i = 0; i < N; ++i) {

            p_v_old = p_v;
            p_x_old = p_x;
            u_v_old = u_v;
            u_x_old = u_x;
            rho_v_old = rho_v;
        }

        for (int pic = 0; pic < maxPicard; pic++) {

            // =======================================================================
            //
            //                           [1. SOLVE WALL]
            //
            // =======================================================================

            #pragma region wall

            std::vector<double> 
                aTW(N, 0.0),                    /// Lower tridiagonal coefficient for wall temperature
                bTW(N, 0.0),                    /// Central tridiagonal coefficient for wall temperature
                cTW(N, 0.0),                    /// Upper tridiagonal coefficient for wall temperature
                dTW(N, 0.0);                    /// Known vector coefficient for wall temperature

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
             * @brief Loop on nodes. 
             * Assembles the coefficients for the tridiagonal system of wall temperature
             */
            for (int i = 1; i < N - 1; ++i) {

                /// Physical properties
                const double cp = steel::cp(T_w_bulk_iter[i]);
                const double rho = steel::rho(T_w_bulk_iter[i]);

                const double k_P = steel::k(T_w_bulk_iter[i]);
                const double k_L = steel::k(T_w_bulk_iter[i - 1]);
                const double k_R = steel::k(T_w_bulk_iter[i + 1]);

                /// Volumetric heat source in the wall due to heat flux at the outer wall [W/m^3]
                const double volum_heat_source_o_w = q_o_w[i] * 2 * r_o / (r_o * r_o - r_i * r_i);

                /// Volumetric heat source in the wall due to heat flux at the wall-wick interface [W/m^3]
                const double volum_heat_source_w_x = q_w_x_wall[i] * 2 * r_i / (r_o * r_o - r_i * r_i);

                /// Wall volumetric heat source [W/m^3]
                Q_flux_wall[i] = volum_heat_source_o_w - volum_heat_source_w_x;

                aTW[i] = - k_L / (rho * cp * dz * dz);
                bTW[i] = 1 / dt + (k_L + k_R) / (rho * cp * dz * dz);
                cTW[i] = - k_R / (rho * cp * dz * dz);
                dTW[i] = T_w_bulk_old[i] / dt + Q_flux_wall[i] / (cp * rho);
            }

            /// Vector of final wall bulk temperatures
            T_w_bulk = solveTridiagonal(aTW, bTW, cTW, dTW);        

            #pragma endregion

            // =======================================================================
            //
            //                           [2. SOLVE WICK]
            //
            // =======================================================================

            #pragma region wick

            /*
            /// Evaluates maximum absolute value of the wick velocity
            const double max_abs_u =
                std::abs(*std::max_element(u_x.begin(), u_x.end(),
                    [](double a, double b) { return std::abs(a) < std::abs(b); }
                ));

            /// Evaluates minimum value of the wick temperature
            const double min_T_wick = *std::min_element(T_x_bulk.begin(), T_x_bulk.end());

            std::cout << "Solving wick! Time elapsed:" << dt * n << "/" << time_total
                << ", max courant number: " << max_abs_u * dt / dz
                << ", max reynolds number: " << max_abs_u * r_i *liquid_sodium::rho(min_T_wick) / liquid_sodium::mu(min_T_wick) << "\n";
            */

            /**
             * Pressure coupling hypotheses: the meniscus in the last cell of the domain is 
             * considered flat, so the pressure of the wick is equal to the pressure of the vapor
             */
            p_outlet_x = p_v[N - 1];

            /// Outer iterations variables reset
            double u_error_x = 1.0;
            int outer_iter_x = 0;

            /// Inner iterations variables definition
            double p_error_x;
            int inner_iter_x;

            /// The coefficient bXU is needed in momentum predictor loop and pressure correction to estimate the velocities at the faces using the Rhie and Chow correction
            std::vector<double>
                aXU(N, 0.0),                                                /// Lower tridiagonal coefficient for wick velocity
                bXU(N, liquid_sodium::rho(T_init) * dz / dt
                    + 2 * liquid_sodium::mu(1000) / dz),                    /// Central tridiagonal coefficient for wick velocity
                cXU(N, 0.0),                                                /// Upper tridiagonal coefficient for wick velocity
                dXU(N, 0.0);                                                /// Known vector coefficient for wick velocity

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

				    /// Physical properties
                    const double rho_P = liquid_sodium::rho(T_x_bulk_iter[i]);    
                    const double rho_L = liquid_sodium::rho(T_x_bulk_iter[i - 1]);
                    const double rho_R = liquid_sodium::rho(T_x_bulk_iter[i + 1]);

                    const double rho_P_old = liquid_sodium::rho(T_x_bulk_old[i]);

                    const double mu_P = liquid_sodium::mu(T_x_bulk_iter[i]);
                    const double mu_L = liquid_sodium::mu(T_x_bulk_iter[i - 1]);
                    const double mu_R = liquid_sodium::mu(T_x_bulk_iter[i + 1]);

                    const double D_l = 0.5 * (mu_P + mu_L) / dz;        
                    const double D_r = 0.5 * (mu_P + mu_R) / dz;

                    const double invbX_L = 1.0 / bXU[i - 1] + 1.0 / bXU[i];
				    const double invbX_R = 1.0 / bXU[i + 1] + 1.0 / bXU[i];

                    /// RC correction for the left face velocity
                    const double rhie_chow_l = 
                        - invbX_L / (8 * dz) * 
                        (p_padded_x[i - 2] - 3 * p_padded_x[i - 1] + 3 * p_padded_x[i] - p_padded_x[i + 1]);
                
                    /// RC correction for the right face velocity 
                    const double rhie_chow_r = 
                        - invbX_R / (8 * dz) * 
                        (p_padded_x[i - 1] - 3 * p_padded_x[i] + 3 * p_padded_x[i + 1] - p_padded_x[i + 2]);

                    /// Left face velocity [m/s] (average + RC correction)
                    const double u_l_face = 0.5 * (u_x[i - 1] + u_x[i]) + rhie_chow_on_off_x * rhie_chow_l;

                    /// Right face velocity [m/s] (average + RC correction)
                    const double u_r_face = 0.5 * (u_x[i] + u_x[i + 1]) + rhie_chow_on_off_x * rhie_chow_r;

				    /// Upwind densities at the faces
                    const double rho_l = (u_l_face >= 0) ? rho_L : rho_P;
                    const double rho_r = (u_r_face >= 0) ? rho_P : rho_R; 

                    const double F_l = rho_l * u_l_face;    /// Mass flux [kg/(m2 s)] of the left face 
                    const double F_r = rho_r * u_r_face;    /// Mass flux [kg/(m2 s)] of the right face 

                    aXU[i] = 
                        - std::max(F_l, 0.0) 
                        - D_l;
                    cXU[i] = 
                        - std::max(-F_r, 0.0) 
                        - D_r;
                    bXU[i] = 
                        + std::max(F_r, 0.0) + std::max(-F_l, 0.0) 
                        + rho_P * dz / dt 
                        + D_l + D_r 
                        + mu_P / K * dz 
                        + CF * mu_P * dz / sqrt(K) * abs(u_x[i]);
                    dXU[i] = 
                        - 0.5 * (p_x[i + 1] - p_x[i - 1])
                        + rho_P_old * u_x_old[i] * dz / dt;
                }

                /// Diffusion coefficients for the first and last node to define BCs
                const double D_first = liquid_sodium::mu(T_x_bulk_iter[0]) / dz;
                const double D_last = liquid_sodium::mu(T_x_bulk_iter[N - 1]) / dz;

                /// Velocity BCs: zero velocity on the first node
			    aXU[0] = 0.0;
                bXU[0] = (liquid_sodium::rho(T_x_bulk_iter[0]) * dz / dt + 2 * D_first);
                cXU[0] = 0.0; 
                dXU[0] = (liquid_sodium::rho(T_x_bulk_iter[0]) * dz / dt + 2 * D_first) * u_inlet_x;
            
                /// Velocity BCs: zero velocity on the last node
                aXU[N - 1] = 0.0; 
                bXU[N - 1] = (liquid_sodium::rho(T_x_bulk_iter[N - 1]) * dz / dt + 2 * D_last);
			    cXU[N - 1] = 0.0;
                dXU[N - 1] = (liquid_sodium::rho(T_x_bulk_iter[N - 1]) * dz / dt + 2 * D_last) * u_outlet_x;

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
                    std::vector<double> 
                        aXP(N, 0.0), 
                        bXP(N, 0.0), 
                        cXP(N, 0.0), 
                        dXP(N, 0.0);

                    std::vector<double> p_prime_x(N, 0.0);    /// Wick correction pressure field [Pa]

                    /**
                     * @brief Calculates the coefficients for the tridiagonal system of the pressure correction p'
                     */
                    for (int i = 1; i < N - 1; i++) {

					    /// Physical properties
                        const double rho_P = liquid_sodium::rho(T_x_bulk_iter[i]);
                        const double rho_L = liquid_sodium::rho(T_x_bulk_iter[i - 1]);
                        const double rho_R = liquid_sodium::rho(T_x_bulk_iter[i + 1]);

                        const double d_l_face = 0.5 * (1.0 / bXU[i - 1] + 1.0 / bXU[i]) / dz;    /// 1/Ap [(m2 s)/kg] average on left face
                        const double d_r_face = 0.5 * (1.0 / bXU[i] + 1.0 / bXU[i + 1]) / dz;    /// 1/Ap [(m2 s)/kg] average on right face

                        /// RC correction for the left face velocity
                        const double rhie_chow_l = -d_l_face / 4 * 
                            (p_padded_x[i - 2] - 3 * p_padded_x[i - 1] + 3 * p_padded_x[i] - p_padded_x[i + 1]);

                        /// RC correction for the right face velocity 
                        const double rhie_chow_r = -d_r_face / 4 * 
                            (p_padded_x[i - 1] - 3 * p_padded_x[i] + 3 * p_padded_x[i + 1] - p_padded_x[i + 2]);

                        /// Density [kg/m3] on the left face
                        const double rho_l = 0.5 * (rho_L + rho_P);

                        /// Diffusion coefficient [s/m] on the left face
                        const double E_l = rho_l * d_l_face;                                

                        /// Density [kg/m3] on the right face
                        const double rho_r = 0.5 * (rho_P + rho_R);

                        /// Diffusion coefficient [s/m] on the right face
                        const double E_r = rho_r * d_r_face;                                

                        /// Left face velocity [m/s] (average + RC correction)
                        const double u_l_star = 0.5 * (u_x[i - 1] + u_x[i]) + rhie_chow_on_off_x * rhie_chow_l;

                        /// Right face velocity [m/s] (average + RC correction)
                        const double u_r_star = 0.5 * (u_x[i] + u_x[i + 1]) + rhie_chow_on_off_x * rhie_chow_r;

                        /// Left face mass flux [kg/(m^2 s)] (upwind discretization)
                        const double phi_l_star = (u_l_star > 0.0) ? rho_L * u_l_star : rho_P * u_l_star;  

                        /// Right face mass flux [kg/(m^2 s)] (upwind discretization)
                        const double phi_r_star = (u_r_star > 0.0) ? rho_P * u_r_star : rho_R * u_r_star; 

                        /// Mass difference of the fluxes across faces [kg/(m^2 s)]
                        const double mass_imbalance = (phi_r_star - phi_l_star); 

                        /// Mass flux [kg/(m^2 s)] due to external source (positive if out of the wick)
                        const double mass_flux = Gamma_xv_wick[i] * dz;

                        aXP[i] = -E_l;                              /// [s/m]
                        cXP[i] = -E_r;                              /// [s/m]
                        bXP[i] = E_l + E_r;                         /// [s/m]
                        dXP[i] = -mass_flux - mass_imbalance;       /// [kg/(m^2 s)]
                    }

                    /// BCs for the correction of pressure: zero gradient at first node
				    aXP[0] = 0.0;
                    bXP[0] = 1.0; 
                    cXP[0] = -1.0; 
                    dXP[0] = 0.0;

                    /// BCs for the correction of pressure: zero at first node
                    aXP[N - 1] = 0.0;
                    bXP[N - 1] = 1.0; 
				    cXP[N - 1] = 0.0;
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
            std::vector<double> 
                aXT(N, 0.0), 
                bXT(N, 0.0), 
                cXT(N, 0.0), 
                dXT(N, 0.0);

            /**
             * @brief Calculates the coefficients for the tridiagonal system of the wick temperature T
             */
            for (int i = 1; i < N - 1; i++) {

                /// Physical properties
                const double rho_P = liquid_sodium::rho(T_x_bulk_iter[i]);
                const double rho_L = liquid_sodium::rho(T_x_bulk_iter[i - 1]);
                const double rho_R = liquid_sodium::rho(T_x_bulk_iter[i + 1]);

                const double rho_P_old = liquid_sodium::rho(T_x_bulk_old[i]);

                const double k_cond_P = liquid_sodium::k(T_x_bulk_iter[i]);
                const double k_cond_L = liquid_sodium::k(T_x_bulk_iter[i - 1]);
                const double k_cond_R = liquid_sodium::k(T_x_bulk_iter[i + 1]);

                const double cp_P = liquid_sodium::cp(T_x_bulk_iter[i]);
                const double cp_L = liquid_sodium::cp(T_x_bulk_iter[i - 1]);
                const double cp_R = liquid_sodium::cp(T_x_bulk_iter[i + 1]);
                 
                const double cp_P_old = liquid_sodium::cp(T_x_bulk_old[i]);

                /// Diffusion coefficient [W/(m^2 K)] of the left face (average)
                const double D_l = 0.5 * (k_cond_P + k_cond_L) / dz;        

                /// Diffusion coefficient [W/(m^2 K)] of the right face (average)
                const double D_r = 0.5 * (k_cond_P + k_cond_R) / dz;       

                const double invbX_L = 1.0 / bXU[i - 1] + 1.0 / bXU[i];
                const double invbX_R = 1.0 / bXU[i + 1] + 1.0 / bXU[i];

                /// RC correction for the left face velocity
                const double rhie_chow_l = -invbX_L / (8 * dz) *
                    (p_padded_x[i - 2] - 3 * p_padded_x[i - 1] + 3 * p_padded_x[i] - p_padded_x[i + 1]);

                /// RC correction for the right face velocity 
                const double rhie_chow_r = -invbX_R / (8 * dz) *
                    (p_padded_x[i - 1] - 3 * p_padded_x[i] + 3 * p_padded_x[i + 1] - p_padded_x[i + 2]);

                /// Left face velocity [m/s] (average + RC correction)
                const double u_l_face = 0.5 * (u_x[i - 1] + u_x[i]) + rhie_chow_on_off_x * rhie_chow_l;

                /// Right face velocity [m/s] (average + RC correction)
                const double u_r_face = 0.5 * (u_x[i] + u_x[i + 1]) + rhie_chow_on_off_x * rhie_chow_r;

                /// Density [kg/m3] of the left face (upwind)
                const double rho_l = (u_l_face >= 0) ? rho_L : rho_P;   

                /// Density [kg/m3] of the right face (upwind)
                const double rho_r = (u_r_face >= 0) ? rho_P : rho_R;     

                /// S. h. at constant pressure [J/(kg K)] of the left face (upwind)
                const double cp_l = (u_l_face >= 0) ? cp_L : cp_P;

                /// S. h. at constant pressure [J/(kg K)] of the right face (upwind)
                const double cp_r = (u_r_face >= 0) ? cp_P : cp_R;         

                /// Mass flux [kg/(m2 s)] of the left face (upwind)
                const double Fl = rho_l * u_l_face;                 

                /// Mass flux [kg/(m2 s)] of the left face (upwind)
                const double Fr = rho_r * u_r_face;                         

                /// Mass flux [W/(m^2 K)] of the left face (upwind)
                const double C_l = (Fl * cp_l);

                /// Mass flux [W/(m^2 K)] of the left face (upwind)
                const double C_r = (Fr * cp_r);                             

                /// Volumetric heat source due to heat flux at the wall-wick interface [W/m^3]
                double volum_heat_source_w_x = q_w_x_wick[i] * 2 * r_i / (r_i * r_i - r_v * r_v);

                /// Volumetric heat source due to heat flux at the wick-vapor interface [W/m^3]
                double volum_heat_source_x_v = q_x_v_wick[i] * 2 * r_v / (r_i * r_i - r_v * r_v);

                /**
                  * Wick volumetric heat source [W/m^3] due to phase change and interface heat flux
                  */ 
                Q_flux_wick[i] = volum_heat_source_w_x - volum_heat_source_x_v;  

                aXT[i] = 
                    - D_l 
                    - std::max(C_l, 0.0);       /// [W/(m2 K)]
                cXT[i] = 
                    - D_r 
                    - std::max(-C_r, 0.0);      /// [W/(m2 K)]
                bXT[i] = 
                    (std::max(C_r, 0.0) + std::max(-C_l, 0.0)) 
                    + D_l + D_r 
                    + rho_P * cp_P * dz / dt;   /// [W/(m2 K)]
                dXT[i] =
                    + rho_P_old * cp_P_old * dz / dt * T_x_bulk_old[i]
                    + Q_flux_wick[i] * dz
                    + Q_mass_wick[i] * dz;              /// [W/m2]
            }

            /// Temperature BCs: zero gradient on the first node
		    aXT[0] = 0.0;
            bXT[0] = 1.0; 
            cXT[0] = -1.0; 
            dXT[0] = 0.0;

            /// Temperature BCs: zero gradient on the last node
            aXT[N - 1] = -1.0; 
            bXT[N - 1] = 1.0; 
		    cXT[N - 1] = 0.0;
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

            for (int i = 0; i < N; ++i) {

                /// Mass flux from the wick to the vapor [kg/(m2 s)]
                phi_x_v[i] = (sigma_e * vapor_sodium::P_sat(T_x_v_iter[i]) / std::sqrt(T_x_v_iter[i]) -
                                sigma_c * Omega * p_v[i] / std::sqrt(T_v_bulk_iter[i])) /
                                (std::sqrt(2 * M_PI * Rv));

                Gamma_xv_vapor[i] = phi_x_v[i] * 2.0 * eps_s / r_v;
                Gamma_xv_wick[i] = phi_x_v[i] * (2.0 * r_v * eps_s) / (r_i * r_i - r_v * r_v);

                /**
                 * Variable b [-], used to calculate omega. 
                 * Ratio of the overrall speed to the most probable velocity of the vapor.
                 */
                const double b = std::abs(-phi_x_v[i] / (p_v[i] * std::sqrt(2.0 / (Rv * T_v_bulk_iter[i]))));

                /**
                  * Linearization of the omega [-] function to correct the net evaporation/condensation mass flux
                  */
                if (b < 0.1192) Omega = 1.0 + b * std::sqrt(M_PI);
                else if (b <= 0.9962) Omega = 0.8959 + 2.6457 * b;
                else Omega = 2.0 * b * std::sqrt(M_PI);

                const double k_bulk_w = steel::k(T_w_bulk_iter[i]);                             /// Wall bulk thermal conductivity [W/(m K)]
                const double k_bulk_x = liquid_sodium::k(T_x_bulk_iter[i]);                     /// Liquid bulk thermal conductivity [W/(m K)]
                const double cp_v = vapor_sodium::cp(T_v_bulk_iter[i]);                         /// Vapor bulk specific heat [J/(kg K)]
				const double k_int_w = steel::k(T_w_x_iter[i]);                                 /// Wall interfacial thermal conductivity [W/(m K)]
				const double k_int_x = liquid_sodium::k(T_w_x_iter[i]);                         /// Wick interfacial thermal conductivity [W/(m K)]     
				const double k_x_v = liquid_sodium::k(T_x_v[i]);                                /// Wick-vapor interface thermal conductivity [W/(m K)]  
				const double k_v_x = vapor_sodium::k(T_x_v[i], p_v[i]);                    /// Vapor-wick interface thermal conductivity [W/(m K)]
                const double k_v_cond = vapor_sodium::k(T_v_bulk_iter[i], p_v[i]);         /// Vapor thermal conductivity [W/(m K)]
                const double k_v_eff = k_v_cond + mu_t[i] * cp_v / Pr_t;                        /// Effective vapor thermal conductivity [W/(m K)]
                const double mu_v = vapor_sodium::mu(T_v_bulk_iter[i]);                         /// Vapor dynamic viscosity [Pa*s]
                const double Dh_v = 2.0 * r_v;                                              /// Hydraulic diameter of the vapor core [m]
                const double Re_v = rho_v[i] * std::fabs(u_v[i]) * Dh_v / mu_v;       /// Reynolds number [-]
                const double Pr_v = cp_v * mu_v / k_v_cond;                                     /// Prandtl number [-]
                const double H_xm = vapor_sodium::h_conv(Re_v, Pr_v, k_v_cond, Dh_v);           /// Convective heat transfer coefficient at the vapor-wick interface [W/m^2/K]
                const double Psat = vapor_sodium::P_sat(T_x_v_iter[i]);                         /// Saturation pressure [Pa]        
                const double dPsat_dT = Psat * std::log(10.0) * (7740.0 / (T_x_v_iter[i] * T_x_v_iter[i]));   /// Derivative of the saturation pressure wrt T [Pa/K]   

                const double fac = (2.0 * r_v * eps_s * beta) / (r_i * r_i);    /// Useful factor in the coefficients calculation [s / m^2]

                double h_xv_v;          /// Specific enthalpy [J/kg] of vapor upon phase change between wick and vapor
                double h_vx_x;          /// Specific enthalpy [J/kg] of wick upon phase change between vapor and wick

                if (phi_x_v[i] > 0.0) {

                    // Evaporation case
                    h_xv_v = vapor_sodium::h(T_x_v_iter[i]);
                    h_vx_x = liquid_sodium::h(T_x_v_iter[i]);

                } else {

                    // Condensation case
                    h_xv_v = vapor_sodium::h(T_v_bulk_iter[i]);
                    h_vx_x = liquid_sodium::h(T_x_v_iter[i])
                        + (vapor_sodium::h(T_v_bulk_iter[i]) - vapor_sodium::h(T_x_v_iter[i]));
                }

                /// Coefficients for the parabolic temperature profiles in wall and wick (check equations)
                const double E1w = 2.0 / 3.0 * (r_o + r_i - 1 / (1 / r_o + 1 / r_i));
                const double E2w = 0.5 * (r_o * r_o + r_i * r_i);
                const double E1x = 2.0 / 3.0 * (r_i + r_v - 1 / (1 / r_i + 1 / r_v));
                const double E2x = 0.5 * (r_i * r_i + r_v * r_v);

                const double E3 = H_xm;
                const double E4 = -k_bulk_x + H_xm * r_v;
                const double E5 = -2.0 * r_v * k_bulk_x + H_xm * r_v * r_v;
                const double E6 = H_xm * T_v_bulk_iter[i] - (h_xv_v - h_vx_x) * phi_x_v[i];

                const double alpha = 1.0 / (2 * r_o * (E1w - r_i) + r_i * r_i - E2w);
                const double delta = T_x_bulk_iter[i] - T_w_bulk_iter[i] + q_o_w[i] / k_bulk_w * (E1w - r_i) - 
                    (E1x - r_i) * (E6 - E3 * T_x_bulk_iter[i]) / (E4 - E1x * E3);
                const double gamma = r_i * r_i + ((E5 - E2x * E3) * (E1x - r_i)) / (E4 - E1x * E3) - E2x;

                ABC[i][5] = (-q_o_w[i] * k_int_w / k_bulk_w +
                    2 * k_int_w * (r_o - r_i) * alpha * delta +
                    k_int_x * (E6 - E3 * T_x_bulk_iter[i]) / (E4 - E1x * E3)) /
                    (2 * (r_i - r_o) * k_int_w * alpha * gamma +
                        (E5 - E2x * E3) / (E4 - E1x * E3) * k_int_x -
                        2 * r_i * k_int_x);
                ABC[i][2] = alpha * (delta + gamma * ABC[i][5]);
                ABC[i][1] = q_o_w[i] / k_bulk_w - 2 * r_o * ABC[i][2];
                ABC[i][0] = T_w_bulk_iter[i] - E1w * q_o_w[i] / k_bulk_w + (2 * r_o * E1w - E2w) * ABC[i][2];
                ABC[i][4] = (E6 - E3 * T_x_bulk_iter[i] - (E5 - E2x * E3) * ABC[i][5]) / (E4 - E1x * E3);
                ABC[i][3] = T_x_bulk_iter[i] - E1x * ABC[i][4] - E2x * ABC[i][5];

                // Update temperatures at the interfaces
                T_o_w[i] = ABC[i][0] + ABC[i][1] * r_o + ABC[i][2] * r_o * r_o; /// Temperature at the outer wall
                T_w_x[i] = ABC[i][0] + ABC[i][1] * r_i + ABC[i][2] * r_i * r_i; /// Temperature at the wall wick interface
                T_x_v[i] = ABC[i][3] + ABC[i][4] * r_v + ABC[i][5] * r_v * r_v; /// Temperature at the wick vapor interface

                std::vector<double> q_raw(N, 0.0);

                for (int i = 0; i < N; ++i) {

                    if (i <= evaporator_nodes) q_raw[i] = q_pp_evaporator;
                    else if (i >= N - condenser_nodes) {

                        double conv = h_conv * (T_o_w_iter[i] - T_env);
                        double irr = emissivity * sigma *
                            (std::pow(T_o_w_iter[i], 4) - std::pow(T_env, 4));

                        q_raw[i] = -(conv + irr);
                    }
                }

                // Light smoothing (3-point binomial filter)
                std::vector<double> q_smooth(N);

                for (int i = 0; i < N; ++i) {

                    double qm = q_raw[i];
                    double ql = (i > 0 ? q_raw[i - 1] : q_raw[i]);
                    double qr = (i < N - 1 ? q_raw[i + 1] : q_raw[i]);

                    // filtro di smoothing: [0.25, 0.5, 0.25]
                    q_smooth[i] = 0.25 * ql + 0.5 * qm + 0.25 * qr;
                }

                q_o_w = q_smooth;
                                                                                       
                q_w_x_wall[i] = k_int_w * (ABC[i][1] + 2.0 * ABC[i][2] * r_i);  /// Heat flux across wall-wick interface (positive if to wick)
                q_w_x_wick[i] = k_int_x * (ABC[i][1] + 2.0 * ABC[i][2] * r_i);  /// Heat flux across wall-wick interface (positive if to wick)
                q_x_v_wick[i] = k_x_v * (ABC[i][4] + 2.0 * ABC[i][5] * r_v);    /// Heat flux across wick-vapor interface (positive if to vapor)
                q_x_v_vapor[i] = k_v_x * (ABC[i][4] + 2.0 * ABC[i][5] * r_v);   /// Heat flux across wick-vapor interface (positive if to vapor)

                Q_mass_vapor[i] = + Gamma_xv_vapor[i] * h_xv_v; /// Volumetric heat source [W/m3] due to evaporation/condensation (to be summed to the vapor)
                Q_mass_wick[i] = - Gamma_xv_wick[i] * h_vx_x;   /// Volumetric heat source [W/m3] due to evaporation/condensation (to be summed to the wick)
            }

            /// Coupling hypotheses: temperature is transferred to the pressure of the sodium vapor
            p_outlet_v = vapor_sodium::P_sat(T_x_v_iter[N - 1]);

            #pragma endregion

            // =======================================================================
            //
            //                           [4. SOLVE VAPOR]
            //
            // =======================================================================

            #pragma region vapor

            /* 
            /// Evaluates maximum absolute value of the vapor velocity
            const double max_abs_u_v =
                std::abs(*std::max_element(u_x.begin(), u_x.end(),
                    [](double a, double b) { return std::abs(a) < std::abs(b); }
                ));

            /// Evaluates maximum density value of the vapor
            const double max_rho_v = *std::max_element(rho_v.begin(), rho_v.end());

            /// Evaluates minimum temperature value of the vapor
            const double min_T_v = *std::min_element(T_v_bulk.begin(), T_v_bulk.end());

            std::cout << "Solving vapor! Time elapsed:" << dt * n << "/" << time_total
                << ", max courant number: " << max_abs_u_v * dt / dz
                << ", max reynolds number: " << max_abs_u_v * r_i * max_rho_v / vapor_sodium::mu(min_T_v) << "\n";
                
            */

            /// Wick vapor coupling hypotheses
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

				    /// Physical properties
                    const double rho_P = rho_v[i];
                    const double rho_L = rho_v[i - 1];
                    const double rho_R = rho_v[i + 1];

                    const double rho_P_old = rho_v_old[i];

                    const double mu_P = vapor_sodium::mu(T_v_bulk_iter[i]);
                    const double mu_L = vapor_sodium::mu(T_v_bulk_iter[i - 1]);
                    const double mu_R = vapor_sodium::mu(T_v_bulk_iter[i + 1]);

                    /// Diffusion coefficient [kg/(m2 s)] of the left face (average)
                    const double D_l = 4.0 / 3.0 * 0.5 * (mu_P + mu_L) / dz;

                    /// Diffusion coefficient [kg/(m2 s)] of the right face (average)
                    const double D_r = 4.0 / 3.0 * 0.5 * (mu_P + mu_R) / dz;    

				    /// Inverse of b coefficients needed for RC correction
                    const double invbV_L = 1.0 / bVU[i - 1] + 1.0 / bVU[i];
                    const double invbV_R = 1.0 / bVU[i + 1] + 1.0 / bVU[i];

                    /// RC correction for the left face velocity
                    const double rhie_chow_l = -invbV_L / (8 * dz) *
                        (p_padded_v[i - 2] - 3 * p_padded_v[i - 1] + 3 * p_padded_v[i] - p_padded_v[i + 1]);
                
                    /// RC correction for the right face velocity
                    const double rhie_chow_r = -invbV_R / (8 * dz) *
                        (p_padded_v[i - 1] - 3 * p_padded_v[i] + 3 * p_padded_v[i + 1] - p_padded_v[i + 2]);

                    /// Left face velocity [m/s] (average + RC correction)
                    const double u_l_face = 0.5 * (u_v[i - 1] + u_v[i])
                        + rhie_chow_on_off_v * rhie_chow_l;                     

                    /// Right face velocity [m/s] (average + RC correction)
                    const double u_r_face = 0.5 * (u_v[i] + u_v[i + 1])
                        + rhie_chow_on_off_v * rhie_chow_r;                     

                    /// Density [kg/m3] of the left face (upwind)
                    const double rho_l = (u_l_face >= 0) ? rho_L : rho_P;

                    /// Density [kg/m3] of the right face (upwind)
                    const double rho_r = (u_r_face >= 0) ? rho_P : rho_R;       

                    /// Mass flux [kg/(m2 s)] of the left face (upwind)
                    const double F_l = rho_l * u_l_face;

                    /// Mass flux [kg/(m2 s)] of the right face (upwind)
                    const double F_r = rho_r * u_r_face;           

                    /// Reynolds number [-]
                    const double Re = u_v[i] * (2 * r_v) * rho_P / mu_P;

                    /// Friction factor [-] (according to THROHPUT)
                    const double f = (Re < 1187.4) ? 64 / Re : 0.3164 * std::pow(Re, -0.25);  

                    /// Friction force [kg/(m2 s)]
                    const double F = 0.25 * f * rho_P * std::abs(u_v[i]) / r_v;

                    aVU[i] = 
                        - std::max(F_l, 0.0) 
                        - D_l;                              /// [kg/(m2 s)]
                    cVU[i] = 
                        - std::max(-F_r, 0.0) 
                        - D_r;                              /// [kg/(m2 s)]
                    bVU[i] = 
                        + (std::max(F_r, 0.0) + std::max(-F_l, 0.0)) 
                        + rho_P * dz / dt 
                        + D_l + D_r 
                        + F * dz;                           /// [kg/(m2 s)]
                    dVU[i] = 
                        - 0.5 * (p_v[i + 1] - p_v[i - 1])
                        + rho_P_old * u_v_old[i] * dz / dt;         /// [kg/(m s2)]
                }

                /// Diffusion coefficients for the first and last node to define BCs
                const double D_first = 4.0 / 3.0 * 0.5 * 
                    (vapor_sodium::mu(T_v_bulk_iter[0]) + vapor_sodium::mu(T_v_bulk_iter[1])) / dz;
                const double D_last = 4.0 / 3.0 * 0.5 * 
                    (vapor_sodium::mu(T_v_bulk_iter[N - 1]) + vapor_sodium::mu(T_v_bulk_iter[N - 2])) / dz;

                /// Velocity BCs needed variables for the first node
                const double u_r_face_first = 0.5 * (u_v[1]);
                const double rho_r_first = (u_r_face_first >= 0) ? rho_v[0] : rho_v[1];
                const double F_r_first = rho_r_first * u_r_face_first;

                /// Velocity BCs needed variables for the last node
                const double u_l_face_last = 0.5 * (u_v[N - 2]);
                const double rho_l_last = (u_l_face_last >= 0) ? rho_v[N - 2] : rho_v[N - 1];
                const double F_l_last = rho_l_last * u_l_face_last;
            
                /// Velocity BCs: zero velocity on the first node
			    aVU[0] = 0.0;
                bVU[0] = std::max(F_r_first, 0.0) + rho_v[0] * dz / dt + 2 * D_first;
                cVU[0] = 0.0; 
                dVU[0] = bVU[0] * u_inlet_v;

                /// Velocity BCs: zero velocity on the last node
                aVU[N - 1] = 0.0;
                bVU[N - 1] = -std::max(-F_l_last, 0.0) + rho_v[N - 1] * dz / dt + 2 * D_last;
			    cVU[N - 1] = 0.0;
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
                    std::vector<double> 
                        aVP(N, 0.0), 
                        bVP(N, 0.0), 
                        cVP(N, 0.0), 
                        dVP(N, 0.0);

                    std::vector<double> p_prime_v(N, 0.0);    /// Vapor correction pressure field [Pa]

                    /**
                     * @brief Calculates the coefficients for the tridiagonal system of the pressure correction p'
                     */
                    for (int i = 1; i < N - 1; i++) {

					    /// Physical properties
                        const double rho_P = rho_v[i];
                        const double rho_L = rho_v[i - 1];
                        const double rho_R = rho_v[i + 1];

                        /// 1/Ap [(m2 s)/kg] average on left face
                        const double d_l_face = 0.5 * (1.0 / bVU[i - 1] + 1.0 / bVU[i]) / dz;

                        /// 1/Ap [(m2 s)/kg] average on right face
                        const double d_r_face = 0.5 * (1.0 / bVU[i] + 1.0 / bVU[i + 1]) / dz;    

                        /// RC correction for the left face velocity
                        const double rhie_chow_l = - d_l_face / 4 * 
                            (p_padded_v[i - 2] - 3 * p_padded_v[i - 1] + 3 * p_padded_v[i] - p_padded_v[i + 1]);
                    
                        /// RC correction for the right face velocity
                        const double rhie_chow_r = - d_r_face / 4 * 
                            (p_padded_v[i - 1] - 3 * p_padded_v[i] + 3 * p_padded_v[i + 1] - p_padded_v[i + 2]);

                        /// Density [kg/m3] on the left face
                        const double rho_l = 0.5 * (rho_v[i - 1] + rho_v[i]);

                        /// Diffusion coefficient [s/m] on the left face
                        const double E_l = rho_l * d_l_face;                                

                        /// Density [kg/m3] on the right face
                        const double rho_r = 0.5 * (rho_v[i] + rho_v[i + 1]);

                        /// Diffusion coefficient [s/m] on the right face
                        const double E_r = rho_r * d_r_face;                                

                        /// Compressibility [kg/J] assuming ideal gas
                        const double psi_i = 1.0 / (Rv * T_v_bulk_iter[i]);

                        /// Left face velocity [m/s] (average + RC correction)
                        const double u_l_star = 0.5 * (u_v[i - 1] + u_v[i]) +
                            rhie_chow_on_off_v * rhie_chow_l;                   

                        /// Right face velocity [m/s] (average + RC correction)
                        const double u_r_star = 0.5 * (u_v[i] + u_v[i + 1]) +
                            rhie_chow_on_off_v * rhie_chow_r;                   

                        /// Left face mass flux [kg/(m^2 s)] (upwind discretization)
                        const double phi_l_star = (u_l_star > 0.0) ? rho_L * u_l_star : rho_P * u_l_star;  

                        /// Right face mass flux [kg/(m^2 s)] (upwind discretization)
                        const double phi_r_star = (u_r_star > 0.0) ? rho_P * u_r_star : rho_R * u_r_star;  

                        /// Mass difference of the fluxes across faces [kg/(m2 s)]
                        const double mass_imbalance = (phi_r_star - phi_l_star); 

                        /// Mass flux [kg/(m2 s)] due to external source (positive if out of the wick)
                        const double mass_flux_source = Gamma_xv_vapor[i] * dz;

                        /// Term [kg/(m2 s)] related to the change in density
                        const double change_density = (rho_P - rho_v_old[i]) * dz / dt;

                        aVP[i] = 
                            - E_l;              /// [s/m]
                        cVP[i] = 
                            - E_r;              /// [s/m]
                        bVP[i] = 
                            + E_l + E_r 
                            + psi_i * dz / dt;  /// [s/m]
                        dVP[i] = 
                            + mass_flux_source 
                            - mass_imbalance 
                            - change_density;   /// [kg/(m^2 s)]
                    }

                    /// BCs for the correction of pressure: zero gradient at first node
				    aVP[0] = 0.0;
                    bVP[0] = 1.0; 
                    cVP[0] = -1.0; 
                    dVP[0] = 0.0;

                    /// BCs for the correction of pressure: zero at last node
                    aVP[N - 1] = 0.0;
                    bVP[N - 1] = 1.0;  
				    cVP[N - 1] = 0.0;
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
                        p_v[i] += p_prime_v[i];        // Note that PISO does not require an under-relaxation factor
                        p_storage_v[i + 1] = p_v[i];

                        p_error_v = std::max(p_error_v, std::fabs(p_v[i] - p_prev));
                    }

                    p_storage_v[0] = p_storage_v[1];
                    p_storage_v[N + 1] = p_outlet_v;

                    #pragma endregion

                    // Update density with new p,T, to get the density connected to the pressure field of timestep n+1
                    eos_update(rho_v, p_v, T_v_bulk_iter);

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
                        const double sonic_limit = std::sqrt(gamma * Rv * T_v_bulk_iter[i]);

                        const double calc_velocity = u_v[i] -
                            (p_prime_v[i + 1] - p_prime_v[i - 1]) / (2.0 * dz * bVU[i]);

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

                const double sigma_k = 0.85;        /// Diffusion coefficient for k [-]
                const double sigma_omega = 0.5;     /// Diffusion coefficient for ω [-]
                const double beta_star = 0.09;      /// Production limiter coefficient [-]
                const double beta = 0.075;          /// Dissipation coefficient for ω [-]
                const double alpha = 5.0 / 9.0;     /// Blending coefficient [-]

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

                std::vector<double> dudz(N, 0.0);       /// Velocity gradient du/dz [1/s]
                std::vector<double> Pk(N, 0.0);         /// Turbulence production rate term [m2/s3]

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
            std::vector<double> 
                aVT(N, 0.0), 
                bVT(N, 0.0), 
                cVT(N, 0.0), 
                dVT(N, 0.0);

            for (int i = 1; i < N - 1; i++) {

			    /// Physical properties
                const double rho_P = rho_v[i];
                const double rho_L = rho_v[i - 1];
                const double rho_R = rho_v[i + 1];

                const double k_cond_P = vapor_sodium::k(T_v_bulk_iter[i], p_v[i]);
                const double k_cond_L = vapor_sodium::k(T_v_bulk_iter[i - 1], p_v[i - 1]);
                const double k_cond_R = vapor_sodium::k(T_v_bulk_iter[i + 1], p_v[i + 1]);

                const double cp_P = vapor_sodium::cp(T_v_bulk_iter[i]);
                const double cp_L = vapor_sodium::cp(T_v_bulk_iter[i - 1]);
                const double cp_R = vapor_sodium::cp(T_v_bulk_iter[i + 1]);

                const double cp_P_old = vapor_sodium::cp(T_v_bulk_old[i]);

                const double mu_P = vapor_sodium::mu(T_v_bulk_iter[i]);

                const double keff_P = k_cond_P + SST_model_turbulence_on_off * (mu_t[i] * cp_P / Pr_t);     
                const double keff_L = k_cond_L + SST_model_turbulence_on_off * (mu_t[i - 1] * cp_L / Pr_t);
                const double keff_R = k_cond_R + SST_model_turbulence_on_off * (mu_t[i + 1] * cp_R / Pr_t);

                /// Diffusion coefficient [W/(m^2 K)] of the left face (average)
                const double D_l = 0.5 * (keff_P + keff_L) / dz;

                /// Diffusion coefficient [W/(m^2 K)] of the right face (average)
                const double D_r = 0.5 * (keff_P + keff_R) / dz;                        

			    const double invbVU_L = 1.0 / bVU[i - 1] + 1.0 / bVU[i];    
			    const double invbVU_R = 1.0 / bVU[i + 1] + 1.0 / bVU[i]; 

                /// RC correction for the left face velocity
                const double rhie_chow_l = - invbVU_L / (8 * dz) *
                    (p_padded_v[i - 2] - 3 * p_padded_v[i - 1] + 3 * p_padded_v[i] - p_padded_v[i + 1]);
            
                /// RC correction for the right face velocity 
                const double rhie_chow_r = - invbVU_R / (8 * dz) *
                    (p_padded_v[i - 1] - 3 * p_padded_v[i] + 3 * p_padded_v[i + 1] - p_padded_v[i + 2]);

                /// Left face velocity [m/s] (average + RC correction)
                const double u_l_face = 0.5 * (u_v[i - 1] + u_v[i]) + rhie_chow_on_off_v * rhie_chow_l;

                /// Right face velocity [m/s] (average + RC correction)
                const double u_r_face = 0.5 * (u_v[i] + u_v[i + 1]) + rhie_chow_on_off_v * rhie_chow_r;

                /// Density [kg/m3] of the left face (upwind)
                const double rho_l = (u_l_face >= 0) ? rho_L : rho_P; 

                /// Density [kg/m3] of the right face (upwind)
                const double rho_r = (u_r_face >= 0) ? rho_P : rho_R;   

                /// S. h. at constant pressure [J/(kg K)] of the left face (upwind)
                const double cp_l = (u_l_face >= 0) ? cp_L : cp_P;

                /// S. h. at constant pressure [J/(kg K)] of the right face (upwind)
                const double cp_r = (u_r_face >= 0) ? cp_P : cp_R;      

                /// Mass flux [kg/(m2 s)] of the left face (upwind)       
                const double Fl = rho_l * u_l_face;

                /// Mass flux [kg/(m2 s)] of the left face (upwind)
                const double Fr = rho_r * u_r_face;                     

                /// Mass flux [W/(m^2 K)] of the left face (upwind)
                const double C_l = (Fl * cp_l);

                /// Mass flux [W/(m^2 K)] of the left face (upwind)
                const double C_r = (Fr * cp_r);                         

                /// Vapor volumetric heat source [W/m^3] due to phase change and heat flux at the interface
                Q_flux_vapor[i] = 2 * q_x_v_vapor[i] / r_v;

                const double dpdz_up = (u_v[i] >= 0.0)
                    ? u_v[i] * (p_v[i] - p_v[i - 1]) / dz
                    : u_v[i] * (p_v[i + 1] - p_v[i]) / dz;

                aVT[i] = 
                    - D_l 
                    - std::max(C_l, 0.0);                   /// [W/(m2 K)]
                cVT[i] = 
                    -D_r 
                    - std::max(-C_r, 0.0);                  /// [W/(m2 K)]
                bVT[i] = 
                    + (std::max(C_r, 0.0) + std::max(-C_l, 0.0)) 
                    + D_l + D_r 
                    + rho_v[i] * cp_P * dz / dt;        /// [W/(m2 K)]

                const double pressure_work = (p_v[i] - p_v_old[i]) / dt + dpdz_up;

                const double viscous_dissipation = 
                    0.5 * mu_P * ((u_v[i + 1] - u_v[i]) * (u_v[i + 1] - u_v[i])
                        + (u_v[i] + u_v[i - 1]) * (u_v[i] + u_v[i - 1])) / (dz * dz);

                dVT[i] = 
                    + rho_v_old[i] * cp_P_old * dz / dt * T_v_bulk_old[i]
                    + pressure_work * dz
                    + viscous_dissipation * dz
                    + Q_flux_vapor[i] * dz
                    + Q_mass_vapor[i] * dz;                     /// [W/m2]
            }

            /// Temperature BCs: zero gradient on the first node
		    aVT[0] = 0.0;
            bVT[0] = 1.0; 
            cVT[0] = -1.0; 
            dVT[0] = 0.0;

            /// Temperature BCs: zero gradient on the last node
            aVT[N - 1] = -1.0; 
            bVT[N - 1] = 1.0; 
		    cVT[N - 1] = 0.0;
            dVT[N - 1] = 0.0;

            T_v_bulk = solveTridiagonal(aVT, bVT, cVT, dVT);

            // Update density with new p,T
            eos_update(rho_v, p_v, T_v_bulk);

            #pragma endregion

            #pragma endregion

            // Calculate Picard error
            double L1 = 0.0;
            for (int i = 0; i < N; ++i) {

				double Aold = T_v_bulk_iter[i];
                double Anew = T_v_bulk[i];
				double denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
				double eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
				L1 += eps;

				Aold = T_x_bulk_iter[i];
				Anew = T_x_bulk[i];
				denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
				eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);

				L1 += eps;

				Aold = T_w_bulk_iter[i];
				Anew = T_w_bulk[i];
				denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
				eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
				L1 += eps;

                Aold = T_o_w_iter[i];
                Anew = T_o_w[i];
                denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                L1 += eps;

                Aold = T_w_x_iter[i];
                Anew = T_w_x[i];
                denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                L1 += eps;

                Aold = T_x_v_iter[i];
                Anew = T_x_v[i];
                denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                L1 += eps;
            }

            if (L1 < picTolerance)
                break;   // Picard converged

            for (int i = 0; i < N; ++i) {

                T_o_w_iter = T_o_w;
                T_w_bulk_iter = T_w_bulk;
                T_w_x_iter = T_w_x;
                T_x_bulk_iter = T_x_bulk;
                T_x_v_iter = T_x_v;
                T_v_bulk_iter = T_v_bulk;
            }
        }

        // Update solution
        for (int i = 0; i < N; ++i) {

            T_o_w_old = T_o_w;
            T_w_bulk_old = T_w_bulk;
            T_w_x_old = T_w_x;
            T_x_bulk_old = T_x_bulk;
            T_x_v_old = T_x_v;
            T_v_bulk_old = T_v_bulk;
        }

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

        const int output_every = 100;

        if(n % output_every == 0){
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

            time_output << time_total << " ";

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

            time_output.flush();
        }

        #pragma endregion
    }

    time_output.close();

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