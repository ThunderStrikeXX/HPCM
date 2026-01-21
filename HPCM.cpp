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
#include <limits>
#include <cstdint>

#include "tdma.h"
#include "steel.h"
#include "liquid_sodium.h"
#include "vapor_sodium.h"
#include "adaptive_dt.h"

int main() {

    // =======================================================================
    //                       [CONSTANTS AND VARIABLES]
    // =======================================================================

    #pragma region constants_and_variables

    // Mathematical constants
    const double M_PI = 3.141592;           // Pi [-]

    // Physical properties
    const double emissivity = 0.5;          // Wall emissivity [-]
    const double sigma = 5.67e-8;           // Stefan-Boltzmann constant [W/m2K4]
    const double Rv = 361.5;                // Gas constant for the sodium vapor [J/(kgK)]
    
    // Environmental boundary conditions
    const double h_conv = 1;                // Convective heat transfer coefficient for external heat removal [W/m^2/K]
    const double power = 119;               // Power at the evaporator side [W]
    const double T_env = 280.0;             // External environmental temperature [K]

    // Evaporation and condensation parameters
    const double eps_s = 1.0;               // Surface fraction of the wick available for phasic interface [-]
    const double sigma_e = 0.05;            // Evaporation accomodation coefficient [-]. 1 means optimal evaporation
    const double sigma_c = 0.05;            // Condensation accomodation coefficient [-]. 1 means optimal condensation
	double Omega = 1.0;                     // Initialization of Omega parameter for evaporation/condensation model [-]

    // Wick permeability parameters
    const double K = 1e-10;                 // Permeability [m2]
    const double CF = 1e5;                  // Forchheimer coefficient [1/m]
            
    // Geometric parameters
    const int N = 20;                                           // Number of axial nodes [-]
    const double L = 0.982; 			                        // Length of the heat pipe [m]
    const double dz = L / N;                                    // Axial discretization step [m]
    const double evaporator_start = 0.020;                      // Evaporator begin [m]
	const double evaporator_end = 0.073;                        // Evaporator end [m]
	const double condenser_length = 0.292;                      // Condenser length [m]
    const double evaporator_nodes = 
        std::floor((evaporator_end - evaporator_start) / dz);   // Number of evaporator nodes [-]
    const double condenser_nodes = 
        std::ceil(condenser_length / dz);                       // Number of condenser nodes [-]
    const double adiabatic_nodes = 
        N - (evaporator_nodes + condenser_nodes);               // Number of adiabatic nodes [-]
    const double r_o = 0.01335;                                 // Outer wall radius [m]
    const double r_i = 0.0112;                                  // Wall-wick interface radius [m]
    const double r_v = 0.01075;                                 // Vapor-wick interface radius [m]
    const double Dh_v = 2.0 * r_v;                              // Hydraulic diameter of the vapor core [m]

    // Evaporator region parameters
    const double Lh = evaporator_end - evaporator_start;
    const double delta_h = 0.01;
    const double Lh_eff = Lh + delta_h;
    const double q0 = power / (2.0 * M_PI * r_o * Lh_eff);      // [W/m2]

    // Condenser region parameters
    const double delta_c = 0.05;
    const double condenser_start = L - condenser_length;
    const double condenser_end = L;

    // Coefficients for the parabolic temperature profiles in wall and wick (check equations)
    const double E1w = 2.0 / 3.0 * (r_o + r_i - 1 / (1 / r_o + 1 / r_i));
    const double E2w = 0.5 * (r_o * r_o + r_i * r_i);
    const double E1x = 2.0 / 3.0 * (r_i + r_v - 1 / (1 / r_i + 1 / r_v));
    const double E2x = 0.5 * (r_i * r_i + r_v * r_v);

    // Time-stepping parameters
    double          dt_user = 1e-1;                 // Initial time step [s] (then it is updated according to the limits)
	double          dt = dt_user;                   // Current time step [s]
    double          time_total = 0.0;               // Total simulation time [s]
    const double    time_simulation = 2000;         // Simulation total number [s]
	double          dt_code = dt_user;              // Time step used in the code [s]
	int             halves = 0;                     // Number of halvings of the time step
    int             n = 0;                          // Iteration number [-]
    double          accelerator = 0.1;              // Adaptive timestep multiplier (maximum value for stability: 5)[-]

	// Picard iteration parameters
	const double max_picard = 100;                  // Maximum number of Picard iterations per time step [-]
	const double pic_tolerance = 1e-3;   	        // Tolerance for Picard iterations [-]   
    int pic = 0;                                    // Outside to check if convergence is reached [-]
    std::vector<double> pic_error(3, 0.0);          // L1 error for picard convergence [-]

    // PISO Wick parameters
    const int tot_simple_iter_x = 5;                // Outer iterations per time-step [-]
    const int tot_piso_iter_x = 10;                 // Inner iterations per outer iteration [-]
    const double momentum_tol_x = 1e-6;             // Tolerance for the momentum equation [-]
    const double continuity_tol_x = 1e-6;           // Tolerance for the continuity equation [-]
    const double temperature_tol_x = 1e-2;          // Tolerance for the energy equation [-]

    // PISO Vapor parameters
    const int tot_simple_iter_v = 5;                // Outer iterations per time-step [-]
    const int tot_piso_iter_v = 10;                 // Inner iterations per outer iteration [-]
    const double momentum_tol_v = 1e-6;             // Tolerance for the outer iterations (velocity) [-]
    const double continuity_tol_v = 1e-6;           // Tolerance for the inner iterations (pressure) [-]
    const double temperature_tol_v = 1e-2;          // Tolerance for the energy equation [-]

    // Constant temperature for initialization
    const double T_init = 800.0;

    std::vector<double> T_o_w(N, T_init);           // Outer wall temperature [K]
    std::vector<double> T_w_bulk(N, T_init);        // Wall bulk temperature [K]
    std::vector<double> T_w_x(N, T_init);           // Wall-wick interface temperature [K]
    std::vector<double> T_x_bulk(N, T_init);        // Wick bulk temperature [K]
    std::vector<double> T_x_v(N, T_init);           // Wick-vapore interface temperature [K]
    std::vector<double> T_v_bulk(N, T_init);        // Vapor bulk temperature [K]

    // Wick fields
    std::vector<double> u_x(N, -0.0001);            // Wick velocity field [m/s]
    std::vector<double> p_x(N);                     // Wick pressure field [Pa]
    std::vector<double> p_prime_x(N, 0.0);          // Wick correction pressure field [Pa]
    std::vector<double> rho_x(N);                   // Wick density field [Pa]
    std::vector<double> p_storage_x(N + 2);         // Wick padded pressure vector for R&C correction [Pa]
    double* p_padded_x = &p_storage_x[1];           // Poìnter to work on the wick pressure padded storage with the same indes

    for (int i = 0; i < N; ++i) p_x[i] = vapor_sodium::P_sat(T_x_v[i]);         // Initialization of the wick pressure
    for (int i = 0; i < N; ++i) rho_x[i] = liquid_sodium::rho(T_x_bulk[i]);     // Initialization of the wick density

    // Vapor fields
    std::vector<double> u_v(N, 0.1);            // Vapor velocity field [m/s]
    std::vector<double> p_v(N);                 // Vapor pressure field [Pa]
    std::vector<double> p_prime_v(N, 0.0);      // Vapor correction pressure field [Pa]
    std::vector<double> rho_v(N);               // Vapor density field [Pa]
    std::vector<double> p_storage_v(N + 2);     // Vapor padded pressure vector for R&C correction [Pa]
    double* p_padded_v = &p_storage_v[1];       // Poìnter to work on the storage with the same indes

    for (int i = 0; i < N; ++i) p_v[i] = vapor_sodium::P_sat(T_x_v[i]);         // Initialization of the vapor pressure

    // Vapor Equation of State update function. Updates density
    auto eos_update = [&](std::vector<double>& rho_, const std::vector<double>& p_, const std::vector<double>& T_) {

        for (int i = 0; i < N; i++) { rho_[i] = std::max(1e-6, p_[i] / (Rv * T_[i])); }

    }; eos_update(rho_v, p_v, T_v_bulk);

    // Heat sources/fluxes at the interfaces
    std::vector<double> q_ow(N);                    // Outer wall heat flux [W/m2]
	std::vector<double> Q_ow(N, 0.0);               // Outer wall heat source [W/m3]
    std::vector<double> Q_wx(N, 0.0);               // Wall heat source due to fluxes [W/m3]
    std::vector<double> Q_xw(N, 0.0);               // Wick heat source due to fluxes [W/m3]
    std::vector<double> Q_xm(N, 0.0);               // Vapor heat source due to fluxes[W/m3]
	std::vector<double> Q_mx(N, 0.0);               // Wick heat source due to fluxes [W/m3]

    std::vector<double> Q_tot_w(N, 0.0);            // Total heat flux in the wall [W/m3]
    std::vector<double> Q_tot_x(N, 0.0);            // Total heat flux in the wick [W/m3]
    std::vector<double> Q_tot_v(N, 0.0);            // Total heat flux in the vapor [W/m3]

    std::vector<double> Q_mass_vapor(N, 0.0);       // Heat volumetric source [W/m3] due to evaporation condensation. To be summed to the vapor
    std::vector<double> Q_mass_wick(N, 0.0);        // Heat volumetric source [W/m3] due to evaporation condensation. To be summed to the wick

	// Mass sources/fluxes at the interfaces
    std::vector<double> phi_x_v(N, 0.0);                // Mass flux [kg/m2/s] at the wick-vapor interface (positive if evaporation)
    std::vector<double> Gamma_xv_vapor(N, 0.0);         // Volumetric mass source [kg / (m^3 s)] (positive if evaporation)
    std::vector<double> Gamma_xv_wick(N, 0.0);          // Volumetric mass source [kg / (m^3 s)] (positive if evaporation)

    // Secondary useful variables
	std::vector<double> saturation_pressure(N, 0.0);    // Saturation pressure field [Pa]
	std::vector<double> sonic_velocity(N, 0.0);         // Sonic velocity field [m/s]

    // Padding pressure storages
    for (int i = 0; i < N; i++) p_storage_x[i + 1] = p_x[i];
    p_storage_x[0] = p_storage_x[1];
    p_storage_x[N + 1] = p_storage_x[N];

    for (int i = 0; i < N; i++) p_storage_v[i + 1] = p_v[i];
    p_storage_v[0] = p_storage_v[1];
    p_storage_v[N + 1] = p_storage_v[N];

    // Old values declaration
    std::vector<double> T_o_w_old;
    std::vector<double> T_w_bulk_old;
    std::vector<double> T_w_x_old;
    std::vector<double> T_x_bulk_old;
    std::vector<double> T_x_v_old;
    std::vector<double> T_v_bulk_old;

    std::vector<double> q_ow_old;

    std::vector<double> Q_ow_old;               
    std::vector<double> Q_wx_old;            
    std::vector<double> Q_xw_old;               
    std::vector<double> Q_xm_old;               
    std::vector<double> Q_mx_old;   

    std::vector<double> Q_mass_wick_old;
    std::vector<double> Q_mass_vapor_old;

    std::vector<double> Gamma_xv_wick_old;
    std::vector<double> Gamma_xv_vapor_old;

    std::vector<double> phi_x_v_old;

    std::vector<double> u_x_old;
    std::vector<double> p_x_old;
    std::vector<double> p_storage_x_old;

    std::vector<double> u_v_old;
    std::vector<double> p_v_old;
    std::vector<double> rho_v_old;
    std::vector<double> p_storage_v_old;

	// Wall physical properties
    std::vector<double> cp_w(N);
    std::vector<double> rho_w(N);
    std::vector<double> k_w(N);

	// Wick physical properties
    std::vector<double> mu_x(N);
    std::vector<double> cp_x(N);
    std::vector<double> k_x(N);

    // Vapor physical properties
    std::vector<double> mu_v(N);
    std::vector<double> cp_v(N);
    std::vector<double> k_v(N);
    std::vector<double> k_v_int(N);

    double h_xv_v;          // Specific enthalpy [J/kg] of vapor upon phase change between wick and vapor
    double h_vx_x;          // Specific enthalpy [J/kg] of wick upon phase change between vapor and wick

    // Iter values (only for Picard loops)
    std::vector<double> T_o_w_iter(N, 0.0);
    std::vector<double> T_w_x_iter(N, 0.0);
    std::vector<double> T_x_v_iter(N, 0.0);

    // Parabolas coefficients vector
    std::vector<double> ABC(6 * N);

    // Wick BCs
    const double u_inlet_x = 0.0;                               // Wick inlet velocity [m/s]
    const double u_outlet_x = 0.0;                              // Wick outlet velocity [m/s]
    double p_outlet_x = vapor_sodium::P_sat(T_x_v[N - 1]);      // Wick outlet pressure [Pa]
    double p_outlet_x_old = p_outlet_x;

    // Vapor BCs
    const double u_inlet_v = 0.0;                               // Vapor inlet velocity [m/s]
    const double u_outlet_v = 0.0;                              // Vapor outlet velocity [m/s]
    double p_outlet_v = vapor_sodium::P_sat(T_v_bulk[N - 1]);   // Vapor outlet pressure [Pa]
    double p_outlet_v_old = p_outlet_v;

    // Models
    const int rhie_chow_on_off_x = 1;             // 0: no wick RC correction, 1: wick with RC correction
    const int rhie_chow_on_off_v = 1;             // 0: no vapor RC correction, 1: vapor with RC correction

    // Tridiagonal coefficients for the wick velocity predictor
    std::vector<double> aXU(N, 0.0);                                      
    std::vector<double> bXU(N, rho_x[0] * dz / dt + 2 * mu_x[0] / dz);             
    std::vector<double> bXU_old = bXU;
    std::vector<double> cXU(N, 0.0);                                       
    std::vector<double> dXU(N, 0.0);                                    

	// Tridiagonal coefficients for the vapor velocity predictor
    std::vector<double> aVU(N, 0.0);                                                  
    std::vector<double> bVU(N, 2 * (4.0 / 3.0 * mu_v[0] / dz) + dz / dt * rho_v[0]);
    std::vector<double> bVU_old = bVU;
    std::vector<double> cVU(N, 0.0);                                                   
    std::vector<double> dVU(N, 0.0);                                                    

    // Tridiagonal coefficients for the wall temperature
    std::vector<double> aTW(N, 0.0);
    std::vector<double> bTW(N, 0.0);
    std::vector<double> cTW(N, 0.0);
    std::vector<double> dTW(N, 0.0);

    // Tridiagonal coefficients for the wick pressure correction
    std::vector<double> aXP(N, 0.0);
    std::vector<double> bXP(N, 0.0);
    std::vector<double> cXP(N, 0.0);
    std::vector<double> dXP(N, 0.0);

    // Tridiagonal coefficients for the vapor pressure correction
    std::vector<double> aVP(N, 0.0);
    std::vector<double> bVP(N, 0.0);
    std::vector<double> cVP(N, 0.0);
    std::vector<double> dVP(N, 0.0);

    // Tridiagonal coefficients for the wick temperature
    std::vector<double> aXT(N, 0.0);
    std::vector<double> bXT(N, 0.0);
    std::vector<double> cXT(N, 0.0);
    std::vector<double> dXT(N, 0.0);

    // Tridiagonal coefficients for the vapor temperature
    std::vector<double> aVT(N, 0.0);
    std::vector<double> bVT(N, 0.0);
    std::vector<double> cVT(N, 0.0);
    std::vector<double> dVT(N, 0.0);

    // Tridiagonal coefficients for the vapor k turbulent value
    std::vector<double> aK(N, 0.0);
    std::vector<double> bK(N, 0.0);
    std::vector<double> cK(N, 0.0);
    std::vector<double> dK(N, 0.0);

    // Tridiagonal coefficients for the vapor omega turbulent value
    std::vector<double> aW(N, 0.0);
    std::vector<double> bW(N, 0.0);
    std::vector<double> cW(N, 0.0);
    std::vector<double> dW(N, 0.0);

	// Residuals for wick loops
    double momentum_res_x = 1.0;
    double temperature_res_x = 1.0;
	double continuity_res_x = 1.0;

    int simple_iter_x = 0;
    int piso_iter_x = 0;

    double p_error_x = 0.0;
	double u_error_x = 0.0;

    // Residuals for vapor loops
    double momentum_res_v = 1.0;
    double temperature_res_v = 1.0;
    double continuity_res_v = 1.0;

    int simple_iter_v = 0;
    int piso_iter_v = 0;

    double p_error_v = 0.0;
    double u_error_v = 0.0;
	double rho_error_v = 0.0;

    // Previous values for residuals calculations
    std::vector<double> T_prev_x(N);
    std::vector<double> T_prev_v(N);

    // TDMA solver
    tdma::Solver tdma_solver(N);

    // Mesh z positions of the begin of the cells
    std::vector<double> mesh(N, 0.0);
    for (int i = 0; i < N; ++i) mesh[i] = i * dz;

    // Mesh z positions of the center of the cells
    std::vector<double> mesh_center(N, 0.0);
    for (int i = 0; i < N; ++i) mesh_center[i] = (i + 0.5) * dz;

    const int output_precision = 6;                             // Output precision
    const int sampling_frequency = 10 / accelerator;         // Sampling frequency

    std::string case_chosen = "case_0";

    // Create result folder
    int new_case = 0;
    while (true) {
        case_chosen = "case_" + std::to_string(new_case);
        if (!std::filesystem::exists(case_chosen)) {
            std::filesystem::create_directory(case_chosen);
            break;
        }
        new_case++;
    }

    std::cout << "Running case " << case_chosen << "\n";

    std::ofstream mesh_output(case_chosen + "/mesh.txt", std::ios::app);
    mesh_output << std::setprecision(output_precision);

    for (int i = 0; i < N; ++i) mesh_output << mesh_center[i] << " ";

    mesh_output.flush();
    mesh_output.close();

    // Old values
    T_o_w_old = T_o_w;
    T_w_bulk_old = T_w_bulk;
    T_w_x_old = T_w_x;
    T_x_bulk_old = T_x_bulk;
    T_x_v_old = T_x_v;
    T_v_bulk_old = T_v_bulk;

    q_ow_old = q_ow;

    Q_ow_old = Q_ow;
    Q_wx_old = Q_wx;
    Q_xw_old = Q_xw;
    Q_xm_old = Q_xm;
    Q_mx_old = Q_mx;

    Q_mass_wick_old = Q_mass_wick;
    Q_mass_vapor_old = Q_mass_vapor;

    Gamma_xv_wick_old = Gamma_xv_wick;
    Gamma_xv_vapor_old = Gamma_xv_vapor;

    phi_x_v_old = phi_x_v;

    u_x_old = u_x;
    p_x_old = p_x;
    p_storage_x_old = p_storage_x;
    bXU = bXU_old;

    u_v_old = u_v;
    p_v_old = p_v;
    p_storage_v_old = p_storage_v;
    rho_v_old = rho_v;
    bVU_old = bVU;

    // Steam outputs
    std::ofstream time_output(case_chosen + "/time.txt", std::ios::app);
    std::ofstream dt_output(case_chosen + "/dt.txt", std::ios::app);
    std::ofstream simulation_time_output(case_chosen + "/simulation_time.txt", std::ios::app);
    std::ofstream clock_time_output(case_chosen + "/clock_time.txt", std::ios::app);

    std::ofstream v_velocity_output(case_chosen + "/vapor_velocity.txt", std::ios::app);
    std::ofstream v_pressure_output(case_chosen + "/vapor_pressure.txt", std::ios::app);
    std::ofstream v_bulk_temperature_output(case_chosen + "/vapor_bulk_temperature.txt", std::ios::app);
    std::ofstream v_rho_output(case_chosen + "/rho_vapor.txt", std::ios::app);

    std::ofstream x_velocity_output(case_chosen + "/wick_velocity.txt", std::ios::app);
    std::ofstream x_pressure_output(case_chosen + "/wick_pressure.txt", std::ios::app);
    std::ofstream x_bulk_temperature_output(case_chosen + "/wick_bulk_temperature.txt", std::ios::app);
    std::ofstream x_rho_output(case_chosen + "/rho_liquid.txt", std::ios::app);

    std::ofstream x_v_temperature_output(case_chosen + "/wick_vapor_interface_temperature.txt", std::ios::app);
    std::ofstream w_x_temperature_output(case_chosen + "/wall_wick_interface_temperature.txt", std::ios::app);
    std::ofstream o_w_temperature_output(case_chosen + "/outer_wall_temperature.txt", std::ios::app);
    std::ofstream w_bulk_temperature_output(case_chosen + "/wall_bulk_temperature.txt", std::ios::app);

    std::ofstream x_v_mass_flux_output(case_chosen + "/wick_vapor_mass_source.txt", std::ios::app);

    std::ofstream Q_ow_output(case_chosen + "/outer_wall_heat_source.txt", std::ios::app);
    std::ofstream Q_wx_output(case_chosen + "/wick_wx_heat_source.txt", std::ios::app);
    std::ofstream Q_xw_output(case_chosen + "/wall_wx_heat_source.txt", std::ios::app);
    std::ofstream Q_xm_output(case_chosen + "/vapor_xv_heat_source.txt", std::ios::app);
    std::ofstream Q_mx_output(case_chosen + "/wick_xv_heat_source.txt", std::ios::app);

    std::ofstream Q_mass_vapor_output(case_chosen + "/vapor_heat_source_mass.txt", std::ios::app);
    std::ofstream Q_mass_wick_output(case_chosen + "/wick_heat_source_mass.txt", std::ios::app);

    std::ofstream saturation_pressure_output(case_chosen + "/saturation_pressure.txt", std::ios::app);
    std::ofstream sonic_velocity_output(case_chosen + "/sonic_velocity.txt", std::ios::app);

    std::ofstream total_heat_source_wall_output(case_chosen + "/total_heat_source_wall.txt", std::ios::app);
    std::ofstream total_heat_source_wick_output(case_chosen + "/total_heat_source_wick.txt", std::ios::app);
    std::ofstream total_heat_source_vapor_output(case_chosen + "/total_heat_source_vapor.txt", std::ios::app);

    std::ofstream total_mass_source_wick_output(case_chosen + "/total_mass_source_wick.txt", std::ios::app);
    std::ofstream total_mass_source_vapor_output(case_chosen + "/total_mass_source_vapor.txt", std::ios::app);

    std::ofstream momentum_res_x_output(case_chosen + "/momentum_res_x.txt", std::ios::app);
    std::ofstream continuity_res_x_output(case_chosen + "/continuity_res_x.txt", std::ios::app);
    std::ofstream temperature_res_x_output(case_chosen + "/temperature_res_x.txt", std::ios::app);

    std::ofstream momentum_res_v_output(case_chosen + "/momentum_res_v.txt", std::ios::app);
    std::ofstream continuity_res_v_output(case_chosen + "/continuity_res_v.txt", std::ios::app);
    std::ofstream temperature_res_v_output(case_chosen + "/temperature_res_v.txt", std::ios::app);

    time_output << std::setprecision(output_precision);
    dt_output << std::setprecision(output_precision);
    simulation_time_output << std::setprecision(output_precision);
    clock_time_output << std::setprecision(output_precision);

    v_velocity_output << std::setprecision(output_precision);
    v_pressure_output << std::setprecision(output_precision);
    v_bulk_temperature_output << std::setprecision(output_precision);
    v_rho_output << std::setprecision(output_precision);

    x_velocity_output << std::setprecision(output_precision);
    x_pressure_output << std::setprecision(output_precision);
    x_bulk_temperature_output << std::setprecision(output_precision);
    x_rho_output << std::setprecision(output_precision);

    x_v_temperature_output << std::setprecision(output_precision);
    w_x_temperature_output << std::setprecision(output_precision);
    o_w_temperature_output << std::setprecision(output_precision);
    w_bulk_temperature_output << std::setprecision(output_precision);

    x_v_mass_flux_output << std::setprecision(output_precision);

    Q_ow_output << std::setprecision(output_precision);
    Q_wx_output << std::setprecision(output_precision);
    Q_xw_output << std::setprecision(output_precision);
    Q_xm_output << std::setprecision(output_precision);
    Q_mx_output << std::setprecision(output_precision);

    Q_mass_vapor_output << std::setprecision(output_precision);
    Q_mass_wick_output << std::setprecision(output_precision);

    saturation_pressure_output << std::setprecision(output_precision);
    sonic_velocity_output << std::setprecision(output_precision);

    total_heat_source_wall_output << std::setprecision(output_precision);
    total_heat_source_wick_output << std::setprecision(output_precision);
    total_heat_source_vapor_output << std::setprecision(output_precision);

    total_mass_source_wick_output << std::setprecision(output_precision);
    total_mass_source_vapor_output << std::setprecision(output_precision);

    momentum_res_x_output << std::setprecision(output_precision);
    continuity_res_x_output << std::setprecision(output_precision);
    temperature_res_x_output << std::setprecision(output_precision);

    momentum_res_v_output << std::setprecision(output_precision);
    continuity_res_v_output << std::setprecision(output_precision);
    temperature_res_v_output << std::setprecision(output_precision);

    std::vector<double> a_Q_mass_vapor(N, 0.0);
    std::vector<double> b_Q_mass_vapor(N, 0.0);

    std::vector<double> a_Q_mass_vapor_old;
    std::vector<double> b_Q_mass_vapor_old;

    a_Q_mass_vapor_old = a_Q_mass_vapor;
    b_Q_mass_vapor_old = b_Q_mass_vapor;

    #pragma endregion

    // Print number of working threads
    std::cout << "Threads: " << omp_get_max_threads() << "\n";  

    // Start computational time measurement
	double start = omp_get_wtime();                             

    // Time stepping loop
    while (time_total < time_simulation) {

        n++;

        // Start computational time iteration
        auto t0 = std::chrono::high_resolution_clock::now();

        // Timestep calculation
        double dt_cand_w = new_dt_w(dz, dt, T_w_bulk, Q_tot_w);
        double dt_cand_x = new_dt_x(dz, dt, u_x, T_x_bulk, Gamma_xv_wick, Q_tot_x);
        double dt_cand_v = new_dt_v(dz, dt, u_v, T_v_bulk, rho_v, Gamma_xv_vapor, Q_tot_v, bVU);

        dt_code = std::min(std::min(dt_cand_w, dt_cand_x), std::min(dt_cand_x, dt_cand_v));
		dt = std::min(dt_user, dt_code);    // Choosing the minimum between user and calculated timestep
		dt *= std::pow(0.5, halves);        // Halving the timestep if Picard failed
        dt *= accelerator;                  // Accelerator multiplier
        if (dt < 1e-12) return 1;

		// Iter = old (for Picard loops)
        T_o_w_iter = T_o_w_old;
        T_w_x_iter = T_w_x_old;
        T_x_v_iter = T_x_v_old;

        // Updating all properties
        for (int i = 0; i < N; ++i) {

            cp_w[i] = steel::cp(T_w_bulk[i]);
            rho_w[i] = steel::rho(T_w_bulk[i]);
            k_w[i] = steel::k(T_w_bulk[i]);

            rho_x[i] = liquid_sodium::rho(T_x_bulk[i]);
            mu_x[i] = liquid_sodium::mu(T_x_bulk[i]);
            cp_x[i] = liquid_sodium::cp(T_x_bulk[i]);
            k_x[i] = liquid_sodium::k(T_x_bulk[i]);

            mu_v[i] = vapor_sodium::mu(T_v_bulk[i]);
            cp_v[i] = vapor_sodium::cp(T_v_bulk[i]);
            k_v[i] = vapor_sodium::k(T_v_bulk[i], p_v[i]);
            k_v_int[i] = vapor_sodium::k(T_x_v[i], p_v[i]);
        }

        for (pic = 0; pic < max_picard; pic++) {

            // =======================================================================
            //                                [WICK]
            // =======================================================================

            #pragma region wick

            // Pressure coupling hypotheses
            p_outlet_x = p_v[N - 1];

            // Momentum and energy residual initialization to access outer loop
            momentum_res_x = 1.0;
            temperature_res_x = 1.0;

            // Outer iterations reset
            simple_iter_x = 0;

			// Outer "PISO" iterations
            while ((simple_iter_x < tot_simple_iter_x) && (momentum_res_x > momentum_tol_x || temperature_res_x > temperature_tol_x)) {

				// ==========  MOMENTUM PREDICTOR 
                #pragma region momentum_predictor

                for (int i = 1; i < N - 1; i++) {

				    // Physical properties
                    const double rho_P = rho_x[i];    
                    const double rho_L = rho_x[i - 1];
                    const double rho_R = rho_x[i + 1];

                    const double rho_P_old = liquid_sodium::rho(T_x_bulk_old[i]);

                    const double mu_P = mu_x[i];
                    const double mu_L = mu_x[i - 1];
                    const double mu_R = mu_x[i + 1];

                    const double D_l = 0.5 * (mu_P + mu_L) / dz;        
                    const double D_r = 0.5 * (mu_P + mu_R) / dz;

                    const double avgInvbLU_L = 0.5 * (1.0 / bXU[i - 1] + 1.0 / bXU[i]); // [m2s/kg]
                    const double avgInvbLU_R = 0.5 * (1.0 / bXU[i + 1] + 1.0 / bXU[i]); // [m2s/kg]

                    const double rc_l = -avgInvbLU_L / 4.0 *
                        (p_padded_x[i - 2] - 3.0 * p_padded_x[i - 1] + 3.0 * p_padded_x[i] - p_padded_x[i + 1]); // [m/s]
                    const double rc_r = -avgInvbLU_R / 4.0 *
                        (p_padded_x[i - 1] - 3.0 * p_padded_x[i] + 3.0 * p_padded_x[i + 1] - p_padded_x[i + 2]); // [m/s]

                    const double u_l_face = 0.5 * (u_x[i - 1] + u_x[i]) + rhie_chow_on_off_x * rc_l;
                    const double u_r_face = 0.5 * (u_x[i] + u_x[i + 1]) + rhie_chow_on_off_x * rc_r;

                    const double rho_l = (u_l_face >= 0) ? rho_L : rho_P;
                    const double rho_r = (u_r_face >= 0) ? rho_P : rho_R; 

                    const double F_l = rho_l * u_l_face;    // Mass flux [kg/(m2 s)] of the left face 
                    const double F_r = rho_r * u_r_face;    // Mass flux [kg/(m2 s)] of the right face 

                    aXU[i] = 
                        - std::max(F_l, 0.0) 
                        - D_l;
                    cXU[i] = 
                        - std::max(-F_r, 0.0) 
                        - D_r;
                    bXU[i] = 
                        + std::max(F_r, 0.0) 
                        + std::max(-F_l, 0.0) 
                        + rho_P * dz / dt 
                        + D_l + D_r 
                        + mu_P / K * dz 
                        + CF * mu_P * dz / sqrt(K) * abs(u_x[i]);
                    dXU[i] = 
                        - 0.5 * (p_x[i + 1] - p_x[i - 1])
                        + rho_P_old * u_x_old[i] * dz / dt;
                }

                // Diffusion coefficients for the first and last node to define BCs
                const double D_first = mu_x[0] / dz;
                const double D_last = mu_x[N - 1] / dz;

                // Velocity BCs: fixed (zero) velocity on the first node
			    aXU[0] = 0.0;
                bXU[0] = (rho_x[0] * dz / dt + 2 * D_first);
                cXU[0] = 0.0; 
                dXU[0] = (rho_x[0] * dz / dt + 2 * D_first) * u_inlet_x;
            
                // Velocity BCs: fixed (zero) velocity on the last node
                aXU[N - 1] = 0.0; 
                bXU[N - 1] = (rho_x[N - 1] * dz / dt + 2 * D_last);
			    cXU[N - 1] = 0.0;
                dXU[N - 1] = (rho_x[N - 1] * dz / dt + 2 * D_last) * u_outlet_x;

                tdma_solver.solve(aXU, bXU, cXU, dXU, u_x);

                #pragma endregion

                // =========== TEMPERATURE CALCULATOR
                #pragma region temperature_calculator

                for (int i = 1; i < N - 1; i++) {

                    // Physical properties
                    const double rho_P = rho_x[i];
                    const double rho_L = rho_x[i - 1];
                    const double rho_R = rho_x[i + 1];

                    const double rho_P_old = liquid_sodium::rho(T_x_bulk_old[i]);

                    const double k_cond_P = k_x[i];
                    const double k_cond_L = k_x[i - 1];
                    const double k_cond_R = k_x[i + 1];

                    const double cp_P = cp_x[i];
                    const double cp_L = cp_x[i - 1];
                    const double cp_R = cp_x[i + 1];

                    const double cp_P_old = liquid_sodium::cp(T_x_bulk_old[i]);

                    const double D_l = 0.5 * (k_cond_P + k_cond_L) / dz;
                    const double D_r = 0.5 * (k_cond_P + k_cond_R) / dz;

                    const double avgInvbLU_L = 0.5 * (1.0 / bXU[i - 1] + 1.0 / bXU[i]);     // [m2s/kg]
                    const double avgInvbLU_R = 0.5 * (1.0 / bXU[i + 1] + 1.0 / bXU[i]);     // [m2s/kg]

                    const double rc_l = -avgInvbLU_L / 4.0 *
                        (p_padded_x[i - 2] - 3.0 * p_padded_x[i - 1] + 3.0 * p_padded_x[i] - p_padded_x[i + 1]);    // [m/s]
                    const double rc_r = -avgInvbLU_R / 4.0 *
                        (p_padded_x[i - 1] - 3.0 * p_padded_x[i] + 3.0 * p_padded_x[i + 1] - p_padded_x[i + 2]);    // [m/s]

                    const double u_l_face = 0.5 * (u_x[i - 1] + u_x[i]) + rhie_chow_on_off_x * rc_l;         // [m/s]
                    const double u_r_face = 0.5 * (u_x[i] + u_x[i + 1]) + rhie_chow_on_off_x * rc_r;         // [m/s]

                    const double rho_l = (u_l_face >= 0) ? rho_L : rho_P;
                    const double rho_r = (u_r_face >= 0) ? rho_P : rho_R;

                    const double cp_l = (u_l_face >= 0) ? cp_L : cp_P;
                    const double cp_r = (u_r_face >= 0) ? cp_P : cp_R;

                    const double Fl = rho_l * u_l_face;
                    const double Fr = rho_r * u_r_face;

                    const double C_l = (Fl * cp_l);
                    const double C_r = (Fr * cp_r);

                    Q_tot_x[i] = Q_wx[i] + Q_mx[i] + Q_mass_wick[i];

                    aXT[i] =
                        -D_l
                        - std::max(C_l, 0.0);       // [W/(m2 K)]
                    cXT[i] =
                        -D_r
                        - std::max(-C_r, 0.0);      // [W/(m2 K)]
                    bXT[i] =
                        +std::max(C_r, 0.0)
                        + std::max(-C_l, 0.0)
                        + D_l + D_r
                        + rho_P * cp_P * dz / dt;   // [W/(m2 K)]
                    dXT[i] =
                        +rho_P_old * cp_P_old * dz / dt * T_x_bulk_old[i]
                        + Q_wx[i] * dz              // Positive if heat is added to the wick
                        + Q_mx[i] * dz              // Positive if heat is added to the wick
                        + Q_mass_wick[i] * dz;      // [W/m2]       
                }

                // Temperature BCs: zero gradient on the first node
                aXT[0] = 0.0;
                bXT[0] = 1.0;
                cXT[0] = -1.0;
                dXT[0] = 0.0;

                // Temperature BCs: zero gradient on the last node
                aXT[N - 1] = -1.0;
                bXT[N - 1] = 1.0;
                cXT[N - 1] = 0.0;
                dXT[N - 1] = 0.0;

                T_prev_x = T_x_bulk;

                tdma_solver.solve(aXT, bXT, cXT, dXT, T_x_bulk);

                #pragma endregion

                // =========== TEMPERATURE RESIDUAL CALCULATOR
                #pragma region temperature_residual_calculator
                temperature_res_x = 0.0;

                for (int i = 0; i < N; ++i) {

                    temperature_res_x = std::max(
                        temperature_res_x,
                        std::abs(T_x_bulk[i] - T_prev_x[i]) / T_prev_x[i]
                    );
                }

                #pragma endregion

                // Continuity residual initialization to access inner loop
                continuity_res_x = 1.0;

                // Inner iterations reset
                piso_iter_x = 0;

				// Inner "SIMPLE" iterations
                while ((piso_iter_x < tot_piso_iter_x) && (continuity_res_x > continuity_tol_x)) {

                    // =========== CONTINUITY SATIFACTOR
                    #pragma region continuity_satisfactor

					// Loop to assemble the linear system for the pressure correction
                    for (int i = 1; i < N - 1; i++) {

					    // Physical properties
                        const double rho_P = rho_x[i];
                        const double rho_L = rho_x[i - 1];
                        const double rho_R = rho_x[i + 1];

                        const double avgInvbLU_L = 0.5 * (1.0 / bXU[i - 1] + 1.0 / bXU[i]);     // [m2s/kg]
                        const double avgInvbLU_R = 0.5 * (1.0 / bXU[i + 1] + 1.0 / bXU[i]);     // [m2s/kg]

                        const double rc_l = -avgInvbLU_L / 4.0 *
                            (p_padded_x[i - 2] - 3.0 * p_padded_x[i - 1] + 3.0 * p_padded_x[i] - p_padded_x[i + 1]);    // [m/s]
                        const double rc_r = -avgInvbLU_R / 4.0 *
                            (p_padded_x[i - 1] - 3.0 * p_padded_x[i] + 3.0 * p_padded_x[i + 1] - p_padded_x[i + 2]);    // [m/s]

                        const double u_l_face = 0.5 * (u_x[i - 1] + u_x[i]) + rhie_chow_on_off_x * rc_l;    // [m/s]
                        const double u_r_face = 0.5 * (u_x[i] + u_x[i + 1]) + rhie_chow_on_off_x * rc_r;    // [m/s]

                        const double rho_left_uw = (u_l_face >= 0.0) ? rho_L : rho_P;
                        const double rho_right_uw = (u_r_face >= 0.0) ? rho_P : rho_R;

                        const double F_l = rho_left_uw * u_l_face;          // [kg/(m2s)]
                        const double F_r = rho_right_uw * u_r_face;         // [kg/(m2s)]

                        const double mass_imbalance = (F_r - F_l);          // [kg/(m2s)]

                        const double mass_flux = - Gamma_xv_wick[i] * dz;   // [kg/(m2s)]

                        const double rho_l_cd = 0.5 * (rho_L + rho_P);      // [kg/m3]
                        const double rho_r_cd = 0.5 * (rho_P + rho_R);      // [kg/m3]

                        const double E_l = rho_l_cd * avgInvbLU_L / dz;     // [s/m]
                        const double E_r = rho_r_cd * avgInvbLU_R / dz;     // [s/m]

                        aXP[i] = -E_l;              // [s/m]
                        cXP[i] = -E_r;              // [s/m]
                        bXP[i] = E_l + E_r;         // [s/m]
                        dXP[i] = 
                            + mass_flux 
                            - mass_imbalance;       // [kg/(m^2 s)]
                    }

                    // BCs for the correction of pressure: zero gradient at first node
				    aXP[0] = 0.0;
                    bXP[0] = 1.0; 
                    cXP[0] = -1.0; 
                    dXP[0] = 0.0;

                    // BCs for the correction of pressure: zero at first node
                    aXP[N - 1] = 0.0;
                    bXP[N - 1] = 1.0; 
				    cXP[N - 1] = 0.0;
                    dXP[N - 1] = 0.0;

                    tdma_solver.solve(aXP, bXP, cXP, dXP, p_prime_x);

                    #pragma endregion

                    // =========== PRESSURE CORRECTOR
                    #pragma region pressure_corrector

                    // Initialization maximum pressure variation
                    p_error_x = 0.0;

                    for (int i = 0; i < N; i++) {

                        double p_prev_x = p_x[i];
                        p_x[i] += p_prime_x[i];         // PIMPLE does not require an under-relaxation factor
                        p_storage_x[i + 1] = p_x[i];

                        p_error_x = std::max(p_error_x, std::abs(p_x[i] - p_prev_x));
                    }

                    // Enforcing boundary conditions
                    p_x[0] = p_x[1];
                    p_storage_x[0] = p_storage_x[1];

                    p_x[N - 1] = p_outlet_x;
                    p_storage_x[N + 1] = p_outlet_x;

                    #pragma endregion

                    // =========== VELOCITY CORRECTOR
                    #pragma region velocity_corrector

                    // Initialization maximum pressure variation
                    u_error_x = 0.0;

                    for (int i = 1; i < N - 1; i++) {

                        double u_prev = u_x[i];
                        u_x[i] = u_x[i] - (p_prime_x[i + 1] - p_prime_x[i - 1]) / (2.0 * bXU[i]);

                        u_error_x = std::max(u_error_x, std::abs(u_x[i] - u_prev));
                    }

                    #pragma endregion

                    // =========== CONTINUITY RESIDUAL CALCULATOR
                    #pragma region continuity_residual_calculator

                    continuity_res_x = 0.0;

                    for (int i = 1; i < N - 1; ++i) {

                        continuity_res_x = std::max(continuity_res_x, std::abs(dXP[i]));
                    }

                    #pragma endregion

                    piso_iter_x++;
                }

                // =========== MOMENTUM RESIDUAL CALCULATOR
                #pragma region momentum_residual_calculator

                momentum_res_x = 0.0;

                for (int i = 1; i < N - 1; ++i) {
                    momentum_res_x = std::max(momentum_res_x, std::abs(aXU[i] * u_x[i - 1] + bXU[i] * u_x[i] + cXU[i] * u_x[i + 1] - dXU[i]));
                }

                momentum_res_x /= N;

                #pragma endregion

                simple_iter_x++;
            }
                
            #pragma endregion

            // =======================================================================
            //                                [VAPOR]
            // =======================================================================

            #pragma region vapor

            // Pressure coupling hypotheses
            p_v[N - 1] = p_outlet_v;

            // Momentum and energy residual initialization to access outer loop
            momentum_res_v = 1.0;
            temperature_res_v = 1.0;

            // Outer iterations reset
            simple_iter_v = 0;

            // Outer "PISO" iterations
            while ((simple_iter_v < tot_simple_iter_v) && (momentum_res_v > momentum_tol_v || temperature_res_v > temperature_tol_v)) {

                // ==========  MOMENTUM PREDICTOR 
                #pragma region momentum_predictor

                for (int i = 1; i < N - 1; i++) {

				    // Physical properties
                    const double rho_P = rho_v[i];
                    const double rho_L = rho_v[i - 1];
                    const double rho_R = rho_v[i + 1];

                    const double rho_P_old = rho_v_old[i];

                    const double mu_P = mu_v[i];
                    const double mu_L = mu_v[i - 1];
                    const double mu_R = mu_v[i + 1];

                    const double D_l = 4.0 / 3.0 * 0.5 * (mu_P + mu_L) / dz;
                    const double D_r = 4.0 / 3.0 * 0.5 * (mu_P + mu_R) / dz;    

                    const double avgInvbVU_L = 0.5 * (1.0 / bVU[i - 1] + 1.0 / bVU[i]); // [m2s/kg]
                    const double avgInvbVU_R = 0.5 * (1.0 / bVU[i + 1] + 1.0 / bVU[i]); // [m2s/kg]

                    const double rc_l = -avgInvbVU_L / 4.0 *
                        (p_padded_v[i - 2] - 3.0 * p_padded_v[i - 1] + 3.0 * p_padded_v[i] - p_padded_v[i + 1]); // [m/s]
                    const double rc_r = -avgInvbVU_R / 4.0 *
                        (p_padded_v[i - 1] - 3.0 * p_padded_v[i] + 3.0 * p_padded_v[i + 1] - p_padded_v[i + 2]); // [m/s]

                    const double u_l_face = 0.5 * (u_v[i - 1] + u_v[i]) + rhie_chow_on_off_v * rc_l;    // [m/s]
                    const double u_r_face = 0.5 * (u_v[i] + u_v[i + 1]) + rhie_chow_on_off_v * rc_r;    // [m/s]

                    const double rho_l = (u_l_face >= 0) ? rho_L : rho_P;
                    const double rho_r = (u_r_face >= 0) ? rho_P : rho_R;       

                    const double F_l = rho_l * u_l_face;
                    const double F_r = rho_r * u_r_face;           

                    const double Re = u_v[i] * (2 * r_v) * rho_P / mu_P;
                    const double f = (Re < 1187.4) ? 64 / Re : 0.3164 * std::pow(Re, -0.25);  
                    const double F = 0.25 * f * rho_P * std::abs(u_v[i]) / r_v;

                    aVU[i] = 
                        - std::max(F_l, 0.0) 
                        - D_l;                              // [kg/(m2 s)]
                    cVU[i] = 
                        - std::max(-F_r, 0.0) 
                        - D_r;                              // [kg/(m2 s)]
                    bVU[i] = 
                        + std::max(F_r, 0.0) 
                        + std::max(-F_l, 0.0) 
                        + rho_P * dz / dt 
                        + D_l + D_r 
                        + F * dz;                           // [kg/(m2 s)]
                    dVU[i] = 
                        - 0.5 * (p_v[i + 1] - p_v[i - 1])
                        + rho_P_old * u_v_old[i] * dz / dt; // [kg/(m s2)]
                }

                /// Diffusion coefficients for the first and last node to define BCs
                const double D_first = (4.0 / 3.0) * mu_v[0] / dz;
                const double D_last = (4.0 / 3.0) * mu_v[N - 1] / dz;

                /// Velocity BCs needed variables for the first node
                const double u_r_face_first = 0.5 * (u_v[1]);
                const double rho_r_first = (u_r_face_first >= 0) ? rho_v[0] : rho_v[1];
                const double F_r_first = rho_r_first * u_r_face_first;

                /// Velocity BCs needed variables for the last node
                const double u_l_face_last = 0.5 * (u_v[N - 2]);
                const double rho_l_last = (u_l_face_last >= 0) ? rho_v[N - 2] : rho_v[N - 1];
                const double F_l_last = rho_l_last * u_l_face_last;

                // Velocity BCs: zero velocity on the first node
			    aVU[0] = 0.0;
                bVU[0] = + (std::max(F_r_first, 0.0) + rho_v[0] * dz / dt + 2 * D_first);
                cVU[0] = 0.0; 
                dVU[0] = bVU[0] * u_inlet_v;

                // Velocity BCs: zero velocity on the last node
                aVU[N - 1] = 0.0;
                bVU[N - 1] = + (- std::max(-F_l_last, 0.0) + rho_v[N - 1] * dz / dt + 2 * D_last);
			    cVU[N - 1] = 0.0;
                dVU[N - 1] = bVU[N - 1] * u_outlet_v;

                tdma_solver.solve(aVU, bVU, cVU, dVU, u_v);

                #pragma endregion

                // ==========  TEMPERATURE CALCULATOR 
                #pragma region temperature_calculator

                for (int i = 1; i < N - 1; i++) {

                    // Physical properties
                    const double rho_P = rho_v[i];
                    const double rho_L = rho_v[i - 1];
                    const double rho_R = rho_v[i + 1];

                    const double k_cond_P = k_v[i];
                    const double k_cond_L = k_v[i - 1];
                    const double k_cond_R = k_v[i + 1];

                    const double cp_P = cp_v[i];
                    const double cp_L = cp_v[i - 1];
                    const double cp_R = cp_v[i + 1];

                    const double cp_P_old = vapor_sodium::cp(T_v_bulk_old[i]);

                    const double mu_P = vapor_sodium::mu(T_v_bulk[i]);

                    const double D_l = 0.5 * (k_cond_P + k_cond_L) / dz;
                    const double D_r = 0.5 * (k_cond_P + k_cond_R) / dz;

                    const double avgInvbVU_v = 0.5 * (1.0 / bVU[i - 1] + 1.0 / bVU[i]);     // [m2s/kg]
                    const double avgInvbVU_R = 0.5 * (1.0 / bVU[i + 1] + 1.0 / bVU[i]);     // [m2s/kg]

                    const double rc_v = -avgInvbVU_v / 4.0 *
                        (p_padded_v[i - 2] - 3.0 * p_padded_v[i - 1] + 3.0 * p_padded_v[i] - p_padded_v[i + 1]);    // [m/s]
                    const double rc_r = -avgInvbVU_R / 4.0 *
                        (p_padded_v[i - 1] - 3.0 * p_padded_v[i] + 3.0 * p_padded_v[i + 1] - p_padded_v[i + 2]);    // [m/s]

                    const double u_l_face = 0.5 * (u_v[i - 1] + u_v[i]) + rhie_chow_on_off_v * rc_v;         // [m/s]
                    const double u_r_face = 0.5 * (u_v[i] + u_v[i + 1]) + rhie_chow_on_off_v * rc_r;         // [m/s]

                    const double rho_l = (u_l_face >= 0) ? rho_v[i - 1] : rho_v[i];     // [kg/m3]
                    const double rho_r = (u_r_face >= 0) ? rho_v[i] : rho_v[i + 1];     // [kg/m3]

                    const double cp_l = (u_l_face >= 0) ? cp_v[i - 1] : cp_v[i];     // [kg/m3]
                    const double cp_r = (u_r_face >= 0) ? cp_v[i] : cp_v[i + 1];     // [kg/m3]

                    const double Fl = rho_l * u_l_face;         // [kg/m2s]
                    const double Fr = rho_r * u_r_face;         // [kg/m2s]

                    const double C_l = (Fl * cp_l);               // [W/(m2K)]
                    const double C_r = (Fr * cp_r);               // [W/(m2K)]

                    const double dpdz_up = u_v[i] * (p_v[i + 1] - p_v[i - 1]) / 2.0;

                    const double dp_dt = (p_v[i] - p_v_old[i]) / dt * dz;

                    const double viscous_dissipation =
                        4.0 / 3.0 * 0.25 * mu_v[i] * ((u_v[i + 1] - u_v[i]) * (u_v[i + 1] - u_v[i])
                            + (u_v[i] + u_v[i - 1]) * (u_v[i] + u_v[i - 1])) / dz;

                    Q_tot_v[i] = dp_dt / dz + dpdz_up / dz + viscous_dissipation + Q_xm[i] + Q_mass_vapor[i];

                    aVT[i] =
                        - D_l
                        - std::max(C_l, 0.0)
                        ;                                   /// [W/(m2K)]

                    cVT[i] =
                        - D_r
                        - std::max(-C_r, 0.0)
                        ;                                   /// [W/(m2K)]

                    bVT[i] =
                        + std::max(C_r, 0.0)
                        + std::max(-C_l, 0.0)
                        + D_l + D_r
                        + rho_v[i] * cp_v[i] * dz / dt;      /// [W/(m2 K)]
                        // - a_Q_mass_vapor[i] * dz;

                    dVT[i] =
                        + rho_v_old[i] * cp_P_old * dz / dt * T_v_bulk_old[i]
                        + dp_dt
                        + dpdz_up
                        + viscous_dissipation * dz
                        + Q_xm[i] * dz;                     // Positive if heat from wick to vapor
                        // + b_Q_mass_vapor[i] * dz
                        // + Q_mass_vapor[i] * dz          // [W/m2]

                    if (time_total < 1.0) {

                        dVT[i] += Q_mass_vapor[i] * dz;           // Approximated evaporation mass flux [kg/(m2s)]
                    }
                    else {

                        bVT[i] += -a_Q_mass_vapor[i] * dz;
                        dVT[i] += b_Q_mass_vapor[i] * dz;
                    }
                }

                // Temperature BCs: zero gradient on the first node
                aVT[0] = 0.0;
                bVT[0] = 1.0;
                cVT[0] = -1.0;
                dVT[0] = 0.0;

                // Temperature BCs: zero gradient on the last node
                aVT[N - 1] = -1.0;
                bVT[N - 1] = 1.0;
                cVT[N - 1] = 0.0;
                dVT[N - 1] = 0.0;

                T_prev_v = T_v_bulk;

                tdma_solver.solve(aVT, bVT, cVT, dVT, T_v_bulk);

                #pragma endregion

                // =========== TEMPERATURE RESIDUAL CALCULATOR
                #pragma region temperature_residual_calculator

                temperature_res_v = 0.0;

                for (int i = 0; i < N; ++i) {

                    temperature_res_v = std::max(
                        temperature_res_v,
                        std::abs(T_v_bulk[i] - T_prev_v[i]) / T_prev_v[i]
                    );
                }

                #pragma endregion

                // Continuity residual initialization to access inner loop
                continuity_res_v = 1.0;

                // Inner iterations reset
                piso_iter_v = 0;

				// Inner iterations for the vapor continuity equation
                while ((piso_iter_v < tot_piso_iter_v) && (continuity_res_v > continuity_tol_v)) {

                    // =========== CONTINUITY SATIFACTOR
                    #pragma region continuity_satisfactor

                    for (int i = 1; i < N - 1; ++i) {

                        const double avgInvbVU_L = 0.5 * (1.0 / bVU[i - 1] + 1.0 / bVU[i]);     // [m2s/kg]
                        const double avgInvbVU_R = 0.5 * (1.0 / bVU[i + 1] + 1.0 / bVU[i]);     // [m2s/kg]

                        const double rc_l = -avgInvbVU_L / 4.0 *
                            (p_padded_v[i - 2] - 3.0 * p_padded_v[i - 1] + 3.0 * p_padded_v[i] - p_padded_v[i + 1]);    // [m/s]
                        const double rc_r = -avgInvbVU_R / 4.0 *
                            (p_padded_v[i - 1] - 3.0 * p_padded_v[i] + 3.0 * p_padded_v[i + 1] - p_padded_v[i + 2]);    // [m/s]

                        const double psi_i = 1.0 / (Rv * T_v_bulk[i]); // [kg/J]

                        const double u_l_star = 0.5 * (u_v[i - 1] + u_v[i]) + rhie_chow_on_off_v * rc_l;    // [m/s]
                        const double u_r_star = 0.5 * (u_v[i] + u_v[i + 1]) + rhie_chow_on_off_v * rc_r;    // [m/s]

                        const double Crho_l = u_l_star >= 0 ? (1.0 / (Rv * T_v_bulk[i - 1])) : (1.0 / (Rv * T_v_bulk[i]));  // [s2/m2]
                        const double Crho_r = u_r_star >= 0 ? (1.0 / (Rv * T_v_bulk[i])) : (1.0 / (Rv * T_v_bulk[i + 1]));  // [s2/m2]

                        const double C_l = Crho_l * u_l_star;       // [s/m]
                        const double C_r = Crho_r * u_r_star;       // [s/m]

                        const double rho_l_upwind = (u_l_star >= 0.0) ? rho_v[i - 1] : rho_v[i];    // [kg/m3]
                        const double rho_r_upwind = (u_r_star >= 0.0) ? rho_v[i] : rho_v[i + 1];    // [kg/m3]

                        const double phi_l = rho_l_upwind * u_l_star;   // [kg/(m2s)]
                        const double phi_r = rho_r_upwind * u_r_star;   // [kg/(m2s)]

                        const double mass_imbalance = (phi_r - phi_l) + (rho_v[i] - rho_v_old[i]) * dz / dt;  // [kg/(m2s)]

                        const double mass_flux = Gamma_xv_vapor[i] * dz;         // [kg/(m2s)]

                        const double E_l = 0.5 * (rho_v[i - 1] * (1.0 / bVU[i - 1]) + rho_v[i] * (1.0 / bVU[i])) / dz; // [s/m]
                        const double E_r = 0.5 * (rho_v[i] * (1.0 / bVU[i]) + rho_v[i + 1] * (1.0 / bVU[i + 1])) / dz; // [s/m]

                        aVP[i] =
                            - E_l
                            - std::max(C_l, 0.0)
                            ;                                   /// [s/m]

                        cVP[i] =
                            - E_r
                            - std::max(-C_r, 0.0)
                            ;                                   /// [s/m]

                        bVP[i] =
                            + E_l + E_r
                            + std::max(C_r, 0.0)
                            + std::max(-C_l, 0.0)
                            + psi_i * dz / dt;                  /// [s/m]

                        dVP[i] = + mass_flux - mass_imbalance;  /// [kg/(m2s)]
                    }

                    // BCs for the correction of pressure: zero gradient at first node
				    aVP[0] = 0.0;
                    bVP[0] = 1.0; 
                    cVP[0] = -1.0; 
                    dVP[0] = 0.0;

                    // BCs for the correction of pressure: zero at last node
                    aVP[N - 1] = 0.0;
                    bVP[N - 1] = 1.0;  
				    cVP[N - 1] = 0.0;
                    dVP[N - 1] = 0.0;

                    tdma_solver.solve(aVP, bVP, cVP, dVP, p_prime_v);

                    #pragma endregion

                    // =========== PRESSURE CORRECTOR
                    #pragma region pressure_corrector

                    p_error_v = 0.0;

                    for (int i = 0; i < N; i++) {

                        double p_prev = p_v[i];
                        p_v[i] += p_prime_v[i];        // PISO does not require an under-relaxation factor
                        p_storage_v[i + 1] = p_v[i];

                        p_error_v = std::max(p_error_v, std::abs(p_v[i] - p_prev));
                    }

                    // Enforcing boundary conditions
                    p_v[0] = p_v[1];
                    p_storage_v[0] = p_storage_v[1];

                    p_v[N - 1] = p_outlet_v;
                    p_storage_v[N + 1] = p_outlet_v;

                    #pragma endregion

                    // =========== VELOCITY CORRECTOR
                    #pragma region velocity_corrector

                    u_error_v = 0.0;

                    for (int i = 1; i < N - 1; ++i) {

                        double u_prev = u_v[i];

                        sonic_velocity[i] = std::sqrt(vapor_sodium::gamma(T_v_bulk[i]) * Rv * T_v_bulk[i]);

                        const double calc_velocity = u_v[i] -
                            (p_prime_v[i + 1] - p_prime_v[i - 1]) / (2.0 * bVU[i]);

                        if (calc_velocity < sonic_velocity[i]) u_v[i] = calc_velocity;
                        else u_v[i] = sonic_velocity[i];

                        u_error_v = std::max(u_error_v, std::abs(u_v[i] - u_prev));
                    }

                    #pragma endregion

                    // =========== DENSITY CORRECTOR
                    #pragma region density_corrector

                    rho_error_v = 0.0;

                    for (int i = 0; i < N; ++i) {
                        double rho_prev = rho_v[i];
                        rho_v[i] += p_prime_v[i] / (Rv * T_v_bulk[i]);
                        rho_error_v = std::max(rho_error_v, std::abs(rho_v[i] - rho_prev));
                    }

                    #pragma endregion

                    // =========== CONTINUITY RESIDUAL CALCULATOR
                    #pragma region continuity_residual_calculator

                    continuity_res_v = 0.0;

                    for (int i = 1; i < N - 1; ++i) {

                        continuity_res_v = std::max(continuity_res_v, std::abs(dVP[i]));
                    }

                    #pragma endregion

                    piso_iter_v++;
                }

                // =========== MOMENTUM RESIDUAL CALCULATOR
                #pragma region momentum_residual_calculator

                momentum_res_v = 0.0;

                for (int i = 1; i < N - 1; ++i) {
                    momentum_res_v = std::max(momentum_res_v, std::abs(aVU[i] * u_v[i - 1] + bVU[i] * u_v[i] + cVU[i] * u_v[i + 1] - dVU[i]));
                }

                #pragma endregion

                // =========== TEMPERATURE RESIDUAL CALCULATOR
                #pragma region temperature_residual_calculator

                temperature_res_v = 0.0;

                for (int i = 1; i < N - 1; ++i) {
                    temperature_res_v = std::max(temperature_res_v, std::abs(T_v_bulk[i] - T_prev_v[i]));
                }

                #pragma endregion

                simple_iter_v++;
            }

            // Update density with new p,T
            eos_update(rho_v, p_v, T_v_bulk);

            #pragma endregion

            // =======================================================================
            //                             [INTERFACES]
            // =======================================================================

            #pragma region interfaces 

            for (int i = 0; i < N; ++i) {

                // Physical properties
                const double Re_v = rho_v[i] * std::abs(u_v[i]) * Dh_v / mu_v[i];       // Reynolds number [-]
                const double Pr_v = cp_v[i] * mu_v[i] / k_v[i];                         // Prandtl number [-]
                const double H_xm = vapor_sodium::h_conv(Re_v, Pr_v, k_v[i], Dh_v);     // Convective heat transfer coefficient at the vapor-wick interface [W/m^2/K]
                saturation_pressure[i] = vapor_sodium::P_sat(T_x_v_iter[i]);            // Saturation pressure [Pa]        
                const double H_xmdPsat_dT =
                    saturation_pressure[i] * std::log(10.0) *
                    (7740.0 / (T_x_v_iter[i] * T_x_v_iter[i]));                         // Derivative of the saturation pressure wrt T [Pa/K]   

                // Enthalpies
                if (phi_x_v[i] > 0.0) {                             // Evaporation case

                    h_xv_v = vapor_sodium::h(T_x_v_iter[i]);
                    h_vx_x = liquid_sodium::h(T_x_v_iter[i]);

                }
                else {                                              // Condensation case

                    h_xv_v = vapor_sodium::h(T_v_bulk[i]);
                    h_vx_x = liquid_sodium::h(T_x_v_iter[i])
                        + (vapor_sodium::h(T_v_bulk[i]) - vapor_sodium::h(T_x_v_iter[i]));
                }

                // Useful constants
                const double E3 = H_xm;
                const double E4 = -k_x[i] + H_xm * r_v;
                const double E5 = -2.0 * r_v * k_x[i] + H_xm * r_v * r_v;
                const double E6 = H_xm * T_v_bulk[i] - (h_xv_v - h_vx_x) * phi_x_v[i];

                const double alpha = 1.0 / (2 * r_o * (E1w - r_i) + r_i * r_i - E2w);
                const double delta = T_x_bulk[i] - T_w_bulk[i] + q_ow[i] / k_w[i] * (E1w - r_i) -
                    (E1x - r_i) * (E6 - E3 * T_x_bulk[i]) / (E4 - E1x * E3);
                const double gamma = r_i * r_i + ((E5 - E2x * E3) * (E1x - r_i)) / (E4 - E1x * E3) - E2x;

                // Parabolas coefficients evaluated explicitly
                ABC[6 * i + 5] = (-q_ow[i] +
                    2 * k_w[i] * (r_o - r_i) * alpha * delta +
                    k_x[i] * (E6 - E3 * T_x_bulk[i]) / (E4 - E1x * E3)) /
                    (2 * (r_i - r_o) * k_w[i] * alpha * gamma +
                        (E5 - E2x * E3) / (E4 - E1x * E3) * k_x[i] -
                        2 * r_i * k_x[i]);
                ABC[6 * i + 2] = alpha * (delta + gamma * ABC[6 * i + 5]);
                ABC[6 * i + 1] = q_ow[i] / k_w[i] - 2 * r_o * ABC[6 * i + 2];
                ABC[6 * i + 0] = T_w_bulk[i] - E1w * q_ow[i] / k_w[i] + (2 * r_o * E1w - E2w) * ABC[6 * i + 2];
                ABC[6 * i + 4] = (E6 - E3 * T_x_bulk[i] - (E5 - E2x * E3) * ABC[6 * i + 5]) / (E4 - E1x * E3);
                ABC[6 * i + 3] = T_x_bulk[i] - E1x * ABC[6 * i + 4] - E2x * ABC[6 * i + 5];

                // Update temperatures at the interfaces
                T_o_w[i] = ABC[6 * i + 0] + ABC[6 * i + 1] * r_o + ABC[6 * i + 2] * r_o * r_o; // Temperature at the outer wall
                T_w_x[i] = ABC[6 * i + 0] + ABC[6 * i + 1] * r_i + ABC[6 * i + 2] * r_i * r_i; // Temperature at the wall wick interface
                T_x_v[i] = ABC[6 * i + 3] + ABC[6 * i + 4] * r_v + ABC[6 * i + 5] * r_v * r_v; // Temperature at the wick vapor interface

                // Condenser power
                double conv = h_conv * (T_o_w[i] - T_env);                  // [W/m2]
                double irr = emissivity * sigma *
                    (std::pow(T_o_w[i], 4) - std::pow(T_env, 4));           // [W/m2]

                // Outer wall power profile
                if (mesh_center[i] >= (evaporator_start - delta_h) && mesh_center[i] < evaporator_start) {
                    double x = (mesh_center[i] - (evaporator_start - delta_h)) / delta_h;
                    q_ow[i] = 0.5 * q0 * (1.0 - std::cos(M_PI * x));
                }
                else if (mesh_center[i] >= evaporator_start && mesh_center[i] <= evaporator_end) {
                    q_ow[i] = q0;
                }
                else if (mesh_center[i] > evaporator_end && mesh_center[i] <= (evaporator_end + delta_h)) {
                    double x = (mesh_center[i] - evaporator_end) / delta_h;
                    q_ow[i] = 0.5 * q0 * (1.0 + std::cos(M_PI * x));
                }
                else if (mesh_center[i] >= condenser_start && mesh_center[i] < condenser_start + delta_c) {
                    double x = (mesh_center[i] - condenser_start) / delta_c;
                    double w = 0.5 * (1.0 - std::cos(M_PI * x));
                    q_ow[i] = -(conv + irr) * w;
                }
                else if (mesh_center[i] >= condenser_start + delta_c) {
                    q_ow[i] = -(conv + irr);
                }

                Q_ow[i] = q_ow[i] * 2 * r_o / (r_o * r_o - r_i * r_i);    // Outer wall heat source [W/m3]
                Q_wx[i] = k_w[i] * (ABC[6 * i + 1] + 2.0 * ABC[6 * i + 2] * r_i) * 2 * r_i / (r_i * r_i - r_v * r_v);            // Heat source to the wick due to wall-wick heat flux [W/m3]
                Q_xw[i] = -k_w[i] * (ABC[6 * i + 1] + 2.0 * ABC[6 * i + 2] * r_i) * 2 * r_i / (r_o * r_o - r_i * r_i);           // Heat source to the wall due to wall-wick heat flux [W/m3]
                Q_xm[i] = H_xm * (ABC[6 * i + 3] + ABC[6 * i + 4] * r_v + ABC[6 * i + 5] * r_v * r_v - T_v_bulk[i]) * 2.0 / r_v;  // Heat source to the vapor due to wick-vapor heat flux [W/m3])
                Q_mx[i] = -k_w[i] * (ABC[6 * i + 4] + 2.0 * ABC[6 * i + 5] * r_v) * 2.0 * r_v / (r_i * r_i - r_v * r_v);           // Heat source to the wick due to wick-vapor heat flux [W/m3]

                Q_mass_vapor[i] = +Gamma_xv_vapor[i] * h_xv_v; // Volumetric heat source [W/m3] due to evaporation/condensation (to be summed to the vapor)
                Q_mass_wick[i] = -Gamma_xv_wick[i] * h_vx_x;   // Volumetric heat source [W/m3] due to evaporation/condensation (to be summed to the wick)


                phi_x_v[i] = (sigma_e * vapor_sodium::P_sat(T_x_v[i]) / std::sqrt(T_x_v[i]) -
                                sigma_c * Omega * p_v[i] / std::sqrt(T_v_bulk[i])) /
                (std::sqrt(2 * M_PI * Rv));                             // Real evaporation mass flux [kg/(m2s)]


                if (time_total < 1.0) {

                    phi_x_v[i] = (sigma_e * vapor_sodium::P_sat(T_x_v[i]) -
                        sigma_c * Omega * p_v[i]) /
                        std::sqrt(2 * M_PI * Rv * T_x_v[i]);                 // Approximated evaporation mass flux [kg/(m2s)]
                }
                else {

                    const double A_Q_mass_vapor = sigma_e * saturation_pressure[i] / std::sqrt(T_x_v[i]);
                    const double B_Q_mass_vapor = sigma_c * Omega * p_v[i];
                    const double C_Q_mass_vapor = std::sqrt(2 * M_PI * Rv);

                    a_Q_mass_vapor[i] = h_xv_v * (2.0 * eps_s / r_v) * B_Q_mass_vapor / (2 * C_Q_mass_vapor) * std::pow(T_v_bulk[i], -3.0 / 2.0);
                    b_Q_mass_vapor[i] = h_xv_v * (2.0 * eps_s / r_v) * phi_x_v[i] - a_Q_mass_vapor[i] * T_v_bulk[i];
                }
               
                Gamma_xv_vapor[i] = phi_x_v[i] * 2.0 * eps_s / r_v;     // Volumetric mass source [kg/m3s] to vapor
                Gamma_xv_wick[i] = phi_x_v[i] * (2.0 * r_v * eps_s)
                    / (r_i * r_i - r_v * r_v);                          // Volumetric mass source [kg/m3s] to wick
            }

            // Coupling hypotheses: temperature is transferred to the pressure of the sodium vapor
            p_outlet_v = vapor_sodium::P_sat(T_x_v_iter[N - 1]);

            #pragma endregion

            // =======================================================================
            //                               [PICARD]
            // =======================================================================

            #pragma region picard

			// Picard error calculation

            double Aold, Anew, denom, eps;
            
            pic_error[0] = 0.0;
            pic_error[1] = 0.0;
            pic_error[2] = 0.0;

            for (int i = 0; i < N; ++i) {

                Aold = T_o_w_iter[i];
                Anew = T_o_w[i];
                denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                pic_error[0] += eps;

                Aold = T_w_x_iter[i];
                Anew = T_w_x[i];
                denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                pic_error[1] += eps;

                Aold = T_x_v_iter[i];
                Anew = T_x_v[i];
                denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                pic_error[2] += eps;

            }

            // Picard error normalization
            pic_error[0] /= N;
            pic_error[1] /= N;
            pic_error[2] /= N;

            if (pic_error[0] < 1e-2 &&
                pic_error[1] < 1e-2 &&
                pic_error[2] < 1e-2) {

				halves = 0;     // Reset halves if Picard converged
                break;          // Picard converged
            }

            // Iter = new
            T_o_w_iter = T_o_w;
            T_w_x_iter = T_w_x;
            T_x_v_iter = T_x_v;

            #pragma endregion
        }

        // =======================================================================
        //                               [WALL]
        // =======================================================================

        #pragma region wall

        // Loop to assembly the linear system for the wall bulk temperature
        for (int i = 1; i < N - 1; ++i) {

            // Physical properties
            const double cp = cp_w[i];
            const double rho = rho_w[i];

            const double k_l = 0.5 * (k_w[i - 1] + k_w[i]);
            const double k_r = 0.5 * (k_w[i + 1] + k_w[i]);

            Q_tot_w[i] = Q_ow[i] + Q_xw[i];

            aTW[i] = -k_l / (rho * cp * dz * dz);
            bTW[i] = 1 / dt + (k_l + k_r) / (rho * cp * dz * dz);
            cTW[i] = -k_r / (rho * cp * dz * dz);
            dTW[i] =
                + T_w_bulk_old[i] / dt
                + Q_ow[i] / (cp * rho)      // Positive if heat is added to the wall
                + Q_xw[i] / (cp * rho);     // Positive if heat is added to the wall
        }

        // BCs for the first node: zero gradient, adiabatic face
        aTW[0] = 0.0;
        bTW[0] = 1.0;
        cTW[0] = -1.0;
        dTW[0] = 0.0;

        // BCs for the last node: zero gradient, adiabatic face
        aTW[N - 1] = -1.0;
        bTW[N - 1] = 1.0;
        cTW[N - 1] = 0.0;
        dTW[N - 1] = 0.0;

        tdma_solver.solve(aTW, bTW, cTW, dTW, T_w_bulk);

        #pragma endregion

		// Convergence reached
        if (pic != max_picard) {

            // Update old values
            T_o_w_old = T_o_w;
            T_w_bulk_old = T_w_bulk;
            T_w_x_old = T_w_x;
            T_x_bulk_old = T_x_bulk;
            T_x_v_old = T_x_v;
            T_v_bulk_old = T_v_bulk;
            p_v_old = p_v;
            p_x_old = p_x;
            u_v_old = u_v;
            u_x_old = u_x;
            rho_v_old = rho_v;

            p_storage_x_old = p_storage_x;
            p_storage_v_old = p_storage_v;

            T_x_v_old = T_x_v;
            T_w_x_old = T_w_x;
            T_o_w_old = T_o_w;

            q_ow_old = q_ow;

            Q_ow_old = Q_ow;
            Q_wx_old = Q_wx;
            Q_xw_old = Q_xw;
            Q_xm_old = Q_xm;
            Q_mx_old = Q_mx;

            Q_mass_wick_old = Q_mass_wick;
            Q_mass_vapor_old = Q_mass_vapor;

            Gamma_xv_wick_old = Gamma_xv_wick;
            Gamma_xv_vapor_old = Gamma_xv_vapor;

            phi_x_v_old = phi_x_v;

            bXU_old = bXU;
            bVU_old = bVU;

            p_outlet_x_old = p_outlet_x;
            p_outlet_v_old = p_outlet_v;

            a_Q_mass_vapor_old = a_Q_mass_vapor;
            b_Q_mass_vapor_old = b_Q_mass_vapor;

            // Update total time elapsed
            time_total += dt;

		// Convergence not reached (max Picard iterations reached)
        } else {
            
            // Rollback to previous time step
            T_o_w = T_o_w_old;
            T_w_bulk = T_w_bulk_old;
            T_w_x = T_w_x_old;
            T_x_bulk = T_x_bulk_old;
            T_x_v = T_x_v_old;
            T_v_bulk = T_v_bulk_old;
            u_x = u_x_old;
            p_x = p_x_old;
            u_v = u_v_old;
            p_v = p_v_old;
            rho_v = rho_v_old;

            p_storage_x = p_storage_x_old;
            p_storage_v = p_storage_v_old;

            T_x_v = T_x_v_old;
            T_w_x = T_w_x_old;
            T_o_w = T_o_w_old;

            q_ow = q_ow_old;

            Q_ow = Q_ow_old;
            Q_wx = Q_wx_old;
            Q_xw = Q_xw_old;
            Q_xm = Q_xm_old;
            Q_mx = Q_mx_old;

            Q_mass_wick = Q_mass_wick_old;
            Q_mass_vapor = Q_mass_vapor_old;

            Gamma_xv_wick = Gamma_xv_wick_old;
            Gamma_xv_vapor = Gamma_xv_vapor_old;

            phi_x_v = phi_x_v_old;

            bXU = bXU_old;
            bVU = bVU_old;

            p_outlet_x = p_outlet_x_old;
            p_outlet_v = p_outlet_v_old;

            a_Q_mass_vapor = a_Q_mass_vapor_old;
            b_Q_mass_vapor = b_Q_mass_vapor_old;

            halves += 1;        // Reduce time step if max Picard iterations reached
			n -= 1;             // Repeat current time step
        }

        // =======================================================================
        //                               [OUTPUT]
        // =======================================================================

        #pragma region output

        if(n % sampling_frequency == 0){
            for (int i = 0; i < N; ++i) {

                v_velocity_output << u_v[i] << " ";
                v_pressure_output << p_v[i] << " ";
                v_bulk_temperature_output << T_v_bulk[i] << " ";
                v_rho_output << rho_v[i] << " ";

                x_velocity_output << u_x[i] << " ";
                x_pressure_output << p_x[i] << " ";
                x_bulk_temperature_output << T_x_bulk[i] << " ";
                x_rho_output << rho_x[i] << " ";

                x_v_temperature_output << T_x_v[i] << " ";
                w_x_temperature_output << T_w_x[i] << " ";
                o_w_temperature_output << T_o_w[i] << " ";
                w_bulk_temperature_output << T_w_bulk[i] << " ";

                x_v_mass_flux_output << phi_x_v[i] << " ";

                Q_ow_output << Q_ow[i] << " ";
                Q_wx_output << Q_wx[i] << " ";
                Q_xw_output << Q_xw[i] << " ";
                Q_xm_output << Q_xm[i] << " ";
                Q_mx_output << Q_mx[i] << " ";

                Q_mass_vapor_output << Q_mass_vapor[i] << " ";
                Q_mass_wick_output << Q_mass_wick[i] << " ";

                saturation_pressure_output << saturation_pressure[i] << " ";
                sonic_velocity_output << sonic_velocity[i] << " ";
            }

            // Time between timesteps [ms]
            auto t_end_timestep = std::chrono::high_resolution_clock::now();
            double simulation_time = std::chrono::duration<double, std::milli>(t_end_timestep - t0).count();

            // Time from the start of the simulation
            double t_clock = omp_get_wtime();
            double clock_time = t_clock - start;

            time_output << time_total << " ";
            dt_output << dt << " ";
            simulation_time_output << simulation_time << " ";
            clock_time_output << clock_time << " ";

            double total_heat_source_wall = 0.0;
            double total_heat_source_wick = 0.0;
            double total_heat_source_vapor = 0.0;

            double total_mass_source_wick = 0.0;
            double total_mass_source_vapor = 0.0;

            for (int i = 1; i < N - 1; ++i) {

                total_heat_source_wall += Q_tot_w[i];
                total_heat_source_wick += Q_tot_x[i];
                total_heat_source_vapor += Q_tot_v[i];

                total_mass_source_wick += Gamma_xv_wick[i];
                total_mass_source_vapor += Gamma_xv_vapor[i];
            }

            total_heat_source_wall_output << total_heat_source_wall << " ";
            total_heat_source_wick_output << total_heat_source_wick << " ";
            total_heat_source_vapor_output << total_heat_source_vapor << " ";

            total_mass_source_wick_output << total_mass_source_wick << " ";
            total_mass_source_vapor_output << total_mass_source_vapor << " ";

            momentum_res_x_output << momentum_res_x << " ";
            continuity_res_x_output << continuity_res_x << " ";
            temperature_res_x_output << temperature_res_x << " ";

            momentum_res_v_output << momentum_res_v << " ";
            continuity_res_v_output << continuity_res_v << " ";
            temperature_res_v_output << temperature_res_v << " ";

            v_velocity_output << "\n";
            v_pressure_output << "\n";
            v_bulk_temperature_output << "\n";
            v_rho_output << "\n";

            x_velocity_output << "\n";
            x_pressure_output << "\n";
            x_bulk_temperature_output << "\n";
            x_rho_output << "\n";

            x_v_temperature_output << "\n";
            w_x_temperature_output << "\n";
            o_w_temperature_output << "\n";
            w_bulk_temperature_output << "\n";

            x_v_mass_flux_output << "\n";

            Q_ow_output << "\n";
            Q_wx_output << "\n";
            Q_xw_output << "\n";
            Q_xm_output << "\n";
            Q_mx_output << "\n";

            Q_mass_vapor_output << "\n";
            Q_mass_wick_output << "\n";

            saturation_pressure_output << "\n";
            sonic_velocity_output << "\n";

            v_velocity_output.flush();
            v_pressure_output.flush();
            v_bulk_temperature_output.flush();
            v_rho_output.flush();

            x_velocity_output.flush();
            x_pressure_output.flush();
            x_bulk_temperature_output.flush();
            x_rho_output.flush();

            x_v_temperature_output.flush();
            w_x_temperature_output.flush();
            o_w_temperature_output.flush();
            w_bulk_temperature_output.flush();

            x_v_mass_flux_output.flush();

            Q_ow_output.flush();
            Q_wx_output.flush();
            Q_xw_output.flush();
            Q_xm_output.flush();
            Q_mx_output.flush();

            Q_mass_vapor_output.flush();
            Q_mass_wick_output.flush();

            saturation_pressure_output.flush();
            sonic_velocity_output.flush();

            time_output.flush();
            simulation_time_output.flush();
            dt_output.flush();
            clock_time_output.flush();

            total_heat_source_wall_output.flush();
            total_heat_source_wick_output.flush();
            total_heat_source_vapor_output.flush();

            total_mass_source_wick_output.flush();
            total_mass_source_vapor_output.flush();

            momentum_res_x_output.flush();
            continuity_res_x_output.flush();
            temperature_res_x_output.flush();

            momentum_res_v_output.flush();
            continuity_res_v_output.flush();
            temperature_res_v_output.flush();
        }

        #pragma endregion
    }

    time_output.close();
    simulation_time_output.close();
    dt_output.close();

    v_velocity_output.close();
    v_pressure_output.close();
    v_bulk_temperature_output.close();
	v_rho_output.close();

    x_velocity_output.close();
    x_pressure_output.close();
    x_bulk_temperature_output.close();
    x_rho_output.close();

    x_v_temperature_output.close();
    w_x_temperature_output.close();
    o_w_temperature_output.close();
    w_bulk_temperature_output.close();

    x_v_mass_flux_output.close();

    Q_ow_output.close();
    Q_wx_output.close();
    Q_xw_output.close();
    Q_xm_output.close();
    Q_mx_output.close();

    Q_mass_vapor_output.close();
    Q_mass_wick_output.close();

    saturation_pressure_output.close();
    sonic_velocity_output.close();

    total_heat_source_wall_output.close();
    total_heat_source_wick_output.close();
    total_heat_source_vapor_output.close();

    total_mass_source_wick_output.close();
    total_mass_source_vapor_output.close();

    momentum_res_x_output.close();
    continuity_res_x_output.close();
    temperature_res_x_output.close();

    momentum_res_v_output.close();
    continuity_res_v_output.close();
    temperature_res_v_output.close();

    double end = omp_get_wtime();
    std::cout << "Execution time: " << end - start;

    return 0;
}