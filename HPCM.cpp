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

#include "tdma.h"
#include "steel.h"
#include "liquid_sodium.h"
#include "vapor_sodium.h"
#include "adaptive_dt.h"

#pragma region case_loading

std::vector<double> read_second_last_row(const std::string& filename, int N) {

    std::ifstream f(filename);
    std::string line, prev, last;

    // prev = riga precedente, last = riga corrente
    while (std::getline(f, line)) {
        prev = last;
        last = line;
    }

    // Se il file ha meno di due righe
    if (prev.empty()) return std::vector<double>(N, 0.0);

    std::vector<double> out;
    out.reserve(N);

    std::string token;
    for (char c : prev) {
        if (c == ' ' || c == '\t') {
            if (!token.empty()) {
                out.push_back(std::stod(token));
                token.clear();
            }
        }
        else {
            token.push_back(c);
        }
    }
    if (!token.empty()) out.push_back(std::stod(token));

    if (out.size() != N) out.resize(N, 0.0);
    return out;
}


std::vector<double> read_last_row(const std::string& filename, int N) {

    std::ifstream f(filename);
    std::string line, last;

    while (std::getline(f, line)) last = line;
    if (last.empty()) return std::vector<double>(N, 0.0);

    std::vector<double> out;
    out.reserve(N);

    std::string token;
    for (char c : last) {
        if (c == ' ' || c == '\t') {
            if (!token.empty()) {
                out.push_back(std::stod(token));
                token.clear();
            }
        }
        else {
            token.push_back(c);
        }
    }
    if (!token.empty()) out.push_back(std::stod(token));

    if (out.size() != N) out.resize(N, 0.0);
    return out;
}

double read_last_value(const std::string& filename) {

    std::ifstream f(filename);
    std::string line, last;

    // Prende l'ultima riga
    while (std::getline(f, line)) last = line;
    if (last.empty()) return 0.0;

    // Scansiona la riga e legge l'ultimo token numerico
    std::string token;
    double last_value = 0.0;

    for (char c : last) {
        if (c == ' ' || c == '\t') {
            if (!token.empty()) {
                last_value = std::stod(token);
                token.clear();
            }
        }
        else {
            token.push_back(c);
        }
    }

    // Ultimo token se presente
    if (!token.empty()) last_value = std::stod(token);

    return last_value;
}

std::string select_case() {

    std::vector<std::string> cases;

    for (const auto& entry : std::filesystem::directory_iterator(".")) {
        if (entry.is_directory()) {
            const std::string name = entry.path().filename().string();
            if (name.rfind("case_", 0) == 0) cases.push_back(name);
        }
    }

    if (cases.empty()) return "";

    std::cout << "Cases found:\n";
    for (size_t i = 0; i < cases.size(); ++i) {
        std::cout << i << ": " << cases[i] << "\n";
    }

    std::cout << "Press ENTER for a new case. Input the number and press ENTER to load a case: ";

    std::string s;
    std::getline(std::cin, s);

    if (s.empty()) return "";

    int idx = std::stoi(s);
    if (idx < 0 || idx >= static_cast<int>(cases.size())) return "";

    return cases[idx];
}

#pragma endregion

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
    const double emissivity = 0.5;          // Wall emissivity [-]
    const double sigma = 5.67e-8;           // Stefan-Boltzmann constant [W/m^2/K^4]
    const double Rv = 361.8;                // Gas constant for the sodium vapor [J/(kg K)]
    
    // Environmental boundary conditions
    const double h_conv = 10;               // Convective heat transfer coefficient for external heat removal [W/m^2/K]
    const double power = 119;               // Power at the evaporator side [W]
    const double T_env = 280.0;             // External environmental temperature [K]

    // Evaporation and condensation parameters
    const double eps_s = 1.0;               // Surface fraction of the wick available for phasic interface [-]
    const double sigma_e = 0.05;            // Evaporation accomodation coefficient [-]. 1 means optimal evaporation
    const double sigma_c = 0.05;            // Condensation accomodation coefficient [-]. 1 means optimal condensation
	double Omega = 1.0;                     // Initialization of Omega parameter for evaporation/condensation model [-]

    // Wick permeability parameters
    const double K = 1e-10;                 // Permeability [m^2]
    const double CF = 1e5;                  // Forchheimer coefficient [1/m]
            
    // Geometric parameters
    const int N = 20;                                          // Number of axial nodes [-]
    const double L = 0.982; 			                        // Length of the heat pipe [m]
    const double dz = L / N;                                    // Axial discretization step [m]
    const double evaporator_start = 0.020;                      // Evaporator begin [m]
	const double evaporator_end = 0.073;                        // Evaporator end [m]
	const double condenser_length = 0.292;                      // Condenser length [m]
    const double evaporator_nodes = 
        std::floor((evaporator_end - evaporator_start) / dz);   // Number of evaporator nodes
    const double condenser_nodes = 
        std::ceil(condenser_length / dz);                       // Number of condenser nodes
    const double adiabatic_nodes = 
        N - (evaporator_nodes + condenser_nodes);               // Number of adiabatic nodes
    const double r_o = 0.01335;                                 // Outer wall radius [m]
    const double r_i = 0.0112;                                  // Wall-wick interface radius [m]
    const double r_v = 0.01075;                                 // Vapor-wick interface radius [m]

    // Time-stepping parameters
    double      dt_user = 1e-4;             // Initial time step [s] (then it is updated according to the limits)
	double      dt = dt_user;               // Current time step [s]
    const int   nSteps = 1e20;              // Number of timesteps [-]
    double      time_total = 0.0;           // Total simulation time [s]
	double      dt_code = dt_user;          // Time step used in the code [s]
	int         halves = 0;                 // Number of halvings of the time step

	// Picard iteration parameters
	const double max_picard = 50;           // Maximum number of Picard iterations per time step [-]
	const double pic_tolerance = 1e-4;   	// Tolerance for Picard iterations [-]   
    std::vector<double> pic_error(6, 0.0);  // L1 error for picard convergence
    int pic = 0;                            // Outside to check if convergence is reached

    // PISO Wick parameters
    const int tot_outer_iter_x = 5;         // Outer iterations per time-step [-]
    const int tot_inner_iter_x = 10;        // Inner iterations per outer iteration [-]
    const double outer_tol_x = 1e-6;        // Tolerance for the outer iterations (velocity) [-]
    const double inner_tol_x = 1e-6;        // Tolerance for the inner iterations (pressure) [-]

    // PISO Vapor parameters
    const int tot_outer_iter_v = 5;         // Outer iterations per time-step [-]
    const int tot_inner_iter_v = 10;        // Inner iterations per outer iteration [-]
    const double outer_tol_v = 1e-6;        // Tolerance for the outer iterations (velocity) [-]
    const double inner_tol_v = 1e-6;        // Tolerance for the inner iterations (pressure) [-]

    // Mesh z positions
    std::vector<double> mesh(N, 0.0);
    for (int i = 0; i < N; ++i) mesh[i] = i * dz;

    // Output precision
    const int global_precision = 8;

    // Constant temperature for initialization
    const double T_init = 800.0;

    std::vector<double> T_o_w(N, T_init);
    std::vector<double> T_w_bulk(N, T_init);
    std::vector<double> T_w_x(N, T_init);
    std::vector<double> T_x_bulk(N, T_init);
    std::vector<double> T_x_v(N, T_init);
    std::vector<double> T_v_bulk(N, T_init);

    // Wick fields
    std::vector<double> u_x(N, -0.0001);        // Wick velocity field [m/s]
    std::vector<double> p_x(N);                 // Wick pressure field [Pa]
    std::vector<double> rho_x(N);               // Wick density field [Pa]
    std::vector<double> p_storage_x(N + 2);     // Wick padded pressure vector for R&C correction [Pa]
    double* p_padded_x = &p_storage_x[1];       // Poìnter to work on the wick pressure padded storage with the same indes

    for (int i = 0; i < N; ++i) p_x[i] = vapor_sodium::P_sat(T_x_v[i]);             // Initialization of the wick pressure
    for (int i = 0; i < N; ++i) rho_x[i] = liquid_sodium::rho(T_x_bulk[i]);     	// Initialization of the wick density

    // Vapor fields
    std::vector<double> u_v(N, 0.1);            // Vapor velocity field [m/s]
    std::vector<double> p_v(N);                 // Vapor pressure field [Pa]
    std::vector<double> rho_v(N);               // Vapor density field [Pa]
    std::vector<double> p_storage_v(N + 2);     // Vapor padded pressure vector for R&C correction [Pa]
    double* p_padded_v = &p_storage_v[1];       // Poìnter to work on the storage with the same indes

    for (int i = 0; i < N; ++i) p_v[i] = vapor_sodium::P_sat(T_x_v[i]);             // Initialization of the vapor pressure

    // Vapor Equation of State update function. Updates density
    auto eos_update = [&](std::vector<double>& rho_, const std::vector<double>& p_, const std::vector<double>& T_) {

        for (int i = 0; i < N; i++) { rho_[i] = std::max(1e-6, p_[i] / (Rv * T_[i])); }

    }; eos_update(rho_v, p_v, T_v_bulk);

    // Heat fluxes at the interfaces
	std::vector<double> Q_ow(N, 0.0);       // Outer wall heat source [W/m3]
    std::vector<double> Q_wx(N, 0.0);       // Wall heat source due to fluxes [W/m3]
    std::vector<double> Q_xw(N, 0.0);       // Wick heat source due to fluxes [W/m3]
    std::vector<double> Q_xm(N, 0.0);       // Vapor heat source due to fluxes[W/m3]
	std::vector<double> Q_mx(N, 0.0);       // Wick heat source due to fluxes [W/m3]

    std::vector<double> Q_mass_vapor(N, 0.0);    // Heat volumetric source [W/m3] due to evaporation condensation. To be summed to the vapor
    std::vector<double> Q_mass_wick(N, 0.0);     // Heat volumetric source [W/m3] due to evaporation condensation. To be summed to the wick

	// Mass sources/fluxes
    std::vector<double> phi_x_v(N, 0.0);                // Mass flux [kg/m2/s] at the wick-vapor interface (positive if evaporation)
    std::vector<double> Gamma_xv_vapor(N, 0.0);         // Volumetric mass source [kg / (m^3 s)] (positive if evaporation)
    std::vector<double> Gamma_xv_wick(N, 0.0);          // Volumetric mass source [kg / (m^3 s)] (positive if evaporation)

	std::vector<double> saturation_pressure(N, 0.0);    // Saturation pressure field [Pa]
	std::vector<double> sonic_velocity(N, 0.0);         // Sonic velocity field [m/s]

    // Old values declaration
    std::vector<double> T_o_w_old;
    std::vector<double> T_w_bulk_old;
    std::vector<double> T_w_x_old;
    std::vector<double> T_x_bulk_old;
    std::vector<double> T_x_v_old;
    std::vector<double> T_v_bulk_old;

    std::vector<double> u_x_old;
    std::vector<double> p_x_old;

    std::vector<double> u_v_old;
    std::vector<double> p_v_old;
    std::vector<double> rho_v_old;

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

	// Select case to load or create a new one
    std::string case_chosen = select_case();

    if(case_chosen.empty()) {

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

        std::ofstream mesh_output(case_chosen + "/mesh.txt", std::ios::app);
        mesh_output << std::setprecision(8);

        for (int i = 0; i < N; ++i) mesh_output << i * dz << " ";

        mesh_output.flush();
        mesh_output.close();

        // Old values
        T_o_w_old = T_o_w;
        T_w_bulk_old = T_w_bulk;
        T_w_x_old = T_w_x;
        T_x_bulk_old = T_x_bulk;
        T_x_v_old = T_x_v;
        T_v_bulk_old = T_v_bulk;

        u_x_old = u_x;
        p_x_old = p_x;

        u_v_old = u_v;
        p_v_old = p_v;
        rho_v_old = rho_v;
        
        /*
		// Linear constant heat distribution for testing
        const double q_max = 2 * power / (M_PI * L * r_o);

        static std::vector<double> z(N);
        for (int j = 0; j < N; ++j) z[j] = (j + 0.5) * dz;

        for (int i = 0; i < N; ++i) {
            Q_ow[i] = q_max * (1 - 2 * (z[i] / L)) * 2 * r_o / (r_o * r_o - r_i * r_i);
        } 
        */

    } else {

        // Load state variables
        u_v = read_last_row(case_chosen + "/vapor_velocity.txt", N);
        p_v = read_last_row(case_chosen + "/vapor_pressure.txt", N);
        T_v_bulk = read_last_row(case_chosen + "/vapor_bulk_temperature.txt", N);
        rho_v = read_last_row(case_chosen + "/rho_vapor.txt", N);

        u_x = read_last_row(case_chosen + "/wick_velocity.txt", N);
        p_x = read_last_row(case_chosen + "/wick_pressure.txt", N);
        T_x_bulk = read_last_row(case_chosen + "/wick_bulk_temperature.txt", N);
        rho_x = read_last_row(case_chosen + "/rho_wick.txt", N);

        T_x_v = read_last_row(case_chosen + "/wick_vapor_interface_temperature.txt", N);
        T_w_x = read_last_row(case_chosen + "/wall_wick_interface_temperature.txt", N);
        T_o_w = read_last_row(case_chosen + "/outer_wall_temperature.txt", N);
        T_w_bulk = read_last_row(case_chosen + "/wall_bulk_temperature.txt", N);

        phi_x_v = read_last_row(case_chosen + "/wick_vapor_mass_source.txt", N);

		Q_ow = read_last_row(case_chosen + "/outer_wall_heat_flux.txt", N);
        Q_wx = read_last_row(case_chosen + "/wall_heat_source_flux.txt", N);
        Q_xw = read_last_row(case_chosen + "/wick_heat_source_flux.txt", N);
        Q_xm = read_last_row(case_chosen + "/vapor_heat_source_flux.txt", N);
        Q_mx = read_last_row(case_chosen + "/vapor_heat_source_flux.txt", N);
        
        Q_mass_vapor = read_last_row(case_chosen + "/vapor_heat_source_mass.txt", N);
        Q_mass_wick = read_last_row(case_chosen + "/wick_heat_source_mass.txt", N);

		time_total = read_last_value(case_chosen + "/time.txt");

        // Old values
        T_o_w_old = read_second_last_row(case_chosen + "/outer_wall_temperature.txt", N);
        T_w_bulk_old = read_second_last_row(case_chosen + "/wall_bulk_temperature.txt", N);
        T_w_x_old = read_second_last_row(case_chosen + "/wall_wick_interface_temperature.txt", N);
        T_x_bulk_old = read_second_last_row(case_chosen + "/wick_bulk_temperature.txt", N);
        T_x_v_old = read_second_last_row(case_chosen + "/wick_vapor_interface_temperature.txt", N);
        T_v_bulk_old = read_second_last_row(case_chosen + "/vapor_bulk_temperature.txt", N);

        u_x_old = read_second_last_row(case_chosen + "/wick_velocity.txt", N);
        p_x_old = read_second_last_row(case_chosen + "/wick_pressure.txt", N);

        u_v_old = read_second_last_row(case_chosen + "/vapor_velocity.txt", N);
        p_v_old = read_second_last_row(case_chosen + "/vapor_pressure.txt", N);
        rho_v_old = read_second_last_row(case_chosen + "/rho_vapor.txt", N);
    }

	// Padding pressure storages
    for (int i = 0; i < N; i++) p_storage_x[i + 1] = p_x[i];
    p_storage_x[0] = p_storage_x[1];
    p_storage_x[N + 1] = p_storage_x[N];

    for (int i = 0; i < N; i++) p_storage_v[i + 1] = p_v[i];
    p_storage_v[0] = p_storage_v[1];
    p_storage_v[N + 1] = p_storage_v[N];

    // Steam outputs
    std::ofstream time_output(case_chosen + "/time.txt", std::ios::app);

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
    std::ofstream Q_wx_output(case_chosen + "/wall_wx_heat_source.txt", std::ios::app);
    std::ofstream Q_xw_output(case_chosen + "/wick_wx_heat_source.txt", std::ios::app);
    std::ofstream Q_xm_output(case_chosen + "/wick_xv_heat_source.txt", std::ios::app);
    std::ofstream Q_mx_output(case_chosen + "/vapor_xv_heat_source.txt", std::ios::app);

    std::ofstream Q_mass_vapor_output(case_chosen + "/vapor_heat_source_mass.txt", std::ios::app);
    std::ofstream Q_mass_wick_output(case_chosen + "/wick_heat_source_mass.txt", std::ios::app);

    std::ofstream saturation_pressure_output(case_chosen + "/saturation_pressure.txt", std::ios::app);
    std::ofstream sonic_velocity_output(case_chosen + "/sonic_velocity.txt", std::ios::app);

    time_output << std::setprecision(global_precision);

    v_velocity_output << std::setprecision(global_precision);
    v_pressure_output << std::setprecision(global_precision);
    v_bulk_temperature_output << std::setprecision(global_precision);
    v_rho_output << std::setprecision(global_precision);

    x_velocity_output << std::setprecision(global_precision);
    x_pressure_output << std::setprecision(global_precision);
    x_bulk_temperature_output << std::setprecision(global_precision);
    x_rho_output << std::setprecision(global_precision);

    x_v_temperature_output << std::setprecision(global_precision);
    w_x_temperature_output << std::setprecision(global_precision);
    o_w_temperature_output << std::setprecision(global_precision);
    w_bulk_temperature_output << std::setprecision(global_precision);

    x_v_mass_flux_output << std::setprecision(global_precision);

    Q_ow_output << std::setprecision(global_precision);
    Q_wx_output << std::setprecision(global_precision);
    Q_xw_output << std::setprecision(global_precision);
    Q_xm_output << std::setprecision(global_precision);
    Q_mx_output << std::setprecision(global_precision);

    Q_mass_vapor_output << std::setprecision(global_precision);
    Q_mass_wick_output << std::setprecision(global_precision);

    saturation_pressure_output << std::setprecision(global_precision);
    sonic_velocity_output << std::setprecision(global_precision);

    // Iter values (only for Picard loops)
    std::vector<double> T_o_w_iter(N, 0.0);
    std::vector<double> T_w_bulk_iter(N, 0.0);
    std::vector<double> T_w_x_iter(N, 0.0);
    std::vector<double> T_x_bulk_iter(N, 0.0);
    std::vector<double> T_x_v_iter(N, 0.0);
    std::vector<double> T_v_bulk_iter(N, 0.0);

    // Wick BCs
    const double u_inlet_x = 0.0;                               // Wick inlet velocity [m/s]
    const double u_outlet_x = 0.0;                              // Wick outlet velocity [m/s]
    double p_outlet_x = vapor_sodium::P_sat(T_x_v[N - 1]);      // Wick outlet pressure [Pa]

    // Vapor BCs
    const double u_inlet_v = 0.0;                               // Vapor inlet velocity [m/s]
    const double u_outlet_v = 0.0;                              // Vapor outlet velocity [m/s]
    double p_outlet_v = vapor_sodium::P_sat(T_v_bulk[N - 1]);   // Vapor outlet pressure [Pa]

    // Turbulence constants for sodium vapor (SST model)
    const double Pr_t = 0.01;                                   // Prandtl turbulent number for sodium vapor [-]
    const double I = 0.05;                                      // Turbulence intensity [-]
    const double L_t = 0.07 * L;                                // Turbulence length scale [m]
    const double k0 = 1.5 * pow(I * u_inlet_v, 2);              // Initial turbulent kinetic energy [m^2/s^2]
    const double omega0 = sqrt(k0) / (0.09 * L_t);              // Initial specific dissipation rate [1/s]
    const double sigma_k = 0.85;                                // k-equation turbulent Prandtl number [-]
    const double sigma_omega = 0.5;                             // ω-equation turbulent Prandtl number [-]
    const double beta_star = 0.09;                              // β* constant for SST model [-]
    const double beta = 0.075;                                  // β constant for turbulence model [-]
    const double alpha = 5.0 / 9.0;                             // α blending coefficient for SST model [-]

    // Turbulence fields for sodium vapor
    std::vector<double> k_turb(N, k0);
    std::vector<double> omega_turb(N, omega0);
    std::vector<double> mu_t(N, 0.0);

    // Models
    const int rhie_chow_on_off_x = 1;             // 0: no wick RC correction, 1: wick with RC correction
    const int rhie_chow_on_off_v = 1;             // 0: no vapor RC correction, 1: vapor with RC correction
    const int SST_model_turbulence_on_off = 0;    // 0: no vapor turbulence, 1: vapor with turbulence

    // Initialization of the vapor velocity tridiagonal coefficients
    std::vector<double> aXU(N, 0.0);                                        // Lower tridiagonal coefficient for wick velocity
    std::vector<double> bXU(N, rho_x[0] * dz / dt + 2 * mu_x[0] / dz);      // Central tridiagonal coefficient for wick velocity                     
    std::vector<double> cXU(N, 0.0);                                        // Upper tridiagonal coefficient for wick velocity
    std::vector<double> dXU(N, 0.0);                                        // Known vector coefficient for wick velocity

	// Initialization of the vapor velocity tridiagonal coefficients
    std::vector<double>  aVU(N, 0.0);                                                   // Lower tridiagonal coefficient for vapor velocity
    std::vector<double> bVU(N, 2 * (4.0 / 3.0 * mu_v[0] / dz) + dz / dt * rho_v[0]);    // Central tridiagonal coefficient for vapor velocity
    std::vector<double> cVU(N, 0.0);                                                    // Upper tridiagonal coefficient for vapor velocity
    std::vector<double> dVU(N, 0.0);                                                    // Known vector for vapor velocity

	// Residuals for wick loops
    double wick_momentum_residual = 1.0;
    double wick_temperature_residual = 1.0;
	double wick_continuity_residual = 1.0;

    int outer_iter_x = 0;
    int inner_iter_x = 0;

    double p_error_x = 0.0;
	double u_error_x = 0.0;

    // Residuals for vapor loops
    double vapor_momentum_residual = 1.0;
    double vapor_temperature_residual = 1.0;
    double vapor_continuity_residual = 1.0;

    int outer_iter_v = 0;
    int inner_iter_v = 0;

    double p_error_v = 0.0;
    double u_error_v = 0.0;
	double rho_error_v = 0.0;

    std::vector<double> T_prev_x(N);
    std::vector<double> T_prev_v(N);

    #pragma endregion

    // Print number of working threads
    std::cout << "Threads: " << omp_get_max_threads() << "\n";

    double start = omp_get_wtime();

    // Time stepping loop
    for (int n = 0; n < nSteps; ++n) { 

        double dt_cand_w = new_dt_w(dz, dt, T_w_bulk, Q_ow);
        double dt_cand_x = new_dt_x(dz, dt, u_x, T_x_bulk, Gamma_xv_wick, Q_wx);
        double dt_cand_v = new_dt_v(dz, dt, u_v, T_v_bulk, rho_v, Gamma_xv_vapor, Q_xm, bVU);

        dt_code = std::min(std::min(dt_cand_w, dt_cand_x), std::min(dt_cand_x, dt_cand_v));

        dt = std::min(dt_user, dt_code);

		dt *= std::pow(0.5, halves);

        T_o_w_iter = T_o_w_old;
        T_w_bulk_iter = T_w_bulk_old;
        T_w_x_iter = T_w_x_old;
        T_x_bulk_iter = T_x_bulk_old;
        T_x_v_iter = T_x_v_old;
        T_v_bulk_iter = T_v_bulk_old;

        for (pic = 0; pic < max_picard; pic++) {

            // Updating all properties
            for(int i = 0; i < N; ++i) {

                cp_w[i] = steel::cp(T_w_bulk_iter[i]);
                rho_w[i] = steel::rho(T_w_bulk_iter[i]);
                k_w[i] = steel::k(T_w_bulk_iter[i]);

				rho_x[i] = liquid_sodium::rho(T_x_bulk_iter[i]);
                mu_x[i] = liquid_sodium::mu(T_x_bulk_iter[i]);
                cp_x[i] = liquid_sodium::cp(T_x_bulk_iter[i]);
				k_x[i] = liquid_sodium::k(T_x_bulk_iter[i]);

                mu_v[i] = vapor_sodium::mu(T_v_bulk_iter[i]);
                cp_v[i] = vapor_sodium::cp(T_v_bulk_iter[i]);
                k_v[i] = vapor_sodium::k(T_v_bulk_iter[i], p_v[i]);
			}

            // =======================================================================
            //
            //                   [0. SOLVE INTERFACES AND FLUXES]
            //
            // =======================================================================

            #pragma region parabolic_profiles 

            /**
             * Temperature distribution coefficients (six coefficients per node, two parabolas)
             * First three coefficients are a_w, b_w, c_w
             * Last three coefficients are a_x, b_x, c_x
             */
            auto ABC = std::make_unique<std::array<double, 6>[] >(N);

            std::vector<double> q_ow(N);
            
            for (int i = 0; i < N; ++i) {

                q_ow[i] = Q_ow[i] * (r_o * r_o - r_i * r_i) / (2 * r_o);  // Mass flux [kg/m2/s] at the wick-vapor interface (positive if evaporation)

                // Mass flux from the wick to the vapor [kg/(m2 s)]
                /*phi_x_v[i] = (sigma_e * vapor_sodium::P_sat(T_x_v[i]) / std::sqrt(T_x_v[i]) -
                    sigma_c * Omega * p_v[i] / std::sqrt(T_v_bulk[i])) /
                    (std::sqrt(2 * M_PI * Rv));*/

                phi_x_v[i] = (sigma_e * vapor_sodium::P_sat(T_x_v[i]) - sigma_c * Omega * p_v[i]) 
                    / std::sqrt(2 * M_PI * Rv * T_x_v[i]);

                Gamma_xv_vapor[i] = phi_x_v[i] * 2.0 * eps_s / r_v;
                Gamma_xv_wick[i] = phi_x_v[i] * (2.0 * r_v * eps_s) / (r_i * r_i - r_v * r_v);

                /**
                 * Variable b [-], used to calculate omega.
                 * Ratio of the overrall speed to the most probable velocity of the vapor.
                 */
                //const double b = std::abs(-phi_x_v[i] / (p_v[i] * std::sqrt(2.0 / (Rv * T_v_bulk_iter[i]))));

                /**
                  * Linearization of the omega [-] function to correct the net evaporation/condensation mass flux
                  */
                /*
                if (b < 0.1192) Omega = 1.0 + b * std::sqrt(M_PI);
                else if (b <= 0.9962) Omega = 0.8959 + 2.6457 * b;
                else Omega = 2.0 * b * std::sqrt(M_PI);
                */

                const double k_bulk_w = steel::k(T_w_bulk_iter[i]);                             // Wall bulk thermal conductivity [W/(m K)]
                const double k_bulk_x = liquid_sodium::k(T_x_bulk_iter[i]);                     // Liquid bulk thermal conductivity [W/(m K)]
                const double cp_v = vapor_sodium::cp(T_v_bulk_iter[i]);                         // Vapor bulk specific heat [J/(kg K)]
                const double k_int_w = steel::k(T_w_x_iter[i]);                                 // Wall interfacial thermal conductivity [W/(m K)]
                const double k_int_x = liquid_sodium::k(T_w_x_iter[i]);                         // Wick interfacial thermal conductivity [W/(m K)]     
                const double k_x_v = liquid_sodium::k(T_x_v[i]);                                // Wick-vapor interface thermal conductivity [W/(m K)]  
                const double k_v_x = vapor_sodium::k(T_x_v[i], p_v[i]);                         // Vapor-wick interface thermal conductivity [W/(m K)]
                const double k_v_cond = vapor_sodium::k(T_v_bulk_iter[i], p_v[i]);              // Vapor thermal conductivity [W/(m K)]
                const double k_v_eff = k_v_cond + mu_t[i] * cp_v / Pr_t;                        // Effective vapor thermal conductivity [W/(m K)]
                const double mu_v = vapor_sodium::mu(T_v_bulk_iter[i]);                         // Vapor dynamic viscosity [Pa*s]
                const double Dh_v = 2.0 * r_v;                                                  // Hydraulic diameter of the vapor core [m]
                const double Re_v = rho_v[i] * std::fabs(u_v[i]) * Dh_v / mu_v;                 // Reynolds number [-]
                const double Pr_v = cp_v * mu_v / k_v_cond;                                     // Prandtl number [-]
                const double H_xm = vapor_sodium::h_conv(Re_v, Pr_v, k_v_cond, Dh_v);           // Convective heat transfer coefficient at the vapor-wick interface [W/m^2/K]
                saturation_pressure[i] = vapor_sodium::P_sat(T_x_v_iter[i]);                    // Saturation pressure [Pa]        
                const double H_xmdPsat_dT = saturation_pressure[i] * std::log(10.0) * (7740.0 / (T_x_v_iter[i] * T_x_v_iter[i]));   // Derivative of the saturation pressure wrt T [Pa/K]   

                const double fac = (2.0 * r_v * eps_s * beta) / (r_i * r_i);    // Useful factor in the coefficients calculation [s / m^2]

                double h_xv_v;          // Specific enthalpy [J/kg] of vapor upon phase change between wick and vapor
                double h_vx_x;          // Specific enthalpy [J/kg] of wick upon phase change between vapor and wick

				// Definition of enthalpies depending on the phase change direction
                if (phi_x_v[i] > 0.0) {

                    // Evaporation case
                    h_xv_v = vapor_sodium::h(T_x_v_iter[i]);
                    h_vx_x = liquid_sodium::h(T_x_v_iter[i]);

                }
                else {

                    // Condensation case
                    h_xv_v = vapor_sodium::h(T_v_bulk_iter[i]);
                    h_vx_x = liquid_sodium::h(T_x_v_iter[i])
                        + (vapor_sodium::h(T_v_bulk_iter[i]) - vapor_sodium::h(T_x_v_iter[i]));
                }

                // Coefficients for the parabolic temperature profiles in wall and wick (check equations)
                const double E1w = 2.0 / 3.0 * (r_o + r_i - 1 / (1 / r_o + 1 / r_i));
                const double E2w = 0.5 * (r_o * r_o + r_i * r_i);
                const double E1x = 2.0 / 3.0 * (r_i + r_v - 1 / (1 / r_i + 1 / r_v));
                const double E2x = 0.5 * (r_i * r_i + r_v * r_v);

                const double E3 = H_xm;
                const double E4 = -k_bulk_x + H_xm * r_v;
                const double E5 = -2.0 * r_v * k_bulk_x + H_xm * r_v * r_v;
                const double E6 = H_xm * T_v_bulk_iter[i] - (h_xv_v - h_vx_x) * phi_x_v[i];

                const double alpha = 1.0 / (2 * r_o * (E1w - r_i) + r_i * r_i - E2w);
                const double delta = T_x_bulk_iter[i] - T_w_bulk_iter[i] + q_ow[i] / k_bulk_w * (E1w - r_i) -
                    (E1x - r_i) * (E6 - E3 * T_x_bulk_iter[i]) / (E4 - E1x * E3);
                const double gamma = r_i * r_i + ((E5 - E2x * E3) * (E1x - r_i)) / (E4 - E1x * E3) - E2x;

                ABC[i][5] = (-q_ow[i] * k_int_w / k_bulk_w +
                    2 * k_int_w * (r_o - r_i) * alpha * delta +
                    k_int_x * (E6 - E3 * T_x_bulk_iter[i]) / (E4 - E1x * E3)) /
                    (2 * (r_i - r_o) * k_int_w * alpha * gamma +
                        (E5 - E2x * E3) / (E4 - E1x * E3) * k_int_x -
                        2 * r_i * k_int_x);
                ABC[i][2] = alpha * (delta + gamma * ABC[i][5]);
                ABC[i][1] = q_ow[i] / k_bulk_w - 2 * r_o * ABC[i][2];
                ABC[i][0] = T_w_bulk_iter[i] - E1w * q_ow[i] / k_bulk_w + (2 * r_o * E1w - E2w) * ABC[i][2];
                ABC[i][4] = (E6 - E3 * T_x_bulk_iter[i] - (E5 - E2x * E3) * ABC[i][5]) / (E4 - E1x * E3);
                ABC[i][3] = T_x_bulk_iter[i] - E1x * ABC[i][4] - E2x * ABC[i][5];

                // Update temperatures at the interfaces
                T_o_w[i] = ABC[i][0] + ABC[i][1] * r_o + ABC[i][2] * r_o * r_o; // Temperature at the outer wall
                T_w_x[i] = ABC[i][0] + ABC[i][1] * r_i + ABC[i][2] * r_i * r_i; // Temperature at the wall wick interface
                T_x_v[i] = ABC[i][3] + ABC[i][4] * r_v + ABC[i][5] * r_v * r_v; // Temperature at the wick vapor interface

                // Evaporator
                const double Lh = evaporator_end - evaporator_start;
                const double delta_h = 0.01;
                const double Lh_eff = Lh + delta_h;
                const double q0 = power / (2.0 * M_PI * r_o * Lh_eff);      // [W/m^2]

                // Condenser
                const double delta_c = 0.05;
                const double condenser_start = L - condenser_length;
                const double condenser_end = L;
                double conv = h_conv * (T_o_w[i] - T_env);          // [W/m^2]
                double irr = emissivity * sigma *
                    (std::pow(T_o_w[i], 4) - std::pow(T_env, 4));   // [W/m^2]


				// Heat flux distribution along the outer wall
                const double zi = (i + 0.5) * dz;

                if (zi >= (evaporator_start - delta_h) && zi < evaporator_start) {
                    double x = (zi - (evaporator_start - delta_h)) / delta_h;
                    q_ow[i] = 0.5 * q0 * (1.0 - std::cos(M_PI * x));
                }
                else if (zi >= evaporator_start && zi <= evaporator_end) {
                    q_ow[i] = q0;
                }
                else if (zi > evaporator_end && zi <= (evaporator_end + delta_h)) {
                    double x = (zi - evaporator_end) / delta_h;
                    q_ow[i] = 0.5 * q0 * (1.0 + std::cos(M_PI * x));
                } 
                else if (zi >= condenser_start && zi < condenser_start + delta_c) {
                    double x = (zi - condenser_start) / delta_c;
                    double w = 0.5 * (1.0 - std::cos(M_PI * x));
                    q_ow[i] = -(conv + irr) * w;
                }
                else if (zi >= condenser_start + delta_c) {
                    q_ow[i] = -(conv + irr);
                }

                /*
                
				// Step heat distribution for testing

                double conv = h_conv * (T_o_w[i] - T_env);          // [W/m^2]
                double irr = emissivity * sigma *
                    (std::pow(T_o_w[i], 4) - std::pow(T_env, 4));   // [W/m^2]

                // Step distribution
                if (zi >= evaporator_start - delta_h && zi <= evaporator_end + delta_h) {
                    q_ow[i] = q0;
                }
                else if (zi >= condenser_start && zi <= condenser_start + delta_c) {
                    q_ow[i] = -(conv + irr);
                }
                else if (zi > condenser_start + delta_c) {
                    q_ow[i] = -(conv + irr);
                }
                else {
                    q_ow[i] = 0.0;   // fuori dalle zone
                }

                */

				Q_ow[i] = q_ow[i] * 2 * r_o / (r_o * r_o - r_i * r_i);    // Outer wall heat source [W/m3]
				Q_wx[i] = k_int_w * (ABC[i][1] + 2.0 * ABC[i][2] * r_i) * 2 * r_i / (r_i * r_i - r_v * r_v);            // Heat source to the wick due to wall-wick heat flux [W/m3]
                Q_xw[i] = -k_int_w * (ABC[i][1] + 2.0 * ABC[i][2] * r_i) * 2 * r_i / (r_o * r_o - r_i * r_i);           // Heat source to the wall due to wall-wick heat flux [W/m3]
				Q_xm[i] = H_xm * (ABC[i][3] + ABC[i][4] * r_v + ABC[i][5] * r_v * r_v - T_v_bulk_iter[i]) * 2.0 / r_v;  // Heat source to the vapor due to wick-vapor heat flux [W/m3])
				Q_mx[i] = -k_int_x * (ABC[i][4] + 2.0 * ABC[i][5] * r_v) * 2.0  * r_v / (r_i * r_i - r_v * r_v);        // Heat source to the wick due to wick-vapor heat flux [W/m3]

                Q_mass_vapor[i] = +Gamma_xv_vapor[i] * h_xv_v; // Volumetric heat source [W/m3] due to evaporation/condensation (to be summed to the vapor)
                Q_mass_wick[i] = -Gamma_xv_wick[i] * h_vx_x;   // Volumetric heat source [W/m3] due to evaporation/condensation (to be summed to the wick)
            }

            // Coupling hypotheses: temperature is transferred to the pressure of the sodium vapor
            p_outlet_v = vapor_sodium::P_sat(T_x_v_iter[N - 1]);

            #pragma endregion

            // =======================================================================
            //
            //                           [1. SOLVE WALL]
            //
            // =======================================================================

            #pragma region wall

            std::vector<double> aTW(N, 0.0);                    // Lower tridiagonal coefficient for wall temperature
            std::vector<double> bTW(N, 0.0);                    // Central tridiagonal coefficient for wall temperature
            std::vector<double> cTW(N, 0.0);                    // Upper tridiagonal coefficient for wall temperature
            std::vector<double> dTW(N, 0.0);                    // Known vector coefficient for wall temperature

			// Loop to assembly the linear system for the wall bulk temperature
            for (int i = 1; i < N - 1; ++i) {

                // Physical properties
                const double cp = cp_w[i];
                const double rho = rho_w[i];

                const double k_l = 0.5 * (k_w[i - 1] + k_w[i]);
                const double k_r = 0.5 * (k_w[i + 1] + k_w[i]);

                aTW[i] = - k_l / (rho * cp * dz * dz);
                bTW[i] = 1 / dt + (k_l + k_r) / (rho * cp * dz * dz);
                cTW[i] = - k_r / (rho * cp * dz * dz);
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

            // Vector of final wall bulk temperatures
            T_w_bulk = tdma::solve(aTW, bTW, cTW, dTW);        

            #pragma endregion

            // =======================================================================
            //
            //                           [2. SOLVE WICK]
            //
            // =======================================================================

            #pragma region wick

            /**
             * Pressure coupling hypotheses: the meniscus in the last cell of the domain is 
             * considered flat, so the pressure of the wick is equal to the pressure of the vapor
             */
            p_outlet_x = p_v[N - 1];

            u_error_x = 1.0;

            wick_momentum_residual = 1.0;
            wick_temperature_residual = 1.0;

            outer_iter_x = 0;

			// Outer iterations for the wick momentum equations
            while (outer_iter_x < tot_outer_iter_x && (wick_momentum_residual > outer_tol_x || wick_temperature_residual > outer_tol_x)) {

				// -----------------------------------------------------------
				// MOMENTUM PREDICTOR: gets u*
				// ----------------------------------------------------------

                #pragma region momentum_predictor

                // Parallelizing here does not save time
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

                u_x = tdma::solve(aXU, bXU, cXU, dXU);

                #pragma endregion

                // Inner iterations variables reset
                p_error_x = 1.0;

				wick_continuity_residual = 1.0;
                inner_iter_x = 0;

				// Inner wick iterations for the continuity equation
                while (inner_iter_x < tot_inner_iter_x && wick_continuity_residual > inner_tol_x) {

                    // -------------------------------------------------------
					// CONTINUITY SATISFACTOR: gets p'
                    // -------------------------------------------------------

                    #pragma region continuity_satisfactor

                    // Tridiagonal coefficients for the pressure correction
                    std::vector<double> aXP(N, 0.0);
                    std::vector<double> bXP(N, 0.0);
                    std::vector<double> cXP(N, 0.0);
                    std::vector<double> dXP(N, 0.0);

                    std::vector<double> p_prime_x(N, 0.0);    // Wick correction pressure field [Pa]

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

                        const double F_l = rho_left_uw * u_l_face;      // [kg/(m2s)]
                        const double F_r = rho_right_uw * u_r_face;     // [kg/(m2s)]

                        const double mass_imbalance = (F_r - F_l);      // [kg/(m2s)]

                        const double mass_flux = - Gamma_xv_wick[i] * dz; // [kg/(m2s)]

                        const double rho_l_cd = 0.5 * (rho_L + rho_P); // [kg/m3]
                        const double rho_r_cd = 0.5 * (rho_P + rho_R); // [kg/m3]

                        const double E_l = rho_l_cd * avgInvbLU_L / dz; // [s/m]
                        const double E_r = rho_r_cd * avgInvbLU_R / dz; // [s/m]

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

                    p_prime_x = tdma::solve(aXP, bXP, cXP, dXP);

                    #pragma endregion

                    // -------------------------------------------------------
					// PRESSURE CORRECTOR: p = p + p'
                    // -------------------------------------------------------

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

                    p_x[0] = p_x[1];
                    p_storage_x[0] = p_storage_x[1];

                    p_x[N - 1] = p_outlet_x;
                    p_storage_x[N + 1] = p_outlet_x;

                    #pragma endregion

                    // -------------------------------------------------------
					// VELCOITY CORRECTOR: v = v - (grad p') / b
                    // -------------------------------------------------------

                    #pragma region velocity_corrector

                    /**
                      * @brief Corrects the velocity with p'
                      */
                    u_error_x = 0.0;

                    for (int i = 1; i < N - 1; i++) {

                        double u_prev = u_x[i];
                        u_x[i] = u_x[i] - (p_prime_x[i + 1] - p_prime_x[i - 1]) / (2.0 * bXU[i]);

                        u_error_x = std::max(u_error_x, std::fabs(u_x[i] - u_prev));
                    }

                    #pragma endregion

                    // -------------------------------------------------------
                    // CONTINUITY RESIDUAL CALCULATION
                    // -------------------------------------------------------

                    #pragma region continuity_residual_calculation

                    double phi_ref = 0.0;
                    double Sm_ref = 0.0;

                    for (int i = 1; i < N - 1; ++i) {

                        const double u_l_face = 0.5 * (u_x[i - 1] + u_x[i]);
                        const double u_r_face = 0.5 * (u_x[i] + u_x[i + 1]);

                        // Upwind densities at faces
                        const double rho_left_uw = (u_l_face >= 0.0) ? rho_x[i - 1] : rho_x[i];
                        const double rho_right_uw = (u_r_face >= 0.0) ? rho_x[i] : rho_x[i + 1];

                        phi_ref = std::max(phi_ref, rho_left_uw * std::abs(u_l_face));
                        phi_ref = std::max(phi_ref, rho_right_uw * std::abs(u_r_face));

                        Sm_ref = std::max(Sm_ref, std::abs(Gamma_xv_wick[i] * dz));
                    }

                    const double cont_ref = std::max({ phi_ref, Sm_ref, 1e-30 });

                    wick_continuity_residual = 0.0;

                    for (int i = 1; i < N - 1; ++i) {

                        const double avgInvbLU_L = 0.5 * (1.0 / bXU[i - 1] + 1.0 / bXU[i]);     // [m2s/kg]
                        const double avgInvbLU_R = 0.5 * (1.0 / bXU[i + 1] + 1.0 / bXU[i]);     // [m2s/kg]

                        const double rc_l = -avgInvbLU_L / 4.0 *
                            (p_padded_x[i - 2] - 3.0 * p_padded_x[i - 1] + 3.0 * p_padded_x[i] - p_padded_x[i + 1]);    // [m/s]
                        const double rc_r = -avgInvbLU_R / 4.0 *
                            (p_padded_x[i - 1] - 3.0 * p_padded_x[i] + 3.0 * p_padded_x[i + 1] - p_padded_x[i + 2]);    // [m/s]

                        const double u_l_face = 0.5 * (u_x[i - 1] + u_x[i]) + rhie_chow_on_off_x * rc_l;    // [m/s]
                        const double u_r_face = 0.5 * (u_x[i] + u_x[i + 1]) + rhie_chow_on_off_x * rc_r;    // [m/s]

                        // Upwind densities at faces
                        const double rho_left_uw = (u_l_face >= 0.0) ? rho_x[i - 1] : rho_x[i];
                        const double rho_right_uw = (u_r_face >= 0.0) ? rho_x[i] : rho_x[i + 1];

                        const double F_l = rho_left_uw * u_l_face;      // [kg/(m2s)]
                        const double F_r = rho_right_uw * u_r_face;     // [kg/(m2s)]

                        const double mass_imbalance = (F_r - F_l);  // [kg/(m2s)]

                        const double mass_flux = - Gamma_xv_wick[i] * dz;           // [kg/(m2s)]

                        wick_continuity_residual =
                            std::max(wick_continuity_residual,
                                std::abs(mass_flux - mass_imbalance) / cont_ref);
                    }

                    #pragma endregion

                    inner_iter_x++;
                }

                // -------------------------------------------------------
                // MOMENTUM RESIDUAL CALCULATION
                // -------------------------------------------------------

                #pragma region momentum_residual_calculation

                double U_ref = 0.0;
                double F_ref = 0.0;

                for (int i = 0; i < N; ++i) {

                    U_ref = std::max(U_ref, std::abs(u_x[i]));
                }

                for (int i = 0; i < N; ++i) {
                    const double F_inertia = rho_x[i] * U_ref * U_ref;
                    const double F_unsteady = rho_x[i] * U_ref * dz / dt;
                    const double F_viscous = mu_x[i] * U_ref / dz;

                    F_ref = std::max({ F_inertia, F_unsteady, F_viscous, 1e-30 });

                }

                wick_momentum_residual = 0.0;

                for (int i = 1; i < N - 1; ++i) {

                    const double avgInvbLU_L = 0.5 * (1.0 / bXU[i - 1] + 1.0 / bXU[i]);     // [m2s/kg]
                    const double avgInvbLU_R = 0.5 * (1.0 / bXU[i + 1] + 1.0 / bXU[i]);     // [m2s/kg]

                    const double rc_l = -avgInvbLU_L / 4.0 *
                        (p_padded_x[i - 2] - 3.0 * p_padded_x[i - 1] + 3.0 * p_padded_x[i] - p_padded_x[i + 1]);    // [m/s]
                    const double rc_r = -avgInvbLU_R / 4.0 *
                        (p_padded_x[i - 1] - 3.0 * p_padded_x[i] + 3.0 * p_padded_x[i + 1] - p_padded_x[i + 2]);    // [m/s]

                    const double D_l = 0.5 * (mu_x[i - 1] + mu_x[i]) / dz;
                    const double D_r = 0.5 * (mu_x[i] + mu_x[i + 1]) / dz;

                    const double u_l_face =
                        0.5 * (u_x[i - 1] + u_x[i]) + rc_l * rhie_chow_on_off_x;
                    const double u_r_face =
                        0.5 * (u_x[i] + u_x[i + 1]) + rc_r * rhie_chow_on_off_x;

                    // Upwind densities at faces
                    const double rho_left_uw = (u_l_face >= 0.0) ? rho_x[i - 1] : rho_x[i];
                    const double rho_right_uw = (u_r_face >= 0.0) ? rho_x[i] : rho_x[i + 1];

                    const double F_l = rho_left_uw * u_l_face;      // [kg/(m2s)]
                    const double F_r = rho_right_uw * u_r_face;     // [kg/(m2s)]

                    const double accum =
                        rho_x[i] * dz / dt * (u_x[i] - u_x_old[i]);

                    const double conv =
                        F_r * u_r_face - F_l * u_l_face;

                    const double diff =
                        D_r * (u_x[i + 1] - u_x[i])
                        - D_l * (u_x[i] - u_x[i - 1]);

                    const double press =
                        0.5 * (p_x[i + 1] - p_x[i - 1]);

                    const double R =
                        accum + conv - diff + press;

                    wick_momentum_residual =
                        std::max(wick_momentum_residual, std::abs(R) / F_ref);
                }

                #pragma endregion

                // -------------------------------------------------------
                // TEMPERATURE CALCULATION: gets T
                // -------------------------------------------------------

                #pragma region temperature_calculator

                // Tridiagonal coefficients for the wick temperature
                std::vector<double> aXT(N, 0.0);
                std::vector<double> bXT(N, 0.0);
                std::vector<double> cXT(N, 0.0);
                std::vector<double> dXT(N, 0.0);

                // Loop to assembly the linear system for the wick bulk temperature
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
                T_x_bulk = tdma::solve(aXT, bXT, cXT, dXT);

                // -------------------------------
                // TEMPERATURE RESIDUAL
                // -------------------------------
                wick_temperature_residual = 0.0;

                for (int i = 0; i < N; ++i) {

                    wick_temperature_residual = std::max(
                        wick_temperature_residual,
                        std::abs(T_x_bulk[i] - T_prev_x[i])
                    );
                }

                #pragma endregion

                outer_iter_x++;  
            }
            
            #pragma endregion

            // =======================================================================
            //
            //                           [3. SOLVE VAPOR]
            //
            // =======================================================================

            #pragma region vapor

            // Wick vapor coupling hypotheses
            p_v[N - 1] = p_outlet_v;

            // Initializing convergence metrics
            vapor_momentum_residual = 1.0;
            vapor_temperature_residual = 1.0;

            // Outer iterations variables reset
            outer_iter_v = 0;

			// Outer iterations for the vapor momentum equations
            while (outer_iter_v < tot_outer_iter_v && (vapor_momentum_residual > outer_tol_v || vapor_temperature_residual > outer_tol_v)) {

				// -----------------------------------------------------------
				// MOMENTUM PREDICTOR: gets u*
				// ----------------------------------------------------------

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
                        + rho_P_old * u_v_old[i] * dz / dt;         // [kg/(m s2)]
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

                u_v = tdma::solve(aVU, bVU, cVU, dVU);

                #pragma endregion

				// -------------------------------------------------------------
                // TEMPERATURE SOLVER: gets T
				// -------------------------------------------------------------

                #pragma region temperature_calculator

                // Energy equation for T (implicit), upwind convection, central diffusion
                std::vector<double> aVT(N, 0.0);
                std::vector<double> bVT(N, 0.0);
                std::vector<double> cVT(N, 0.0);
                std::vector<double> dVT(N, 0.0);

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

                    const double mu_P = vapor_sodium::mu(T_v_bulk_iter[i]);

                    const double keff_P = k_cond_P + SST_model_turbulence_on_off * (mu_t[i] * cp_P / Pr_t);
                    const double keff_L = k_cond_L + SST_model_turbulence_on_off * (mu_t[i - 1] * cp_L / Pr_t);
                    const double keff_R = k_cond_R + SST_model_turbulence_on_off * (mu_t[i + 1] * cp_R / Pr_t);

                    const double D_l = 0.5 * (keff_P + keff_L) / dz;
                    const double D_r = 0.5 * (keff_P + keff_R) / dz;

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
                        + rho_v[i] * cp_v[i] * dz / dt;     /// [W/(m2 K)]

                    dVT[i] =
                        + rho_v_old[i] * cp_P_old * dz / dt * T_v_bulk_old[i]
                        + dp_dt
                        + dpdz_up
                        + viscous_dissipation * dz
                        + Q_xm[i] * dz                      // Positive if heat from wick to vapor
                        + Q_mass_vapor[i] * dz;             // [W/m2]
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
                T_v_bulk = tdma::solve(aVT, bVT, cVT, dVT);

                #pragma endregion

                // Initializing convergence metrics
                vapor_continuity_residual = 1.0;

                // Inner iterations variables reset
                inner_iter_v = 0;

				// Inner iterations for the vapor continuity equation
                while (inner_iter_v < tot_inner_iter_v && vapor_continuity_residual > inner_tol_v) {

                    // -------------------------------------------------------
					// CONTINUITY SATISFACTOR: gets p'
                    // -------------------------------------------------------

                    #pragma region continuity_satisfactor

                    // Tridiagonal coefficients for the pressure correction
                    std::vector<double> aVP(N, 0.0);
                    std::vector<double> bVP(N, 0.0);
                    std::vector<double> cVP(N, 0.0);
                    std::vector<double> dVP(N, 0.0);

                    std::vector<double> p_prime_v(N, 0.0);    // Vapor correction pressure field [Pa]

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
                            +E_l + E_r
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

                    p_prime_v = tdma::solve(aVP, bVP, cVP, dVP);

                    #pragma endregion

                    // -------------------------------------------------------
                    // PRESSURE CORRECTOR
                    // -------------------------------------------------------

                    #pragma region pressure_corrector

                    p_error_v = 0.0;

                    for (int i = 0; i < N; i++) {

                        double p_prev = p_v[i];
                        p_v[i] += p_prime_v[i];        // Note that PISO does not require an under-relaxation factor
                        p_storage_v[i + 1] = p_v[i];

                        p_error_v = std::max(p_error_v, std::fabs(p_v[i] - p_prev));
                    }

                    p_v[0] = p_v[1];
                    p_storage_v[0] = p_storage_v[1];

                    p_v[N - 1] = p_outlet_v;
                    p_storage_v[N + 1] = p_outlet_v;

                    #pragma endregion

                    // -------------------------------------------------------
                    // VELOCITY CORRECTOR
                    // -------------------------------------------------------

                    #pragma region velocity_corrector

                    u_error_v = 0.0;

                    for (int i = 1; i < N - 1; ++i) {
                        double u_prev = u_v[i];

                        sonic_velocity[i] = std::sqrt(vapor_sodium::gamma(T_v_bulk_iter[i]) * Rv * T_v_bulk_iter[i]);

                        const double calc_velocity = u_v[i] -
                            (p_prime_v[i + 1] - p_prime_v[i - 1]) / (2.0 * bVU[i]);

                        if (calc_velocity < sonic_velocity[i]) {

                            u_v[i] = calc_velocity;

                        }
                        else {

                            //std::cout << "Sonic limit reached, limiting velocity" << "\n";
                            u_v[i] = sonic_velocity[i];

                        }

                        u_error_v = std::max(u_error_v, std::fabs(u_v[i] - u_prev));
                    }

                    #pragma endregion

                    // -------------------------------------------------------
                    // DENSITY CORRECTOR
                    // -------------------------------------------------------

                    #pragma region density_corrector

                    rho_error_v = 0.0;

                    for (int i = 0; i < N; ++i) {
                        double rho_prev = rho_v[i];
                        rho_v[i] += p_prime_v[i] / (Rv * T_v_bulk_iter[i]);
                        rho_error_v = std::max(rho_error_v, std::fabs(rho_v[i] - rho_prev));
                    }

                    #pragma endregion

                    // -------------------------------------------------------
                    // CONTINUITY RESIDUAL CALCULATION
                    // -------------------------------------------------------

                    #pragma region continuity_residual_calculation

                    vapor_continuity_residual = 0.0;

                    for (int i = 1; i < N - 1; ++i) {
                        vapor_continuity_residual = std::max(vapor_continuity_residual, std::fabs(dVP[i]));
                    }

                    #pragma endregion

                    inner_iter_v++;
                }

                // -------------------------------------------------------
                // MOMENTUM RESIDUAL CALCULATION
                // -------------------------------------------------------

                #pragma region momentum_residual_calculation

                vapor_momentum_residual = 0.0;

                for (int i = 1; i < N - 1; ++i) {
                    vapor_momentum_residual = std::max(vapor_momentum_residual, std::fabs(aVU[i] * u_v[i - 1] + bVU[i] * u_v[i] + cVU[i] * u_v[i + 1] - dVU[i]));
                }

                #pragma endregion

                // -------------------------------------------------------
                // TEMPERATURE RESIDUAL CALCULATION
                // -------------------------------------------------------

                #pragma region temperature_residual_calculation

                vapor_temperature_residual = 0.0;

                for (int i = 1; i < N - 1; ++i) {
                    vapor_temperature_residual = std::max(vapor_temperature_residual, std::fabs(T_v_bulk[i] - T_prev_v[i]));
                }

                #pragma endregion

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

                const double sigma_k = 0.85;        // Diffusion coefficient for k [-]
                const double sigma_omega = 0.5;     // Diffusion coefficient for ω [-]
                const double beta_star = 0.09;      // Production limiter coefficient [-]
                const double beta = 0.075;          // Dissipation coefficient for ω [-]
                const double alpha = 5.0 / 9.0;     // Blending coefficient [-]

                // Tridiagonal coefficients for k
                std::vector<double> aK(N, 0.0), 
                                        bK(N, 0.0), 
                                        cK(N, 0.0), 
                                        dK(N, 0.0);

                // Tridiagonal coefficients for omega
                std::vector<double> aW(N, 0.0), 
                                        bW(N, 0.0), 
                                        cW(N, 0.0), 
                                        dW(N, 0.0);

                std::vector<double> dudz(N, 0.0);       // Velocity gradient du/dz [1/s]
                std::vector<double> Pk(N, 0.0);         // Turbulence production rate term [m2/s3]

                // Compute velocity gradient and turbulence production term
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

                // k BCs, constant value at the inlet
                bK[0] = 1.0; 
                cK[0] = 0.0; 
                dK[0] = k_turb[0];

                // k BCs, constant value at the outlet
                aK[N - 1] = 0.0; 
                bK[N - 1] = 1.0; 
                dK[N - 1] = k_turb[N - 1];

                k_turb = tdma::solve(aK, bK, cK, dK);

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

                // w BCs, constant value at the inlet
                bW[0] = 1.0;  
                cW[0] = 0.0;
                dW[0] = omega_turb[0];

                // w BCs, constant value at the outlet
                aW[N - 1] = 0.0; 
                bW[N - 1] = 1.0; 
                dW[N - 1] = omega_turb[N - 1];

                omega_turb = tdma::solve(aW, bW, cW, dW);

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

            // Update density with new p,T
            eos_update(rho_v, p_v, T_v_bulk);

            #pragma endregion

            // =======================================================================
            //
            //                              [4. PICARD]
            //
            // =======================================================================

            #pragma region picard

			// Picard error calculation

            double Aold, Anew, denom, eps;
            
            pic_error[0] = 0.0;
            pic_error[1] = 0.0;
            pic_error[2] = 0.0;
            pic_error[3] = 0.0;
			pic_error[4] = 0.0;
			pic_error[5] = 0.0;

            for (int i = 0; i < N; ++i) {

				Aold = T_v_bulk_iter[i];
                Anew = T_v_bulk[i];
				denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
				eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
				pic_error[0] += eps;

				Aold = T_x_bulk_iter[i];
				Anew = T_x_bulk[i];
				denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
				eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                pic_error[1] += eps;

				Aold = T_w_bulk_iter[i];
				Anew = T_w_bulk[i];
				denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
				eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                pic_error[2] += eps;

                Aold = T_o_w_iter[i];
                Anew = T_o_w[i];
                denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                pic_error[3] += eps;

                Aold = T_w_x_iter[i];
                Anew = T_w_x[i];
                denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                pic_error[4] += eps;

                Aold = T_x_v_iter[i];
                Anew = T_x_v[i];
                denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                pic_error[5] += eps;
            }

            // Picard error normalization
            pic_error[0] /= N;
            pic_error[1] /= N;
            pic_error[2] /= N;
            pic_error[3] /= N;
            pic_error[4] /= N;
            pic_error[5] /= N;

            if (pic_error[0] < 1e-4 &&
                pic_error[1] < 1e-4 &&
                pic_error[2] < 1e-4 &&
                pic_error[3] < 1e-4 &&
                pic_error[4] < 1e-4 &&
                pic_error[5] < 1e-4) {

				halves = 0;     // Reset halves if Picard converged
                break;          // Picard converged
            }

            // Iter = new
            T_o_w_iter = T_o_w;
            T_w_bulk_iter = T_w_bulk;
            T_w_x_iter = T_w_x;
            T_x_bulk_iter = T_x_bulk;
            T_x_v_iter = T_x_v;
            T_v_bulk_iter = T_v_bulk;

            #pragma endregion
        }

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

            halves += 1;        // Reduce time step if max Picard iterations reached
			n -= 1;             // Repeat current time step
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

        const int output_every = 1000;

        if(n % output_every == 0){
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

            time_output << time_total << " ";

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
        }

        #pragma endregion
    }

    time_output.close();

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

    double end = omp_get_wtime();
    std::cout << "Execution time: " << end - start;

    return 0;
}