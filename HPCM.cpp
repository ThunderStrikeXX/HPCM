#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <array>
#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <string>
#include <limits>
#include <cstdint>
#include <regex>

#include "tdma.h"
#include "steel.h"
#include "liquid_sodium.h"
#include "vapor_sodium.h"
#include "adaptive_dt.h"


// =======================================================================
//                       [CONSTANTS AND VARIABLES]
// =======================================================================

#pragma region constants_and_variables

// Mathematical constants
const double pi = 3.141592;                 // Pi [-]

// Physical properties
const double emissivity = 0.5;              // Wall emissivity [-]
const double sigma = 5.67e-8;               // Stefan-Boltzmann constant [W/(m2K4)]
const double Rv = 361.5;                    // Gas constant for the sodium vapor [J/(kgK)]
const double gamma = 1.67;                  // Vapor sodium gamma constant

// Environmental boundary conditions
const double h_conv = 1;                    // Convective heat transfer coefficient for external heat removal [W/(m2K)]
const double power = 1000;                  // Power at the evaporator side [W]
const double T_env = 280.0;                 // External environmental temperature [K]

// Evaporation and condensation parameters
const double eps_s = 1.0;                   // Surface fraction of the liquid available for phasic interface [-]
const double sigma_e = 0.05;                // Evaporation accomodation coefficient [-]. 1 means optimal evaporation
const double sigma_c = 0.05;                // Condensation accomodation coefficient [-]. 1 means optimal condensation
double Omega = 1.0;                         // Initialization of Omega parameter for evaporation/condensation model [-]

// Wick permeability parameters
const double K = 1e-10;                     // Permeability [m^2]
const double CF = 1e5;                      // Forchheimer coefficient [1/m]
double const eps_v = 1.0;                   // Surface fraction of the wick available for liquid passage [-]

// Geometric parameters
const int N = 22;                                                   // Number of axial nodes (two cells are ghost boundaries) [-]
const double length = 1; 			                                // Length of the heat pipe [m]
const double dz = length / (N - 2);                                 // Axial discretization step [m]
const double r_o = 0.01335;                                         // Outer wall radius [m]
const double r_i = 0.0112;                                          // Wall-liquid interface radius [m]
const double r_v = 0.01075;                                         // Vapor-liquid interface radius [m]
const double Dh_v = 2.0 * r_v;                                      // Hydraulic diameter of the vapor core [m]
const double vol_wall_cell = (r_o * r_o - r_i * r_i) * pi * dz;     // Volume of the wall cell [m3]
const double vol_liquid_cell = (r_i * r_i - r_v * r_v) * pi * dz;   // Volume of the liquid cell [m3]
const double vol_vapor_cell = r_v * r_v * pi * dz;                  // Volume of the vapor cell [m3]
const double A_interface_cell = 2 * pi * r_i * dz;                  // Interfacial area between vapor and liquid for a cell [m2]     
const double core_section = pi * r_v * r_v;
const double wick_section = eps_v * pi * (r_i * r_i - r_v * r_v);

// Evaporator region parameters
const double evaporator_start = 0.020;                           // Evaporator begin [m]
const double evaporator_end = 0.073;                             // Evaporator end [m]
const double condenser_length = 0.292;                           // Condenser length [m]
const double evaporator_length = evaporator_end - evaporator_start;             // Evaporator length [m]
const double delta_h = 0.01;                                     // Evaporator ramp [m]
const double evaporator_length_eff = evaporator_length + delta_h;                              // Evaporator effective length [m]
const double q0 = power / (2.0 * pi * r_o * evaporator_length_eff);             // Evaporator heat flux [W/m2]

// Condenser region parameters
const double delta_c = 0.05;                                     // Condenser ramp [m]
const double condenser_start = length - condenser_length;             // Condenser begin [m]

// Coefficients for the parabolic temperature profiles in wall and liquid
const double E1w = 2.0 / 3.0 * (r_o + r_i - 1 / (1 / r_o + 1 / r_i));    // [m]
const double E2w = 0.5 * (r_o * r_o + r_i * r_i);                        // [m2]
const double E1x = 2.0 / 3.0 * (r_i + r_v - 1 / (1 / r_i + 1 / r_v));    // [m]
const double E2x = 0.5 * (r_i * r_i + r_v * r_v);                        // [m2]

// Time-stepping parameters
double           dt_user = 1e-4;                // Initial time step [s] (then it is updated according to the limits)
double           dt = dt_user;                  // Current time step [s]
double           time_total = 0.0;              // Total simulation time [s]
const double     time_simulation = 5000;        // Simulation total number [s]
double           dt_code = dt_user;             // Time step used in the code [s]
int              halves = 0;                    // Number of halvings of the time step [-]
const double     accelerator = 1;               // Adaptive timestep multiplier (TO BE FIXED, WHEN r_down IT ACCUMULATES) [-]

// Picard iteration parameters
int pic = 0;                                    // Outside to check if convergence is reached [-]
const double max_picard = 100;                  // Maximum number of Picard iterations per time step [-]
std::vector<double> pic_error(3, 0.0);          // L1 error for picard convergence [K, K, K]
std::vector<double> pic_tolerance(3, 1e-2);     // Picard convergence tolerance [K, K, K]

// PISO Liquid parameters
const int tot_simple_iter_x = 10;                    // Outer iterations per time-step [-]
const int tot_piso_iter_x = 10;                     // Inner iterations per outer iteration [-]
const double momentum_tol_x = 1e-8;              // Tolerance for the momentum equation [-]
const double continuity_tol_x = 1e-8;            // Tolerance for the continuity equation [-]
const double temperature_tol_x = 1e-3;           // Tolerance for the energy equation [-]

// PISO Vapor parameters
const int tot_simple_iter_v = 10;                   // Outer iterations per time-step [-]
const int tot_piso_iter_v = 10;                     // Inner iterations per outer iteration [-]
const double momentum_tol_v = 1e-8;              // Tolerance for the outer iterations (velocity) [-]
const double continuity_tol_v = 1e-8;            // Tolerance for the inner iterations (pressure) [-]
const double temperature_tol_v = 1e-3;           // Tolerance for the energy equation [-]

const double T_init = 1000;                      // Initial uniform temperature [K]

const double dT_init = 10.0;   // Ampiezza variazione iniziale [K]

const double T_left = T_init + dT_init;
const double T_right = T_init - dT_init;

// Temperature vectors
std::vector<double> T_o_w(N);     // Outer wall temperature [K]
std::vector<double> T_w_bulk(N);  // Wall bulk temperature [K]
std::vector<double> T_w_x(N);     // Wall–liquid interface temperature [K]
std::vector<double> T_x_bulk(N);  // Liquid bulk temperature [K]
std::vector<double> T_x_v(N);     // Liquid–vapor interface temperature [K]
std::vector<double> T_v_bulk(N);  // Vapor bulk temperature [K]

// Enthalpy vectors
std::vector<double> h_x(N);       // Liquid enthalpy [J/kg]
std::vector<double> h_v(N);       // Vapor enthalpy [J/kg]

// Liquid fields
std::vector<double> u_x(N, -0.0001);             // Liquid velocity field [m/s]
std::vector<double> p_x(N);                      // Liquid pressure field [Pa]
std::vector<double> p_prime_x(N, 0.0);           // Liquid correction pressure field [Pa]
std::vector<double> rho_x(N);                    // Liquid density field [Pa]
std::vector<double> p_storage_x(N + 2);          // Liquid padded pressure vector for R&C correction [Pa]
double* p_padded_x = &p_storage_x[1];            // Pointer to work on the liquid pressure padded storage with the same indexes

// Vapor fields
std::vector<double> u_v(N, 10.0);                // Vapor velocity field [m/s]
std::vector<double> p_v(N);                      // Vapor pressure field [Pa]
std::vector<double> p_prime_v(N, 0.0);           // Vapor correction pressure field [Pa]
std::vector<double> rho_v(N);                    // Vapor density field [Pa]
std::vector<double> p_storage_v(N + 2);          // Vapor padded pressure vector for R&C correction [Pa]
double* p_padded_v = &p_storage_v[1];            // Pointer to work on the storage with the same indexes

std::vector<double> DPcap(N, 0.0);

std::vector<double> alpha_l(N, 0.1);
std::vector<double> alpha_v(N, 0.9);

std::vector<double> c_l(N, 0.0);
std::vector<double> c_v(N, 0.0);

std::vector<double> mesh(N, 0.0);
std::vector<double> mesh_center(N, 0.0);

// Heat sources/fluxes at the interfaces
std::vector<double> q_ow(N);                         // Outer wall heat flux [W/m2]
std::vector<double> Q_ow(N, 0.0);                    // Outer wall heat source [W/m3]
std::vector<double> Q_wx(N, 0.0);                    // Wall heat source due to fluxes [W/m3]
std::vector<double> Q_xw(N, 0.0);                    // Liquid heat source due to fluxes [W/m3]
std::vector<double> Q_xm(N, 0.0);                    // Vapor heat source due to fluxes [W/m3]
std::vector<double> Q_mx(N, 0.0);                    // Liquid heat source due to fluxes [W/m3]

std::vector<double> Q_tot_w(N, 0.0);                 // Total heat flux in the wall [W/m3]
std::vector<double> Q_tot_x(N, 0.0);                 // Total heat flux in the liquid [W/m3]
std::vector<double> Q_tot_v(N, 0.0);                 // Total heat flux in the vapor [W/m3]

std::vector<double> heat_balance_surface(N, 0.0);       // Flux balance at vapor-liquid interface post-linearization [W/m2] 
std::vector<double> wall_liquid_heat_balance(N, 0.0);   // Wall-liquid interface heat fluxes difference (should be zero) [W/m2]
std::vector<double> liquid_vapor_heat_balance(N, 0.0);  // Vapor-liquid interface heat fluxes difference (should be zero) [W/m2]

std::vector<double> Q_mass_vapor(N, 0.0);            // Heat volumetric source [W/m3] due to evaporation condensation. To be summed to the vapor
std::vector<double> Q_mass_liquid(N, 0.0);           // Heat volumetric source [W/m3] due to evaporation condensation. To be summed to the liquid

// Mass sources/fluxes at the interfaces
std::vector<double> phi_x_v(N, 0.0);                 // Mass flux [kg/(m2s)] at the liquid-vapor interface (positive if evaporation)
std::vector<double> Gamma_v(N, 0.0);          // Volumetric mass source [kg/(m3s)] (positive if evaporation)
std::vector<double> Gamma_x(N, 0.0);         // Volumetric mass source [kg/(m3s)] (positive if evaporation)

// Secondary useful variables
std::vector<double> saturation_pressure(N, 0.0);     // Saturation pressure field [Pa]
std::vector<double> sonic_velocity(N, 0.0);          // Sonic velocity field [m/s]

std::vector<double> phi_x(N + 1, 0.0);           // Liquid face mass flux [kg/m2s]
std::vector<double> phi_v(N + 1, 0.0);           // Vapor face mass flux [kg/m2s]

// Old values declaration
std::vector<double> T_o_w_old;
std::vector<double> T_w_bulk_old;
std::vector<double> T_w_x_old;
std::vector<double> T_x_bulk_old;
std::vector<double> T_x_v_old;
std::vector<double> T_v_bulk_old;

std::vector<double> alpha_l_old;
std::vector<double> alpha_v_old;

std::vector<double> h_x_old = h_x;
std::vector<double> h_v_old = h_v;

std::vector<double> q_ow_old;

std::vector<double> Q_ow_old;
std::vector<double> Q_wx_old;
std::vector<double> Q_xw_old;
std::vector<double> Q_xm_old;
std::vector<double> Q_mx_old;

std::vector<double> Q_mass_liquid_old;
std::vector<double> Q_mass_vapor_old;

std::vector<double> Gamma_x_old;
std::vector<double> Gamma_v_old;

std::vector<double> phi_x_old;
std::vector<double> phi_v_old;

std::vector<double> u_x_old;
std::vector<double> p_x_old;
std::vector<double> p_storage_x_old;

std::vector<double> u_v_old;
std::vector<double> p_v_old;
std::vector<double> rho_v_old;
std::vector<double> p_storage_v_old;

// Wall physical properties
std::vector<double> cp_w(N);         // Wall specific heat at constant pressure [J/kgK]
std::vector<double> rho_w(N);        // Wall density [kg/m3]
std::vector<double> k_w(N);          // Wall thermal conductivity [W/mK]

// Liquid physical properties 
std::vector<double> mu_x(N);         // Liquid dynamic viscosity [Pa s]
std::vector<double> cp_x(N);         // Liquid specific heat at constant pressure [J/kgK]
std::vector<double> k_x(N);          // Liquid thermal conductivity [W/mK]

// Vapor physical properties
std::vector<double> mu_v(N);         // Vapor dynamic viscosity [Pa s]
std::vector<double> cp_v(N);         // Vapor specific heat at constant pressure [J/kgK]
std::vector<double> k_v(N);          // Vapor thermal conductivity in the vapor bulk [W/mK]
std::vector<double> k_v_int(N);      // Vapor thermal conductivity at the vapor-liquid interface [W/mK]
std::vector<double> Re_v(N);         // Vapor Reynolds number [-]
std::vector<double> HTC(N);          // Vapor heat transfer coefficient [W/m2K]

double h_x_phase;               // Specific enthalpy [J/kg] of liquid upon phase change between vapor and liquid
double h_v_phase;               // Specific enthalpy [J/kg] of vapor upon phase change between liquid and vapor

// Iter values (only for Picard loops)
std::vector<double> T_o_w_iter(N, 0.0);
std::vector<double> T_w_x_iter(N, 0.0);
std::vector<double> T_x_v_iter(N, 0.0);

// Parabolas coefficients vector 
std::vector<double> ABC(6 * N);

// Liquid BCs
const double u_inlet_x = 0.0;                               // Liquid inlet velocity [m/s]
const double u_outlet_x = 0.0;                              // Liquid outlet velocity [m/s]
double p_outlet_x = vapor_sodium::P_sat(T_x_v[N - 1]);      // Liquid outlet pressure [Pa]

// Vapor BCs
const double u_inlet_v = 0.0;                               // Vapor inlet velocity [m/s]
const double u_outlet_v = 0.0;                              // Vapor outlet velocity [m/s]
double p_outlet_v = vapor_sodium::P_sat(T_v_bulk[N - 1]);   // Vapor outlet pressure [Pa]

// Models
const int rhie_chow_on_off_x = 1;             // 0: no liquid RC correction, 1: liquid with RC correction
const int rhie_chow_on_off_v = 1;             // 0: no vapor RC correction, 1: vapor with RC correction

// Tridiagonal coefficients for the liquid velocity predictor
std::vector<double> aXU(N, 0.0);
std::vector<double> bXU(N, 0.0);
std::vector<double> bXU_old(N, 0.0);
std::vector<double> cXU(N, 0.0);
std::vector<double> dXU(N, 0.0);

// Tridiagonal coefficients for the vapor velocity predictor
std::vector<double> aVU(N, 0.0);
std::vector<double> bVU(N, 0.0);
std::vector<double> bVU_old(N, 0.0);
std::vector<double> cVU(N, 0.0);
std::vector<double> dVU(N, 0.0);

// Tridiagonal coefficients for the wall temperature
std::vector<double> aTW(N, 0.0);
std::vector<double> bTW(N, 0.0);
std::vector<double> cTW(N, 0.0);
std::vector<double> dTW(N, 0.0);

// Tridiagonal coefficients for the liquid pressure correction
std::vector<double> aXP(N, 0.0);
std::vector<double> bXP(N, 0.0);
std::vector<double> cXP(N, 0.0);
std::vector<double> dXP(N, 0.0);

// Tridiagonal coefficients for the vapor pressure correction
std::vector<double> aVP(N, 0.0);
std::vector<double> bVP(N, 0.0);
std::vector<double> cVP(N, 0.0);
std::vector<double> dVP(N, 0.0);

// Tridiagonal coefficients for the liquid temperature
std::vector<double> aXT(N, 0.0);
std::vector<double> bXT(N, 0.0);
std::vector<double> cXT(N, 0.0);
std::vector<double> dXT(N, 0.0);

// Tridiagonal coefficients for the vapor temperature
std::vector<double> aVT(N, 0.0);
std::vector<double> bVT(N, 0.0);
std::vector<double> cVT(N, 0.0);
std::vector<double> dVT(N, 0.0);

// Tridiagonal coefficients for the alpha_l
std::vector<double> aXA(N, 0.0);
std::vector<double> bXA(N, 0.0);
std::vector<double> cXA(N, 0.0);
std::vector<double> dXA(N, 0.0);

// Residuals for liquid loops
double momentum_res_x = 1.0;
double temperature_res_x = 1.0;
double continuity_res_x = 1.0;

// Index for liquid outer and inner iterations
int simple_iter_x = 0;
int piso_iter_x = 0;

// Errors for liquid pressure and velocity
double p_error_x = 0.0;
double u_error_x = 0.0;

// Residuals for mass, monentum and enthalpy equations
double momentum_res_v = 1.0;
double temperature_res_v = 1.0;
double continuity_res_v = 1.0;

// Index for vapor outer and inner iterations
int simple_iter_v = 0;
int piso_iter_v = 0;

// Errors for vapor pressure, velocity and density
double p_error_v = 0.0;
double u_error_v = 0.0;
double rho_error_v = 0.0;

// Previous values for residuals calculations
std::vector<double> T_prev_x(N);
std::vector<double> T_prev_v(N);

// Printing parameters
double t_last_print = 0.0;                   // Time from last print [s]
const double print_interval = 1e-3;           // Time interval for printing [s]

// TDMA solver
tdma::Solver tdma_solver(N);

#pragma endregion

// =======================================================================
//  WICK GEOMETRY & VOID FRACTION BOUNDS
// =======================================================================

// Pore geometry (assumed known constants)
const double R_pore = 1e-6;                                                 // Pore radius [m]
const double D_v = 2 * r_v;                                                 // Inner wick diameter [m]
const double D_i = 2 * r_i;                                                 // Outer wick diameter [m]
const double V_pore_hemi = (2.0 / 3.0) * pi * R_pore * R_pore * R_pore;     // Volume of a pore [m3]
const double A_pore = pi * R_pore * R_pore;                                 // Surface of a pore [m2]
const double flow_section = core_section + wick_section;                    // Flow section [m2]
const double alpha_wick_i_0 = core_section / flow_section;

inline double N_pore(double D) {

    return eps_v * pi * D / (A_pore * flow_section);
}

const double alpha_wick_i_plus = alpha_wick_i_0 + N_pore(D_i) * V_pore_hemi;

// =======================================================================
//  CONTACT ANGLE & CAPILLARY RADIUS
// =======================================================================

inline double compute_xi(double alpha_v) {

    if (alpha_v <= alpha_wick_i_0 || alpha_v >= alpha_wick_i_plus) {  // Overflow or dryout → flat surface
        return 1.0;
    }
    else if (alpha_v < alpha_wick_i_plus) {

        const double C1_arg = 3.0 * (alpha_v - alpha_wick_i_0)
            / (pi * R_pore * R_pore * R_pore * N_pore(D_v));
        const double C1 = C1_arg * C1_arg;

        const double inner = C1 * (1.0 + C1 * (3.0 + C1 * (3.0 + C1)));
        const double C2 = std::cbrt((C1 + 1.0) * (C1 + 1.0) + std::sqrt(inner));

        return 1.0 / C2 + C2 / (C1 + 1.0) - 1.0;
    }
    else {
        return 0.0;
    }
}

inline double compute_R_cap(double xi) {

    const double cos_theta = std::sqrt(1.0 - xi * xi);

    if (cos_theta < 1e-12) return 1e12;
    return R_pore / cos_theta;
}

// =======================================================================
//  CAPILLARY PRESSURE
// =======================================================================

double compute_Dpcap(double alpha_v, double T_x_v) {

    const double xi = compute_xi(alpha_v);
    const double surf_ten = liquid_sodium::surf_ten(T_x_v);

    if (std::abs(xi - 1.0) < 1e-12) {

        return 0.0;
    }

    const double R_cap = compute_R_cap(xi);
    return 2.0 * surf_ten / R_cap;
}

// =======================================================================
//  INTERFACIAL AREA DENSITY
// =======================================================================

double compute_a_int(double alpha_v) {

    if (alpha_v <= alpha_wick_i_0) {

        return (2.0 * pi / flow_section) * std::sqrt(alpha_v * flow_section / pi);
    }
    else if (alpha_v <= alpha_wick_i_plus) {

        const double xi = compute_xi(alpha_v);
        return (2.0 * pi * R_pore * R_pore * N_pore(D_v)) / (1.0 + xi);
    }
    else {

        return (pi / flow_section) * (D_i * D_i - 4.0 * (1.0 - alpha_v) * flow_section / pi);
    }
}

int main() {

    // Variables initialization

    for (int i = 0; i < N; ++i) {

        double xi = (N > 1) ? static_cast<double>(i) / (N - 1) : 0.0;

        double T_lin = T_left + (T_right - T_left) * xi;

        T_o_w[i] = T_lin;
        T_w_bulk[i] = T_lin;
        T_w_x[i] = T_lin;
        T_x_bulk[i] = T_lin;
        T_x_v[i] = T_lin;
        T_v_bulk[i] = T_lin;

        h_x[i] = liquid_sodium::h_l_linear(T_lin);
        h_v[i] = vapor_sodium::h_g_linear(T_lin);

        p_x[i] = vapor_sodium::P_sat(T_x_v[i]);
        p_v[i] = vapor_sodium::P_sat(T_x_v[i]);

        rho_x[i] = liquid_sodium::rho(T_x_bulk[i]);
        rho_v[i] = std::max(1e-6, p_v[i] / (Rv * T_v_bulk[i]));

        p_storage_x[i + 1] = p_x[i];
        p_storage_v[i + 1] = p_v[i];

        cp_w[i] = steel::cp(T_w_bulk[i]);
        rho_w[i] = steel::rho(T_w_bulk[i]);
        k_w[i] = steel::k(T_w_bulk[i]);

        mu_x[i] = liquid_sodium::mu(T_x_bulk[i]);
        cp_x[i] = liquid_sodium::cp_l_linear();
        k_x[i] = liquid_sodium::k(T_x_bulk[i]);

        mu_v[i] = vapor_sodium::mu(T_v_bulk[i]);
        k_v[i] = vapor_sodium::k(T_v_bulk[i], p_v[i]);
        cp_v[i] = vapor_sodium::cp_g_linear();
        k_v_int[i] = vapor_sodium::k(T_x_v[i], p_v[i]);

        bXU[i] = 2 * mu_x[i] / dz + rho_x[i] * dz / dt;
        bVU[i] = 2 * (4.0 / 3.0 * mu_v[i] / dz) + rho_v[i] * dz / dt;

        mesh[i] = (i - 1) * dz;
        mesh_center[i] = (i - 0.5) * dz;
    }

    // Enforcing boundary conditions
    
    p_storage_x[0] = p_storage_x[1];
    p_storage_x[N + 1] = p_storage_x[N];

    p_storage_v[0] = p_storage_v[1];
    p_storage_v[N + 1] = p_storage_v[N];

    const int output_precision = 6;                             // Output precision [-]

    // Case name input
    std::string case_name, case_chosen;
    const std::regex valid("^[A-Za-z0-9_-]+$");

    for (;;) {
        std::cout << "Enter case name: ";
        std::cin >> case_name;

        case_chosen = "case_" + case_name;

        if (!std::regex_match(case_name, valid)) continue;
        if (std::filesystem::exists(case_chosen)) continue;

        std::filesystem::create_directory(case_chosen);
        break;
    }

    std::cout << "Running case " << case_chosen << "\n";

    // Mesh center cells position output
    std::ofstream mesh_output(case_chosen + "/mesh.txt", std::ios::app);
    mesh_output << std::setprecision(output_precision);

    for (int i = 1; i < N - 1; ++i) mesh_output << mesh_center[i] << " ";

    mesh_output.flush();
    mesh_output.close();

    // Old values
    T_o_w_old = T_o_w;
    T_w_bulk_old = T_w_bulk;
    T_w_x_old = T_w_x;
    T_x_bulk_old = T_x_bulk;
    T_x_v_old = T_x_v;
    T_v_bulk_old = T_v_bulk;

    h_v_old = h_v;
    h_x_old = h_x;

    alpha_l_old = alpha_l;
    alpha_v_old = alpha_v;

    q_ow_old = q_ow;

    Q_ow_old = Q_ow;
    Q_wx_old = Q_wx;
    Q_xw_old = Q_xw;
    Q_xm_old = Q_xm;
    Q_mx_old = Q_mx;

    Q_mass_liquid_old = Q_mass_liquid;
    Q_mass_vapor_old = Q_mass_vapor;

    Gamma_x_old = Gamma_x;
    Gamma_v_old = Gamma_v;

    phi_x_old = phi_x;
    phi_v_old = phi_v;

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

    std::ofstream x_velocity_output(case_chosen + "/liquid_velocity.txt", std::ios::app);
    std::ofstream x_pressure_output(case_chosen + "/liquid_pressure.txt", std::ios::app);
    std::ofstream x_bulk_temperature_output(case_chosen + "/liquid_bulk_temperature.txt", std::ios::app);
    std::ofstream x_rho_output(case_chosen + "/rho_liquid.txt", std::ios::app);

    std::ofstream x_v_temperature_output(case_chosen + "/liquid_vapor_interface_temperature.txt", std::ios::app);
    std::ofstream w_x_temperature_output(case_chosen + "/wall_liquid_interface_temperature.txt", std::ios::app);
    std::ofstream o_w_temperature_output(case_chosen + "/outer_wall_temperature.txt", std::ios::app);
    std::ofstream w_bulk_temperature_output(case_chosen + "/wall_bulk_temperature.txt", std::ios::app);

    std::ofstream x_v_mass_flux_output(case_chosen + "/liquid_vapor_mass_source.txt", std::ios::app);

    std::ofstream Q_ow_output(case_chosen + "/outer_wall_heat_source.txt", std::ios::app);
    std::ofstream Q_wx_output(case_chosen + "/liquid_wx_heat_source.txt", std::ios::app);
    std::ofstream Q_xw_output(case_chosen + "/wall_wx_heat_source.txt", std::ios::app);
    std::ofstream Q_xm_output(case_chosen + "/vapor_xv_heat_source.txt", std::ios::app);
    std::ofstream Q_mx_output(case_chosen + "/liquid_xv_heat_source.txt", std::ios::app);

    std::ofstream Q_mass_vapor_output(case_chosen + "/vapor_heat_source_mass.txt", std::ios::app);
    std::ofstream Q_mass_liquid_output(case_chosen + "/liquid_heat_source_mass.txt", std::ios::app);

    std::ofstream saturation_pressure_output(case_chosen + "/saturation_pressure.txt", std::ios::app);
    std::ofstream sonic_velocity_output(case_chosen + "/sonic_velocity.txt", std::ios::app);

    std::ofstream total_heat_source_wall_output(case_chosen + "/total_heat_source_wall.txt", std::ios::app);
    std::ofstream total_heat_source_liquid_output(case_chosen + "/total_heat_source_liquid.txt", std::ios::app);
    std::ofstream total_heat_source_vapor_output(case_chosen + "/total_heat_source_vapor.txt", std::ios::app);

    std::ofstream total_mass_source_liquid_output(case_chosen + "/total_mass_source_liquid.txt", std::ios::app);
    std::ofstream total_mass_source_vapor_output(case_chosen + "/total_mass_source_vapor.txt", std::ios::app);

    std::ofstream momentum_res_x_output(case_chosen + "/momentum_res_x.txt", std::ios::app);
    std::ofstream continuity_res_x_output(case_chosen + "/continuity_res_x.txt", std::ios::app);
    std::ofstream temperature_res_x_output(case_chosen + "/temperature_res_x.txt", std::ios::app);

    std::ofstream momentum_res_v_output(case_chosen + "/momentum_res_v.txt", std::ios::app);
    std::ofstream continuity_res_v_output(case_chosen + "/continuity_res_v.txt", std::ios::app);
    std::ofstream temperature_res_v_output(case_chosen + "/temperature_res_v.txt", std::ios::app);

    std::ofstream global_heat_balance_output(case_chosen + "/global_heat_balance.txt", std::ios::app);
    std::ofstream heat_balance_surface_output(case_chosen + "/heat_balance_surface.txt", std::ios::app);
    std::ofstream wall_liquid_heat_balance_output(case_chosen + "/wall_liquid_heat_balance.txt", std::ios::app);
    std::ofstream liquid_vapor_heat_balance_output(case_chosen + "/liquid_vapor_heat_balance.txt", std::ios::app);

    std::ofstream reynolds_output(case_chosen + "/reynolds_vapor.txt", std::ios::app);
    std::ofstream HTC_output(case_chosen + "/HTC.txt", std::ios::app);
    std::ofstream DPcap_output(case_chosen + "/DPcap.txt", std::ios::app);

    std::ofstream alpha_l_output(case_chosen + "/alpha_l.txt", std::ios::app);
    std::ofstream alpha_v_output(case_chosen + "/alpha_v.txt", std::ios::app);

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
    Q_mass_liquid_output << std::setprecision(output_precision);

    saturation_pressure_output << std::setprecision(output_precision);
    sonic_velocity_output << std::setprecision(output_precision);

    total_heat_source_wall_output << std::setprecision(output_precision);
    total_heat_source_liquid_output << std::setprecision(output_precision);
    total_heat_source_vapor_output << std::setprecision(output_precision);

    total_mass_source_liquid_output << std::setprecision(output_precision);
    total_mass_source_vapor_output << std::setprecision(output_precision);

    momentum_res_x_output << std::setprecision(output_precision);
    continuity_res_x_output << std::setprecision(output_precision);
    temperature_res_x_output << std::setprecision(output_precision);

    momentum_res_v_output << std::setprecision(output_precision);
    continuity_res_v_output << std::setprecision(output_precision);
    temperature_res_v_output << std::setprecision(output_precision);

    heat_balance_surface_output << std::setprecision(output_precision);
    global_heat_balance_output << std::setprecision(output_precision);
    wall_liquid_heat_balance_output << std::setprecision(output_precision);
    liquid_vapor_heat_balance_output << std::setprecision(output_precision);

    reynolds_output << std::setprecision(output_precision);
    HTC_output << std::setprecision(output_precision);
    DPcap_output << std::setprecision(output_precision);

    alpha_l_output << std::setprecision(output_precision);
    alpha_v_output << std::setprecision(output_precision);

    // Start computational time measurement of whole simulation
    auto t_start_simulation = std::chrono::high_resolution_clock::now();

    // Time stepping loop
    while (time_total < time_simulation) {

        // Start computational time iteration
        auto t_start_timestep = std::chrono::high_resolution_clock::now();

        // Timestep calculation
        double dt_cand_w = new_dt_w(dt, T_w_bulk, Q_tot_w);
        double dt_cand_x = new_dt_x(dt, u_x, T_x_bulk, Gamma_x, Q_tot_x);
        double dt_cand_v = new_dt_v(dz, dt, u_v, T_v_bulk, rho_v, Gamma_v, Q_tot_v, bVU);

        dt_code = std::min(std::min(dt_cand_w, dt_cand_x), 
            std::min(dt_cand_x, dt_cand_v));                    // Choosing the minimum amongst all the candidates

		dt = std::min(dt_user, dt_code);                        // Choosing the minimum between user and calculated timestep
		dt *= std::pow(0.5, halves);                            // Halving the timestep if Picard failed
        dt *= accelerator;                                      // Accelerator multiplier
        if (dt < 1e-12) {

            std::cout << "Time " << time_total << ", convergence not achieved. \n";
            std::cout << "Timestep " << dt << " < 1e-12 \n";

            std::cout << "Absolute error outer wall temperature: " << pic_error[0] << "\n";
            std::cout << "Absolute error wall-wick temperature: " << pic_error[1] << "\n";
            std::cout << "Absolute error wick-vapor temperature: " << pic_error[2] << "\n";

            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cin.get();

            return 1;
        }
		// Iter = old (for Picard loops)
        T_o_w_iter = T_o_w_old;
        T_w_x_iter = T_w_x_old;
        T_x_v_iter = T_x_v_old;

        // Ramping for power to heat pipe

        constexpr double T_ramp = 0;   // End of the ramp [s]

        double ramp = 1.0;

        if (time_total <= 0.0) {
            ramp = 0.0;
        }
        else if (time_total < T_ramp) {
            ramp = 0.5 * (1.0 - std::cos(pi * time_total / T_ramp));
        }
        else {
            ramp = 1.0;
        }

        // Updating all properties
        for (int i = 0; i < N; ++i) {

            cp_w[i] = steel::cp(T_w_bulk[i]);
            rho_w[i] = steel::rho(T_w_bulk[i]);
            k_w[i] = steel::k(T_w_bulk[i]);

            rho_x[i] = liquid_sodium::rho(T_x_bulk[i]);
            mu_x[i] = liquid_sodium::mu(T_x_bulk[i]);
            cp_x[i] = liquid_sodium::cp_l_linear();
            k_x[i] = liquid_sodium::k(T_x_bulk[i]);

            mu_v[i] = vapor_sodium::mu(T_v_bulk[i]);
            k_v[i] = vapor_sodium::k(T_v_bulk[i], p_v[i]);
            cp_v[i] = vapor_sodium::cp_g_linear();
            k_v_int[i] = vapor_sodium::k(T_x_v[i], p_v[i]);
        }

        for (pic = 0; pic < max_picard; pic++) {

           // =======================================================================
           //                                [WICK]
           // =======================================================================

            #pragma region liquid

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

                    aXU[i] =
                        -std::max(phi_x[i], 0.0)
                        - D_l;
                    cXU[i] =
                        -std::max(-phi_x[i + 1], 0.0)
                        - D_r;
                    bXU[i] =
                        +std::max(phi_x[i + 1], 0.0)
                        + std::max(-phi_x[i], 0.0)
                        + rho_P * dz / dt
                        + D_l + D_r
                        + mu_P / K * dz
                        + CF * mu_P * dz / sqrt(K) * abs(u_x[i]);
                    dXU[i] =
                        -0.5 * (p_x[i + 1] - p_x[i - 1])
                        + rho_P_old * u_x_old[i] * dz / dt;
                }

                // Diffusion coefficients for the first and last node to define BCs
                const double D_first = mu_x[0] / dz;
                const double D_last = mu_x[N - 1] / dz;

                // Velocity BCs: fixed (zero) velocity on the first face
                aXU[0] = 0.0;
                bXU[0] = (rho_x[0] * dz / dt + 2 * D_first);
                cXU[0] = (rho_x[0] * dz / dt + 2 * D_first);
                dXU[0] = 0.0;

                // Velocity BCs: fixed (zero) velocity on the last face
                aXU[N - 1] = (rho_x[N - 1] * dz / dt + 2 * D_last);
                bXU[N - 1] = (rho_x[N - 1] * dz / dt + 2 * D_last);
                cXU[N - 1] = 0.0;
                dXU[N - 1] = 0.0;

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

                    const double C_l = phi_x[i];   // [W/m2]
                    const double C_r = phi_x[i + 1];

                    const double k_cond_P = k_x[i];
                    const double k_cond_L = k_x[i - 1];
                    const double k_cond_R = k_x[i + 1];

                    const double cp_P = cp_x[i];
                    const double cp_L = cp_x[i - 1];
                    const double cp_R = cp_x[i + 1];

                    const double cp_l = 0.5 * (cp_P + cp_L);
                    const double cp_r = 0.5 * (cp_P + cp_R);

                    const double D_l = 0.5 * (k_cond_P + k_cond_L) / dz;
                    const double D_r = 0.5 * (k_cond_P + k_cond_R) / dz;

                    Q_tot_x[i] = Q_wx[i] + Q_mx[i] + Q_mass_liquid[i];

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
                        + rho_P * dz / dt;                                  // [W/(m2 K)]

                    dXT[i] =
                        +rho_P_old * dz / dt * h_x_old[i]
                        + Q_wx[i] * dz                  // Positive if heat is added to the liquid
                        + Q_mx[i] * dz                  // Positive if heat is added to the liquid
                        + Q_mass_liquid[i] * dz;        // [W/m2]       
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

                tdma_solver.solve(aXT, bXT, cXT, dXT, h_x);

                // Recovering temperature from enthalpy
                for (int i = 0; i < N; i++) {

                    T_x_bulk[i] = liquid_sodium::T_from_h_l_linear(h_x[i]);
                }

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

                        const double mass_imbalance = (phi_x[i + 1] - phi_x[i]);          // [kg/(m2s)]

                        const double mass_flux = -Gamma_x[i] * dz;   // [kg/(m2s)]

                        const double rho_l_cd = 0.5 * (rho_L + rho_P);      // [kg/m3]
                        const double rho_r_cd = 0.5 * (rho_P + rho_R);      // [kg/m3]

                        const double E_l = rho_l_cd * avgInvbLU_L / dz;     // [s/m]
                        const double E_r = rho_r_cd * avgInvbLU_R / dz;     // [s/m]

                        aXP[i] = -E_l;              // [s/m]
                        cXP[i] = -E_r;              // [s/m]
                        bXP[i] = E_l + E_r;         // [s/m]
                        dXP[i] =
                            +mass_flux
                            - mass_imbalance;       // [kg/(m2s)]
                    }

                    // BCs for the correction of pressure: zero gradient at first face
                    aXP[0] = 0.0;
                    bXP[0] = 1.0;
                    cXP[0] = -1.0;
                    dXP[0] = 0.0;

                    // BCs for the correction of pressure: zero at first face
                    aXP[N - 1] = 1.0;
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

                    p_x[N - 1] = p_x[N - 2];
                    p_storage_x[N + 1] = p_storage_x[N];

                    double omega = 0.5;
                    for (int i = 0; i < N; ++i) {
                        double p_target = p_v[i] - DPcap[i];
                        p_x[i] = (1.0 - omega) * p_x[i] + omega * p_v[i];
                    }

                    #pragma endregion

                    // =========== VELOCITY CORRECTOR
                    #pragma region velocity_corrector

                    u_error_x = 0.0;

                    for (int i = 1; i < N - 1; i++) {

                        double u_prev = u_x[i];
                        u_x[i] = u_x[i] - (p_prime_x[i + 1] - p_prime_x[i - 1]) / (2.0 * bXU[i]);

                        u_error_x = std::max(u_error_x, std::abs(u_x[i] - u_prev));
                    }

                    #pragma endregion

                    // =========== FLUX CORRECTOR
                    #pragma region flux_corrector

                    for (int i = 1; i < N; ++i) {

                        const double avgInvbXU = 0.5 * (1.0 / bXU[i - 1] + 1.0 / bXU[i]); // [m2s/kg]

                        // Correzione incrementale coerente con la matrice p'
                        const double rho_face = (phi_x[i] >= 0.0) ? rho_x[i - 1] : rho_x[i];
                        phi_x[i] -= rho_face * avgInvbXU * (p_prime_x[i] - p_prime_x[i - 1]) / dz;

                    }

                    #pragma endregion

                    // =========== CONTINUITY RESIDUAL CALCULATOR
                    #pragma region continuity_residual_calculator

                    continuity_res_x = 0.0;

                    for (int i = 1; i < N - 1; ++i) {

                        const double mass_imbalance = (phi_x[i + 1] - phi_x[i]);          // [kg/(m2s)]
                        const double mass_flux = -Gamma_x[i] * dz;   // [kg/(m2s)]
                        dXP[i] = +mass_flux - mass_imbalance;       // [kg/(m2s)]

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

            // Apply Rhie and Chow correction OUTSIDE loops
            for (int i = 1; i < N; ++i) {
                const double avgInvbXU = 0.5 * (1.0 / bXU[i - 1] + 1.0 / bXU[i]);
                double rc = -avgInvbXU / 4.0 * (p_padded_x[i - 2] - 3 * p_padded_x[i - 1]
                    + 3 * p_padded_x[i] - p_padded_x[i + 1]);
                const double u_face = 0.5 * (u_x[i - 1] + u_x[i]) + rc;
                const double rho_face = (u_face >= 0.0) ? rho_x[i - 1] : rho_x[i];
                phi_x[i] = rho_face * u_face;
            }

            #pragma endregion

            // =======================================================================
            //                                [VAPOR]
            // =======================================================================

            #pragma region vapor

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

                    const double Re = u_v[i] * (2 * r_v) * rho_P / mu_P;
                    const double f = (Re < 1187.4) ? 64 / Re : 0.3164 * std::pow(Re, -0.25);
                    const double F = 0.25 * f * rho_P * std::abs(u_v[i]) / r_v;

                    aVU[i] =
                        -std::max(phi_v[i], 0.0)
                        - D_l;                              // [kg/(m2 s)]
                    cVU[i] =
                        -std::max(-phi_v[i + 1], 0.0)
                        - D_r;                              // [kg/(m2 s)]
                    bVU[i] =
                        +std::max(phi_v[i + 1], 0.0)
                        + std::max(-phi_v[i], 0.0)
                        + rho_P * dz / dt
                        + D_l + D_r
                        + F * dz;                           // [kg/(m2 s)]
                    dVU[i] =
                        -0.5 * (p_v[i + 1] - p_v[i - 1])
                        + rho_P_old * u_v_old[i] * dz / dt; // [kg/(m s2)]
                }

                /// Diffusion coefficients for the first and last node to define BCs
                const double D_first = (4.0 / 3.0) * mu_v[0] / dz;
                const double D_last = (4.0 / 3.0) * mu_v[N - 1] / dz;

                // Velocity BCs: zero velocity on the first node
                aVU[0] = 0.0;
                bVU[0] = +(rho_v[0] * dz / dt + 2 * D_first);
                cVU[0] = +(rho_v[0] * dz / dt + 2 * D_first);
                dVU[0] = 0.0;

                // Velocity BCs: zero velocity on the last node
                aVU[N - 1] = +(rho_v[N - 1] * dz / dt + 2 * D_last);
                bVU[N - 1] = +(rho_v[N - 1] * dz / dt + 2 * D_last);
                cVU[N - 1] = 0.0;
                dVU[N - 1] = 0.0;

                tdma_solver.solve(aVU, bVU, cVU, dVU, u_v);

                #pragma endregion

                // ==========  TEMPERATURE CALCULATOR 
                #pragma region temperature_calculator

                for (int i = 1; i < N - 1; i++) {

                    // Physical properties
                    const double rho_P = rho_v[i];
                    const double rho_L = rho_v[i - 1];
                    const double rho_R = rho_v[i + 1];

                    const double cp_P = cp_v[i];
                    const double cp_L = cp_v[i - 1];
                    const double cp_R = cp_v[i + 1];

                    const double cp_l = (phi_v[i] >= 0) ? cp_L : cp_P;
                    const double cp_r = (phi_v[i + 1] >= 0) ? cp_P : cp_R;

                    const double k_cond_P = k_v[i];
                    const double k_cond_L = k_v[i - 1];
                    const double k_cond_R = k_v[i + 1];

                    const double D_l = 0.5 * (k_cond_P + k_cond_L) / dz;
                    const double D_r = 0.5 * (k_cond_P + k_cond_R) / dz;

                    const double C_l = phi_v[i];
                    const double C_r = phi_v[i + 1];

                    const double dpdz_up = u_v[i] * (p_v[i + 1] - p_v[i - 1]) / 2.0;

                    const double dp_dt = (p_v[i] - p_v_old[i]) / dt * dz;

                    const double viscous_dissipation =
                        4.0 / 3.0 * 0.25 * mu_v[i] * ((u_v[i + 1] - u_v[i]) * (u_v[i + 1] - u_v[i])
                            + (u_v[i] + u_v[i - 1]) * (u_v[i] + u_v[i - 1])) / dz;

                    Q_tot_v[i] = /*dp_dt / dz + dpdz_up / dz + viscous_dissipation +*/ Q_xm[i] + Q_mass_vapor[i];

                    aVT[i] =
                        -D_l
                        - std::max(C_l, 0.0)
                        ;                                   /// [W/(m2K)]

                    cVT[i] =
                        -D_r
                        - std::max(-C_r, 0.0)
                        ;                                   /// [W/(m2K)]

                    bVT[i] =
                        +std::max(C_r, 0.0)
                        + std::max(-C_l, 0.0)
                        + D_l + D_r
                        + rho_v[i] * dz / dt;               /// [W/(m2 K)]

                    dVT[i] =
                        +rho_v_old[i] * dz / dt * h_v_old[i]
                        // + dp_dt
                        // + dpdz_up
                        // + viscous_dissipation * dz
                        +Q_xm[i] * dz                      // Positive if heat from liquid to vapor
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

                tdma_solver.solve(aVT, bVT, cVT, dVT, h_v);

                // Recovering temperture from enthalpy
                for (int i = 0; i < N; i++) {

                    T_v_bulk[i] = vapor_sodium::T_from_h_g_linear(h_v[i]);

                }

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

                        const double psi_i = 1.0 / (Rv * T_v_bulk[i]); // [kg/J]

                        const double Crho_l = phi_v[i] >= 0 ? (1.0 / (Rv * T_v_bulk[i - 1])) : (1.0 / (Rv * T_v_bulk[i]));  // [s2/m2]
                        const double Crho_r = phi_v[i + 1] >= 0 ? (1.0 / (Rv * T_v_bulk[i])) : (1.0 / (Rv * T_v_bulk[i + 1]));  // [s2/m2]

                        const double C_l = Crho_l * phi_v[i] / rho_v[i];       // [s/m]
                        const double C_r = Crho_r * phi_v[i + 1] / rho_v[i + 1];       // [s/m]

                        const double rho_l_upwind = (phi_v[i] >= 0.0) ? rho_v[i - 1] : rho_v[i];    // [kg/m3]
                        const double rho_r_upwind = (phi_v[i + 1] >= 0.0) ? rho_v[i] : rho_v[i + 1];    // [kg/m3]

                        const double mass_imbalance = (phi_v[i + 1] - phi_v[i]) + (rho_v[i] - rho_v_old[i]) * dz / dt;  // [kg/(m2s)]

                        const double mass_flux = Gamma_v[i] * dz;         // [kg/(m2s)]

                        const double E_l = 0.5 * (rho_v[i - 1] * (1.0 / bVU[i - 1]) + rho_v[i] * (1.0 / bVU[i])) / dz; // [s/m]
                        const double E_r = 0.5 * (rho_v[i] * (1.0 / bVU[i]) + rho_v[i + 1] * (1.0 / bVU[i + 1])) / dz; // [s/m]

                        aVP[i] =
                            -E_l
                            - std::max(C_l, 0.0)
                            ;                                   /// [s/m]

                        cVP[i] =
                            -E_r
                            - std::max(-C_r, 0.0)
                            ;                                   /// [s/m]

                        bVP[i] =
                            +E_l + E_r
                            + std::max(C_r, 0.0)
                            + std::max(-C_l, 0.0)
                            + psi_i * dz / dt;                  /// [s/m]

                        dVP[i] = +mass_flux - mass_imbalance;  /// [kg/(m2s)]
                    }

                    // BCs for the correction of pressure: zero gradient at first node
                    aVP[0] = 0.0;
                    bVP[0] = 1.0;
                    cVP[0] = -1.0;
                    dVP[0] = 0.0;

                    // BCs for the correction of pressure: zero at last node
                    aVP[N - 1] = -1.0;
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

                    p_v[N - 1] = p_v[N - 2];
                    p_storage_v[N + 1] = p_storage_v[N];

                    #pragma endregion

                    // =========== VELOCITY CORRECTOR
                    #pragma region velocity_corrector

                    u_error_v = 0.0;

                    for (int i = 1; i < N - 1; ++i) {

                        double u_prev = u_v[i];

                        // sonic_velocity[i] = std::sqrt(vapor_sodium::gamma(T_v_bulk[i]) * Rv * T_v_bulk[i]);

                        const double calc_velocity = u_v[i] -
                            (p_prime_v[i + 1] - p_prime_v[i - 1]) / (2.0 * bVU[i]);

                        u_v[i] = calc_velocity;

                        // if (calc_velocity < sonic_velocity[i]) 
                        // else u_v[i] = sonic_velocity[i];

                        u_error_v = std::max(u_error_v, std::abs(u_v[i] - u_prev));
                    }

                    #pragma endregion

                    // =========== FLUX CORRECTOR
                    #pragma region flux_corrector

                    for (int i = 1; i < N; ++i) {

                        const double avgInvbVU = 0.5 * (1.0 / bVU[i - 1] + 1.0 / bVU[i]); // [m2s/kg]

                        // Correzione incrementale coerente con la matrice p'
                        const double rho_face = (phi_v[i] >= 0.0) ? rho_v[i - 1] : rho_v[i];
                        phi_v[i] -= rho_face * avgInvbVU * (p_prime_v[i] - p_prime_v[i - 1]) / dz;

                    }

                    phi_v[0] = u_inlet_v * rho_v[0];
                    phi_v[1] = u_inlet_v * rho_v[0];

                    phi_v[N - 1] = u_outlet_v * rho_v[N - 1];
                    phi_v[N] = u_outlet_v * rho_v[N - 1];

                    #pragma endregion

                    // =========== DENSITY CORRECTOR
                    #pragma region density_corrector

                    rho_error_v = 0.0;

                    for (int i = 0; i < N; ++i) {
                        double rho_prev = rho_v[i];
                        rho_v[i] += p_prime_v[i] / (Rv * T_v_bulk[i]);
                        rho_error_v = std::max(rho_error_v, std::abs(rho_v[i] - rho_prev));
                    }

                    // Enforcing density BCs on ghost cells
                    rho_v[0] = rho_v[1];
                    rho_v[N - 1] = rho_v[N - 2];

                    #pragma endregion

                    // =========== CONTINUITY RESIDUAL CALCULATOR
                    #pragma region continuity_residual_calculator

                    continuity_res_v = 0.0;

                    for (int i = 1; i < N - 1; ++i) {

                        const double mass_imbalance = (phi_v[i + 1] - phi_v[i]) + (rho_v[i] - rho_v_old[i]) * dz / dt;  // [kg/(m2s)]
                        const double mass_flux = Gamma_v[i] * dz;         // [kg/(m2s)]
                        dVP[i] = +mass_flux - mass_imbalance;  /// [kg/(m2s)]

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

            // Apply Rhie and Chow correction OUTSIDE loops
            for (int i = 1; i < N; ++i) {
                const double avgInvbVU = 0.5 * (1.0 / bVU[i - 1] + 1.0 / bVU[i]);
                double rc = -avgInvbVU / 4.0 * (p_padded_v[i - 2] - 3 * p_padded_v[i - 1]
                    + 3 * p_padded_v[i] - p_padded_v[i + 1]);
                const double u_face = 0.5 * (u_v[i - 1] + u_v[i]) + rc;
                const double rho_face = (u_face >= 0.0) ? rho_v[i - 1] : rho_v[i];
                phi_v[i] = rho_face * u_face;
            }

            // Update density with new p,T
            for(int i = 0; i < N; ++i) rho_v[i] = std::max(1e-6, p_v[i] / (Rv * T_v_bulk[i]));

            #pragma endregion

            // =======================================================================
            //                          [VOF EQUATION]
            // =======================================================================

            #pragma region vof_equation

            for (int i = 1; i < N - 1; i++) {

                // --- Sound velocities c_k
                c_l[i] = 2660.7 - 0.37667 * T_x_bulk[i] - 9.0356e-5 * T_x_bulk[i] * T_x_bulk[i];
                c_v[i] = std::sqrt(Rv * gamma * T_v_bulk[i]);

                // --- Acoustic impedances Z_k
                const double Z_l = rho_x[i] * c_l[i];           
                const double Z_v = rho_v[i] * c_v[i];          
                const double Z_sum = Z_l + Z_v;

                // --- Interface velocity u_int
                const double dalpha_l_dx = (alpha_l[i + 1] - alpha_l[i - 1]) / (2.0 * dz);
                const double sgn_dalpha = (dalpha_l_dx > 0.0) ? 1.0 : ((dalpha_l_dx < 0.0) ? -1.0 : 0.0);
                const double u_int = (Z_l * u_x[i] + Z_v * u_v[i]) / Z_sum
                    + sgn_dalpha * (p_v[i] - p_x[i]) / Z_sum;

                // --- Interface pressure p_int
                const double p_int = (Z_l * p_v[i] + Z_v * p_x[i]) / Z_sum;

                // --- Interface thermodynamic state 
                const double rho_int = liquid_sodium::rho(T_x_v[i]);     
                const double h_l_int = liquid_sodium::h_l_linear(T_x_v[i]); 
                const double h_v_int = vapor_sodium::h_g_linear(T_x_v[i]); 
                const double E_l_int = h_l_int - p_int / rho_int + 0.5 * u_int * u_int;
                const double E_v_int = h_v_int - p_int / rho_int + 0.5 * u_int * u_int;

                // --- Interfacial area density a_int
                const double a_int = compute_a_int(alpha_l[i]);

                // --- Pressure relaxation rate Theta
                const double Theta = a_int / Z_sum;

                // --- Capillary pressure DPcap
                DPcap[i] = compute_Dpcap(alpha_l[i], T_x_v[i]);

                // Tridiagonal coefficients: a*alpha_{i-1} + b*alpha_i + c*alpha_{i+1} = d
                aXA[i] = -u_int / dz;                                 // west neighbor
                cXA[i] = u_int / dz;                               // east neighbor (upwind: no east contribution)
                bXA[i] = 1 / dt;                       // diagonal
                dXA[i] = 
                    + alpha_l_old[i] / dt 
                    + a_int / Z_sum * (p_x[i] + DPcap[i] - p_v[i])
                    - Gamma_v[i] * a_int / rho_int;            // RHS
            }

            aXA[0] = 1.0;
            bXA[0] = -1.0;
            cXA[0] = 0.0;
            dXA[0] = 0.0;

            aXA[N - 1] = 0.0;
            bXA[N - 1] = -1.0;
            cXA[N - 1] = 1.0;
            dXA[N - 1] = 0.0;

            tdma_solver.solve(aXA, bXA, cXA, dXA, alpha_l);

            for (int i = 0; i < N; ++i) alpha_v[i] = 1 - alpha_l[i];


            bool alpha_valid = true;

            for (int i = 0; i < N; ++i) {

                if (alpha_l[i] < 0.0 && alpha_l[i] > -1e-3) {

                    alpha_l[i] = 0.0;
                    alpha_v[i] = 1.0;

                }
                else if (alpha_v[i] < 0.0 && alpha_v[i] > -1e-3) {

                    alpha_v[i] = 0.0;
                    alpha_l[i] = 0.0;
                }
                else if (alpha_l[i] < -1e-3 || alpha_v[i] < -1e-3) {

                    alpha_valid = false;

                }
            }

            if (alpha_valid == false) {

                pic = max_picard;
                break;
            }

            #pragma endregion

            // =======================================================================
            //                             [INTERFACES]
            // =======================================================================

            #pragma region interfaces 

            for (int i = 1; i < N - 1; ++i) {

                // Physical properties
                Re_v[i] = rho_v[i] * std::abs(u_v[i]) * Dh_v / mu_v[i];         // Reynolds number [-]
                const double Pr_v = cp_v[i] * mu_v[i] / k_v[i];              // Prandtl number [-]
                HTC[i] = vapor_sodium::h_conv(Re_v[i], Pr_v, k_v[i], Dh_v);     // Convective HTC at the vapor-liquid interface [W/(m2K)]

                // Enthalpies
                if (Gamma_v[i] > 0.0) {                          // Evaporation case

                    h_v_phase = vapor_sodium::h_g_linear(T_x_v_iter[i]);
                    h_x_phase = liquid_sodium::h_l_linear(T_x_v_iter[i]);

                }
                else {                                                  // Condensation case

                    h_v_phase = vapor_sodium::h_g_linear(T_x_v_iter[i]);
                    h_x_phase = liquid_sodium::h_l_linear(T_x_v_iter[i])
                        + (vapor_sodium::h_g_linear(T_v_bulk[i]) - vapor_sodium::h_g_linear(T_x_v_iter[i]));
                }

                const double enthalpy_difference = (h_v_phase - h_x_phase);

                // Useful constants
                const double E3 = HTC[i];
                const double E4 = -k_x[i] + HTC[i] * r_v;
                const double E5 = -2.0 * r_v * k_x[i] + HTC[i] * r_v * r_v;
                const double E6 = HTC[i] * T_v_bulk[i] - enthalpy_difference * phi_x_v[i];

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

                // ABC[6 * i + 0] = a_w
                // ABC[6 * i + 1] = b_w
                // ABC[6 * i + 2] = c_w
                // ABC[6 * i + 3] = a_x
                // ABC[6 * i + 4] = b_x
                // ABC[6 * i + 5] = c_x

                // Update temperatures at the interfaces
                T_o_w[i] = ABC[6 * i + 0] + ABC[6 * i + 1] * r_o + ABC[6 * i + 2] * r_o * r_o; // Temperature at the outer wall [K]
                T_w_x[i] = ABC[6 * i + 0] + ABC[6 * i + 1] * r_i + ABC[6 * i + 2] * r_i * r_i; // Temperature at the wall liquid interface [K]
                T_x_v[i] = ABC[6 * i + 3] + ABC[6 * i + 4] * r_v + ABC[6 * i + 5] * r_v * r_v; // Temperature at the liquid vapor interface [K]

                double conv = h_conv * (T_o_w[i] - T_env);       // [W/m2]
                double irr = emissivity * sigma * (std::pow(T_o_w[i], 4) - std::pow(T_env, 4));   // [W/m2]
               
                double sum_w = 0.0;
                int n_evap = 0;

                // Evaporator
                for (int i = 0; i < N; ++i) {
                    double w = 0.0;

                    if (mesh_center[i] >= evaporator_start - delta_h &&
                        mesh_center[i] <= evaporator_end + delta_h) {

                        ++n_evap;

                        if (mesh_center[i] >= evaporator_start - delta_h &&
                            mesh_center[i] < evaporator_start) {

                            double x = (mesh_center[i] - (evaporator_start - delta_h)) / delta_h;
                            w = 0.5 * (1.0 - std::cos(pi * x));
                        }
                        else if (mesh_center[i] >= evaporator_start &&
                            mesh_center[i] <= evaporator_end) {

                            w = 1.0;
                        }
                        else if (mesh_center[i] > evaporator_end &&
                            mesh_center[i] <= evaporator_end + delta_h) {

                            double x = (mesh_center[i] - evaporator_end) / delta_h;
                            w = 0.5 * (1.0 + std::cos(pi * x));
                        }
                    }

                    q_ow[i] = w;        // Evaporator weight
                    sum_w += w;
                }

                // Evaporator normalization 
                const double s = (sum_w > 0.0) ? q0 / sum_w : 0.0;
                for (int i = 0; i < N; ++i) {

                    q_ow[i] *= s;
                    Q_ow[i] = q_ow[i] * 2 * r_o * evaporator_length_eff / ((r_o * r_o - r_i * r_i) * dz);
                }


                // Evaporator power ramp
                for (int i = 0; i < N; ++i) {

                    q_ow[i] *= ramp;
                    Q_ow[i] *= ramp;
                }

                // Condenser
                for (int i = 0; i < N; ++i) {

                    if (mesh_center[i] >= condenser_start &&
                        mesh_center[i] < condenser_start + delta_c) {

                        double x = (mesh_center[i] - condenser_start) / delta_c;
                        q_ow[i] = -(conv + irr) * 0.5 * (1.0 - std::cos(pi * x));
                        Q_ow[i] = q_ow[i] * 2 * r_o / (r_o * r_o - r_i * r_i);
                    }
                    else if (mesh_center[i] >= condenser_start + delta_c) {
                        q_ow[i] = -(conv + irr);
                        Q_ow[i] = q_ow[i] * 2 * r_o / (r_o * r_o - r_i * r_i);
                    }
                }

                // Liquid to wall, heat source due to heat flux [W/m3]
                Q_xw[i] = -k_w[i] * (ABC[6 * i + 1] + 2.0 * ABC[6 * i + 2] * r_i) * 2 * r_i / (r_o * r_o - r_i * r_i);           

                // Wall to liquid, heat source due to heat flux [W/m3]
                Q_wx[i] = k_w[i] * (ABC[6 * i + 1] + 2.0 * ABC[6 * i + 2] * r_i) * 2 * r_i / (r_i * r_i - r_v * r_v);
                
                // Liquid to mixture, heat source due to heat flux [W/m3] 
                Q_xm[i] = HTC[i] * (ABC[6 * i + 3] + ABC[6 * i + 4] * r_v + ABC[6 * i + 5] * r_v * r_v - T_v_bulk[i]) * 2.0 / r_v;  
                
                // Mixture to liquid, heat source due to heat flux [W/m3] 
                Q_mx[i] = -k_x[i] * (ABC[6 * i + 4] + 2.0 * ABC[6 * i + 5] * r_v) * 2.0 * r_v / (r_i * r_i - r_v * r_v);

                // Volumetric heat source [W/m3] due to evaporation/condensation (to be summed to the vapor)
                Q_mass_vapor[i] = +Gamma_v[i] * h_v_phase; 

                // Volumetric heat source [W/m3] due to evaporation/condensation (to be summed to the liquid)
                Q_mass_liquid[i] = -Gamma_x[i] * h_x_phase;

                // Real evaporation mass flux [kg/(m2s)]
                phi_x_v[i] = eps_s * (sigma_e * vapor_sodium::P_sat(T_x_v[i]) / std::sqrt(T_x_v[i]) -
                    sigma_c * Omega * p_v[i] / std::sqrt(T_v_bulk[i])) /
                    std::sqrt(2 * pi * Rv);   

                // Volumetric mass source [kg/m3s] to vapor
                Gamma_v[i] = phi_x_v[i] * 2.0 / r_v;

                // Volumetric mass source [kg/m3s] to liquid
                Gamma_x[i] = phi_x_v[i] * (2.0 * r_v) / (r_i * r_i - r_v * r_v);

                heat_balance_surface[i] =
                    - k_x[i] * (ABC[6 * i + 4] + 2 * ABC[6 * i + 5] * r_v)
                    + HTC[i] * (ABC[6 * i + 3] + ABC[6 * i + 4] * r_v + ABC[6 * i + 5] * r_v * r_v - T_v_bulk[i])
                    + enthalpy_difference * phi_x_v[i];
            }

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

            if (pic_error[0] < pic_tolerance[0] &&
                pic_error[1] < pic_tolerance[1] &&
                pic_error[2] < pic_tolerance[2]) {

                halves = std::max(0, --halves);     // Double timestep if converged
                break;                              // Picard converged
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

            alpha_l_old = alpha_l;
            alpha_v_old = alpha_v;

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

            Q_mass_liquid_old = Q_mass_liquid;
            Q_mass_vapor_old = Q_mass_vapor;

            Gamma_x_old = Gamma_x;
            Gamma_v_old = Gamma_v;

            phi_x_old = phi_x;
            phi_v_old = phi_v;

            bXU_old = bXU;
            bVU_old = bVU;

            h_x_old = h_x;
            h_v_old = h_v;

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

            alpha_l = alpha_l_old;
            alpha_v = alpha_v_old;

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

            Q_mass_liquid = Q_mass_liquid_old;
            Q_mass_vapor = Q_mass_vapor_old;

            Gamma_x = Gamma_x_old;
            Gamma_v = Gamma_v_old;

            phi_x = phi_x_old;
            phi_v = phi_v_old;

            bXU = bXU_old;
            bVU = bVU_old;

            h_x = h_x_old;
            h_v = h_v_old;

            halves += 1;                        // Reduce time step if max Picard iterations reached
        }

        // =======================================================================
        //                               [OUTPUT]
        // =======================================================================

        #pragma region output

        if (time_total >= t_last_print + print_interval) {
            for (int i = 1; i < N - 1; ++i) {

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

                x_v_mass_flux_output << phi_x_v[i] * A_interface_cell << " ";

                Q_ow_output << Q_ow[i] * vol_wall_cell << " ";
                Q_wx_output << Q_wx[i] * vol_liquid_cell << " ";
                Q_xw_output << Q_xw[i] * vol_wall_cell << " ";
                Q_xm_output << Q_xm[i] * vol_vapor_cell << " ";
                Q_mx_output << Q_mx[i] * vol_liquid_cell << " ";

                Q_mass_vapor_output << Q_mass_vapor[i] * vol_vapor_cell << " ";
                Q_mass_liquid_output << Q_mass_liquid[i] * vol_liquid_cell << " ";

                saturation_pressure[i] = vapor_sodium::P_sat(T_x_v_iter[i]);            // Saturation pressure [Pa] 

                saturation_pressure_output << saturation_pressure[i] << " ";
                sonic_velocity_output << sonic_velocity[i] << " ";

                heat_balance_surface_output << heat_balance_surface[i] << " ";

                wall_liquid_heat_balance_output << wall_liquid_heat_balance[i] << " ";
                liquid_vapor_heat_balance_output << liquid_vapor_heat_balance[i] << " ";

                reynolds_output << Re_v[i] << " ";
                HTC_output << HTC[i] << " ";
                DPcap_output << p_v[i] - p_x[i] << " "; 

                alpha_l_output << alpha_l[i] << " ";
                alpha_v_output << alpha_v[i] << " ";
            }

            // Time between timesteps [ms]
            auto t_now = std::chrono::high_resolution_clock::now();
            double simulation_time = std::chrono::duration<double, std::milli>(t_now - t_start_timestep).count();

            // Time from the start of the simulation
            double clock_time = std::chrono::duration<double>(t_now - t_start_simulation).count();

            time_output << time_total << " ";
            dt_output << dt << " ";
            simulation_time_output << simulation_time << " ";
            clock_time_output << clock_time << " ";

            double total_heat_source_wall = 0.0;
            double total_heat_source_liquid = 0.0;
            double total_heat_source_vapor = 0.0;

            double total_mass_source_liquid = 0.0;
            double total_mass_source_vapor = 0.0;

            double global_heat_balance = 0.0;

            for (int i = 1; i < N - 1; ++i) {

                wall_liquid_heat_balance[i] = Q_wx[i] * (r_i * r_i - r_v * r_v) + Q_xw[i] * (r_o * r_o - r_i * r_i);
                liquid_vapor_heat_balance[i] = (Q_xm[i] + Q_mass_vapor[i]) * (r_v * r_v) + (Q_mx[i] + Q_mass_liquid[i]) * (r_i * r_i - r_v * r_v);

                global_heat_balance += Q_ow[i];
                total_heat_source_wall += Q_tot_w[i];
                total_heat_source_liquid += Q_tot_x[i];
                total_heat_source_vapor += Q_tot_v[i];

                total_mass_source_liquid += Gamma_x[i];
                total_mass_source_vapor += Gamma_v[i];
            }

            global_heat_balance *= vol_wall_cell;
            total_heat_source_wall *= vol_wall_cell;
            total_heat_source_liquid *= vol_liquid_cell;
            total_heat_source_vapor *= vol_vapor_cell;

            total_heat_source_wall_output << total_heat_source_wall << " ";
            total_heat_source_liquid_output << total_heat_source_liquid << " ";
            total_heat_source_vapor_output << total_heat_source_vapor << " ";

            total_mass_source_liquid *= vol_liquid_cell;
            total_mass_source_vapor *= vol_vapor_cell;

            total_mass_source_liquid_output << total_mass_source_liquid << " ";
            total_mass_source_vapor_output << total_mass_source_vapor << " ";

            global_heat_balance_output << global_heat_balance << " ";

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
            Q_mass_liquid_output << "\n";

            saturation_pressure_output << "\n";
            sonic_velocity_output << "\n";

            heat_balance_surface_output << "\n";

            wall_liquid_heat_balance_output << "\n";
            liquid_vapor_heat_balance_output << "\n";

            reynolds_output << "\n";
            HTC_output << "\n";
            DPcap_output << "\n";

            alpha_l_output << "\n";
            alpha_v_output << "\n";

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
            Q_mass_liquid_output.flush();

            saturation_pressure_output.flush();
            sonic_velocity_output.flush();

            time_output.flush();
            simulation_time_output.flush();
            dt_output.flush();
            clock_time_output.flush();

            total_heat_source_wall_output.flush();
            total_heat_source_liquid_output.flush();
            total_heat_source_vapor_output.flush();

            total_mass_source_liquid_output.flush();
            total_mass_source_vapor_output.flush();

            momentum_res_x_output.flush();
            continuity_res_x_output.flush();
            temperature_res_x_output.flush();

            momentum_res_v_output.flush();
            continuity_res_v_output.flush();
            temperature_res_v_output.flush();

            heat_balance_surface_output.flush();

            global_heat_balance_output.flush();

            wall_liquid_heat_balance_output.flush();
            liquid_vapor_heat_balance_output.flush();

            reynolds_output.flush();
            HTC_output.flush();
            DPcap_output.flush();

            alpha_l_output.flush();
            alpha_v_output.flush();

            t_last_print += print_interval;
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
    Q_mass_liquid_output.close();

    saturation_pressure_output.close();
    sonic_velocity_output.close();

    total_heat_source_wall_output.close();
    total_heat_source_liquid_output.close();
    total_heat_source_vapor_output.close();

    total_mass_source_liquid_output.close();
    total_mass_source_vapor_output.close();

    momentum_res_x_output.close();
    continuity_res_x_output.close();
    temperature_res_x_output.close();

    momentum_res_v_output.close();
    continuity_res_v_output.close();
    temperature_res_v_output.close();

    heat_balance_surface_output.close();

    global_heat_balance_output.close();

    wall_liquid_heat_balance_output.close();
    liquid_vapor_heat_balance_output.close();

    reynolds_output.close();
    HTC_output.close();
    DPcap_output.close();

    alpha_l_output.close();
    alpha_v_output.close();

    return 0;
}