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

#include "steel.h"
#include "liquid_sodium.h"
#include "vapor_sodium.h"

// Wall
double new_dt_w(
    double dt_old,
    const std::vector<double>& T,
    const std::vector<double>& St
);

// Wick
double new_dt_x(
    double dt_old,
    const std::vector<double>& u,
    const std::vector<double>& T,
    const std::vector<double>& Sm,
    const std::vector<double>& Qf
);

// Vapor
double new_dt_v(
    double dz,
    double dt_old,
    const std::vector<double>& u,
    const std::vector<double>& T,
    const std::vector<double>& rho,
    const std::vector<double>& Sm,
    const std::vector<double>& Qf,
    const std::vector<double>& bVU
);