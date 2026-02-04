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

#include "numeric_types.h"

// Wall
data_type new_dt_w(
    data_type dt_old,
    const std::vector<data_type>& T,
    const std::vector<data_type>& St
);

// Wick
data_type new_dt_x(
    data_type dt_old,
    const std::vector<data_type>& u,
    const std::vector<data_type>& T,
    const std::vector<data_type>& Sm,
    const std::vector<data_type>& Qf
);

// Vapor
data_type new_dt_v(
    data_type dz,
    data_type dt_old,
    const std::vector<data_type>& u,
    const std::vector<data_type>& T,
    const std::vector<data_type>& rho,
    const std::vector<data_type>& Sm,
    const std::vector<data_type>& Qf,
    const std::vector<data_type>& bVU
);