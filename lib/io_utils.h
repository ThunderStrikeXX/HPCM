#pragma once

#include <vector>
#include <string>

// Lettura righe da file
std::vector<double> read_second_last_row(const std::string& filename, int N);
std::vector<double> read_last_row(const std::string& filename, int N);
double read_last_value(const std::string& filename);

// Selezione caso da directory
std::string select_case();