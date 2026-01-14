#include "io_utils.h"

#include <fstream>
#include <filesystem>
#include <iostream>

// -----------------------------------------------------------------------------
// Legge la penultima riga
std::vector<double> read_second_last_row(const std::string& filename, int N) {

    std::ifstream f(filename);
    std::string line, prev, last;

    while (std::getline(f, line)) {
        prev = last;
        last = line;
    }

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

    if (out.size() != static_cast<size_t>(N)) out.resize(N, 0.0);
    return out;
}

// -----------------------------------------------------------------------------
// Legge l'ultima riga
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

    if (out.size() != static_cast<size_t>(N)) out.resize(N, 0.0);
    return out;
}

// -----------------------------------------------------------------------------
// Legge l'ultimo valore dell'ultima riga
double read_last_value(const std::string& filename) {

    std::ifstream f(filename);
    std::string line, last;

    while (std::getline(f, line)) last = line;
    if (last.empty()) return 0.0;

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
    if (!token.empty()) last_value = std::stod(token);

    return last_value;
}

// -----------------------------------------------------------------------------
// Selezione del case
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
