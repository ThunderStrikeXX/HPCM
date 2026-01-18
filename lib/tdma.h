#pragma once
#include <vector>

namespace tdma {

    class Solver {
    public:
        Solver(int n) : n_(n), c_star_(n) {}

        void solve(
            const std::vector<double>& a,
            const std::vector<double>& b,
            const std::vector<double>& c,
            std::vector<double>& d,
            std::vector<double>& x
        );

    private:
        int n_;
        std::vector<double> c_star_;
    };

}