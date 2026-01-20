#pragma once
#include <vector>

namespace tdma {

    class Solver {
    public:
        explicit Solver(int n)
            : n_(n), c_star_(n), d_star_(n) {
        }

        void solve(
            const std::vector<double>& a,
            const std::vector<double>& b,
            const std::vector<double>& c,
            const std::vector<double>& d,
            std::vector<double>& x);

    private:
        int n_;
        std::vector<double> c_star_;
        std::vector<double> d_star_;
    };

}
