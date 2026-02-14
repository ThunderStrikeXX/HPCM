#pragma once
#include <vector>
#include <cstddef>

namespace tdma {

    class Solver {
    public:
        explicit Solver(std::size_t n)
            : c_star_(n), d_star_(n), n_(n) {
        }

        void solve(
            const std::vector<double>& a,
            const std::vector<double>& b,
            const std::vector<double>& c,
            const std::vector<double>& d,
            std::vector<double>& x);

    private:
        std::vector<double> c_star_;
        std::vector<double> d_star_;
        std::size_t n_;
    };

}