#pragma once
#include <vector>
#include <cstddef>

using data_type = float;

namespace tdma {

    class Solver {
    public:
        explicit Solver(std::size_t n)
            : c_star_(n), d_star_(n), n_(n) {
        }

        void solve(
            const std::vector<data_type>& a,
            const std::vector<data_type>& b,
            const std::vector<data_type>& c,
            const std::vector<data_type>& d,
            std::vector<data_type>& x);

    private:
        std::vector<data_type> c_star_;
        std::vector<data_type> d_star_;
        std::size_t n_;
    };

}