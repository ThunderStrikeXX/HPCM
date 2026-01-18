#include "tdma.h"
#include <stdexcept>

namespace tdma {

    void Solver::solve(
        const std::vector<double>& a,
        const std::vector<double>& b,
        const std::vector<double>& c,
        std::vector<double>& d,
        std::vector<double>& x) {
        const int n = n_;
        if (a.size() != n || b.size() != n || c.size() != n || d.size() != n || x.size() != n)
            throw std::runtime_error("TDMA: size mismatch");

        c_star_[0] = c[0] / b[0];
        d[0] = d[0] / b[0];

        for (int i = 1; i < n; ++i) {
            const double m = b[i] - a[i] * c_star_[i - 1];
            const double invm = 1.0 / m;
            c_star_[i] = c[i] * invm;
            d[i] = (d[i] - a[i] * d[i - 1]) * invm;
        }

        x[n - 1] = d[n - 1];
        for (int i = n - 2; i >= 0; --i)
            x[i] = d[i] - c_star_[i] * x[i + 1];
    }
}