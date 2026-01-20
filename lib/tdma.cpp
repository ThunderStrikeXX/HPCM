#include "tdma.h"
#include <stdexcept>

namespace tdma {

    void Solver::solve(
        const std::vector<double>& a,
        const std::vector<double>& b,
        const std::vector<double>& c,
        const std::vector<double>& d,
        std::vector<double>& x) {
        const int n = n_;
        if (a.size() != n || b.size() != n || c.size() != n || d.size() != n || x.size() != n)
            throw std::runtime_error("TDMA: size mismatch");

        // Forward sweep (no modification of d)
        double invb = 1.0 / b[0];
        c_star_[0] = c[0] * invb;
        d_star_[0] = d[0] * invb;

        for (int i = 1; i < n; ++i) {
            const double invm = 1.0 / (b[i] - a[i] * c_star_[i - 1]);
            c_star_[i] = c[i] * invm;
            d_star_[i] = (d[i] - a[i] * d_star_[i - 1]) * invm;
        }

        // Back substitution
        x[n - 1] = d_star_[n - 1];
        for (int i = n - 2; i >= 0; --i)
            x[i] = d_star_[i] - c_star_[i] * x[i + 1];
    }

}
