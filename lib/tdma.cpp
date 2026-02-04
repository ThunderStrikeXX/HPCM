#include "tdma.h"
#include <stdexcept>

#include "numeric_types.h"

namespace tdma {

    void Solver::solve(
        const std::vector<data_type>& a,
        const std::vector<data_type>& b,
        const std::vector<data_type>& c,
        const std::vector<data_type>& d,
        std::vector<data_type>& x) {
        const std::size_t n = n_;

        // Forward sweep (no modification of d)
        data_type invb = 1.0 / b[0];
        c_star_[0] = c[0] * invb;
        d_star_[0] = d[0] * invb;

        for (std::size_t i = 1; i < n; ++i) {
            const data_type invm = 1.0 / (b[i] - a[i] * c_star_[i - 1]);
            c_star_[i] = c[i] * invm;
            d_star_[i] = (d[i] - a[i] * d_star_[i - 1]) * invm;
        }

        // Back substitution
        x[n - 1] = d_star_[n - 1];
        for (std::size_t i = n - 1; i-- > 0; ) {
            x[i] = d_star_[i] - c_star_[i] * x[i + 1];
        }
    }

}
