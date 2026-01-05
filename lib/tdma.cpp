#include "tdma.h"

#include <stdexcept>

namespace tdma {

std::vector<double> solve(
    const std::vector<double>& a,
    const std::vector<double>& b,
    const std::vector<double>& c,
    const std::vector<double>& d)
{
    const int n = b.size();
    if (a.size()!=n || c.size()!=n || d.size()!=n)
        throw std::runtime_error("TDMA: size mismatch");

    std::vector<double> c_star(n), d_star(n), x(n);

    c_star[0] = c[0] / b[0];
    d_star[0] = d[0] / b[0];

    for (int i = 1; i < n; ++i) {
        const double m = b[i] - a[i] * c_star[i - 1];
        c_star[i] = c[i] / m;
        d_star[i] = (d[i] - a[i] * d_star[i - 1]) / m;
    }

    x[n - 1] = d_star[n - 1];
    for (int i = n - 2; i >= 0; --i)
        x[i] = d_star[i] - c_star[i] * x[i + 1];

    return x;
}

}