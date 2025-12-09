#pragma once
#include <vector>

/**
 * @brief Solves a tridiagonal system of linear equations A*x = d using the Thomas Algorithm (TDMA).
 *
 * The method consists of two main phases: forward elimination and back substitution,
 * which is optimized for the sparse tridiagonal structure.
 *
 * @param a The sub-diagonal vector (size N, with a[0] being zero/unused).
 * @param b The main diagonal vector (size N). Must contain non-zero elements.
 * @param c The super-diagonal vector (size N, with c[N-1] being zero/unused).
 * @param d The right-hand side vector (size N).
 * @return std::vector<double> The solution vector 'x' (size N).
 * * @note This implementation assumes the system is diagonally dominant or otherwise
 * stable, as it does not include pivoting. The vectors 'a', 'b', 'c', and 'd' must
 * all have the same size N, corresponding to the size of the system.
 */
std::vector<double> solveTridiagonal(const std::vector<double>& a,
    const std::vector<double>& b,
    const std::vector<double>& c,
    const std::vector<double>& d) {

    int n = b.size();
    std::vector<double> c_star(n, 0.0);
    std::vector<double> d_star(n, 0.0);
    std::vector<double> x(n, 0.0);

    c_star[0] = c[0] / b[0];
    d_star[0] = d[0] / b[0];

    for (int i = 1; i < n; i++) {

        double m = b[i] - a[i] * c_star[i - 1];
        c_star[i] = c[i] / m;
        d_star[i] = (d[i] - a[i] * d_star[i - 1]) / m;
    }

    x[n - 1] = d_star[n - 1];

    for (int i = n - 2; i >= 0; i--)
        x[i] = d_star[i] - c_star[i] * x[i + 1];

    return x;
}