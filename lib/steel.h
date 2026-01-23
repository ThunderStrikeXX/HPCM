#pragma once
#include <array>
#include <cstddef>

using data_type = double;

namespace steel {

    /// Temperature values of the Cp table [K]
    constexpr std::array<data_type, 15> T = { {
        300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700
    } };

    /// Specific heat values of the Cp table [J kg^-1 K^-1]
    constexpr std::array<data_type, 15> Cp_J_kgK = { {
        510.0296,523.4184,536.8072,550.1960,564.0032,
        577.3920,590.7808,604.1696,617.5584,631.3656,
        644.7544,658.1432,671.5320,685.3392,698.7280
    } };

    inline data_type cp(data_type Tquery) {
        if (Tquery <= T.front()) return Cp_J_kgK.front();
        if (Tquery >= T.back())  return Cp_J_kgK.back();

        int i = static_cast<int>((Tquery - 300.0) / 100.0);
        if (i < 0) i = 0;

        int iMax = static_cast<int>(T.size()) - 2;
        if (i > iMax) i = iMax;

        data_type x0 = 300.0 + 100.0 * i, x1 = x0 + 100.0;
        data_type y0 = Cp_J_kgK[static_cast<std::size_t>(i)];
        data_type y1 = Cp_J_kgK[static_cast<std::size_t>(i + 1)];
        data_type t = (Tquery - x0) / (x1 - x0);

        return y0 + t * (y1 - y0);
    }

    inline data_type rho(data_type Tquery) {
        return (7.9841 - 2.6560e-4 * Tquery - 1.158e-7 * Tquery * Tquery) * 1e3;
    }

    inline data_type k(data_type Tquery) {
        return (8.116e-2 + 1.618e-4 * Tquery) * 100.0;
    }
}