#include "sigcore.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace sigcore {

void fft_radix2(std::vector<cplx>& a) {
    const std::size_t n = a.size();
    if (n == 0 || (n & (n - 1)) != 0) {
        throw std::invalid_argument("fft length must be a power of two");
    }
    // Bit-reversal permutation.
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }
    const double pi = std::acos(-1.0);
    std::vector<cplx> tw;
    for (std::size_t size = 2; size <= n; size <<= 1) {
        const std::size_t half = size >> 1;
        tw.resize(half);
        for (std::size_t k = 0; k < half; ++k) {
            tw[k] = std::polar(1.0, -2.0 * pi * static_cast<double>(k) / static_cast<double>(size));
        }
        for (std::size_t start = 0; start < n; start += size) {
            for (std::size_t k = 0; k < half; ++k) {
                const cplx even = a[start + k];
                const cplx odd = a[start + k + half] * tw[k];
                a[start + k] = even + odd;
                a[start + k + half] = even - odd;
            }
        }
    }
}

std::vector<std::size_t> cfar_ca_detect(const std::vector<double>& x, long train, long guard,
                                        double alpha) {
    if (train < 1) {
        throw std::invalid_argument("train must be >= 1");
    }
    if (guard < 0) {
        throw std::invalid_argument("guard must be >= 0");
    }
    if (!(alpha > 0.0) || !std::isfinite(alpha)) {
        throw std::invalid_argument("alpha must be a positive finite number");
    }
    std::vector<std::size_t> out;
    const long n = static_cast<long>(x.size());
    if (n == 0) {
        return out;
    }
    std::vector<double> cs(static_cast<std::size_t>(n) + 1, 0.0);
    for (long i = 0; i < n; ++i) {
        cs[i + 1] = cs[i] + x[i];
    }
    auto clamp = [n](long v) { return v < 0 ? 0L : (v > n ? n : v); };
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (long i = 0; i < n; ++i) {
        const long lead_lo = clamp(i - guard - train);
        const long lead_hi = clamp(i - guard);
        const long lag_lo = clamp(i + guard + 1);
        const long lag_hi = clamp(i + guard + train + 1);
        const long count = (lead_hi - lead_lo) + (lag_hi - lag_lo);
        if (count == 0) {
            continue;
        }
        const double sum = (cs[lead_hi] - cs[lead_lo]) + (cs[lag_hi] - cs[lag_lo]);
        const double noise = sum / static_cast<double>(count);
        const double thr = alpha * noise;
        const double left = i > 0 ? x[i - 1] : neg_inf;
        const double right = i < n - 1 ? x[i + 1] : neg_inf;
        if (x[i] > thr && x[i] > left && x[i] >= right) {
            out.push_back(static_cast<std::size_t>(i));
        }
    }
    return out;
}

std::vector<double> conv2d(const std::vector<double>& img, std::size_t h, std::size_t w,
                           const std::vector<double>& ker, std::size_t kh, std::size_t kw) {
    if (h == 0 || w == 0 || img.size() != h * w) {
        throw std::invalid_argument("image buffer does not match h*w");
    }
    if (kh == 0 || kw == 0 || kh % 2 == 0 || kw % 2 == 0 || ker.size() != kh * kw) {
        throw std::invalid_argument("kernel must be odd-sized and match kh*kw");
    }
    const long ci = static_cast<long>(kh / 2);
    const long cj = static_cast<long>(kw / 2);
    const long H = static_cast<long>(h);
    const long W = static_cast<long>(w);
    std::vector<double> out(h * w, 0.0);
    for (long y = 0; y < H; ++y) {
        for (long x = 0; x < W; ++x) {
            double acc = 0.0;
            for (long a = 0; a < static_cast<long>(kh); ++a) {
                const long yy = y + a - ci;
                if (yy < 0 || yy >= H) {
                    continue;
                }
                for (long b = 0; b < static_cast<long>(kw); ++b) {
                    const long xx = x + b - cj;
                    if (xx < 0 || xx >= W) {
                        continue;
                    }
                    // Flipped kernel: ker[(kh-1-a), (kw-1-b)].
                    const double k = ker[(kh - 1 - static_cast<std::size_t>(a)) * kw +
                                         (kw - 1 - static_cast<std::size_t>(b))];
                    acc += k * img[static_cast<std::size_t>(yy) * w + static_cast<std::size_t>(xx)];
                }
            }
            out[static_cast<std::size_t>(y) * w + static_cast<std::size_t>(x)] = acc;
        }
    }
    return out;
}

double nearest_rank_percentile(std::vector<double> values, double p) {
    if (values.empty()) {
        throw std::invalid_argument("no values");
    }
    if (!(p > 0.0 && p <= 100.0)) {
        throw std::invalid_argument("p must lie in (0, 100]");
    }
    std::sort(values.begin(), values.end());
    double rank = std::ceil(p / 100.0 * static_cast<double>(values.size()));
    if (rank < 1.0) {
        rank = 1.0;
    }
    return values[static_cast<std::size_t>(rank) - 1];
}

}  // namespace sigcore
