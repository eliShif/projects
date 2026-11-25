// sigcore: standalone C++17 core mirroring the Python reference in sigvision/.
//   - radix-2 FFT
//   - cell-averaging CFAR peak detection
//   - 2-D "same" convolution with zero padding
//   - nearest-rank percentile (used by the bench command)
// All functions throw std::invalid_argument on invalid input.
#pragma once

#include <complex>
#include <cstddef>
#include <vector>

namespace sigcore {

using cplx = std::complex<double>;

// In-place iterative radix-2 FFT. Size must be a power of two (>= 1).
void fft_radix2(std::vector<cplx>& a);

// Indices where x[i] > alpha * mean(training cells), x[i] > x[i-1], x[i] >= x[i+1].
// Window bounds are clipped to the array; cells with no training cells are skipped.
std::vector<std::size_t> cfar_ca_detect(const std::vector<double>& x, long train, long guard,
                                        double alpha);

// out[y,x] = sum_{i,j} K[i,j] * I[y + kh/2 - i, x + kw/2 - j]; zero outside the image.
// Kernel dimensions must be odd and non-zero; buffers are row-major.
std::vector<double> conv2d(const std::vector<double>& img, std::size_t h, std::size_t w,
                           const std::vector<double>& ker, std::size_t kh, std::size_t kw);

// Nearest-rank percentile, p in (0, 100]; values must be non-empty.
double nearest_rank_percentile(std::vector<double> values, double p);

}  // namespace sigcore
