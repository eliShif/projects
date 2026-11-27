// Classical image/signal kernels with a text protocol, used to cross-check the
// NumPy reference in portfolio/classical_vision.py.
//
// Usage:  classical_filters <command> [--repeat N]   (input on stdin, result on stdout)
//
// Commands and input tokens (whitespace separated, decimal numbers):
//   conv2d  R C KR KC <R*C image values> <KR*KC kernel values>
//           same-size zero-padded cross-correlation; kernel dimensions must be odd
//   sobel   R C <R*C image values>         Sobel gradient magnitude, zero padded
//   fir     N M <N signal values> <M taps> causal FIR: y[n] = sum_k h[k] x[n-k]
//   fft     N <N real values>              N a power of two; outputs N/2+1 (re im) pairs
//   stft    N F H <N values>               F a power of two; periodic Hann window,
//                                          hop H; outputs frames x (F/2+1) magnitudes
//
// Output: the result tokens, then "timing_ns <total> <repeats>". Errors go to
// stderr with exit code 2. The timing covers only the compute loop (no parsing).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct InputError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

constexpr std::int64_t kMaxElements = 1LL << 24;
constexpr std::int64_t kMaxRepeat = 1000000;
constexpr double kPi = 3.14159265358979323846;

class Reader {
 public:
  explicit Reader(std::istream& in) : in_(in) {}

  std::int64_t integer(const std::string& name) {
    const std::string token = next(name);
    std::size_t used = 0;
    long long value = 0;
    try {
      value = std::stoll(token, &used);
    } catch (const std::exception&) {
      throw InputError("invalid integer for " + name);
    }
    if (used != token.size()) throw InputError("invalid integer for " + name);
    return value;
  }

  double real(const std::string& name) {
    const std::string token = next(name);
    std::size_t used = 0;
    double value = 0.0;
    try {
      value = std::stod(token, &used);
    } catch (const std::exception&) {
      throw InputError("invalid number in " + name);
    }
    if (used != token.size() || !std::isfinite(value)) throw InputError("invalid number in " + name);
    return value;
  }

  std::vector<double> values(std::int64_t count, const std::string& name) {
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(count));
    for (std::int64_t i = 0; i < count; ++i) out.push_back(real(name));
    return out;
  }

  void expect_end() {
    std::string extra;
    if (in_ >> extra) throw InputError("unexpected trailing input");
  }

 private:
  std::string next(const std::string& name) {
    std::string token;
    if (!(in_ >> token)) throw InputError("truncated input while reading " + name);
    return token;
  }

  std::istream& in_;
};

std::int64_t checked_count(std::int64_t a, std::int64_t b, const std::string& what) {
  if (a <= 0 || b <= 0) throw InputError(what + " dimensions must be positive");
  if (a > kMaxElements / b) throw InputError(what + " is too large");
  return a * b;
}

bool power_of_two(std::int64_t n) { return n > 0 && (n & (n - 1)) == 0; }

template <typename F>
std::int64_t timed(std::int64_t repeat, F&& body) {
  const auto start = std::chrono::steady_clock::now();
  for (std::int64_t i = 0; i < repeat; ++i) body();
  const auto stop = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
}

std::vector<double> correlate2d(const std::vector<double>& image, std::int64_t rows, std::int64_t cols,
                                const std::vector<double>& kernel, std::int64_t krows, std::int64_t kcols) {
  std::vector<double> out(static_cast<std::size_t>(rows * cols), 0.0);
  const std::int64_t hr = krows / 2;
  const std::int64_t hc = kcols / 2;
  for (std::int64_t i = 0; i < rows; ++i) {
    for (std::int64_t j = 0; j < cols; ++j) {
      double acc = 0.0;
      for (std::int64_t a = 0; a < krows; ++a) {
        const std::int64_t ii = i + a - hr;
        if (ii < 0 || ii >= rows) continue;
        for (std::int64_t b = 0; b < kcols; ++b) {
          const std::int64_t jj = j + b - hc;
          if (jj < 0 || jj >= cols) continue;
          acc += image[static_cast<std::size_t>(ii * cols + jj)] * kernel[static_cast<std::size_t>(a * kcols + b)];
        }
      }
      out[static_cast<std::size_t>(i * cols + j)] = acc;
    }
  }
  return out;
}

std::vector<double> sobel_magnitude(const std::vector<double>& image, std::int64_t rows, std::int64_t cols) {
  static const std::vector<double> gx{-1, 0, 1, -2, 0, 2, -1, 0, 1};
  static const std::vector<double> gy{-1, -2, -1, 0, 0, 0, 1, 2, 1};
  std::vector<double> horizontal = correlate2d(image, rows, cols, gx, 3, 3);
  const std::vector<double> vertical = correlate2d(image, rows, cols, gy, 3, 3);
  for (std::size_t i = 0; i < horizontal.size(); ++i) horizontal[i] = std::hypot(horizontal[i], vertical[i]);
  return horizontal;
}

std::vector<double> fir(const std::vector<double>& signal, const std::vector<double>& taps) {
  std::vector<double> out(signal.size(), 0.0);
  for (std::size_t n = 0; n < signal.size(); ++n) {
    double acc = 0.0;
    const std::size_t limit = std::min(taps.size(), n + 1);
    for (std::size_t k = 0; k < limit; ++k) acc += taps[k] * signal[n - k];
    out[n] = acc;
  }
  return out;
}

using Complex = std::complex<double>;

void fft_in_place(std::vector<Complex>& a) {
  const std::size_t n = a.size();
  for (std::size_t i = 1, j = 0; i < n; ++i) {
    std::size_t bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(a[i], a[j]);
  }
  for (std::size_t len = 2; len <= n; len <<= 1) {
    const double angle = -2.0 * kPi / static_cast<double>(len);
    for (std::size_t i = 0; i < n; i += len) {
      for (std::size_t k = 0; k < len / 2; ++k) {
        const Complex w = std::polar(1.0, angle * static_cast<double>(k));
        const Complex u = a[i + k];
        const Complex v = a[i + k + len / 2] * w;
        a[i + k] = u + v;
        a[i + k + len / 2] = u - v;
      }
    }
  }
}

std::vector<Complex> rfft(const std::vector<double>& signal) {
  std::vector<Complex> buffer(signal.begin(), signal.end());
  fft_in_place(buffer);
  buffer.resize(signal.size() / 2 + 1);
  return buffer;
}

std::vector<double> stft_magnitude(const std::vector<double>& signal, std::size_t frame, std::size_t hop) {
  const std::size_t frames = 1 + (signal.size() - frame) / hop;
  const std::size_t bins = frame / 2 + 1;
  std::vector<double> window(frame);
  for (std::size_t i = 0; i < frame; ++i)
    window[i] = 0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(i) / static_cast<double>(frame));
  std::vector<double> out(frames * bins);
  std::vector<double> chunk(frame);
  for (std::size_t t = 0; t < frames; ++t) {
    for (std::size_t i = 0; i < frame; ++i) chunk[i] = signal[t * hop + i] * window[i];
    const std::vector<Complex> spectrum = rfft(chunk);
    for (std::size_t k = 0; k < bins; ++k) out[t * bins + k] = std::abs(spectrum[k]);
  }
  return out;
}

void print_timing(std::int64_t total_ns, std::int64_t repeat) {
  std::cout << "timing_ns " << total_ns << ' ' << repeat << '\n';
}

void print_values(const std::vector<double>& values, std::int64_t columns) {
  std::int64_t in_line = 0;
  for (const double v : values) {
    std::cout << v;
    if (++in_line == columns) {
      std::cout << '\n';
      in_line = 0;
    } else {
      std::cout << ' ';
    }
  }
  if (in_line != 0) std::cout << '\n';
}

int run(const std::string& command, std::int64_t repeat) {
  Reader reader(std::cin);
  std::cout << std::setprecision(17);
  if (command == "conv2d") {
    const std::int64_t rows = reader.integer("rows");
    const std::int64_t cols = reader.integer("cols");
    const std::int64_t krows = reader.integer("kernel rows");
    const std::int64_t kcols = reader.integer("kernel cols");
    const std::int64_t count = checked_count(rows, cols, "image");
    const std::int64_t kcount = checked_count(krows, kcols, "kernel");
    if (krows % 2 == 0 || kcols % 2 == 0) throw InputError("kernel dimensions must be odd");
    const std::vector<double> image = reader.values(count, "image");
    const std::vector<double> kernel = reader.values(kcount, "kernel");
    reader.expect_end();
    std::vector<double> out;
    const std::int64_t ns = timed(repeat, [&] { out = correlate2d(image, rows, cols, kernel, krows, kcols); });
    std::cout << rows << ' ' << cols << '\n';
    print_values(out, cols);
    print_timing(ns, repeat);
  } else if (command == "sobel") {
    const std::int64_t rows = reader.integer("rows");
    const std::int64_t cols = reader.integer("cols");
    const std::int64_t count = checked_count(rows, cols, "image");
    const std::vector<double> image = reader.values(count, "image");
    reader.expect_end();
    std::vector<double> out;
    const std::int64_t ns = timed(repeat, [&] { out = sobel_magnitude(image, rows, cols); });
    std::cout << rows << ' ' << cols << '\n';
    print_values(out, cols);
    print_timing(ns, repeat);
  } else if (command == "fir") {
    const std::int64_t n = reader.integer("signal length");
    const std::int64_t m = reader.integer("tap count");
    if (n <= 0 || m <= 0) throw InputError("signal and taps must be nonempty");
    if (n > kMaxElements || m > kMaxElements) throw InputError("input is too large");
    const std::vector<double> signal = reader.values(n, "signal");
    const std::vector<double> taps = reader.values(m, "taps");
    reader.expect_end();
    std::vector<double> out;
    const std::int64_t ns = timed(repeat, [&] { out = fir(signal, taps); });
    std::cout << n << '\n';
    print_values(out, 8);
    print_timing(ns, repeat);
  } else if (command == "fft") {
    const std::int64_t n = reader.integer("signal length");
    if (!power_of_two(n)) throw InputError("fft length must be a positive power of two");
    if (n > kMaxElements) throw InputError("input is too large");
    const std::vector<double> signal = reader.values(n, "signal");
    reader.expect_end();
    std::vector<Complex> out;
    const std::int64_t ns = timed(repeat, [&] { out = rfft(signal); });
    std::cout << out.size() << '\n';
    for (const Complex& c : out) std::cout << c.real() << ' ' << c.imag() << '\n';
    print_timing(ns, repeat);
  } else if (command == "stft") {
    const std::int64_t n = reader.integer("signal length");
    const std::int64_t frame = reader.integer("frame length");
    const std::int64_t hop = reader.integer("hop");
    if (n <= 0 || n > kMaxElements) throw InputError("signal length out of range");
    if (!power_of_two(frame)) throw InputError("frame length must be a positive power of two");
    if (hop <= 0) throw InputError("hop must be positive");
    if (frame > n) throw InputError("frame length exceeds signal length");
    const std::vector<double> signal = reader.values(n, "signal");
    reader.expect_end();
    std::vector<double> out;
    const std::int64_t ns = timed(repeat, [&] {
      out = stft_magnitude(signal, static_cast<std::size_t>(frame), static_cast<std::size_t>(hop));
    });
    const std::int64_t bins = frame / 2 + 1;
    std::cout << out.size() / static_cast<std::size_t>(bins) << ' ' << bins << '\n';
    print_values(out, bins);
    print_timing(ns, repeat);
  } else {
    throw InputError("unknown command: " + command);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2 && argc != 4) throw InputError("usage: classical_filters <command> [--repeat N]");
    const std::string command = argv[1];
    std::int64_t repeat = 1;
    if (argc == 4) {
      if (std::string(argv[2]) != "--repeat") throw InputError("usage: classical_filters <command> [--repeat N]");
      std::size_t used = 0;
      const std::string text = argv[3];
      try {
        repeat = std::stoll(text, &used);
      } catch (const std::exception&) {
        throw InputError("invalid repeat count");
      }
      if (used != text.size() || repeat < 1 || repeat > kMaxRepeat) throw InputError("repeat must be in [1, 1000000]");
    }
    return run(command, repeat);
  } catch (const InputError& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "internal error: " << error.what() << '\n';
    return 3;
  }
}
