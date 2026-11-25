// sigcore CLI: text protocol on stdin/stdout, used by sigvision/cpp_bridge.py.
//
//   sigcore fft      stdin: n, then n lines "re im"          stdout: n, then n lines "re im"
//   sigcore cfar     stdin: n train guard alpha, then n x    stdout: count, then indices
//   sigcore conv2d   stdin: h w kh kw, image (h*w), kernel   stdout: h w, then h*w values
//   sigcore bench <fft|cfar|conv2d> <repeats>   (same stdin as the kind)
//       stdout: kind=.. repeats=.. p50_us=.. p95_us=.. peak_rss_kb=.. checksum=..
//
// Errors go to stderr as "error: <message>" with exit status 2.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "sigcore.hpp"

namespace {

using sigcore::cplx;

constexpr long long kMaxCount = 1LL << 26;

template <class T>
T read_value(std::istream& in, const char* what) {
    T v{};
    if (!(in >> v)) {
        throw std::runtime_error(std::string("malformed input while reading ") + what);
    }
    return v;
}

std::size_t read_size(std::istream& in, const char* what) {
    const long long v = read_value<long long>(in, what);
    if (v < 0 || v > kMaxCount) {
        throw std::runtime_error(std::string("size out of range: ") + what);
    }
    return static_cast<std::size_t>(v);
}

struct FftInput {
    std::vector<cplx> a;
};

struct CfarInput {
    std::vector<double> x;
    long train = 0;
    long guard = 0;
    double alpha = 0.0;
};

struct ConvInput {
    std::size_t h = 0, w = 0, kh = 0, kw = 0;
    std::vector<double> img, ker;
};

FftInput read_fft(std::istream& in) {
    FftInput r;
    const std::size_t n = read_size(in, "fft length");
    r.a.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double re = read_value<double>(in, "fft real part");
        const double im = read_value<double>(in, "fft imaginary part");
        r.a.emplace_back(re, im);
    }
    return r;
}

CfarInput read_cfar(std::istream& in) {
    CfarInput r;
    const std::size_t n = read_size(in, "cfar length");
    r.train = read_value<long>(in, "cfar train");
    r.guard = read_value<long>(in, "cfar guard");
    r.alpha = read_value<double>(in, "cfar alpha");
    r.x.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        r.x.push_back(read_value<double>(in, "cfar sample"));
    }
    return r;
}

ConvInput read_conv(std::istream& in) {
    ConvInput r;
    r.h = read_size(in, "image height");
    r.w = read_size(in, "image width");
    r.kh = read_size(in, "kernel height");
    r.kw = read_size(in, "kernel width");
    if (r.h * r.w > static_cast<std::size_t>(kMaxCount) ||
        r.kh * r.kw > static_cast<std::size_t>(kMaxCount)) {
        throw std::runtime_error("buffer too large");
    }
    r.img.reserve(r.h * r.w);
    for (std::size_t i = 0; i < r.h * r.w; ++i) {
        r.img.push_back(read_value<double>(in, "image sample"));
    }
    r.ker.reserve(r.kh * r.kw);
    for (std::size_t i = 0; i < r.kh * r.kw; ++i) {
        r.ker.push_back(read_value<double>(in, "kernel sample"));
    }
    return r;
}

long peak_rss_kb() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmHWM:", 0) == 0) {
            std::istringstream iss(line.substr(6));
            long kb = -1;
            if (iss >> kb) {
                return kb;
            }
        }
    }
    return -1;  // not available on this platform
}

// Each runner executes the kernel once and returns a checksum of its output.
double run_fft(const FftInput& in, bool print) {
    std::vector<cplx> a = in.a;
    sigcore::fft_radix2(a);
    double checksum = 0.0;
    if (print) {
        std::cout << a.size() << "\n";
    }
    for (const cplx& v : a) {
        checksum += std::abs(v);
        if (print) {
            std::cout << v.real() << " " << v.imag() << "\n";
        }
    }
    return checksum;
}

double run_cfar(const CfarInput& in, bool print) {
    const std::vector<std::size_t> idx = sigcore::cfar_ca_detect(in.x, in.train, in.guard, in.alpha);
    double checksum = static_cast<double>(idx.size());
    if (print) {
        std::cout << idx.size() << "\n";
    }
    for (std::size_t i : idx) {
        checksum += static_cast<double>(i);
        if (print) {
            std::cout << i << "\n";
        }
    }
    return checksum;
}

double run_conv(const ConvInput& in, bool print) {
    const std::vector<double> out = sigcore::conv2d(in.img, in.h, in.w, in.ker, in.kh, in.kw);
    double checksum = 0.0;
    if (print) {
        std::cout << in.h << " " << in.w << "\n";
    }
    for (std::size_t y = 0; y < in.h; ++y) {
        for (std::size_t x = 0; x < in.w; ++x) {
            const double v = out[y * in.w + x];
            checksum += v;
            if (print) {
                std::cout << v << (x + 1 == in.w ? "\n" : " ");
            }
        }
    }
    return checksum;
}

int run_bench(const std::string& kind, const std::string& repeats_text) {
    long repeats = 0;
    try {
        std::size_t used = 0;
        repeats = std::stol(repeats_text, &used);
        if (used != repeats_text.size()) {
            throw std::invalid_argument("trailing characters");
        }
    } catch (const std::exception&) {
        throw std::runtime_error("repeats must be an integer");
    }
    if (repeats < 1 || repeats > 1000000) {
        throw std::runtime_error("repeats out of range (1..1000000)");
    }

    std::vector<double> samples_us;
    samples_us.reserve(static_cast<std::size_t>(repeats));
    double checksum = 0.0;
    auto measure = [&](auto&& fn) {
        fn();  // warm-up, not timed
        for (long r = 0; r < repeats; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            checksum = fn();
            const auto t1 = std::chrono::steady_clock::now();
            samples_us.push_back(
                static_cast<double>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) /
                1000.0);
        }
    };

    if (kind == "fft") {
        const FftInput in = read_fft(std::cin);
        measure([&] { return run_fft(in, false); });
    } else if (kind == "cfar") {
        const CfarInput in = read_cfar(std::cin);
        measure([&] { return run_cfar(in, false); });
    } else if (kind == "conv2d") {
        const ConvInput in = read_conv(std::cin);
        measure([&] { return run_conv(in, false); });
    } else {
        throw std::runtime_error("unknown bench kind: " + kind);
    }

    std::cout << std::setprecision(17) << "kind=" << kind << " repeats=" << repeats
              << " p50_us=" << sigcore::nearest_rank_percentile(samples_us, 50.0)
              << " p95_us=" << sigcore::nearest_rank_percentile(samples_us, 95.0)
              << " peak_rss_kb=" << peak_rss_kb() << " checksum=" << checksum << "\n";
    return 0;
}

int usage() {
    std::cerr << "usage: sigcore fft|cfar|conv2d  |  sigcore bench <fft|cfar|conv2d> <repeats>\n";
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            return usage();
        }
        const std::string cmd = argv[1];
        std::cout << std::setprecision(17);
        if (cmd == "bench") {
            if (argc != 4) {
                return usage();
            }
            return run_bench(argv[2], argv[3]);
        }
        if (argc != 2) {
            return usage();
        }
        if (cmd == "fft") {
            run_fft(read_fft(std::cin), true);
        } else if (cmd == "cfar") {
            run_cfar(read_cfar(std::cin), true);
        } else if (cmd == "conv2d") {
            run_conv(read_conv(std::cin), true);
        } else {
            return usage();
        }
        std::cout.flush();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
}
