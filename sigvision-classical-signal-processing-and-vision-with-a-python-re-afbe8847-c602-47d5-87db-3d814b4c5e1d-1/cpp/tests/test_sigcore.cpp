// Dependency-free unit tests for sigcore. Exit status 0 = all passed.

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "sigcore.hpp"

using sigcore::cplx;

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        ++g_checks;                                                          \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

#define CHECK_THROWS(expr)                                                   \
    do {                                                                     \
        ++g_checks;                                                          \
        bool threw = false;                                                  \
        try {                                                                \
            expr;                                                            \
        } catch (const std::invalid_argument&) {                             \
            threw = true;                                                    \
        }                                                                    \
        if (!threw) {                                                        \
            std::fprintf(stderr, "FAIL %s:%d: expected invalid_argument: %s\n", __FILE__, \
                         __LINE__, #expr);                                   \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

static std::vector<cplx> naive_dft(const std::vector<cplx>& x) {
    const std::size_t n = x.size();
    const double pi = std::acos(-1.0);
    std::vector<cplx> out(n);
    for (std::size_t k = 0; k < n; ++k) {
        cplx acc(0.0, 0.0);
        for (std::size_t t = 0; t < n; ++t) {
            const double ang = -2.0 * pi * static_cast<double>(k * t % n) / static_cast<double>(n);
            acc += x[t] * cplx(std::cos(ang), std::sin(ang));
        }
        out[k] = acc;
    }
    return out;
}

static void test_fft_impulse() {
    std::vector<cplx> a(8, cplx(0.0, 0.0));
    a[0] = cplx(1.0, 0.0);
    sigcore::fft_radix2(a);
    for (const cplx& v : a) {
        CHECK(std::abs(v - cplx(1.0, 0.0)) < 1e-12);
    }
}

static void test_fft_matches_naive_dft() {
    std::vector<cplx> x;
    for (int i = 0; i < 16; ++i) {
        x.emplace_back(std::sin(0.7 * i) + 0.1 * i, std::cos(0.3 * i));
    }
    const std::vector<cplx> ref = naive_dft(x);
    std::vector<cplx> got = x;
    sigcore::fft_radix2(got);
    double max_err = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        max_err = std::fmax(max_err, std::abs(got[i] - ref[i]));
    }
    CHECK(max_err < 1e-9);
}

static void test_fft_single_sample_and_invalid_sizes() {
    std::vector<cplx> one(1, cplx(3.0, -2.0));
    sigcore::fft_radix2(one);
    CHECK(std::abs(one[0] - cplx(3.0, -2.0)) < 1e-15);
    for (std::size_t n : {0u, 3u, 6u, 12u}) {
        std::vector<cplx> bad(n, cplx(1.0, 0.0));
        CHECK_THROWS(sigcore::fft_radix2(bad));
    }
}

static std::vector<double> flat_with_spike(std::size_t n, std::size_t pos, double v) {
    std::vector<double> x(n, 1.0);
    x[pos] = v;
    return x;
}

static void test_cfar_spike_and_edges() {
    auto d = sigcore::cfar_ca_detect(flat_with_spike(64, 32, 100.0), 4, 1, 5.0);
    CHECK(d.size() == 1 && d[0] == 32);
    d = sigcore::cfar_ca_detect(flat_with_spike(64, 0, 100.0), 4, 1, 5.0);
    CHECK(d.size() == 1 && d[0] == 0);
    d = sigcore::cfar_ca_detect(flat_with_spike(64, 63, 100.0), 4, 1, 5.0);
    CHECK(d.size() == 1 && d[0] == 63);
    d = sigcore::cfar_ca_detect(std::vector<double>(64, 1.0), 4, 1, 5.0);
    CHECK(d.empty());
    d = sigcore::cfar_ca_detect(std::vector<double>(), 4, 1, 5.0);
    CHECK(d.empty());
}

static void test_cfar_guard_cells_prevent_masking() {
    std::vector<double> x(64, 1.0);
    x[32] = 100.0;
    x[33] = 100.0;
    auto without_guard = sigcore::cfar_ca_detect(x, 4, 0, 10.0);
    CHECK(without_guard.empty());
    auto with_guard = sigcore::cfar_ca_detect(x, 4, 1, 10.0);
    CHECK(with_guard.size() == 1 && with_guard[0] == 32);
}

static void test_cfar_invalid_parameters() {
    const std::vector<double> x(16, 1.0);
    CHECK_THROWS(sigcore::cfar_ca_detect(x, 0, 1, 5.0));
    CHECK_THROWS(sigcore::cfar_ca_detect(x, 4, -1, 5.0));
    CHECK_THROWS(sigcore::cfar_ca_detect(x, 4, 1, 0.0));
    CHECK_THROWS(sigcore::cfar_ca_detect(x, 4, 1, -2.0));
    CHECK_THROWS(sigcore::cfar_ca_detect(x, 4, 1, std::nan("")));
}

static void test_conv2d_identity_and_delta() {
    std::vector<double> img(5 * 6);
    for (std::size_t i = 0; i < img.size(); ++i) {
        img[i] = static_cast<double>(i % 7) - 2.5;
    }
    const std::vector<double> identity = {0, 0, 0, 0, 1, 0, 0, 0, 0};
    const std::vector<double> out = sigcore::conv2d(img, 5, 6, identity, 3, 3);
    bool same = out.size() == img.size();
    for (std::size_t i = 0; same && i < img.size(); ++i) {
        same = std::fabs(out[i] - img[i]) < 1e-15;
    }
    CHECK(same);

    // Convolving a delta with a kernel reproduces the (unflipped) kernel around the delta.
    std::vector<double> delta(7 * 7, 0.0);
    delta[3 * 7 + 3] = 1.0;
    const std::vector<double> ker = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    const std::vector<double> resp = sigcore::conv2d(delta, 7, 7, ker, 3, 3);
    bool match = true;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            match = match && std::fabs(resp[(2 + i) * 7 + (2 + j)] - ker[i * 3 + j]) < 1e-15;
        }
    }
    CHECK(match);
}

static void test_conv2d_sobel_on_ramp() {
    const std::size_t h = 5, w = 5;
    std::vector<double> img(h * w);
    for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = 0; x < w; ++x) {
            img[y * w + x] = static_cast<double>(x);
        }
    }
    const std::vector<double> sobel_x = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    const std::vector<double> gx = sigcore::conv2d(img, h, w, sobel_x, 3, 3);
    CHECK(std::fabs(gx[2 * w + 2] - 8.0) < 1e-12);
    CHECK(std::fabs(gx[1 * w + 3] - 8.0) < 1e-12);
}

static void test_conv2d_invalid_arguments() {
    const std::vector<double> img(4 * 4, 1.0);
    CHECK_THROWS(sigcore::conv2d(img, 4, 4, std::vector<double>(4, 1.0), 2, 2));  // even kernel
    CHECK_THROWS(sigcore::conv2d(img, 4, 4, std::vector<double>(8, 1.0), 3, 3));  // size mismatch
    CHECK_THROWS(sigcore::conv2d(img, 4, 5, std::vector<double>(9, 1.0), 3, 3));  // image mismatch
    CHECK_THROWS(sigcore::conv2d(std::vector<double>(), 0, 0, std::vector<double>(1, 1.0), 1, 1));
}

static void test_percentile() {
    std::vector<double> v;
    for (int i = 10; i >= 1; --i) {
        v.push_back(static_cast<double>(i));
    }
    CHECK(sigcore::nearest_rank_percentile(v, 50.0) == 5.0);
    CHECK(sigcore::nearest_rank_percentile(v, 95.0) == 10.0);
    CHECK(sigcore::nearest_rank_percentile(v, 100.0) == 10.0);
    CHECK(sigcore::nearest_rank_percentile(v, 1.0) == 1.0);
    CHECK_THROWS(sigcore::nearest_rank_percentile(v, 0.0));
    CHECK_THROWS(sigcore::nearest_rank_percentile(std::vector<double>(), 50.0));
}

int main() {
    test_fft_impulse();
    test_fft_matches_naive_dft();
    test_fft_single_sample_and_invalid_sizes();
    test_cfar_spike_and_edges();
    test_cfar_guard_cells_prevent_masking();
    test_cfar_invalid_parameters();
    test_conv2d_identity_and_delta();
    test_conv2d_sobel_on_ramp();
    test_conv2d_invalid_arguments();
    test_percentile();
    std::printf("sigcore tests: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
