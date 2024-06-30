
#include <nanobench.h>
#include <doctest/doctest.h>
#include <random>
#include <xmmintrin.h>
#include <immintrin.h>

uint32_t uint32_sqrt_native_double(uint32_t num) {
    auto val = std::sqrt(static_cast<double>(num));
    return static_cast<uint32_t>(val);
}

uint32_t uint32_sqrt_native_single(uint32_t num) {
    auto val = std::sqrt(static_cast<float>(num));
    return static_cast<uint32_t>(val);
}

uint32_t uint32_sqrt_binary_sqrt(uint32_t num) {
    unsigned long rem = 0;
    int root = 0;
    int i;

    for (i = 0; i < 16; i++) {
        root <<= 1;
        rem <<= 2;
        rem += num >> 30;
        num <<= 2;

        if (root < rem) {
            root++;
            rem -= root;
            root++;
        }
    }

    return root >> 1;
}

uint32_t uint32_sqrt_binary_search(uint32_t num); // Implementation omitted, see binary_search.cpp

uint32_t uint32_sqrt_bit_guessing(uint32_t num) {
    uint8_t shift = std::bit_width(num);
    shift += shift & 1;

    uint32_t result = 0;

    do {
        shift -= 2;
        result <<= 1;
        result |= 1;
        result ^= result * result > (num >> shift);
    } while (shift != 0);

    return result;
}

uint32_t uint32_sqrt_newton_guess_helper(uint32_t num)
{
    uint32_t log2floor = std::bit_width(num) - 1;
    return 1 << (log2floor >> 1);
}

uint32_t uint32_sqrt_newton_guessing(uint32_t num) {
    uint32_t a = uint32_sqrt_newton_guess_helper(num);
    uint32_t b = num;

    // compute unsigned difference
    while (std::max(a, b) - std::min(a, b) > 1) {
        b = num / a;
        a = (a + b) / 2;
    }

    return a - (a * a > num);
}

#if __SSE2_MATH__
uint32_t uint32_sqrt_intrinsic_simd_double(uint32_t num) {
    __v2df xmm0 = {0, 0};
    xmm0 = __builtin_ia32_cvtsi2sd(xmm0, num);
    xmm0 = __builtin_ia32_sqrtsd(xmm0);
    return __builtin_ia32_cvttsd2si(xmm0);
}

uint32_t uint32_sqrt_intrinsic_simd_single(uint32_t num) {
    __v4sf xmm0 = {0, 0, 0, 0};
    xmm0 = __builtin_ia32_cvtsi2ss(xmm0, num);
    xmm0 = __builtin_ia32_sqrtss(xmm0);
    return __builtin_ia32_cvttss2si(xmm0);
}
#endif

template<size_t size>
union vec {
    typedef double mmd_t __attribute__ ((__vector_size__ (size), __may_alias__));
    typedef float mms_t __attribute__ ((__vector_size__ (size), __may_alias__));
    typedef uint32_t u32_t __attribute__ ((vector_size (size)));
    typedef float f32_t __attribute__ ((vector_size (size)));
    typedef double f64_t __attribute__ ((vector_size (size)));
    u32_t u32;
    f32_t f32;
    f64_t f64;
    mmd_t mmd;
    mms_t mms;
};

#if __SSE2__
vec<8> uint32_sqrt_intrinsic_simd_double_x2(uint32_t num) {
    vec<8> input = {num, num};
    vec<16> float_in{};
    float_in.f64 = __builtin_convertvector(input.u32, typeof(float_in.f64));
    vec<16> float_out{};
    float_out.mmd = _mm_sqrt_pd(float_in.mmd);
    vec<8> out{};
    out.u32 = __builtin_convertvector(float_out.f64, typeof(out.u32));
    return out;
}
#endif
#ifdef __AVX2__
vec<16> uint32_sqrt_intrinsic_simd_double_x4(uint32_t num) {
    vec<16> input = {num, num};
    vec<32> float_in{};
    float_in.f64 = __builtin_convertvector(input.u32, typeof(float_in.f64));
    vec<32> float_out{};
    float_out.mmd = _mm256_sqrt_pd(float_in.mmd);
    vec<16> out{};
    out.u32 = __builtin_convertvector(float_out.f64, typeof(out.u32));
    return out;
}
#endif
#ifdef __AVX512F__
vec<32> uint32_sqrt_intrinsic_simd_double_x8(uint32_t num) {
    vec<32> input = {num, num};
    vec<64> float_in{};
    float_in.f64 = __builtin_convertvector(input.u32, typeof(float_in.f64));
    vec<64> float_out{};
    float_out.mmd = _mm512_sqrt_pd(float_in.mmd);
    vec<32> out{};
    out.u32 = __builtin_convertvector(float_out.f64, typeof(out.u32));
    return out;
}
#endif

vec<16> uint32_sqrt_intrinsic_simd_single_x4(uint32_t num) {
    vec<16> input = {num, num};
    vec<16> float_in{};
    float_in.f32 = __builtin_convertvector(input.u32, typeof(float_in.f32));
    vec<16> float_out{};
    float_out.mms = _mm_sqrt_ps(float_in.mms);
    vec<16> out{};
    out.u32 = __builtin_convertvector(float_out.f32, typeof(out.u32));
    return out;
}
#ifdef __AVX2__
vec<32> uint32_sqrt_intrinsic_simd_single_x8(uint32_t num) {
    vec<32> input = {num, num};
    vec<32> float_in{};
    float_in.f32 = __builtin_convertvector(input.u32, typeof(float_in.f32));
    vec<32> float_out{};
    float_out.mms = _mm256_sqrt_ps(float_in.mms);
    vec<32> out{};
    out.u32 = __builtin_convertvector(float_out.f32, typeof(out.u32));
    return out;
}
#endif
#ifdef __AVX512F__
vec<64> uint32_sqrt_intrinsic_simd_single_x16(uint32_t num) {
    vec<64> input = {num, num};
    vec<64> float_in{};
    float_in.f32 = __builtin_convertvector(input.u32, typeof(float_in.f32));
    vec<64> float_out{};
    float_out.mms = _mm512_sqrt_ps(float_in.mms);
    vec<64> out{};
    out.u32 = __builtin_convertvector(float_out.f32, typeof(out.u32));
    return out;
}
#endif

template <typename F>
void uint32_bench(ankerl::nanobench::Bench& bench, F f, std::string_view name) {
    ankerl::nanobench::Rng rng;

    bench.run(name.data(), [&]() {
        uint32_t random = rng();
        auto result = f(random);
        ankerl::nanobench::doNotOptimizeAway(result);
    });
}

TEST_CASE("uint32_sqrt") {
    ankerl::nanobench::Bench b;
    b.title("sqrt for int32")
        .unit("uint32_t")
        .warmup(100)
        .relative(true);

    b.performanceCounters(true);
    //b.epochs(1000); // Set the number of epochs here for more accuracy

    uint32_bench(b, uint32_sqrt_native_double,          "native sqrt(double)");
    uint32_bench(b, uint32_sqrt_native_single,          "[INACCURATE] native sqrt(float)");
    uint32_bench(b, uint32_sqrt_binary_sqrt,            "binary sqrt");
    uint32_bench(b, uint32_sqrt_binary_search,          "binary search");
    uint32_bench(b, uint32_sqrt_bit_guessing,           "bit guessing (fast inverse)");
    uint32_bench(b, uint32_sqrt_newton_guessing,        "newton guessing (fast inverse)");

#if __SSE2__
    uint32_bench(b, uint32_sqrt_intrinsic_simd_double,  "simd sqrt(double)");
    uint32_bench(b, uint32_sqrt_intrinsic_simd_single,  "[INACCURATE] simd sqrt(float)");
    uint32_bench(b.batch(2), uint32_sqrt_intrinsic_simd_double_x2,   "parallel simd sqrt(double)x2");
#endif
#ifdef __AVX2__
    uint32_bench(b.batch(4), uint32_sqrt_intrinsic_simd_double_x4,   "parallel simd sqrt(double)x4");
#endif
#if __SSE2__
    uint32_bench(b.batch(4), uint32_sqrt_intrinsic_simd_single_x4,   "[INACCURATE] parallel simd sqrt(float)x4");
#endif
#ifdef __AVX512F__
    uint32_bench(b.batch(8), uint32_sqrt_intrinsic_simd_double_x8,   "parallel simd sqrt(double)x8");
#endif
#ifdef __AVX2__
    uint32_bench(b.batch(8), uint32_sqrt_intrinsic_simd_single_x8,   "[INACCURATE] parallel simd sqrt(float)x8");
#endif
#ifdef __AVX512F__
    uint32_bench(b.batch(16), uint32_sqrt_intrinsic_simd_single_x16,  "[INACCURATE] parallel simd sqrt(float)x16");
#endif
}
