// Analytic Kaiser-windowed sinc interpolation, with zero extension. This is a
// new FIR implementation; it does not reproduce resampy's lookup-table filter.
#ifndef PFFDTD_POST_RESAMPLE_H
#define PFFDTD_POST_RESAMPLE_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pffdtd_post {

static_assert(std::numeric_limits<double>::radix==2 &&
              std::numeric_limits<double>::digits==53,
              "Native/CUDA resampling requires binary64 double");

#ifdef __CUDACC__
#define PFFDTD_RESAMPLE_INLINE __host__ __device__ __forceinline__
#else
#define PFFDTD_RESAMPLE_INLINE inline
#endif

namespace detail {

// I0(x) = sum_k ((x*x/4)^k/(k!)^2). For 0<=x<=12.9846, truncating
// after k=32 leaves a relative tail below 1e-24. Horner evaluation uses
// positive terms and compile-time divisors, without a table or device loop.
template<unsigned K>
struct ResampleI0 {
   static PFFDTD_RESAMPLE_INLINE constexpr double value(double squared_argument)
   {
      return 1.0+(squared_argument/(K*K))*ResampleI0<K+1>::value(squared_argument);
   }
};

template<>
struct ResampleI0<33> {
   static PFFDTD_RESAMPLE_INLINE constexpr double value(double)
   {
      return 1.0;
   }
};

constexpr double resample_beta=12.9846;
constexpr double resample_rolloff=0.917347441;
constexpr double resample_support=50.0;
constexpr double resample_pi=3.141592653589793238462643383279502884;
constexpr double resample_i0_inverse=
   1.0/ResampleI0<1>::value(resample_beta*resample_beta*0.25);
constexpr double resample_exact_integer=9007199254740992.0; // 2^53

inline double resample_ratio(double fs_in, double fs_out)
{
   if (!std::isfinite(fs_in) || !std::isfinite(fs_out) || fs_in<=0 || fs_out<=0)
      throw std::invalid_argument("Resampling rates must be finite and positive");
   const double ratio=fs_out/fs_in;
   // This also excludes division overflow, underflow to zero, and a support
   // interval that a signed 64-bit GPU tap index cannot represent safely.
   const double scale=ratio<1.0 ? ratio : 1.0;
   const double radius=resample_support/scale;
   if (!std::isfinite(ratio) || ratio<=0 || ratio>resample_exact_integer ||
       !std::isfinite(radius) ||
       radius>static_cast<double>(std::numeric_limits<std::int64_t>::max()/4))
      throw std::invalid_argument("Resampling ratio or FIR support is not representable");
   return ratio;
}

inline std::size_t resample_product(std::size_t channels, std::size_t samples)
{
   if (!channels || !samples ||
       channels>std::numeric_limits<std::size_t>::max()/samples)
      throw std::invalid_argument("Invalid resampling buffer shape");
   const std::size_t count=channels*samples;
   // vector addressing and CUDA byte extents must both remain representable.
   if (count>std::numeric_limits<std::size_t>::max()/sizeof(double) ||
       count>static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())/
             sizeof(double))
      throw std::invalid_argument("Resampling byte extent is not representable");
   return count;
}

// An exact two-limb product is enough for a binary64 mantissa multiplied by
// a <=2^53 sample count. Keeping this host-only avoids compiler extensions
// and resolves floor decisions even one product unit below an integer.
struct ResampleProduct { std::uint64_t high,low; };

inline ResampleProduct resample_integer_product(std::uint64_t a, std::uint64_t b)
{
   const std::uint64_t mask=0xffffffffULL;
   const std::uint64_t a0=a&mask,a1=a>>32,b0=b&mask,b1=b>>32;
   const std::uint64_t first=a0*b0;
   const std::uint64_t second=a1*b0+(first>>32);
   const std::uint64_t third=(second&mask)+a0*b1;
   return {a1*b1+(second>>32)+(third>>32),(third<<32)|(first&mask)};
}

inline unsigned resample_bits(std::uint64_t value)
{
   unsigned bits=0;
   while (value) { value>>=1; ++bits; }
   return bits;
}

inline ResampleProduct resample_align(ResampleProduct value, unsigned shift)
{
   if (shift==0) return value;
   if (shift>=64) return {value.low<<(shift-64),0};
   return {(value.high<<shift)|(value.low>>(64-shift)),value.low<<shift};
}

inline int resample_compare(ResampleProduct a, int exponent_a,
                            ResampleProduct b, int exponent_b)
{
   const unsigned bits_a=a.high ? 64+resample_bits(a.high) : resample_bits(a.low);
   const unsigned bits_b=b.high ? 64+resample_bits(b.high) : resample_bits(b.low);
   if (!bits_a || !bits_b) return bits_a ? 1 : (bits_b ? -1 : 0);
   const int magnitude_a=static_cast<int>(bits_a)+exponent_a;
   const int magnitude_b=static_cast<int>(bits_b)+exponent_b;
   if (magnitude_a!=magnitude_b) return magnitude_a<magnitude_b ? -1 : 1;
   a=resample_align(a,128-bits_a); b=resample_align(b,128-bits_b);
   if (a.high!=b.high) return a.high<b.high ? -1 : 1;
   return a.low<b.low ? -1 : (a.low>b.low ? 1 : 0);
}

struct ResampleRate { std::uint64_t mantissa; int exponent; };

inline ResampleRate resample_rate(double rate)
{
   int exponent=0;
   const double fraction=std::frexp(rate,&exponent);
   return {static_cast<std::uint64_t>(std::ldexp(fraction,53)),exponent-53};
}

} // namespace detail

// These coefficient functions require a ratio already checked by
// resample_length. They contain no allocation, host-only code or exceptions.
// The FIR support is open: |distance| >= radius has coefficient zero.
PFFDTD_RESAMPLE_INLINE double resample_radius(double ratio)
{
   return detail::resample_support/(ratio<1.0 ? ratio : 1.0);
}

PFFDTD_RESAMPLE_INLINE double resample_weight(double distance, double ratio)
{
   const double scale=ratio<1.0 ? ratio : 1.0;
   const double radius=detail::resample_support/scale;
   const double absolute=::fabs(distance);
   if (absolute>=radius) return 0.0;
   const double cutoff=detail::resample_rolloff*scale;
   const double argument=detail::resample_pi*cutoff*distance;
   const double sinc=argument==0.0 ? 1.0 : ::sin(argument)/argument;
   const double position=distance/radius;
   // I0's squared argument avoids a square root per tap. Clipping protects
   // the endpoint from a negative roundoff when position is close to +/-1.
   const double window_argument=detail::resample_beta*detail::resample_beta*0.25*
                               (1.0-position*position);
   const double window=detail::ResampleI0<1>::value(
      window_argument>0.0 ? window_argument : 0.0)*detail::resample_i0_inverse;
   return cutoff*sinc*window;
}

#undef PFFDTD_RESAMPLE_INLINE

inline std::size_t resample_length(std::size_t samples, double fs_in, double fs_out)
{
   detail::resample_ratio(fs_in,fs_out);
   // All integer sample positions must be exactly representable in the
   // double-precision phase used by both native and CUDA gather kernels.
   if (!samples || static_cast<long double>(samples)>detail::resample_exact_integer)
      throw std::invalid_argument("Resampling input length is not representable");
   detail::resample_product(1,samples);
   if (fs_in==fs_out) return samples;

   const std::uint64_t phase_limit=9007199254740992ULL;
   const std::uint64_t byte_limit=std::numeric_limits<std::size_t>::max()/sizeof(double);
   const std::uint64_t address_limit=static_cast<std::uint64_t>(
      std::numeric_limits<std::ptrdiff_t>::max())/sizeof(double);
   const std::uint64_t max_count=phase_limit<byte_limit ?
      (phase_limit<address_limit ? phase_limit : address_limit) :
      (byte_limit<address_limit ? byte_limit : address_limit);
   const detail::ResampleRate in=detail::resample_rate(fs_in),
                              out=detail::resample_rate(fs_out);
   const detail::ResampleProduct numerator=detail::resample_integer_product(samples,out.mantissa);
   if (detail::resample_compare(numerator,out.exponent,
         detail::resample_integer_product(1,in.mantissa),in.exponent)<0 ||
       detail::resample_compare(numerator,out.exponent,
         detail::resample_integer_product(max_count+1,in.mantissa),in.exponent)>=0)
      throw std::invalid_argument("Resampling output length is not representable or is zero");

   long double estimate=static_cast<long double>(samples)*static_cast<long double>(fs_out);
   // On platforms where long double has binary64's exponent range, this
   // product may overflow although its quotient is small and representable.
   estimate=std::isfinite(estimate) ? estimate/static_cast<long double>(fs_in) :
      static_cast<long double>(samples)*(static_cast<long double>(fs_out)/fs_in);
   std::uint64_t count=estimate<1 ? 1 :
      (estimate>=max_count ? max_count : static_cast<std::uint64_t>(std::floor(estimate)));
   while (detail::resample_compare(numerator,out.exponent,
         detail::resample_integer_product(count,in.mantissa),in.exponent)<0) --count;
   while (detail::resample_compare(numerator,out.exponent,
         detail::resample_integer_product(count+1,in.mantissa),in.exponent)>=0) ++count;
   return static_cast<std::size_t>(count);
}

// Planar [channel*samples+n] input, time origin zero, no delay compensation
// beyond centering the symmetric FIR at n/ratio. Missing edge taps are zero;
// no per-output DC normalization is applied. Equal rates bypass the FIR.
inline std::vector<double> resample_serial(const std::vector<double>& planar,
      std::size_t channels, std::size_t samples, double fs_in, double fs_out)
{
   const std::size_t samples_out=resample_length(samples,fs_in,fs_out);
   const std::size_t input_count=detail::resample_product(channels,samples);
   const std::size_t output_count=detail::resample_product(channels,samples_out);
   if (planar.size()!=input_count)
      throw std::invalid_argument("Resampling input does not match its planar shape");
   for (double sample : planar)
      if (!std::isfinite(sample))
         throw std::invalid_argument("Resampling input sample must be finite");
   if (fs_in==fs_out) return planar;

   const double ratio=detail::resample_ratio(fs_in,fs_out);
   const double radius=resample_radius(ratio);
   std::vector<double> output(output_count,0.0);
   for (std::size_t channel=0; channel<channels; ++channel) {
      const std::size_t input_base=channel*samples;
      const std::size_t output_base=channel*samples_out;
      for (std::size_t n=0; n<samples_out; ++n) {
         const double position=static_cast<double>(n)/ratio;
         // Clip in floating point before any integral conversion. Validated
         // lengths are <=2^53, so these bounds are exact integer coordinates.
         const double first=position-radius>0.0 ? std::ceil(position-radius) : 0.0;
         const double last=position+radius<static_cast<double>(samples-1) ?
            std::floor(position+radius) : static_cast<double>(samples-1);
         double value=0.0;
         if (first<=last) {
            const std::size_t begin=static_cast<std::size_t>(first);
            const std::size_t end=static_cast<std::size_t>(last);
            for (std::size_t tap=begin; tap<=end; ++tap)
               value+=planar[input_base+tap]*resample_weight(
                  position-static_cast<double>(tap),ratio);
         }
         if (!std::isfinite(value))
            throw std::runtime_error("Resampling produced a non-finite output sample");
         output[output_base+n]=value;
      }
   }
   return output;
}

} // namespace pffdtd_post
#endif
