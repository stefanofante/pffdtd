// Native coefficient design and serial DSP reference. CUDA uses the same
// normalized DF-II-transposed sample update; parallel scans need separate gates.
#ifndef PFFDTD_POST_DSP_H
#define PFFDTD_POST_DSP_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pffdtd_post {

struct Sos {
   double b0, b1, b2, a1, a2;
};

#ifdef __CUDACC__
#define PFFDTD_POST_INLINE __host__ __device__ __forceinline__
#else
#define PFFDTD_POST_INLINE inline
#endif

PFFDTD_POST_INLINE double sos_sample(const Sos &sos, double x, double &z1, double &z2)
{
   const double y=sos.b0*x+z1;
   z1=sos.b1*x-sos.a1*y+z2;
   z2=sos.b2*x-sos.a2*y;
   return y;
}

#undef PFFDTD_POST_INLINE

namespace detail {

inline void validate_design(unsigned order, double fs, double cut)
{
   if (order==0 || order>16)
      throw std::invalid_argument("Butterworth order must be in 1..16");
   if (!std::isfinite(fs) || fs<=0 || !std::isfinite(1.0/fs) ||
       !std::isfinite(2.0*fs))
      throw std::invalid_argument("Sample rate must be positive and representable");
   if (!std::isfinite(cut) || cut<0 || cut>=0.5*fs)
      throw std::invalid_argument("Cutoff must be finite and below Nyquist");
}

inline std::vector<Sos> design(unsigned order, double fs, double cut,
                                bool highpass, bool integrate)
{
   validate_design(order,fs,cut);
   if (cut==0) {
      if (!highpass) throw std::invalid_argument("Lowpass cutoff must be positive");
      if (integrate) return {{0.5/fs,0.5/fs,0.0,-1.0,0.0}};
      return {};
   }

   const double pi=std::acos(-1.0);
   // The legacy combined highpass/integrator starts from an analog 2*pi*cut
   // Butterworth design. Ordinary digital HP/LP prewarp their cutoff instead.
   const double normalized=integrate ? pi*(cut/fs) : std::tan(pi*(cut/fs));
   if (!std::isfinite(normalized) || normalized<=0)
      throw std::invalid_argument("Cutoff cannot be represented in the coefficient design");
   std::vector<Sos> sections;
   sections.reserve((order+1)/2);
   double gain=integrate ? 0.5/fs : 1.0;

   // Put the most damped poles first. Conjugate pairs are represented directly
   // as real quadratics; no high-order polynomial expansion is required.
   if (order%2) {
      const double pole=(1.0-normalized)/(1.0+normalized);
      if (!(std::abs(pole)<1.0))
         throw std::invalid_argument("Cutoff is too close to an endpoint for stable coefficients");
      sections.push_back({1.0,(highpass && !integrate) ? -1.0 : 1.0,0.0,-pole,0.0});
      gain*=highpass ? 1.0/(1.0+normalized) : normalized/(1.0+normalized);
   }
   for (unsigned pair=order/2; pair>0; --pair) {
      const unsigned k=pair-1;
      const double angle=pi*(2.0*k+order+1.0)/(2.0*order);
      const std::complex<double> analog=normalized*std::complex<double>(std::cos(angle),std::sin(angle));
      const std::complex<double> denominator=1.0-analog;
      const std::complex<double> pole=(1.0+analog)/denominator;
      if (!(std::abs(pole)<1.0))
         throw std::invalid_argument("Cutoff is too close to an endpoint for stable coefficients");
      Sos section={1.0,highpass ? -2.0 : 2.0,1.0,-2.0*pole.real(),std::norm(pole)};
      if (highpass && integrate && order%2==0 && pair==order/2) {
         // Remove one analog zero at the origin. Its missing degree becomes a
         // digital zero at -1, paired here with one of the zeros at +1.
         section.b1=0.0; section.b2=-1.0;
      }
      sections.push_back(section);
      const double divisor=std::norm(denominator);
      gain*=highpass ? 1.0/divisor : normalized*normalized/divisor;
   }
   if (!std::isfinite(gain) || gain<=0)
      throw std::invalid_argument("Filter gain cannot be represented");
   sections.front().b0*=gain;
   sections.front().b1*=gain;
   sections.front().b2*=gain;
   return sections;
}

inline bool finite(const Sos &sos)
{
   return std::isfinite(sos.b0) && std::isfinite(sos.b1) && std::isfinite(sos.b2) &&
          std::isfinite(sos.a1) && std::isfinite(sos.a2);
}

} // namespace detail

inline std::vector<Sos> design_highpass(unsigned order, double fs, double cut, bool integrate)
{
   return detail::design(order,fs,cut,true,integrate);
}

inline std::vector<Sos> design_lowpass(unsigned order, double fs, double cut)
{
   return detail::design(order,fs,cut,false,false);
}

// Planar [channel*samples+n] data, zero initial state for every channel/section.
// Reverse uses the same zero-state recurrence in reverse time, without padding
// or a steady-state initializer (matching two separate legacy sosfilt calls).
inline void filter_serial(double *data, std::size_t channels, std::size_t samples,
                           const std::vector<Sos> &sections, bool reverse=false)
{
   for (const Sos &sos : sections)
      if (!detail::finite(sos)) throw std::invalid_argument("SOS coefficient is not finite");
   if (channels==0 || samples==0) return;
   if (!data || channels>std::numeric_limits<std::size_t>::max()/samples)
      throw std::invalid_argument("Invalid planar sample buffer");
   const std::size_t count=channels*samples;
   for (std::size_t i=0; i<count; ++i)
      if (!std::isfinite(data[i])) throw std::invalid_argument("Input sample is not finite");
   std::vector<double> z1(sections.size()),z2(sections.size());
   for (std::size_t channel=0; channel<channels; ++channel) {
      std::fill(z1.begin(),z1.end(),0.0);
      std::fill(z2.begin(),z2.end(),0.0);
      for (std::size_t t=0; t<samples; ++t) {
         const std::size_t n=reverse ? samples-1-t : t;
         double value=data[channel*samples+n];
         for (std::size_t section=0; section<sections.size(); ++section)
            value=sos_sample(sections[section],value,z1[section],z2[section]);
         if (!std::isfinite(value)) throw std::runtime_error("Filter produced a nonfinite output");
         data[channel*samples+n]=value;
      }
   }
}

} // namespace pffdtd_post
#endif
