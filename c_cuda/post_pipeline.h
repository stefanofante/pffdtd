// Native receiver reconstruction and RIR processing. The windowed-sinc
// resampler is a new implementation, not a bitwise reproduction of resampy.
#ifndef PFFDTD_POST_PIPELINE_H
#define PFFDTD_POST_PIPELINE_H

#include "post_dsp.h"
#include "post_resample.h"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pffdtd_post {

struct PostOptions {
   double lowcut=10.0, lowpass=0.0, output_rate=48000.0;
   unsigned lowcut_order=8, lowpass_order=8;
   bool symmetric_lowpass=false, integrate=false;
};

struct PostResult {
   std::vector<double> raw, filtered;
   std::size_t channels=0, samples=0, samples_out=0;
   double fs_out=0;
};

inline std::size_t post_product(std::size_t a, std::size_t b)
{
   if (b && a>std::numeric_limits<std::size_t>::max()/b)
      throw std::invalid_argument("Postprocessing element count overflows");
   const std::size_t count=a*b;
   if (count>std::numeric_limits<std::size_t>::max()/sizeof(double))
      throw std::invalid_argument("Postprocessing byte count overflows");
   return count;
}

inline void validate_post_input(const std::vector<double>& input,
      const std::vector<double>& weights, std::size_t channels,
      std::size_t corners, std::size_t samples, double fs,
      const PostOptions& options)
{
   if (!channels || !corners || !samples || !std::isfinite(fs) || fs<=0 ||
       input.size()!=post_product(post_product(channels,corners),samples) ||
       weights.size()!=post_product(channels,corners))
      throw std::invalid_argument("Invalid receiver data shape or sample rate");
   if (!std::isfinite(options.lowcut) || options.lowcut<0 ||
       options.lowcut>=fs/2 || !options.lowcut_order || options.lowcut_order>16 ||
       !std::isfinite(options.output_rate) || options.output_rate<0 ||
       !std::isfinite(options.lowpass) || options.lowpass<0 ||
       !options.lowpass_order || options.lowpass_order>16 ||
       (options.symmetric_lowpass && options.lowpass_order%2))
      throw std::invalid_argument("Invalid filter order, cutoff or output rate");
   const double fs_out=options.output_rate ? options.output_rate : fs;
   if (options.lowpass>=fs_out/2)
      throw std::invalid_argument("Lowpass cutoff must be below output Nyquist");
   for (double value : input)
      if (!std::isfinite(value)) throw std::invalid_argument("Non-finite raw receiver sample");
   for (double weight : weights)
      if (!std::isfinite(weight)) throw std::invalid_argument("Non-finite receiver weight");
   post_product(channels,resample_length(samples,fs,fs_out));
}

inline std::vector<double> recombine_serial(const std::vector<double>& input,
      const std::vector<double>& weights, std::size_t channels,
      std::size_t corners, std::size_t samples)
{
   std::vector<double> result(post_product(channels,samples),0.0);
   for (std::size_t channel=0; channel<channels; ++channel)
      for (std::size_t n=0; n<samples; ++n) {
         double sum=0;
         for (std::size_t corner=0; corner<corners; ++corner) {
            const std::size_t receiver=channel*corners+corner;
            sum+=weights[receiver]*input[receiver*samples+n];
         }
         result[channel*samples+n]=sum;
      }
   return result;
}

inline std::vector<Sos> post_lowpass_sections(double fs, const PostOptions& options)
{
   return options.lowpass>0 ? design_lowpass(
      options.symmetric_lowpass ? options.lowpass_order/2 : options.lowpass_order,
      fs,options.lowpass) : std::vector<Sos>();
}

inline void check_post_finite(const std::vector<double>& values)
{
   for (double value : values)
      if (!std::isfinite(value))
         throw std::runtime_error("Postprocessing produced a non-finite sample");
}

inline PostResult process_cpu(const std::vector<double>& input,
      const std::vector<double>& weights, std::size_t channels,
      std::size_t corners, std::size_t samples, double fs,
      const PostOptions& options)
{
   validate_post_input(input,weights,channels,corners,samples,fs,options);
   PostResult result;
   result.channels=channels; result.samples=samples;
   result.fs_out=options.output_rate ? options.output_rate : fs;
   result.raw=recombine_serial(input,weights,channels,corners,samples);
   result.filtered=result.raw;
   filter_serial(result.filtered.data(),channels,samples,
                 design_highpass(options.lowcut_order,fs,options.lowcut,options.integrate));
   result.filtered=resample_serial(result.filtered,channels,samples,fs,result.fs_out);
   result.samples_out=result.filtered.size()/channels;
   const std::vector<Sos> lowpass=post_lowpass_sections(result.fs_out,options);
   filter_serial(result.filtered.data(),channels,result.samples_out,lowpass);
   if (options.symmetric_lowpass)
      filter_serial(result.filtered.data(),channels,result.samples_out,lowpass,true);
   check_post_finite(result.raw); check_post_finite(result.filtered);
   return result;
}

} // namespace pffdtd_post
#endif
