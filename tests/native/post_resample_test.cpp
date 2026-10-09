// Analytic checks of the native, new Kaiser FIR. No external DSP package or
// generated numerical fixture is needed; the long-double reference evaluates
// the Bessel power series independently of the production Horner polynomial.
#include "post_resample.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

std::size_t cases=0;

void require(bool condition, const char* message)
{
   if (!condition) {
      std::fprintf(stderr,"FAIL: %s\n",message);
      std::exit(EXIT_FAILURE);
   }
   ++cases;
}

void close(double actual, double expected, double tolerance, const char* message)
{
   if (!std::isfinite(actual) || std::abs(actual-expected)>tolerance) {
      std::fprintf(stderr,"FAIL: %s: got %.17g, expected %.17g, tolerance %.3g\n",
                   message,actual,expected,tolerance);
      std::exit(EXIT_FAILURE);
   }
   ++cases;
}

template<typename F>
void invalid(F function, const char* message)
{
   bool caught=false;
   try { function(); }
   catch (const std::invalid_argument&) { caught=true; }
   require(caught,message);
}

long double i0_reference(long double x)
{
   long double sum=1,term=1;
   for (unsigned k=1; k<100; ++k) {
      term*=x*x/(4*static_cast<long double>(k)*k);
      const long double next=sum+term;
      if (next==sum) break;
      sum=next;
   }
   return sum;
}

long double weight_reference(long double distance, long double ratio)
{
   const long double beta=12.9846L,rolloff=0.917347441L;
   const long double scale=std::min(1.0L,ratio);
   // Support endpoints, like output phase, follow the binary64 CPU/CUDA
   // contract. Keep the coefficient evaluation itself in long double.
   const long double radius=50.0/std::min(1.0,static_cast<double>(ratio));
   if (std::abs(distance)>=radius) return 0;
   const long double pi=std::acos(-1.0L),argument=pi*rolloff*scale*distance;
   const long double sinc=argument==0 ? 1 : std::sin(argument)/argument;
   const long double position=distance/radius;
   return rolloff*scale*sinc*i0_reference(beta*std::sqrt(1-position*position))/
          i0_reference(beta);
}

std::vector<double> gather_reference(const std::vector<double>& input,
      std::size_t channels, std::size_t samples, double fs_in, double fs_out)
{
   const std::size_t output_samples=static_cast<std::size_t>(std::floor(
      static_cast<long double>(samples)*fs_out/fs_in));
   // The independent reference retains the shared double phase convention;
   // coefficients and reductions are evaluated in long double.
   const double ratio=fs_out/fs_in;
   const long double radius=50.0/std::min(1.0,ratio);
   std::vector<double> result(channels*output_samples,0);
   for (std::size_t channel=0; channel<channels; ++channel)
      for (std::size_t n=0; n<output_samples; ++n) {
         const long double position=static_cast<double>(n)/ratio;
         long double value=0;
         for (std::size_t tap=0; tap<samples; ++tap)
            if (std::abs(position-tap)<radius)
               value+=static_cast<long double>(input[channel*samples+tap])*
                      weight_reference(position-tap,ratio);
         result[channel*output_samples+n]=static_cast<double>(value);
      }
   return result;
}

void check_coefficients()
{
   using namespace pffdtd_post;
   for (unsigned n=0; n<=4096; ++n) {
      const double x=12.9846*static_cast<double>(n)/4096;
      const double actual=detail::ResampleI0<1>::value(x*x*0.25);
      const double expected=static_cast<double>(i0_reference(x));
      close(actual/expected,1.0,2e-15,"I0 polynomial accuracy");
   }
   const double ratios[]={0.009,0.125,0.5,44100.0/48000,1.0,1.5,4.0};
   for (double ratio : ratios) {
      const double radius=resample_radius(ratio);
      close(radius,50/std::min(1.0,ratio),0,"support radius");
      close(resample_weight(0,ratio),0.917347441*std::min(1.0,ratio),2e-16,
            "impulse coefficient at the time origin");
      for (unsigned k=0; k<=1000; ++k) {
         const double distance=radius*static_cast<double>(k)/1000;
         const double actual=resample_weight(distance,ratio);
         const double expected=static_cast<double>(weight_reference(distance,ratio));
         close(actual,expected,2e-15,"sinc/Kaiser coefficient accuracy");
         close(resample_weight(-distance,ratio),actual,0,"coefficient symmetry");
      }
      close(resample_weight(radius,ratio),0,0,"support endpoint is zero");
      close(resample_weight(2*radius,ratio),0,0,"outside support is zero");
   }
}

void check_lengths_and_shapes()
{
   using namespace pffdtd_post;
   require(resample_length(48000,48000,44100)==44100,"48000-to-44100 exact length");
   require(resample_length(44100,44100,48000)==48000,"44100-to-48000 exact length");
   require(resample_length(7,3,2)==4,"fractional length uses floor");
   require(resample_length(1000001,9000009000000002.0,4500009000000001.0)==500000,
           "floor remains exact one product unit below an integer");
   const std::uint64_t odd_counts[]={5,101,1001,1000001,100000001};
   for (std::uint64_t count : odd_counts) {
      const std::uint64_t base=(9007199254740992ULL-4)/count,k=(count+1)/2;
      // N*Fs_out = K*Fs_in +/- 1. These exact integer rates exercise floor
      // boundaries far closer than long double can distinguish by division.
      require(resample_length(static_cast<std::size_t>(count),
                 static_cast<double>(count*base+2),static_cast<double>(k*base+1))==k-1,
              "family of exact floors one product unit below an integer");
      require(resample_length(static_cast<std::size_t>(count),
                 static_cast<double>(count*base-2),static_cast<double>(k*base-1))==k,
              "family of exact floors one product unit above an integer");
      require(resample_length(static_cast<std::size_t>(count),
                 static_cast<double>(count*base),static_cast<double>(k*base))==k,
              "family of exact integral length ratios");
   }
   require(resample_length(1,1,2)==2,"one input sample may upsample");
   require(resample_length(8,1,std::nextafter(0.5,0.0))==3,"length just below threshold");
   require(resample_length(8,1,std::nextafter(0.5,1.0))==4,"length just above threshold");
   require(resample_length(100,1e300,2e300)==200,"large representable sample rates");
   require(resample_length(100,1e-300,2e-300)==200,"small representable sample rates");
   require(resample_length(7,std::numeric_limits<double>::denorm_min(),
                           std::numeric_limits<double>::denorm_min())==7,
           "equal subnormal rates are valid");
   const double bad[]={0,-1,std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()};
   for (double rate : bad) {
      invalid([&] { resample_length(100,rate,1); },"invalid input rate rejected");
      invalid([&] { resample_length(100,1,rate); },"invalid output rate rejected");
   }
   invalid([] { resample_length(0,1,1); },"zero input length rejected");
   invalid([] { resample_length(1,2,1); },"zero output length rejected");
   invalid([] { resample_length(100,1,1e100); },"unrepresentable upsampling ratio rejected");
   invalid([] { resample_length(100,1e100,1); },"unrepresentable FIR support rejected");
   invalid([] { resample_length(100,std::numeric_limits<double>::min(),
                                     std::numeric_limits<double>::max()); },
           "division overflow rejected");
   invalid([] { resample_length(100,std::numeric_limits<double>::max(),
                                     std::numeric_limits<double>::denorm_min()); },
           "division underflow rejected");
   invalid([] { resample_length(std::numeric_limits<std::size_t>::max(),1,1); },
           "unrepresentable input address or phase rejected");
   if (std::numeric_limits<std::size_t>::digits>=54)
      invalid([] { resample_length(static_cast<std::size_t>(9007199254740992ULL),1,2); },
              "unrepresentable output phase rejected");

   const std::vector<double> input={0.0,-0.0,1.0,-2.0,0.25,0.5};
   std::vector<double> copy=resample_serial(input,2,3,48000,48000);
   require(copy.size()==input.size() &&
           std::memcmp(copy.data(),input.data(),input.size()*sizeof(double))==0,
           "equal rate preserves all finite sample bits including signed zero");
   copy[0]=1;
   require(input[0]==0,"identity result owns an independent buffer");
   invalid([&] { resample_serial(input,0,3,1,1); },"zero channels rejected");
   invalid([&] { resample_serial(input,2,0,1,1); },"zero samples rejected");
   invalid([&] { resample_serial(input,3,3,1,1); },"wrong planar extent rejected");
   invalid([&] { resample_serial(input,std::numeric_limits<std::size_t>::max(),3,1,1); },
           "shape multiplication overflow rejected");
   invalid([&] { resample_serial(input,std::numeric_limits<std::size_t>::max()/8,1,1,1); },
           "unrepresentable byte extent rejected");
   for (double value : bad) {
      if (std::isfinite(value)) continue;
      std::vector<double> nonfinite=input; nonfinite[1]=value;
      invalid([&] { resample_serial(nonfinite,2,3,1,1); },
              "nonfinite samples rejected even in identity path");
   }
}

void check_impulse_reference_and_edges()
{
   using namespace pffdtd_post;
   const std::size_t samples=257;
   std::vector<double> input(2*samples,0);
   input[100]=1;
   input[samples+110]=-0.75;
   const double rates[]={0.5,1.5,160.0/147};
   for (double ratio : rates) {
      const std::vector<double> actual=resample_serial(input,2,samples,1,ratio);
      const std::vector<double> expected=gather_reference(input,2,samples,1,ratio);
      require(actual.size()==expected.size(),"reference planar extent");
      for (std::size_t n=0; n<actual.size(); ++n)
         close(actual[n],expected[n],3e-15,"independent long-double impulse reference");
   }
   const std::vector<double> result=resample_serial(input,2,samples,2,3);
   const std::size_t out=result.size()/2;
   for (std::size_t n=30; n+15<out; ++n)
      close(result[out+n+15],-0.75*result[n],2e-15,
            "integer input delay maps to exact output delay without channel mixing");
   const std::vector<double> zero(2*samples,0);
   for (double ratio : rates) {
      const std::vector<double> output=resample_serial(zero,2,samples,1,ratio);
      require(std::all_of(output.begin(),output.end(),[](double x) { return x==0; }),
              "zero input stays zero");
   }
   const std::vector<double> single=resample_serial({2.0},1,1,1,3);
   for (std::size_t n=0; n<single.size(); ++n)
      close(single[n],2*resample_weight(static_cast<double>(n)/3,3),1e-15,
            "one-sample signal has zero extension");
   const std::vector<double> constant(2048,1);
   const std::vector<double> half=resample_serial(constant,1,constant.size(),2,1);
   close(half[half.size()/2],1,2e-6,"interior DC response");
   close(half[0],(1+0.917347441*0.5)/2,2e-6,
         "DC edge attenuation follows zero extension without local normalization");
   require(half[0]<0.8,"edge attenuation must not be normalized away");
   // Length and radius checks accept extreme ratios when their phase, index
   // range and output count remain representable, without allocating a signal.
   const double tiny_ratio=50.0/1e12;
   const std::size_t nt=static_cast<std::size_t>(2.0/tiny_ratio);
   require(resample_length(nt,1,tiny_ratio)>=1,"very small representable ratio length");
   require(std::isfinite(resample_radius(tiny_ratio)),"very wide support remains finite");
}

void check_tone(double ratio, double cycles_per_input, bool passband)
{
   using namespace pffdtd_post;
   const std::size_t samples=2048;
   const double pi=std::acos(-1.0),phase=0.371;
   std::vector<double> input(samples);
   for (std::size_t n=0; n<samples; ++n)
      input[n]=std::sin(2*pi*cycles_per_input*static_cast<double>(n)+phase);
   const std::vector<double> output=resample_serial(input,1,samples,1,ratio);
   const double radius=resample_radius(ratio);
   double error_square=0,output_square=0;
   std::size_t used=0;
   for (std::size_t n=0; n<output.size(); ++n) {
      const double position=static_cast<double>(n)/ratio;
      if (position<=radius+2 || position>=samples-1-radius-2) continue;
      const double expected=passband ? std::sin(2*pi*cycles_per_input*position+phase) : 0;
      const double error=output[n]-expected;
      error_square+=error*error;
      output_square+=output[n]*output[n];
      ++used;
   }
   require(used>100,"enough interior tone samples");
   close(std::sqrt(error_square/used),0,3e-6,
         passband ? "passband sinusoid amplitude and phase" : "downsampling suppresses alias");
   if (passband)
      require(std::sqrt(output_square/used)>0.69,"passband signal has not been attenuated");
}

void check_dense_reference()
{
   using namespace pffdtd_post;
   const std::size_t samples=67,channels=3;
   std::vector<double> input(samples*channels);
   for (std::size_t n=0; n<input.size(); ++n)
      input[n]=static_cast<double>((n*37+11)%101-50.0)/64;
   const double ratios[]={0.125,0.37,1.25,1.999};
   for (double ratio : ratios) {
      const std::vector<double> actual=resample_serial(input,channels,samples,1,ratio);
      const std::vector<double> expected=gather_reference(input,channels,samples,1,ratio);
      require(actual.size()==expected.size(),"dense reference planar extent");
      for (std::size_t n=0; n<actual.size(); ++n)
         close(actual[n],expected[n],5e-15,"dense signal independent convolution reference");
   }
}

void check_upsampling_image()
{
   using namespace pffdtd_post;
   const std::size_t samples=4096;
   const double pi=std::acos(-1.0);
   std::vector<double> input(samples);
   for (std::size_t n=0; n<samples; ++n)
      input[n]=std::sin(2*pi*0.1*static_cast<double>(n));
   const std::vector<double> output=resample_serial(input,1,samples,1,2);
   // 6000 points contain an integer number of periods at both the desired
   // output frequency .05 and the image .45. Their projections are orthogonal
   // and the selected interval is entirely outside the FIR edge transients.
   double desired=0,image_sine=0,image_cosine=0;
   const std::size_t first=512,count=6000;
   for (std::size_t n=first; n<first+count; ++n) {
      const double phase=2*pi*static_cast<double>(n);
      desired+=output[n]*std::sin(0.05*phase);
      image_sine+=output[n]*std::sin(0.45*phase);
      image_cosine+=output[n]*std::cos(0.45*phase);
   }
   close(2*desired/count,1,3e-6,"upsampling preserves desired sinusoid");
   close(2*std::sqrt(image_sine*image_sine+image_cosine*image_cosine)/count,0,3e-6,
         "upsampling suppresses the spectral image");
}

} // namespace

int main()
{
   check_coefficients();
   check_lengths_and_shapes();
   check_impulse_reference_and_edges();
   check_dense_reference();
   check_upsampling_image();
   check_tone(0.5,0.10,true);
   check_tone(44100.0/48000,0.25,true);
   check_tone(1.5,0.25,true);
   check_tone(160.0/147,0.10,true);
   check_tone(0.5,0.4,false);
   check_tone(0.25,0.20,false);
   std::printf("PASS: native Kaiser FIR resampling (%zu checks)\n",cases);
   return EXIT_SUCCESS;
}
