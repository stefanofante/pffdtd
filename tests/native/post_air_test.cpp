// Independent ISO formulae, direct transforms and legacy time-loop recurrences.
#include "post_air.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

typedef std::complex<long double> Complex;
const long double pi=std::acos(-1.0L);
std::size_t checks=0;
double fft_error=0,stokes_error=0,ola_error=0,modal_error=0,fast_modal_error=0,fast_long_double_error=0;

void require(bool condition,const char* message)
{
   if (!condition) { std::fprintf(stderr,"FAIL: %s\n",message); std::exit(EXIT_FAILURE); }
   ++checks;
}

template<typename Operation> void invalid(Operation operation)
{
   bool caught=false;
   try { operation(); } catch (const std::invalid_argument&) { caught=true; }
   require(caught,"invalid air-filter input was accepted");
}

struct IsoReference {
   long double c,gamma,frO,frN,almO,almN,classical;
};

IsoReference iso(const pffdtd_post::AirOptions& options)
{
   const long double tk=static_cast<long double>(options.temperature)+273.15L;
   const long double ratio=tk/293.15L,p=options.pressure/101.325L;
   const long double vapor=options.humidity*std::pow(10.0L,-6.8346L*std::pow(273.16L/tk,1.261L)+4.6151L)/p;
   const long double oxygen=p*(24+40400*vapor*(0.02L+vapor)/(0.391L+vapor));
   const long double nitrogen=p/std::sqrt(ratio)*(9+280*vapor*std::exp(-4.17L*(std::pow(ratio,-1.0L/3)-1)));
   const long double scale=40*pi/(35*std::log(10.0L));
   const long double almO=scale*0.209L*std::pow(2239.1L/tk,2)*std::exp(-2239.1L/tk);
   const long double almN=scale*0.781L*std::pow(3352.0L/tk,2)*std::exp(-3352.0L/tk);
   return {343.2L*std::sqrt(ratio),almO*std::log(10.0L)/(20*pi*pi*oxygen),oxygen,nitrogen,
           almO,almN,1.6e-10L*std::sqrt(ratio)/p};
}

long double absorption(long double frequency,const IsoReference& coefficients)
{
   const long double f2=frequency*frequency;
   const long double vibrational=2*f2/coefficients.c*(
      coefficients.almO*coefficients.frO/(coefficients.frO*coefficients.frO+f2)+
      coefficients.almN*coefficients.frN/(coefficients.frN*coefficients.frN+f2));
   return (coefficients.classical*f2+vibrational)*std::log(10.0L)/20;
}

void iso_contracts()
{
   pffdtd_post::AirOptions options;
   for (double temperature : {-20.0,0.0,20.0,50.0})
      for (double humidity : {10.0,50.0,100.0})
         for (double pressure : {70.0,101.325,150.0,200.0}) {
            options.temperature=temperature; options.humidity=humidity; options.pressure=pressure;
            const auto actual=pffdtd_post::air_coefficients(options);
            const auto expected=iso(options);
            require(std::abs(actual.c-expected.c)<1e-12L,"ISO sound speed differs");
            require(std::abs(actual.frO-expected.frO)<2e-12L*expected.frO,"ISO oxygen relaxation differs");
            require(std::abs(actual.frN-expected.frN)<2e-12L*expected.frN,"ISO nitrogen relaxation differs");
            require(std::abs(actual.gamma_p-expected.gamma)<2e-12L*expected.gamma,"Stokes coefficient differs");
            long double previous=0;
            for (double frequency : {0.0,1.0,100.0,1000.0,10000.0,24000.0,48000.0}) {
               const double value=pffdtd_post::air_absorption_np(frequency,actual);
               const long double reference=absorption(frequency,expected);
               require(std::isfinite(value) && value>=0 && value>=previous,"ISO absorption is not passive/monotonic");
               require(std::abs(value-reference)<=2e-15L+2e-12L*reference,"ISO absorption differs from independent form");
               if (frequency==0) require(value==0,"ISO zero-frequency absorption is not exactly zero");
               previous=value;
               // Independent ISO9613-1 Eq.(5); the legacy appendix constants
               // differ slightly, hence its documented one-percent gate.
               const long double tk=temperature+273.15L,tr=tk/293.15L,p=pressure/101.325L,f2=frequency*frequency;
               const long double equation5=8.686L*f2*(1.84e-11L*std::sqrt(tr)/p+
                  std::pow(tr,-2.5L)*(0.01275L*std::exp(-2239.1L/tk)/(expected.frO+f2/expected.frO)+
                                     0.1068L*std::exp(-3352.0L/tk)/(expected.frN+f2/expected.frN)))*std::log(10.0L)/20;
               require(std::abs(reference-equation5)<=0.01L*equation5+1e-20L,"ISO appendix disagrees with Eq.(5)");
            }
         }
   options=pffdtd_post::AirOptions();
   require(pffdtd_post::air_coefficients(options).c==343.2,"standard-temperature sound speed differs");
   const double standard=pffdtd_post::air_absorption_np(1000,pffdtd_post::air_coefficients(options));
   options.pressure=70;
   require(std::abs(pffdtd_post::air_absorption_np(1000,pffdtd_post::air_coefficients(options))-standard)>1e-6,
           "pressure option was ignored");
}

std::vector<Complex> dft(const std::vector<Complex>& input,bool inverse)
{
   const std::size_t length=input.size();
   std::vector<Complex> result(length);
   for (std::size_t k=0; k<length; ++k)
      for (std::size_t n=0; n<length; ++n) {
         const long double angle=(inverse ? 2 : -2)*pi*k*n/length;
         result[k]+=input[n]*Complex(std::cos(angle),std::sin(angle))/(inverse ? static_cast<long double>(length) : 1);
      }
   return result;
}

void transform_contracts()
{
   for (std::size_t length : {0u,1u,2u,3u,5u,7u,16u,31u,32u,97u,127u,257u,1009u}) {
      std::vector<std::complex<double>> values(length);
      std::vector<Complex> reference(length);
      std::vector<double> real(length);
      for (std::size_t n=0; n<length; ++n) {
         values[n]=std::complex<double>(static_cast<double>((n*13+7)%29)/32-.25,
                                       static_cast<double>((n*7+3)%17)/32-.125);
         reference[n]=Complex(values[n].real(),values[n].imag()); real[n]=values[n].real();
      }
      const auto expected=dft(reference,false);
      pffdtd_post::native_fft(values);
      for (std::size_t k=0; k<length; ++k) {
         const double error=static_cast<double>(std::abs(Complex(values[k].real(),values[k].imag())-expected[k]));
         fft_error=std::max(fft_error,error);
         require(error<2e-10,"native FFT differs from direct DFT");
      }
      pffdtd_post::native_fft(values,true);
      for (std::size_t n=0; n<length; ++n)
         require(std::abs(values[n]-std::complex<double>(static_cast<double>(reference[n].real()),static_cast<double>(reference[n].imag())))<1e-12,
                 "native FFT roundtrip differs");
      const auto original=real;
      pffdtd_post::dct2_ortho(real);
      for (std::size_t q=0; q<length; ++q) {
         long double expected_mode=0;
         for (std::size_t n=0; n<length; ++n) expected_mode+=original[n]*std::cos(pi*(n+0.5L)*q/length);
         expected_mode*=q ? std::sqrt(2.0L/length) : 1/std::sqrt(static_cast<long double>(length));
         require(std::abs(real[q]-expected_mode)<2e-12,"native orthonormal DCT differs");
      }
      pffdtd_post::idct2_ortho(real);
      for (std::size_t n=0; n<length; ++n) require(std::abs(real[n]-original[n])<1e-12,"DCT roundtrip differs");
   }
}

std::vector<double> signal(std::size_t channels,std::size_t samples)
{
   std::vector<double> input(channels*samples);
   for (std::size_t channel=0; channel<channels; ++channel)
      for (std::size_t n=0; n<samples; ++n)
         input[channel*samples+n]=channel==0 ? (n==0 ? 1 : (n+1==samples ? -0.5 : 0)) :
            (channel==1 ? static_cast<double>((n*17+3)%43)/64-.25 : 0.125);
   return input;
}

void compare(const pffdtd_post::AirResult& actual,const std::vector<long double>& expected,
             std::size_t samples,double& largest,const char* message)
{
   require(actual.samples==samples && actual.data.size()==expected.size(),"air output shape differs");
   for (std::size_t n=0; n<expected.size(); ++n) {
      const double error=static_cast<double>(std::abs(actual.data[n]-expected[n]));
      largest=std::max(largest,error);
      require(std::isfinite(actual.data[n]) && error<2e-9+2e-9*std::abs(expected[n]),message);
   }
}

void stokes_contracts()
{
   pffdtd_post::AirOptions options; options.mode="stokes";
   for (double fs : {32000.0,48000.0,96000.0})
      for (std::size_t samples : {1u,2u,17u,257u,1025u,4097u}) {
         const auto input=signal(3,samples); const auto coefficients=iso(options);
         const long double ts=1/static_cast<long double>(fs),g=coefficients.gamma;
         const std::size_t first=std::min(samples,static_cast<std::size_t>(std::ceil(ts/(2*pi*g))));
         const long double width_factor=0.1L*std::log(10.0L)*options.stokes_db*g*ts;
         const std::size_t output_samples=samples+static_cast<std::size_t>(std::ceil(std::sqrt(width_factor*(samples-1))/ts));
         std::vector<long double> expected(3*output_samples);
         for (std::size_t channel=0; channel<3; ++channel) {
            for (std::size_t n=0; n<first; ++n) expected[channel*output_samples+n]=input[channel*samples+n];
            for (std::size_t n=first; n<samples; ++n) {
               const std::size_t width=static_cast<std::size_t>(std::ceil(std::sqrt(width_factor*n)/ts));
               const long double factor=ts/std::sqrt(n*2*ts*g*pi)*input[channel*samples+n];
               for (std::size_t out=n-width; out<=n+width; ++out) {
                  const long double offset=(static_cast<long double>(n)-out)*ts;
                  expected[channel*output_samples+out]+=factor*std::exp(-offset*offset/(n*2*ts*g));
               }
            }
         }
         const auto actual=pffdtd_post::process_air_cpu(input,3,samples,fs,options);
         compare(actual,expected,output_samples,stokes_error,"Stokes Gaussian scatter differs");
         if (first==samples)
            for (std::size_t channel=0; channel<3; ++channel)
               for (std::size_t n=0; n<samples; ++n)
                  require(actual.data[channel*output_samples+n]==input[channel*samples+n],"Stokes short-input early copy differs");
      }
}

void ola_contracts()
{
   pffdtd_post::AirOptions options; options.mode="OLA";
   for (std::size_t window : {4u,8u,16u,18u,32u})
      for (std::size_t samples : {1u,7u,33u,65u}) {
         options.window=window; const double fs=48000; const auto input=signal(3,samples);
         const std::size_t base=window/4,remainder=window%4;
         const std::size_t hop=base+(remainder>2 || (remainder==2 && base%2));
         std::size_t fft_length=1; while (fft_length<window) fft_length*=2;
         const std::size_t frames=(samples+window+hop-1)/hop,output_samples=(frames-1)*hop;
         const auto coefficients=iso(options);
         std::vector<long double> expected(3*output_samples),wa(window),ws(window);
         for (std::size_t n=0; n<window; ++n) {
            wa[n]=(1-std::cos(2*pi*n/window))/2; ws[n]=wa[n]/(3.0L/8*window/hop);
         }
         for (std::size_t channel=0; channel<3; ++channel) {
            std::vector<long double> padded(window+output_samples),output(padded.size());
            for (std::size_t n=0; n<samples; ++n) padded[window+n]=input[channel*samples+n];
            for (std::size_t frame=0; frame<frames; ++frame) {
               const std::size_t start=frame*hop;
               const long double distance=coefficients.c/fs*(static_cast<long double>(start)-window/2.0L);
               if (distance<0) {
                  for (std::size_t n=0; n<window; ++n) output[start+n]+=ws[n]*padded[start+n];
               } else {
                  std::vector<Complex> data(fft_length);
                  for (std::size_t n=0; n<window; ++n) data[n]=wa[n]*padded[start+n];
                  auto spectrum=dft(data,false);
                  for (std::size_t k=0; k<fft_length; ++k)
                     spectrum[k]*=std::exp(-absorption(static_cast<long double>(std::min(k,fft_length-k))*fs/fft_length,coefficients)*distance);
                  const auto filtered=dft(spectrum,true);
                  for (std::size_t n=0; n<window; ++n) output[start+n]+=ws[n]*filtered[n].real();
               }
            }
            for (std::size_t n=0; n<output_samples; ++n) expected[channel*output_samples+n]=output[window+n];
         }
         compare(pffdtd_post::process_air_cpu(input,3,samples,fs,options),expected,output_samples,ola_error,
                 "OLA differs from independent direct-DFT/window/padding reference");
         const auto plan=pffdtd_post::air_ola_plan(samples,fs,options);
         require(plan.hop==hop && plan.frames==frames && plan.pad>=window-hop && plan.pad<window,"OLA metadata differs");
      }
   options.window=4;
   const auto early=pffdtd_post::process_air_cpu({1},1,1,1e-7,options);
   require(std::abs(early.data[0]-7.0/6)<1e-10,"OLA negative-distance bypass contract differs");
}

void modal_contracts()
{
   pffdtd_post::AirOptions options; options.mode="modal";
   const double fs=48000;
   for (std::size_t samples : {1u,2u,3u,7u,16u,31u,64u,97u,129u,257u})
      for (std::size_t pad : {0u,3u,17u}) {
         options.modal_pad=static_cast<double>(pad)/fs;
         const std::size_t length=samples+static_cast<std::size_t>(std::ceil(options.modal_pad/(1.0/fs)));
         const auto input=signal(3,samples); const auto coefficients=iso(options);
         std::vector<long double> a1(length),a2(length),f1(length),f2(length);
         for (std::size_t q=0; q<length; ++q) {
            const long double theta=pi*q/length,sigma=coefficients.c*absorption(static_cast<long double>(q)*fs/(2*length),coefficients)/fs;
            const long double fm=q ? std::sqrt(2.0L/length)*std::cos(theta/2) : 1/std::sqrt(static_cast<long double>(length));
            a1[q]=2*std::exp(-sigma)*std::cos(theta); a2[q]=-std::exp(-2*sigma);
            f1[q]=fm*(1+sigma/2)/(1+sigma); f2[q]=fm*(1-sigma/2)/(1+sigma);
         }
         std::vector<long double> expected(3*length);
         for (std::size_t channel=0; channel<3; ++channel) {
            std::vector<long double> p0(length),p1(length),padded(length),u(length+1);
            for (std::size_t n=0; n<samples; ++n) padded[n]=input[channel*samples+n];
            for (std::size_t n=0; n<length; ++n) u[n+1]=padded[length-1-n];
            for (std::size_t t=0; t<length; ++t) {
               for (std::size_t q=0; q<length; ++q) p0[q]=a1[q]*p1[q]+a2[q]*p0[q]+f1[q]*u[t+1]-f2[q]*u[t];
               if (t+1<length) p0.swap(p1);
            }
            for (std::size_t n=0; n<length; ++n) {
               long double value=p0[0]/std::sqrt(static_cast<long double>(length));
               for (std::size_t q=1; q<length; ++q) value+=std::sqrt(2.0L/length)*p0[q]*std::cos(pi*(n+0.5L)*q/length);
               expected[channel*length+n]=value;
            }
         }
         const auto actual=pffdtd_post::process_air_cpu(input,3,samples,fs,options);
         compare(actual,expected,length,modal_error,"Modal differs from independent legacy recurrence/direct IDCT");
         for (std::size_t channel=0; channel<3; ++channel) {
            long double source_sum=0,output_sum=0;
            for (std::size_t n=0; n<samples; ++n) source_sum+=input[channel*samples+n];
            for (std::size_t n=0; n<length; ++n) output_sum+=actual.data[channel*length+n];
            require(std::abs(source_sum-output_sum)<2e-10L,"modal zero mode did not preserve DC sum");
         }
      }
}

void invalid_contracts()
{
   pffdtd_post::AirOptions options;
   const auto input=signal(2,17);
   const auto identity=pffdtd_post::process_air_cpu(input,2,17,48000,options);
   require(identity.samples==17 && identity.data==input,"none air mode did not bypass exactly");
   options.mode="unknown";
   invalid([&]{pffdtd_post::process_air_cpu(input,2,17,48000,options);});
   options=pffdtd_post::AirOptions();
   invalid([&]{pffdtd_post::process_air_cpu(input,1,17,48000,options);});
   invalid([&]{pffdtd_post::process_air_cpu({},0,17,48000,options);});
   invalid([&]{pffdtd_post::process_air_cpu({},1,0,48000,options);});
   const double nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity();
   for (double fs : {0.0,-1.0,nan,inf}) invalid([&]{pffdtd_post::process_air_cpu(input,2,17,fs,options);});
   invalid([&]{pffdtd_post::process_air_cpu({nan},1,1,48000,options);});
   for (double temperature : {-21.0,51.0,nan,inf}) { options.temperature=temperature; invalid([&]{pffdtd_post::air_coefficients(options);}); }
   options=pffdtd_post::AirOptions();
   for (double humidity : {9.0,101.0,nan,inf}) { options.humidity=humidity; invalid([&]{pffdtd_post::air_coefficients(options);}); }
   options=pffdtd_post::AirOptions();
   for (double pressure : {0.0,-1.0,201.0,nan,inf}) { options.pressure=pressure; invalid([&]{pffdtd_post::air_coefficients(options);}); }
   options=pffdtd_post::AirOptions();
   const auto coefficients=pffdtd_post::air_coefficients(options);
   for (double frequency : {-1.0,nan,inf}) invalid([&]{pffdtd_post::air_absorption_np(frequency,coefficients);});
   for (double db : {0.0,-1.0,nan,inf}) { options.stokes_db=db; invalid([&]{pffdtd_post::air_stokes_plan(17,48000,options);}); }
   options=pffdtd_post::AirOptions();
   for (double pad : {-1.0,nan,inf}) { options.modal_pad=pad; invalid([&]{pffdtd_post::air_modal_plan(17,48000,options);}); }
   options=pffdtd_post::AirOptions(); options.window=3;
   invalid([&]{pffdtd_post::air_ola_plan(17,48000,options);});
   options.window=std::numeric_limits<std::size_t>::max();
   invalid([&]{pffdtd_post::air_ola_plan(17,48000,options);});
   options=pffdtd_post::AirOptions();
   invalid([&]{pffdtd_post::air_modal_plan(std::numeric_limits<std::size_t>::max(),48000,options);});
   options.modal_method="unknown";
   invalid([&]{pffdtd_post::air_modal_plan(17,48000,options);});
   options=pffdtd_post::AirOptions();
   for (double tolerance : {0.0,-1.0,nan,inf}) {
      options.modal_tolerance=tolerance;
      invalid([&]{pffdtd_post::air_modal_plan(17,48000,options);});
   }
}

void fast_modal_contracts()
{
   pffdtd_post::AirOptions options; options.mode="modal";
   std::size_t fast_cases=0,largest_nodes=0;
   for (std::size_t samples : {1025u,2048u,4097u,8192u})
      for (unsigned atmosphere=0; atmosphere<2; ++atmosphere) {
         options.temperature=atmosphere ? 50 : 20;
         options.humidity=atmosphere ? 10 : 50;
         options.pressure=atmosphere ? 70 : 101.325;
         const double fs=atmosphere ? 96000 : 48000;
         options.modal_pad=17.0/fs;
         const auto input=signal(3,samples);
         const auto modal=pffdtd_post::air_modal_plan(samples,fs,options);
         const double norm=pffdtd_post::air_max_channel_l1(input,3,samples);
         const auto plan=pffdtd_post::air_modal_fft_plan(modal,samples,norm,options.modal_tolerance);
         require(plan.usable,"large modal test did not select the FFT method");
         require(plan.nodes.size()<=4096 && plan.output_error_bound<=options.modal_tolerance,
                 "modal interpolation planner exceeded its error/cost contract");
         largest_nodes=std::max(largest_nodes,plan.nodes.size());
         for (std::size_t q=1; q<modal.samples; q+=std::max(std::size_t(1),modal.samples/100)) {
            long double sum=0;
            for (std::size_t j=0; j<plan.nodes.size(); ++j)
               sum+=pffdtd_post::air_modal_fft_weight(plan,modal,q,j);
            require(std::abs(sum-1)<2e-14L,"Chebyshev interpolation weights do not sum to one");
            for (std::size_t n : {std::size_t(0),std::size_t(1),std::size_t(257),samples-1}) {
               long double estimate=0;
               for (std::size_t j=0; j<plan.nodes.size(); ++j)
                  estimate+=pffdtd_post::air_modal_fft_weight(plan,modal,q,j)*
                     std::exp(-static_cast<long double>(plan.nodes[j])*n);
               const long double expected=std::exp(-static_cast<long double>(modal.sigma[q])*n);
               require(std::abs(estimate-expected)<=plan.kernel_error_bound+2e-14L,
                       "Chebyshev exponential kernel exceeded its analytical tail bound plus roundoff");
            }
            const long double theta=modal.omega[q],sigma=modal.sigma[q];
            const long double expected_sine=(static_cast<long double>(modal.f1[q])*std::cos(theta)-
               static_cast<long double>(modal.f2[q])*std::exp(sigma))/std::sin(theta);
            require(std::abs(plan.sine_coeff[q]-expected_sine)<2e-11L,
                    "stable modal closed-form coefficient differs");
         }
         const auto recurrence=pffdtd_post::process_air_cpu(input,3,samples,fs,options);
         options.modal_method="fft";
         const auto fast=pffdtd_post::process_air_cpu(input,3,samples,fs,options);
         options.modal_method="recurrence";
         require(fast.samples==recurrence.samples && fast.data.size()==recurrence.data.size(),"modal FFT output shape differs");
         for (std::size_t n=0; n<fast.data.size(); ++n) {
            const double error=std::abs(fast.data[n]-recurrence.data[n]);
            fast_modal_error=std::max(fast_modal_error,error);
            require(error<2e-8+2e-8*std::abs(recurrence.data[n]),
                    "modal FFT differs from recurrence beyond the floating-point validation gate");
         }
         if (samples==8192 && atmosphere==0) {
            // Recompute poles and state in long double to distinguish exact
            // modal/FFT algebra from accumulated double-recurrence roundoff.
            const auto coefficients=iso(options);
            const std::size_t length=modal.samples;
            std::vector<double> transformed(length);
            double recurrence_long_double_error=0;
            for (std::size_t channel=0; channel<3; ++channel) {
               long double sum=0;
               for (std::size_t n=0; n<samples; ++n) sum+=input[channel*samples+n];
               transformed[0]=static_cast<double>(sum/std::sqrt(static_cast<long double>(length)));
               for (std::size_t q=1; q<length; ++q) {
                  const long double theta=pi*q/length;
                  const long double sigma=coefficients.c*absorption(static_cast<long double>(q)*fs/(2*length),coefficients)/fs;
                  const long double fm=std::sqrt(2.0L/length)*std::cos(theta/2);
                  const long double a1=2*std::exp(-sigma)*std::cos(theta),a2=-std::exp(-2*sigma);
                  const long double f1=fm*(1+sigma/2)/(1+sigma),f2=fm*(1-sigma/2)/(1+sigma);
                  long double current=0,previous=0,source_previous=0;
                  for (std::size_t t=0; t<samples; ++t) {
                     const long double source=input[channel*samples+samples-1-t];
                     const long double next=a1*current+a2*previous+f1*source-f2*source_previous;
                     previous=current; current=next; source_previous=source;
                  }
                  transformed[q]=static_cast<double>(current);
               }
               // FFT/DCT conventions were independently gated by direct sums
               // above; use the same final inverse to isolate the state error.
               pffdtd_post::idct2_ortho(transformed);
               for (std::size_t n=0; n<length; ++n) {
                  const double error=std::abs(transformed[n]-fast.data[channel*length+n]);
                  fast_long_double_error=std::max(fast_long_double_error,error);
                  recurrence_long_double_error=std::max(recurrence_long_double_error,
                     std::abs(transformed[n]-recurrence.data[channel*length+n]));
                  require(error<2e-11,"modal FFT differs from long-double recurrence beyond its precision gate");
               }
            }
            std::printf("PASS: long modal state reference; FFT error %.3e, double recurrence error %.3e\n",
                        fast_long_double_error,recurrence_long_double_error);
         }
         ++fast_cases;
      }
   options=pffdtd_post::AirOptions(); options.mode="modal";
   const auto small=signal(1,17);
   const auto modal=pffdtd_post::air_modal_plan(17,48000,options);
   require(!pffdtd_post::air_modal_fft_plan(modal,17,pffdtd_post::air_max_channel_l1(small,1,17),1e-12).usable,
           "modal FFT did not retain the small-input recurrence fallback");
   const auto reference=pffdtd_post::process_air_cpu(small,1,17,48000,options);
   options.modal_method="fft";
   require(pffdtd_post::process_air_cpu(small,1,17,48000,options).data==reference.data,
           "modal FFT fallback differs from recurrence");
   std::printf("PASS: %zu fast modal cases, at most %zu Chebyshev FFT nodes; error %.3e\n",
               fast_cases,largest_nodes,fast_modal_error);
}

} // namespace

int main()
{
   iso_contracts(); transform_contracts(); stokes_contracts(); ola_contracts(); modal_contracts(); invalid_contracts(); fast_modal_contracts();
   std::printf("PASS: native air/FFT contracts (%zu checks); max errors FFT %.3e, Stokes %.3e, OLA %.3e, modal %.3e\n",
               checks,fft_error,stokes_error,ola_error,modal_error);
   return EXIT_SUCCESS;
}
