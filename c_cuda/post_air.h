// Native air attenuation matching the legacy Stokes, STFT/OLA and modal
// algorithms. Pressure is used in the ISO9613 humidity/relaxation formulae.
#ifndef PFFDTD_POST_AIR_H
#define PFFDTD_POST_AIR_H

#include "native_fft.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace pffdtd_post {

struct AirOptions {
   std::string mode="none";
   double temperature=20.0,humidity=50.0,pressure=101.325;
   double stokes_db=120.0,modal_pad=0.0;
   std::size_t window=1024;
   std::string modal_method="recurrence";
   double modal_tolerance=1e-12;
};

struct AirResult {
   std::vector<double> data;
   std::size_t samples=0;
};

struct AirCoefficients {
   double c,gamma_p,gamma,frO,frN,almO,almN,classical_db;
};

namespace air_detail {

inline std::size_t add(std::size_t a,std::size_t b)
{
   if (b>std::numeric_limits<std::size_t>::max()-a)
      throw std::invalid_argument("Air-filter sample count overflows");
   return a+b;
}

inline std::size_t product(std::size_t a,std::size_t b)
{
   if (b && a>std::numeric_limits<std::size_t>::max()/b)
      throw std::invalid_argument("Air-filter element count overflows");
   const std::size_t count=a*b;
   if (count>std::numeric_limits<std::size_t>::max()/sizeof(double))
      throw std::invalid_argument("Air-filter byte count overflows");
   return count;
}

inline std::size_t ceiling(double value)
{
   if (!std::isfinite(value) || value<0 ||
       static_cast<long double>(std::ceil(value))>
          static_cast<long double>(std::numeric_limits<std::size_t>::max()))
      throw std::invalid_argument("Air-filter sample count is not representable");
   return static_cast<std::size_t>(std::ceil(value));
}

inline void rate(double fs)
{
   if (!std::isfinite(fs) || fs<=0 || !std::isfinite(1.0/fs) || 1.0/fs<=0)
      throw std::invalid_argument("Air-filter sample rate must be positive and finite");
}

inline void finite_result(const std::vector<double>& values)
{
   for (double value : values)
      if (!std::isfinite(value))
         throw std::runtime_error("Air filter produced a nonfinite sample");
}

} // namespace air_detail

inline std::string air_mode(const AirOptions& options)
{
   std::string mode=options.mode;
   for (char& c : mode) c=static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
   if (mode!="none" && mode!="stokes" && mode!="ola" && mode!="modal")
      throw std::invalid_argument("Air mode must be none, stokes, ola or modal");
   return mode;
}

inline AirCoefficients air_coefficients(const AirOptions& options)
{
   if (!std::isfinite(options.temperature) || options.temperature<-20 || options.temperature>50 ||
       !std::isfinite(options.humidity) || options.humidity<10 || options.humidity>100 ||
       !std::isfinite(options.pressure) || options.pressure<=0 || options.pressure>200)
      throw std::invalid_argument("ISO9613 requires temperature -20..50 C, humidity 10..100%, pressure (0,200] kPa");
   const double pi=std::acos(-1.0),tk=options.temperature+273.15;
   const double tr=tk/293.15,p=options.pressure/101.325;
   const double c=343.2*std::sqrt(tr);
   // Relative humidity times saturation pressure, divided by ambient pressure.
   // At standard pressure this equals the legacy expression with p fixed to 1.
   const double h=options.humidity*std::pow(10.0,-6.8346*std::pow(273.16/tk,1.261)+4.6151)/p;
   const double frO=p*(24+4.04e4*h*(0.02+h)/(0.391+h));
   const double frN=p*std::pow(tr,-0.5)*(9+280*h*std::exp(-4.17*(std::pow(tr,-1.0/3)-1)));
   const double factor=2*pi/35*(10*std::log10(std::exp(2.0)));
   const double almO=factor*0.209*std::pow(2239.1/tk,2)*std::exp(-2239.1/tk);
   const double almN=factor*0.781*std::pow(3352.0/tk,2)*std::exp(-3352.0/tk);
   AirCoefficients result={c,almO/(pi*pi*frO)*std::log(10.0)/20,
      std::log(10.0)*1.6e-11/(4*pi*pi)*c*std::sqrt(tr)/p,
      frO,frN,almO,almN,1.6e-10*std::sqrt(tr)/p};
   if (!std::isfinite(result.c) || !std::isfinite(result.gamma_p) || result.gamma_p<=0 ||
       !std::isfinite(result.gamma) || result.gamma<=0 || !std::isfinite(frO) || frO<=0 ||
       !std::isfinite(frN) || frN<=0 || !std::isfinite(result.classical_db))
      throw std::invalid_argument("ISO9613 coefficients are not representable");
   return result;
}

inline double air_absorption_np(double frequency,const AirCoefficients& coefficients)
{
   if (!std::isfinite(frequency) || frequency<0)
      throw std::invalid_argument("ISO9613 frequency must be finite and nonnegative");
   const double rO=frequency/coefficients.frO,rN=frequency/coefficients.frN;
   const double db=coefficients.classical_db*frequency*frequency+
      coefficients.almO*(frequency/coefficients.c)*(2*rO/(1+rO*rO))+
      coefficients.almN*(frequency/coefficients.c)*(2*rN/(1+rN*rN));
   const double result=db*std::log(10.0)/20;
   if (!std::isfinite(result) || result<0)
      throw std::invalid_argument("ISO9613 absorption is not representable");
   return result;
}

struct AirStokesPlan {
   std::size_t samples,first;
   double ts,gamma,width_factor;
};

inline std::size_t air_stokes_width(const AirStokesPlan& plan,std::size_t n)
{
   return air_detail::ceiling(std::sqrt(plan.width_factor*static_cast<double>(n))/plan.ts);
}

inline AirStokesPlan air_stokes_plan(std::size_t samples,double fs,const AirOptions& options)
{
   air_detail::rate(fs);
   if (!samples || !std::isfinite(options.stokes_db) || options.stokes_db<=0)
      throw std::invalid_argument("Stokes requires samples and a positive finite truncation level");
   const double pi=std::acos(-1.0),ts=1.0/fs,g=air_coefficients(options).gamma_p;
   AirStokesPlan plan={0,0,ts,g,0.1*std::log(10.0)*options.stokes_db*g*ts};
   const double first=ts/(2*pi*g);
   if (!std::isfinite(first) || first>=static_cast<double>(samples)) plan.first=samples;
   else plan.first=air_detail::ceiling(first);
   if (!plan.first) throw std::invalid_argument("Stokes start cannot be represented at this sample rate");
   const std::size_t padding=air_stokes_width(plan,samples-1);
   plan.samples=air_detail::add(samples,padding);
   air_detail::product(1,plan.samples);
   if (plan.first<samples && air_stokes_width(plan,plan.first)>plan.first)
      throw std::invalid_argument("Stokes kernel extends before time zero at this sample rate");
   return plan;
}

struct AirOlaPlan {
   std::size_t samples,window,hop,fft_size,frames,pad;
   double c;
   std::vector<double> analysis,synthesis,absorption;
};

inline AirOlaPlan air_ola_plan(std::size_t samples,double fs,const AirOptions& options)
{
   air_detail::rate(fs);
   if (!samples || options.window<4)
      throw std::invalid_argument("OLA requires samples and a window of at least four samples");
   const AirCoefficients coefficients=air_coefficients(options);
   AirOlaPlan plan={0,options.window,options.window/4,1,0,0,coefficients.c,{},{},{}};
   // NumPy iround uses ties-to-even; compute round(window/4) with integers.
   const std::size_t remainder=options.window%4;
   if (remainder>2 || (remainder==2 && plan.hop%2)) ++plan.hop;
   while (plan.fft_size<plan.window) {
      if (plan.fft_size>std::numeric_limits<std::size_t>::max()/2)
         throw std::invalid_argument("OLA FFT length overflows");
      plan.fft_size*=2;
   }
   const std::size_t framed=air_detail::add(samples,plan.window);
   plan.frames=framed/plan.hop+(framed%plan.hop!=0);
   plan.samples=air_detail::product(plan.frames-1,plan.hop);
   plan.pad=plan.samples-samples;
   if (plan.pad<plan.window-plan.hop || plan.pad>=plan.window)
      throw std::invalid_argument("OLA padding is inconsistent");
   air_detail::product(1,air_detail::add(plan.window,plan.samples));
   air_detail::product(2,plan.fft_size);
   plan.analysis.resize(plan.window); plan.synthesis.resize(plan.window);
   plan.absorption.resize(plan.fft_size/2+1);
   const double pi=std::acos(-1.0),normalization=3.0/8*static_cast<double>(plan.window)/plan.hop;
   for (std::size_t n=0; n<plan.window; ++n) {
      const double weight=0.5*(1-std::cos(2*pi*static_cast<double>(n)/plan.window));
      plan.analysis[n]=weight; plan.synthesis[n]=weight/normalization;
   }
   for (std::size_t k=0; k<plan.absorption.size(); ++k)
      plan.absorption[k]=air_absorption_np(fs*(static_cast<double>(k)/plan.fft_size),coefficients);
   return plan;
}

struct AirModalPlan {
   std::size_t samples;
   std::vector<double> a1,a2,f1,f2,sigma,omega;
};

inline AirModalPlan air_modal_plan(std::size_t samples,double fs,const AirOptions& options)
{
   air_detail::rate(fs);
   if (!samples || !std::isfinite(options.modal_pad) || options.modal_pad<0 ||
       (options.modal_method!="recurrence" && options.modal_method!="fft") ||
       !std::isfinite(options.modal_tolerance) || options.modal_tolerance<=0)
      throw std::invalid_argument("Invalid modal padding, method or tolerance");
   const AirCoefficients coefficients=air_coefficients(options);
   AirModalPlan plan;
   plan.samples=air_detail::add(samples,air_detail::ceiling(options.modal_pad/(1.0/fs)));
   air_detail::product(1,plan.samples);
   plan.a1.resize(plan.samples); plan.a2.resize(plan.samples);
   plan.f1.resize(plan.samples); plan.f2.resize(plan.samples);
   plan.sigma.resize(plan.samples); plan.omega.resize(plan.samples);
   const double pi=std::acos(-1.0),ts=1.0/fs;
   for (std::size_t q=0; q<plan.samples; ++q) {
      const double theta=pi*(static_cast<double>(q)/plan.samples);
      const double sigma=coefficients.c*air_absorption_np(fs*(0.5*static_cast<double>(q)/plan.samples),coefficients)*ts;
      const double fm=q==0 ? 1/std::sqrt(static_cast<double>(plan.samples)) :
         std::sqrt(2.0/plan.samples)*std::cos(theta/2);
      plan.a1[q]=2*std::exp(-sigma)*std::cos(theta);
      plan.a2[q]=-std::exp(-2*sigma);
      plan.sigma[q]=sigma; plan.omega[q]=theta;
      plan.f1[q]=fm*(1+sigma/2)/(1+sigma);
      plan.f2[q]=fm*(1-sigma/2)/(1+sigma);
      if (!std::isfinite(plan.f1[q]) || !std::isfinite(plan.f2[q]))
         throw std::invalid_argument("Modal coefficients are not representable");
   }
   return plan;
}

struct AirModalFftPlan {
   bool usable=false;
   double sigma_max=0,kernel_error_bound=0,output_error_bound=0;
   std::vector<double> nodes,barycentric,normalizers,real_coeff,sine_coeff;
   std::vector<int> exact_node;
};

inline double air_max_channel_l1(const std::vector<double>& input,
      std::size_t channels,std::size_t samples)
{
   if (!channels || !samples || input.size()!=air_detail::product(channels,samples))
      throw std::invalid_argument("Invalid modal norm input shape");
   double largest=0;
   for (std::size_t channel=0; channel<channels; ++channel) {
      long double norm=0;
      for (std::size_t n=0; n<samples; ++n) {
         const double value=input[channel*samples+n];
         if (!std::isfinite(value)) throw std::invalid_argument("Modal input sample is not finite");
         norm+=std::abs(static_cast<long double>(value));
      }
      // Positive-sum roundoff bound, followed by upward conversion to double.
      const long double safety=1-4*std::numeric_limits<long double>::epsilon()*samples;
      const double upper=safety>0 ? static_cast<double>(norm/safety) : std::numeric_limits<double>::infinity();
      largest=std::max(largest,upper==0 ? 0 : std::nextafter(upper,std::numeric_limits<double>::infinity()));
   }
   return largest;
}

namespace air_detail {

// sigma-(1-sigma/2)*expm1(sigma) starts at sigma^3/12. Evaluating
// its series avoids cancellation when low-frequency damping is very small.
inline double modal_damping_delta(double sigma)
{
   if (sigma>=0.125) return sigma-(1-sigma/2)*std::expm1(sigma);
   double term=sigma*sigma*sigma/12,sum=term;
   for (unsigned m=3; m<32; ++m) {
      term*=sigma*(m-1)/((m+1.0)*(m-2));
      sum+=term;
   }
   return sum;
}

inline double modal_kernel_log_bound(double a,std::size_t nodes)
{
   if (a==0) return -std::numeric_limits<double>::infinity();
   const double k=static_cast<double>(nodes);
   return std::log(4.0)+k*k/(std::hypot(a,k)+a)-k*std::asinh(k/a);
}

} // namespace air_detail

inline AirModalFftPlan air_modal_fft_plan(const AirModalPlan& modal,
      std::size_t samples_in,double max_channel_l1,double tolerance)
{
   if (!samples_in || samples_in>modal.samples || !std::isfinite(tolerance) || tolerance<=0 ||
       std::isnan(max_channel_l1) || max_channel_l1<0 || modal.sigma.size()!=modal.samples ||
       modal.omega.size()!=modal.samples || modal.f1.size()!=modal.samples)
      throw std::invalid_argument("Invalid modal FFT planning input");
   AirModalFftPlan plan;
   if (!std::isfinite(max_channel_l1) || max_channel_l1==0 || modal.samples<2) return plan;
   plan.sigma_max=*std::max_element(modal.sigma.begin(),modal.sigma.end());
   // Extreme damping would overflow the closed-form exp(sigma) coefficient;
   // the original damped recurrence remains well defined in that case.
   if (!std::isfinite(plan.sigma_max) || plan.sigma_max<16*std::numeric_limits<double>::min() ||
       plan.sigma_max>50) return plan;
   plan.real_coeff.assign(modal.samples,0.0); plan.sine_coeff.assign(modal.samples,0.0);
   double gain_sum=0;
   for (std::size_t q=1; q<modal.samples; ++q) {
      const double sigma=modal.sigma[q],theta=modal.omega[q],half_sine=std::sin(theta/2);
      plan.real_coeff[q]=modal.f1[q];
      plan.sine_coeff[q]=(modal.f1[q]/(1+sigma/2))*
         (-(2+sigma)*half_sine*half_sine+air_detail::modal_damping_delta(sigma))/std::sin(theta);
      gain_sum+=std::abs(plan.real_coeff[q])+std::abs(plan.sine_coeff[q]);
   }
   const double gain_safety=1-8*std::numeric_limits<double>::epsilon()*modal.samples;
   const double amplification=gain_safety>0 ? max_channel_l1*std::sqrt(2.0/modal.samples)*gain_sum/gain_safety :
      std::numeric_limits<double>::infinity();
   if (!std::isfinite(amplification) || amplification<=0) return plan;
   const double a=(static_cast<double>(samples_in)-1)*plan.sigma_max/2;
   if (!std::isfinite(a)) return plan;
   const double target=std::log(tolerance)-std::log(amplification);
   std::size_t count=2;
   double log_bound=air_detail::modal_kernel_log_bound(a,count);
   while (count<4096 && log_bound>target) {
      ++count; log_bound=air_detail::modal_kernel_log_bound(a,count);
   }
   if (log_bound>target || count*(1+std::log2(2.0*modal.samples))>samples_in/4.0) return plan;
   plan.kernel_error_bound=std::exp(log_bound);
   plan.output_error_bound=std::nextafter(std::exp(log_bound+std::log(amplification)),
      std::numeric_limits<double>::infinity());
   plan.nodes.resize(count); plan.barycentric.resize(count);
   const double pi=std::acos(-1.0);
   for (std::size_t j=0; j<count; ++j) {
      const double sine=std::sin(pi*static_cast<double>(j)/(2*(count-1)));
      plan.nodes[j]=plan.sigma_max*sine*sine;
      plan.barycentric[j]=(j%2 ? -1.0 : 1.0)*(j==0 || j+1==count ? 0.5 : 1.0);
   }
   plan.nodes.front()=0; plan.nodes.back()=plan.sigma_max;
   plan.normalizers.resize(modal.samples); plan.exact_node.assign(modal.samples,-1);
   for (std::size_t q=0; q<modal.samples; ++q) {
      const double point=modal.sigma[q]/plan.sigma_max;
      double denominator=0;
      for (std::size_t j=0; j<count; ++j) {
         const double distance=point-plan.nodes[j]/plan.sigma_max;
         if (distance==0) { plan.exact_node[q]=static_cast<int>(j); break; }
         denominator+=plan.barycentric[j]/distance;
      }
      plan.normalizers[q]=denominator;
      if (plan.exact_node[q]<0 && (!std::isfinite(denominator) || denominator==0)) return plan;
   }
   // Chebyshev coefficients of exp(-a*(1+x)) are scaled Bessel I_k(a).
   // Their interpolation tail is bounded by 4*P(Skellam(a/2,a/2)>=K).
   // Chernoff optimization gives the log bound above. Multiplying by each
   // channel's L1 norm and IDCT coefficient magnitudes bounds output error.
   // This bound covers interpolation, not floating-point FFT roundoff.
   plan.usable=true;
   return plan;
}

inline double air_modal_fft_weight(const AirModalFftPlan& plan,
      const AirModalPlan& modal,std::size_t q,std::size_t node)
{
   if (!plan.usable || q>=modal.samples || q>=plan.exact_node.size() ||
       node>=plan.nodes.size())
      throw std::invalid_argument("Invalid modal FFT interpolation index");
   if (plan.exact_node[q]>=0) return static_cast<std::size_t>(plan.exact_node[q])==node ? 1.0 : 0.0;
   return (plan.barycentric[node]/(modal.sigma[q]/plan.sigma_max-plan.nodes[node]/plan.sigma_max))/plan.normalizers[q];
}

inline AirResult process_air_modal_fft(const std::vector<double>& input,
      std::size_t channels,std::size_t samples,const AirModalPlan& modal,const AirModalFftPlan& plan)
{
   if (!plan.usable || !channels || !samples || samples>modal.samples ||
       input.size()!=air_detail::product(channels,samples))
      throw std::invalid_argument("Invalid modal FFT processing input");
   for (double value : input)
      if (!std::isfinite(value)) throw std::invalid_argument("Modal FFT input sample is not finite");
   AirResult result;
   result.samples=modal.samples;
   result.data.resize(air_detail::product(channels,result.samples));
   const std::size_t extent=air_detail::product(2,modal.samples);
   const NativeFftPlan fft(extent);
   std::vector<std::complex<double>> work(extent);
   std::vector<double> modes(modal.samples),compensation(modal.samples);
   for (std::size_t channel=0; channel<channels; ++channel) {
      std::fill(modes.begin(),modes.end(),0.0); std::fill(compensation.begin(),compensation.end(),0.0);
      for (std::size_t j=0; j<plan.nodes.size(); ++j) {
         std::fill(work.begin(),work.end(),std::complex<double>(0,0));
         for (std::size_t n=0; n<samples; ++n)
            work[n]=input[channel*samples+n]*std::exp(-plan.nodes[j]*static_cast<double>(n));
         fft.transform(work,false);
         for (std::size_t q=1; q<modal.samples; ++q) {
            const double term=air_modal_fft_weight(plan,modal,q,j)*
               (plan.real_coeff[q]*work[q].real()-plan.sine_coeff[q]*work[q].imag());
            const double corrected=term-compensation[q],next=modes[q]+corrected;
            compensation[q]=(next-modes[q])-corrected; modes[q]=next;
         }
      }
      double sum=0;
      for (std::size_t n=0; n<samples; ++n) sum+=input[channel*samples+n];
      modes[0]=sum/std::sqrt(static_cast<double>(modal.samples));
      idct2_ortho(modes);
      std::copy(modes.begin(),modes.end(),result.data.begin()+channel*result.samples);
   }
   air_detail::finite_result(result.data);
   return result;
}

inline AirResult process_air_cpu(const std::vector<double>& input,std::size_t channels,
      std::size_t samples,double fs,const AirOptions& options)
{
   air_detail::rate(fs);
   if (!channels || !samples || input.size()!=air_detail::product(channels,samples))
      throw std::invalid_argument("Invalid planar air-filter input shape");
   for (double value : input)
      if (!std::isfinite(value)) throw std::invalid_argument("Air-filter input sample is not finite");
   const std::string mode=air_mode(options);
   AirResult result;
   if (mode=="none") { result.data=input; result.samples=samples; return result; }
   if (mode=="stokes") {
      const AirStokesPlan plan=air_stokes_plan(samples,fs,options);
      result.samples=plan.samples;
      result.data.assign(air_detail::product(channels,result.samples),0.0);
      const double pi=std::acos(-1.0),two_ts_gamma=2*plan.ts*plan.gamma;
      for (std::size_t channel=0; channel<channels; ++channel) {
         std::copy(input.begin()+channel*samples,input.begin()+channel*samples+plan.first,
                   result.data.begin()+channel*result.samples);
         // Deterministic reference scatter: increasing source index matches the
         // legacy addition order. CUDA gathers these contributions per output.
         for (std::size_t n=plan.first; n<samples; ++n) {
            const std::size_t width=air_stokes_width(plan,n);
            const double gain=plan.ts/std::sqrt(static_cast<double>(n)*two_ts_gamma*pi)*input[channel*samples+n];
            for (std::size_t out=n-width; out<=n+width; ++out) {
               const double dt=(static_cast<double>(n)-static_cast<double>(out))*plan.ts;
               result.data[channel*result.samples+out]+=gain*std::exp(-dt*dt/(static_cast<double>(n)*two_ts_gamma));
            }
         }
      }
   }
   else if (mode=="ola") {
      const AirOlaPlan plan=air_ola_plan(samples,fs,options);
      result.samples=plan.samples;
      result.data.assign(air_detail::product(channels,result.samples),0.0);
      const NativeFftPlan fft(plan.fft_size);
      std::vector<double> padded(air_detail::add(plan.window,plan.samples));
      std::vector<double> output(padded.size());
      std::vector<std::complex<double>> frame(plan.fft_size);
      for (std::size_t channel=0; channel<channels; ++channel) {
         std::fill(padded.begin(),padded.end(),0.0); std::fill(output.begin(),output.end(),0.0);
         std::copy(input.begin()+channel*samples,input.begin()+(channel+1)*samples,padded.begin()+plan.window);
         for (std::size_t m=0; m<plan.frames; ++m) {
            const std::size_t start=m*plan.hop;
            const double distance=plan.c/fs*(static_cast<double>(start)-0.5*plan.window);
            if (distance<0) {
               for (std::size_t n=0; n<plan.window; ++n)
                  output[start+n]+=plan.synthesis[n]*padded[start+n];
               continue;
            }
            std::fill(frame.begin(),frame.end(),std::complex<double>(0,0));
            for (std::size_t n=0; n<plan.window; ++n)
               frame[n]=plan.analysis[n]*padded[start+n];
            fft.transform(frame,false);
            for (std::size_t k=0; k<frame.size(); ++k) {
               const std::size_t bin=std::min(k,frame.size()-k);
               frame[k]*=std::exp(-plan.absorption[bin]*distance);
            }
            fft.transform(frame,true);
            for (std::size_t n=0; n<plan.window; ++n)
               output[start+n]+=plan.synthesis[n]*frame[n].real();
         }
         std::copy(output.begin()+plan.window,output.end(),result.data.begin()+channel*result.samples);
      }
   }
   else {
      const AirModalPlan plan=air_modal_plan(samples,fs,options);
      if (options.modal_method=="fft") {
         const AirModalFftPlan fast=air_modal_fft_plan(plan,samples,
            air_max_channel_l1(input,channels,samples),options.modal_tolerance);
         if (fast.usable) return process_air_modal_fft(input,channels,samples,plan,fast);
      }
      result.samples=plan.samples;
      result.data.resize(air_detail::product(channels,result.samples));
      std::vector<double> modes(plan.samples);
      for (std::size_t channel=0; channel<channels; ++channel) {
         double sum=0;
         for (std::size_t n=0; n<samples; ++n) sum+=input[channel*samples+n];
         modes[0]=sum/std::sqrt(static_cast<double>(plan.samples));
         for (std::size_t q=1; q<plan.samples; ++q) {
            double previous=0,current=0,source_previous=0;
            // Reversing padded data would prepend zeros. Skip them exactly:
            // they cannot change zero state, while q still uses padded length.
            for (std::size_t t=0; t<samples; ++t) {
               const double source=input[channel*samples+samples-1-t];
               const double next=plan.a1[q]*current+plan.a2[q]*previous+
                  plan.f1[q]*source-plan.f2[q]*source_previous;
               previous=current; current=next; source_previous=source;
            }
            modes[q]=current;
         }
         // The mode-dependent complex radii vary with q; ordinary FFT
         // convolution cannot replace the O(Nout*Nin) modal recurrence.
         idct2_ortho(modes);
         std::copy(modes.begin(),modes.end(),result.data.begin()+channel*result.samples);
      }
   }
   air_detail::finite_result(result.data);
   return result;
}

} // namespace pffdtd_post
#endif
