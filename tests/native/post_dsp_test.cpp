// Independent analytical transfer functions and a long-double direct-form-I
// recurrence check the native SOS design/reference. Chunked affine scheduling is
// also modeled in double precision; CUDA runtime equivalence remains a gate.
#include <post_dsp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using pffdtd_post::Sos;
typedef std::complex<long double> Complex;

void require(bool condition, const char *message)
{
   if (!condition) { std::fprintf(stderr,"FAIL: %s\n",message); std::exit(EXIT_FAILURE); }
}

void near(double actual, double expected, double tolerance, const char *message)
{
   require(std::isfinite(actual) && std::abs(actual-expected)<=tolerance,message);
}

template<typename F>
void invalid(F operation)
{
   bool caught=false;
   try { operation(); }
   catch (const std::invalid_argument &) { caught=true; }
   require(caught,"invalid DSP input was accepted");
}

Complex response(const std::vector<Sos> &sections, long double frequency, long double fs)
{
   const long double phase=-2*std::acos(-1.0L)*frequency/fs;
   const Complex delay(std::cos(phase),std::sin(phase));
   Complex result(1,0);
   for (const Sos &s : sections)
      result*=(static_cast<long double>(s.b0)+static_cast<long double>(s.b1)*delay+static_cast<long double>(s.b2)*delay*delay)
                /(1.0L+static_cast<long double>(s.a1)*delay+static_cast<long double>(s.a2)*delay*delay);
   return result;
}

// Evaluate the continuous Butterworth prototype at the bilinear frequency,
// rather than reproducing the production pole-pair/SOS coefficient algorithm.
Complex analog_response(unsigned order, long double fs, long double cut,
                        long double frequency, bool highpass, bool integrate)
{
   const long double pi=std::acos(-1.0L);
   const long double omega=2*fs*std::tan(pi*frequency/fs);
   const long double cutoff=integrate ? 2*pi*cut : 2*fs*std::tan(pi*cut/fs);
   const Complex s(0,omega/cutoff);
   Complex denominator(1,0),numerator(1,0);
   for (unsigned k=1; k<=order; ++k) {
      const long double angle=pi*(2*k+order-1)/(2*order);
      denominator*=s-Complex(std::cos(angle),std::sin(angle));
      if (highpass) numerator*=s;
   }
   Complex result=numerator/denominator;
   if (integrate) result/=Complex(0,omega);
   return result;
}

void stable(const std::vector<Sos> &sections)
{
   for (const Sos &s : sections) {
      require(std::isfinite(s.b0) && std::isfinite(s.b1) && std::isfinite(s.b2),
              "nonfinite numerator coefficient");
      const Complex discriminant(static_cast<long double>(s.a1)*s.a1-4.0L*s.a2,0);
      const Complex root=std::sqrt(discriminant);
      require(std::abs((-static_cast<long double>(s.a1)+root)/2.0L)<1 &&
              std::abs((-static_cast<long double>(s.a1)-root)/2.0L)<1,
              "designed SOS pole is not strictly stable");
   }
}

void known_coefficients()
{
   const double fs=48000,cut=3000,pi=std::acos(-1.0);
   for (bool integrate : {false,true}) {
      const double q=integrate ? pi*cut/fs : std::tan(pi*cut/fs);
      const auto one=pffdtd_post::design_highpass(1,fs,cut,integrate);
      const double g=(integrate ? 0.5/fs : 1.0)/(1+q);
      require(one.size()==1,"first-order HP section count differs");
      near(one[0].b0,g,2e-15,"first-order HP gain differs");
      near(one[0].b1,integrate ? g : -g,2e-15,"first-order HP zero differs");
      near(one[0].a1,(q-1)/(q+1),2e-15,"first-order HP pole differs");
      near(one[0].a2,0,0,"first-order HP has an extra pole");

      const auto two=pffdtd_post::design_highpass(2,fs,cut,integrate);
      const double d=1+std::sqrt(2.0)*q+q*q;
      const double gain=(integrate ? 0.5/fs : 1.0)/d;
      near(two[0].b0,gain,2e-15,"second-order HP gain differs");
      near(two[0].b1,integrate ? 0 : -2*gain,2e-15,"second-order HP zeros differ");
      near(two[0].b2,integrate ? -gain : gain,2e-15,"second-order HP missing-degree zero differs");
      near(two[0].a1,2*(q*q-1)/d,2e-15,"second-order HP a1 differs");
      near(two[0].a2,(1-std::sqrt(2.0)*q+q*q)/d,2e-15,"second-order HP a2 differs");
   }
   const double q=std::tan(pi*cut/fs),d=1+std::sqrt(2.0)*q+q*q;
   const auto one=pffdtd_post::design_lowpass(1,fs,cut),two=pffdtd_post::design_lowpass(2,fs,cut);
   near(one[0].b0,q/(1+q),2e-15,"first-order LP gain differs");
   near(one[0].b1,one[0].b0,0,"first-order LP zero differs");
   near(two[0].b0,q*q/d,2e-15,"second-order LP gain differs");
   near(two[0].b1,2*two[0].b0,0,"second-order LP zeros differ");
   near(two[0].a1,2*(q*q-1)/d,2e-15,"second-order LP pole differs");
}

void transfer_functions()
{
   long double largest_relative=0;
   unsigned cases=0;
   for (double fs : {32000.0,48000.0,96000.0})
      for (double cut : {10.0,300.0,fs*0.125,fs*0.45})
         for (unsigned order=1; order<=16; ++order)
            for (int mode=0; mode<3; ++mode) {
               const bool highpass=mode!=0,integrate=mode==2;
               const auto sections=highpass ? pffdtd_post::design_highpass(order,fs,cut,integrate)
                                            : pffdtd_post::design_lowpass(order,fs,cut);
               require(sections.size()==(order+1)/2,"SOS section count differs");
               stable(sections);
               for (double frequency : {cut*0.1,cut*0.5,cut,cut*2,fs*0.01,fs*0.125,fs*0.4}) {
                  if (frequency>=fs/2) continue;
                  const Complex expected=analog_response(order,fs,cut,frequency,highpass,integrate);
                  const Complex actual=response(sections,frequency,fs);
                  const long double error=std::abs(actual-expected),magnitude=std::abs(expected);
                  if (magnitude>1e-10L) largest_relative=std::max(largest_relative,error/magnitude);
                  if (error>2e-12L+5e-8L*magnitude) {
                     std::fprintf(stderr,"TF mismatch order=%u fs=%.0f cut=%.2f f=%.2f mode=%d error=%.3Le mag=%.3Le\n",
                                  order,fs,cut,frequency,mode,error,magnitude);
                     require(false,"SOS transfer function differs from analog/bilinear prototype");
                  }
               }
               ++cases;
            }
   std::printf("PASS: %u Butterworth designs, orders1..16; analog/bilinear relative error %.3Le\n",cases,largest_relative);
}

std::vector<long double> reference_dfi(const std::vector<double> &input,std::size_t channels,
                                      std::size_t samples,const std::vector<Sos> &sections,bool reverse)
{
   std::vector<long double> result(input.begin(),input.end());
   for (std::size_t channel=0; channel<channels; ++channel)
      for (const Sos &s : sections) {
         long double x1=0,x2=0,y1=0,y2=0;
         for (std::size_t t=0; t<samples; ++t) {
            const std::size_t n=reverse ? samples-1-t : t,index=channel*samples+n;
            const long double x=result[index];
            const long double y=static_cast<long double>(s.b0)*x+s.b1*x1+s.b2*x2-s.a1*y1-s.a2*y2;
            x2=x1; x1=x; y2=y1; y1=y; result[index]=y;
         }
      }
   return result;
}

struct Matrix { double a,b,c,d; };
Matrix multiply(const Matrix &x,const Matrix &y)
{
   return {x.a*y.a+x.b*y.c,x.a*y.b+x.b*y.d,x.c*y.a+x.d*y.c,x.c*y.b+x.d*y.d};
}

Matrix power(const Sos &s,std::size_t exponent)
{
   Matrix result={1,0,0,1},base={-s.a1,1,-s.a2,0};
   while (exponent) {
      if (exponent&1) result=multiply(result,base);
      exponent>>=1;
      if (exponent) base=multiply(base,base);
   }
   return result;
}

// The first zero-state pass builds B for each chunk; affine prefixes provide
// incoming states, and the second pass emits each chunk's samples. Inputs are
// immutable until both the zero-state and prefix passes are complete.
void filter_chunked(std::vector<double> &data,std::size_t channels,std::size_t samples,
                    const std::vector<Sos> &sections,std::size_t block,bool reverse)
{
   const std::size_t chunks=(samples+block-1)/block;
   std::vector<std::array<double,2>> zero(chunks),incoming(chunks);
   for (std::size_t channel=0; channel<channels; ++channel)
      for (const Sos &s : sections) {
         for (std::size_t chunk=0; chunk<chunks; ++chunk) {
            double z1=0,z2=0;
            const std::size_t start=chunk*block,end=std::min(samples,start+block);
            for (std::size_t t=start; t<end; ++t) {
               const std::size_t n=reverse ? samples-1-t : t;
               pffdtd_post::sos_sample(s,data[channel*samples+n],z1,z2);
            }
            zero[chunk]={{z1,z2}};
         }
         double z1=0,z2=0;
         for (std::size_t chunk=0; chunk<chunks; ++chunk) {
            incoming[chunk]={{z1,z2}};
            const Matrix transition=power(s,std::min(block,samples-chunk*block));
            const double next1=transition.a*z1+transition.b*z2+zero[chunk][0];
            const double next2=transition.c*z1+transition.d*z2+zero[chunk][1];
            z1=next1; z2=next2;
         }
         for (std::size_t chunk=0; chunk<chunks; ++chunk) {
            z1=incoming[chunk][0]; z2=incoming[chunk][1];
            const std::size_t start=chunk*block,end=std::min(samples,start+block);
            for (std::size_t t=start; t<end; ++t) {
               const std::size_t n=reverse ? samples-1-t : t;
               data[channel*samples+n]=pffdtd_post::sos_sample(s,data[channel*samples+n],z1,z2);
            }
         }
      }
}

std::vector<double> signal(std::size_t channels,std::size_t samples)
{
   std::vector<double> data(channels*samples);
   for (std::size_t channel=0; channel<channels; ++channel)
      for (std::size_t n=0; n<samples; ++n) {
         if (channel==0) data[channel*samples+n]=n==0 ? 1 : (n==1 ? -1 : 0);
         else if (channel==1) data[channel*samples+n]=static_cast<double>((n*13+17)%31)/64.0-0.25;
         else data[channel*samples+n]=0.125;
      }
   return data;
}

void recurrences()
{
   const std::vector<std::vector<Sos>> filters={
      pffdtd_post::design_highpass(8,48000,10,true),pffdtd_post::design_highpass(6,48000,300,false),
      pffdtd_post::design_lowpass(7,48000,1000),pffdtd_post::design_lowpass(8,48000,4800),
      pffdtd_post::design_highpass(4,48000,0,true),pffdtd_post::design_highpass(4,48000,0,false)};
   long double largest_reference=0;
   double largest_chunk=0;
   unsigned cases=0;
   for (const auto &sections : filters)
      for (std::size_t samples : {1u,2u,17u,255u,256u,257u,1025u,4097u,65537u})
         for (bool reverse : {false,true}) {
            const auto input=signal(3,samples);
            const auto expected=reference_dfi(input,3,samples,sections,reverse);
            auto actual=input;
            pffdtd_post::filter_serial(actual.data(),3,samples,sections,reverse);
            for (std::size_t i=0; i<actual.size(); ++i) {
               const long double error=std::abs(static_cast<long double>(actual[i])-expected[i]);
               largest_reference=std::max(largest_reference,error);
               require(error<=2e-8L+2e-8L*std::abs(expected[i]),"serial filter differs from direct-form-I reference");
            }
            for (std::size_t block : {1u,7u,64u,256u}) {
               auto chunked=input;
               filter_chunked(chunked,3,samples,sections,block,reverse);
               for (std::size_t i=0; i<actual.size(); ++i) {
                  const double error=std::abs(chunked[i]-actual[i]);
                  largest_chunk=std::max(largest_chunk,error);
                  require(std::isfinite(chunked[i]) && error<=2e-7+2e-7*std::abs(actual[i]),
                          "affine chunk contract exceeds its numerical error gate");
               }
            }
            ++cases;
         }
   std::printf("PASS: %u forward/reverse recurrence cases; DFI error %.3Le, affine chunks error %.3e\n",
               cases,largest_reference,largest_chunk);
}

void integration_and_gain()
{
   const auto integrator=pffdtd_post::design_highpass(8,32,0,true);
   std::vector<double> impulse={1,0,0,0,0};
   pffdtd_post::filter_serial(impulse.data(),1,impulse.size(),integrator);
   near(impulse[0],1.0/64,0,"trapezoidal integrator initial impulse differs");
   for (std::size_t i=1; i<impulse.size(); ++i) near(impulse[i],1.0/32,0,"trapezoidal integrator tail differs");
   std::vector<double> differentiated={1,-1,0,0,0};
   pffdtd_post::filter_serial(differentiated.data(),1,differentiated.size(),integrator);
   near(differentiated[0],1.0/64,0,"differentiated impulse first integrated sample differs");
   near(differentiated[1],1.0/64,0,"differentiated impulse second integrated sample differs");
   for (std::size_t i=2; i<differentiated.size(); ++i) near(differentiated[i],0,0,"differentiated impulse integral did not stop");
   for (unsigned order : {1u,2u,7u,8u,16u}) {
      std::vector<double> dc(200000,1.0);
      pffdtd_post::filter_serial(dc.data(),1,dc.size(),pffdtd_post::design_lowpass(order,48000,1000));
      near(dc.back(),1,1e-10,"lowpass steady DC gain differs");
      std::fill(dc.begin(),dc.end(),1.0);
      pffdtd_post::filter_serial(dc.data(),1,dc.size(),pffdtd_post::design_highpass(order,48000,10,false));
      near(dc.back(),0,order==16 ? 1e-8 : 1e-9,"highpass steady DC rejection differs");
   }
}

void invalid_inputs()
{
   const double nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity();
   for (unsigned order : {0u,17u}) {
      invalid([&]{ pffdtd_post::design_highpass(order,48000,10,true); });
      invalid([&]{ pffdtd_post::design_lowpass(order,48000,10); });
   }
   for (double fs : {0.0,-1.0,nan,inf})
      invalid([&]{ pffdtd_post::design_highpass(8,fs,10,true); });
   for (double cut : {-1.0,24000.0,25000.0,nan,inf}) {
      invalid([&]{ pffdtd_post::design_highpass(8,48000,cut,true); });
      invalid([&]{ pffdtd_post::design_lowpass(8,48000,cut); });
   }
   invalid([&]{ pffdtd_post::design_lowpass(8,48000,0); });
   invalid([&]{ pffdtd_post::filter_serial(nullptr,1,1,{}); });
   double sample=nan;
   invalid([&]{ pffdtd_post::filter_serial(&sample,1,1,{}); });
   sample=0;
   invalid([&]{ pffdtd_post::filter_serial(&sample,1,1,{{nan,0,0,0,0}}); });
   pffdtd_post::filter_serial(nullptr,0,10,{});
   pffdtd_post::filter_serial(nullptr,10,0,{});
}

} // namespace

int main()
{
   known_coefficients(); transfer_functions(); recurrences(); integration_and_gain(); invalid_inputs();
   std::puts("post_dsp_test: PASS native coefficient/recurrence contracts; no Python or CUDA runtime execution");
   return EXIT_SUCCESS;
}
