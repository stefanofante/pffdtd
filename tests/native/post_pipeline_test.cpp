#include "post_pipeline.h"
#include "post_chunked.h"
#include <algorithm>
#include <cstdio>
#include <stdexcept>

using namespace pffdtd_post;
namespace {
std::size_t checks=0;
void require(bool condition, const char* message)
{ ++checks; if (!condition) throw std::runtime_error(message); }

// Model the CUDA chunk schedule on the host, against the independently
// scheduled serial recurrence. Chunk carries must be seeded before writes.
void chunk_model(std::vector<double>& values, std::size_t channels,
      std::size_t samples, const std::vector<Sos>& sections, bool reverse)
{
   const std::size_t length=64, chunks=(samples-1)/length+1;
   for (const Sos& section : sections) {
      const PostMatrix transition=post_transition_power(section,length);
      for (std::size_t channel=0; channel<channels; ++channel) {
         std::vector<PostState> response(chunks), initial(chunks);
         for (std::size_t chunk=0; chunk<chunks; ++chunk) {
            double z1=0,z2=0;
            for (std::size_t n=chunk*length; n<std::min(samples,(chunk+1)*length); ++n)
               sos_sample(section,values[channel*samples+(reverse ? samples-1-n : n)],z1,z2);
            response[chunk]={z1,z2};
         }
         PostState state={0,0};
         for (std::size_t chunk=0; chunk<chunks; ++chunk) {
            initial[chunk]=state;
            state=post_matrix_apply(transition,state);
            state.z1+=response[chunk].z1; state.z2+=response[chunk].z2;
         }
         for (std::size_t chunk=0; chunk<chunks; ++chunk) {
            double z1=initial[chunk].z1,z2=initial[chunk].z2;
            for (std::size_t n=chunk*length; n<std::min(samples,(chunk+1)*length); ++n) {
               const std::size_t index=channel*samples+(reverse ? samples-1-n : n);
               values[index]=sos_sample(section,values[index],z1,z2);
            }
         }
      }
   }
}

double max_error=0;
void compare_chunk(std::size_t samples, const std::vector<Sos>& sections, bool reverse)
{
   const std::size_t channels=3;
   std::vector<double> serial(channels*samples);
   for (std::size_t channel=0; channel<channels; ++channel)
      for (std::size_t n=0; n<samples; ++n)
         serial[channel*samples+n]=(channel+1)*
            (0.3*std::sin(0.023*n)+0.2*std::cos(0.13*n)+(n%67==0 ? 0.125 : 0));
   std::vector<double> chunked=serial;
   filter_serial(serial.data(),channels,samples,sections,reverse);
   chunk_model(chunked,channels,samples,sections,reverse);
   double peak=0,error=0;
   for (std::size_t i=0; i<serial.size(); ++i) {
      require(std::isfinite(chunked[i]),"Non-finite chunk response");
      peak=std::max(peak,std::abs(serial[i]));
      error=std::max(error,std::abs(serial[i]-chunked[i]));
   }
   max_error=std::max(max_error,error/(peak+1e-30));
   if (error>1e-12+1e-8*peak) {
      std::fprintf(stderr,"chunk error Nt=%zu reverse=%d sections=%zu peak=%.17g error=%.17g\n",
                   samples,reverse,sections.size(),peak,error);
      throw std::runtime_error("Chunk recurrence exceeds numerical gate");
   }
}

void pipeline_contract()
{
   const std::size_t channels=2, corners=3, samples=513;
   const double fs=48000;
   std::vector<double> input(channels*corners*samples);
   const std::vector<double> weights={0.25,0.25,0.5,0.125,0.5,0.375};
   for (std::size_t receiver=0; receiver<channels*corners; ++receiver)
      std::fill(input.begin()+receiver*samples,input.begin()+(receiver+1)*samples,
                static_cast<double>(receiver+1));
   PostOptions options; options.lowcut=0; options.output_rate=0;
   const PostResult raw=process_cpu(input,weights,channels,corners,samples,fs,options);
   require(raw.samples_out==samples && raw.fs_out==fs,"Native rate/length changed");
   for (std::size_t n=0; n<samples; ++n) {
      require(raw.raw[n]==2.25 && raw.filtered[n]==2.25,"Weighted first receiver wrong");
      require(raw.raw[samples+n]==5.25 && raw.filtered[samples+n]==5.25,"Weighted second receiver wrong");
   }
   options.integrate=true;
   const PostResult integrated=process_cpu(input,weights,channels,corners,samples,fs,options);
   for (std::size_t channel=0; channel<channels; ++channel)
      for (std::size_t n=0; n<samples; ++n) {
         const double value=channel==0 ? 2.25 : 5.25;
         const double expected=value*(n+0.5)/fs;
         require(std::abs(integrated.filtered[channel*samples+n]-expected)<1e-14,
                 "Trapezoidal integration differs from constant analytic response");
      }
   options.lowcut=10; options.output_rate=24000; options.lowpass=3000;
   options.symmetric_lowpass=true;
   const PostResult processed=process_cpu(input,weights,channels,corners,samples,fs,options);
   require(processed.channels==2 && processed.samples_out==256 && processed.fs_out==24000,
           "Full pipeline output shape/rate wrong");
   check_post_finite(processed.filtered);
   auto rejects=[&](const PostOptions& invalid) {
      bool rejected=false;
      try { process_cpu(input,weights,channels,corners,samples,fs,invalid); }
      catch (const std::invalid_argument&) { rejected=true; }
      require(rejected,"Invalid pipeline option accepted");
   };
   PostOptions invalid=options; invalid.lowcut=fs/2; rejects(invalid);
   invalid=options; invalid.lowpass=invalid.output_rate/2; rejects(invalid);
   invalid=options; invalid.lowpass_order=7; rejects(invalid);
   invalid=options; invalid.output_rate=std::numeric_limits<double>::infinity(); rejects(invalid);
   invalid=options; invalid.lowcut_order=0; rejects(invalid);
   bool rejected=false;
   try { process_cpu(input,weights,channels,4,samples,fs,options); }
   catch (const std::invalid_argument&) { rejected=true; }
   require(rejected,"Wrong receiver shape accepted");
}
}

int main()
{
   try {
      pipeline_contract();
      const std::size_t lengths[]={1,63,64,65,513,3073,100003};
      for (std::size_t length : lengths)
         for (unsigned order=1; order<=16; ++order)
            for (bool reverse : {false,true}) {
               compare_chunk(length,design_highpass(order,48000,10,true),reverse);
               compare_chunk(length,design_highpass(order,48000,1000,false),reverse);
               compare_chunk(length,design_lowpass(order,48000,4000),reverse);
            }
      compare_chunk(100003,design_highpass(8,48000,0,true),false);
      std::printf("PASS: native post pipeline and affine chunk model, %zu checks; max relative peak error %.3g; native contracts, no GPU execution\n",checks,max_error);
      return 0;
   }
   catch (const std::exception& error) {
      std::fprintf(stderr,"post_pipeline_test: %s\n",error.what()); return 1;
   }
}
