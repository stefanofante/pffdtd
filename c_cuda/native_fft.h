// Dependency-free native FFT and orthonormal cosine transforms. A reusable
// plan keeps twiddles and the Bluestein kernel between independent transforms.
#ifndef PFFDTD_NATIVE_FFT_H
#define PFFDTD_NATIVE_FFT_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pffdtd_post {

class NativeFftPlan {
   typedef std::complex<double> Complex;
   std::size_t length_, transform_size_;
   std::vector<Complex> twiddles_, chirp_, kernel_;

   static bool power_of_two(std::size_t n)
   {
      return n && !(n&(n-1));
   }

   static Complex unit_circle(double angle)
   {
      return Complex(std::cos(angle),std::sin(angle));
   }

   // Both operands are below modulus. This form also works without a wider
   // integer type when their sum cannot be represented by size_t.
   static std::size_t modular_add(std::size_t a, std::size_t b,
                                  std::size_t modulus)
   {
      return a>=modulus-b ? a-(modulus-b) : a+b;
   }

   void radix2(std::vector<Complex> &data, bool inverse) const
   {
      const std::size_t n=data.size();
      if (n<2) return;
      for (std::size_t i=1,j=0; i<n; ++i) {
         std::size_t bit=n>>1;
         while (j&bit) { j^=bit; bit>>=1; }
         j^=bit;
         if (i<j) std::swap(data[i],data[j]);
      }
      for (std::size_t width=2;; width*=2) {
         const std::size_t half=width/2, step=n/width;
         for (std::size_t base=0; base<n; base+=width) {
            for (std::size_t j=0; j<half; ++j) {
               const Complex twiddle=inverse ? std::conj(twiddles_[j*step]) :
                                               twiddles_[j*step];
               const Complex even=data[base+j];
               const Complex odd=data[base+j+half]*twiddle;
               data[base+j]=even+odd;
               data[base+j+half]=even-odd;
            }
         }
         if (width==n) break; // Avoid overflow in the final width doubling.
      }
      if (inverse) {
         const double normalization=1.0/static_cast<double>(n);
         for (Complex &value : data) value*=normalization;
      }
   }

public:
   explicit NativeFftPlan(std::size_t length)
      : length_(length),transform_size_(length)
   {
      const std::size_t max_elements=std::vector<Complex>().max_size();
      if (length>max_elements)
         throw std::length_error("FFT length exceeds the native buffer limit");
      if (length<2) return;
      if (!power_of_two(length)) {
         if (length>std::numeric_limits<std::size_t>::max()/2)
            throw std::length_error("Bluestein convolution extent overflows");
         const std::size_t needed=length+(length-1);
         transform_size_=1;
         while (transform_size_<needed) {
            if (transform_size_>max_elements/2)
               throw std::length_error("Bluestein convolution exceeds the native buffer limit");
            transform_size_*=2;
         }
      }
      twiddles_.resize(transform_size_/2);
      const double pi=std::acos(-1.0);
      for (std::size_t k=0; k<twiddles_.size(); ++k)
         twiddles_[k]=unit_circle(-2.0*pi*(static_cast<double>(k)/transform_size_));
      if (power_of_two(length)) return;

      chirp_.resize(length);
      kernel_.assign(transform_size_,Complex());
      const std::size_t modulus=2*length;
      std::size_t square=0,odd=1;
      for (std::size_t k=0; k<length; ++k) {
         // k*k is reduced modulo 2*N before conversion to floating point.
         // Incremental modular squares avoid integer overflow and preserve
         // chirp phase accuracy for long, non-power-of-two recordings.
         chirp_[k]=unit_circle(-pi*(static_cast<double>(square)/length));
         kernel_[k]=std::conj(chirp_[k]);
         if (k) kernel_[transform_size_-k]=kernel_[k];
         if (k+1<length) {
            square=modular_add(square,odd,modulus);
            odd+=2;
         }
      }
      radix2(kernel_,false);
   }

   std::size_t length() const { return length_; }

   // Forward: sum_n x[n] exp(-i*2*pi*k*n/N). Inverse includes division by N.
   // The plan is immutable, so separate buffers may use it concurrently.
   void transform(std::vector<Complex> &data, bool inverse=false) const
   {
      if (data.size()!=length_)
         throw std::invalid_argument("FFT input length differs from its plan");
      for (const Complex &value : data)
         if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
            throw std::invalid_argument("FFT input is not finite");
      if (length_<2) return;
      if (chirp_.empty()) {
         radix2(data,inverse);
      } else {
         std::vector<Complex> work(transform_size_);
         for (std::size_t k=0; k<length_; ++k)
            work[k]=(inverse ? std::conj(data[k]) : data[k])*chirp_[k];
         radix2(work,false);
         for (std::size_t k=0; k<transform_size_; ++k) work[k]*=kernel_[k];
         radix2(work,true);
         const double normalization=inverse ? 1.0/static_cast<double>(length_) : 1.0;
         for (std::size_t k=0; k<length_; ++k) {
            const Complex value=work[k]*chirp_[k];
            data[k]=inverse ? std::conj(value)*normalization : value;
         }
      }
      for (const Complex &value : data)
         if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
            throw std::runtime_error("FFT output is not finite");
   }
};

inline void native_fft(std::vector<std::complex<double>> &data, bool inverse=false)
{
   NativeFftPlan(data.size()).transform(data,inverse);
}

namespace detail {

inline std::size_t cosine_fft_extent(std::size_t length)
{
   if (length>std::vector<std::complex<double>>().max_size()/2)
      throw std::length_error("Cosine transform extent exceeds the native buffer limit");
   return 2*length;
}

} // namespace detail

// Orthonormal DCT-II: scale(0)=sqrt(1/N), scale(k>0)=sqrt(2/N).
inline void dct2_ortho(std::vector<double> &data)
{
   const std::size_t n=data.size();
   if (!n) return;
   const std::size_t extent=detail::cosine_fft_extent(n);
   std::vector<std::complex<double>> work(extent);
   for (std::size_t k=0; k<n; ++k) work[k]=work[extent-1-k]=data[k];
   native_fft(work);
   const double pi=std::acos(-1.0);
   const double zero_scale=std::sqrt(1.0/static_cast<double>(n));
   const double other_scale=std::sqrt(2.0/static_cast<double>(n));
   for (std::size_t k=0; k<n; ++k) {
      const double angle=-pi*(static_cast<double>(k)/extent);
      const std::complex<double> phase(std::cos(angle),std::sin(angle));
      data[k]=0.5*(work[k]*phase).real()*(k ? other_scale : zero_scale);
   }
}

// The inverse orthonormal DCT-III, with the same ordering and normalization.
inline void idct2_ortho(std::vector<double> &data)
{
   const std::size_t n=data.size();
   if (!n) return;
   const std::size_t extent=detail::cosine_fft_extent(n);
   std::vector<std::complex<double>> work(extent);
   const double pi=std::acos(-1.0);
   const double zero_scale=std::sqrt(static_cast<double>(n));
   const double other_scale=std::sqrt(0.5*static_cast<double>(n));
   for (std::size_t k=0; k<n; ++k) {
      const double angle=pi*(static_cast<double>(k)/extent);
      const std::complex<double> phase(std::cos(angle),std::sin(angle));
      work[k]=2.0*data[k]*(k ? other_scale : zero_scale)*phase;
      if (k) work[extent-k]=std::conj(work[k]);
   }
   native_fft(work,true);
   for (std::size_t k=0; k<n; ++k) data[k]=work[k].real();
}

} // namespace pffdtd_post
#endif
