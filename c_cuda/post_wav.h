// Receiver WAV export compatible with the legacy float32, per-receiver files.
#ifndef PFFDTD_POST_WAV_H
#define PFFDTD_POST_WAV_H

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace pffdtd_post {
namespace detail {

inline std::runtime_error wav_error(const std::string &operation)
{
   return std::runtime_error(operation+": "+std::strerror(errno));
}

class WavFd {
   int fd_;
public:
   explicit WavFd(int fd) : fd_(fd) {}
   ~WavFd() { if (fd_>=0) ::close(fd_); }
   int get() const { return fd_; }
   void close()
   {
      const int fd=fd_;
      fd_=-1;
      if (::close(fd)<0) throw wav_error("Cannot close WAV file");
   }
   WavFd(const WavFd &)=delete;
   WavFd &operator=(const WavFd &)=delete;
};

inline void wav_put16(unsigned char *data, std::uint16_t value)
{
   data[0]=static_cast<unsigned char>(value);
   data[1]=static_cast<unsigned char>(value>>8);
}

inline void wav_put32(unsigned char *data, std::uint32_t value)
{
   for (unsigned i=0; i<4; ++i) data[i]=static_cast<unsigned char>(value>>(8*i));
}

inline void wav_write_all(int fd, const unsigned char *bytes, std::size_t count)
{
   while (count) {
      const ssize_t written=::write(fd,bytes,count);
      if (written<0) {
         if (errno==EINTR) continue;
         throw wav_error("Cannot write WAV file");
      }
      if (!written) throw std::runtime_error("WAV write made no progress");
      bytes+=written;
      count-=static_cast<std::size_t>(written);
   }
}

struct WavFile {
   std::string name, temporary;
   struct stat original={}, staged={};
   bool exists=false, published=false;
};

class WavStaging {
   int directory_;
   std::vector<WavFile> &files_;
public:
   WavStaging(int directory, std::vector<WavFile> &files)
      : directory_(directory), files_(files) {}
   ~WavStaging()
   {
      for (const WavFile &file : files_)
         if (!file.temporary.empty()) ::unlinkat(directory_,file.temporary.c_str(),0);
   }
   WavStaging(const WavStaging &)=delete;
   WavStaging &operator=(const WavStaging &)=delete;
};

inline bool wav_same_file(const struct stat &first, const struct stat &second)
{
   return first.st_dev==second.st_dev && first.st_ino==second.st_ino;
}

inline int wav_temporary(int directory, std::string &name)
{
   static std::atomic<std::uint64_t> sequence(0);
   for (unsigned attempt=0; attempt<128; ++attempt) {
      name=".pffdtd-wav-"+std::to_string(static_cast<long long>(::getpid()))+
           "-"+std::to_string(sequence.fetch_add(1))+".tmp";
      const int fd=::openat(directory,name.c_str(),O_CREAT|O_EXCL|O_WRONLY|O_CLOEXEC|O_NOFOLLOW,0666);
      if (fd>=0) return fd;
      if (errno!=EEXIST) {
         name.clear();
         throw wav_error("Cannot create temporary WAV file");
      }
   }
   // An occupied name belongs to another writer and must not be removed.
   name.clear();
   throw std::runtime_error("Cannot reserve a temporary WAV filename");
}

inline void wav_encode(int fd, const double *data, std::size_t samples,
                       std::uint32_t rate, double peak, bool normalised)
{
   // SciPy's IEEE-float writer uses an 18-byte fmt chunk and a fact chunk.
   unsigned char header[58]={0};
   const std::uint32_t bytes=static_cast<std::uint32_t>(samples*4);
   std::memcpy(header,"RIFF",4);
   wav_put32(header+4,50+bytes);
   std::memcpy(header+8,"WAVEfmt ",8);
   wav_put32(header+16,18);
   wav_put16(header+20,3); // WAVE_FORMAT_IEEE_FLOAT
   wav_put16(header+22,1); // one receiver per file
   wav_put32(header+24,rate);
   wav_put32(header+28,rate*4);
   wav_put16(header+32,4);
   wav_put16(header+34,32);
   wav_put16(header+36,0);
   std::memcpy(header+38,"fact",4);
   wav_put32(header+42,4);
   wav_put32(header+46,static_cast<std::uint32_t>(samples));
   std::memcpy(header+50,"data",4);
   wav_put32(header+54,bytes);
   wav_write_all(fd,header,sizeof(header));

   unsigned char buffer[16384];
   for (std::size_t begin=0; begin<samples;) {
      const std::size_t count=std::min(samples-begin,sizeof(buffer)/4);
      for (std::size_t i=0; i<count; ++i) {
         const double value=normalised ? (peak>0 ? data[begin+i]/peak : 0.0) : data[begin+i];
         const float sample=static_cast<float>(value);
         std::uint32_t bits;
         std::memcpy(&bits,&sample,sizeof(bits));
         wav_put32(buffer+4*i,bits);
      }
      wav_write_all(fd,buffer,count*4);
      begin+=count;
   }
}

} // namespace detail

// Input is contiguous planar data: channels rows, each with samples values.
// All destinations are checked and every file is staged before publication.
// Each published name is atomic; default publication is exclusive and rolls
// back this call's new files if a concurrent writer causes a later collision.
inline std::vector<std::string> export_wav(const std::string &directory,
   const std::vector<double> &planar, std::size_t channels, std::size_t samples,
   double fs, bool overwrite=false)
{
   static_assert(std::numeric_limits<float>::is_iec559 &&
                 std::numeric_limits<float>::digits==24 && sizeof(float)==4,
                 "WAV export requires IEEE binary32 float");
   if (directory.empty() || directory.find('\0')!=std::string::npos)
      throw std::invalid_argument("WAV output directory must be nonempty");
   if (!std::isfinite(fs) || fs<=0 || std::floor(fs)!=fs ||
       fs>std::numeric_limits<std::uint32_t>::max()/4)
      throw std::invalid_argument("WAV sample rate must be a positive representable integer");
   if (!channels || !samples || samples>(std::numeric_limits<std::uint32_t>::max()-50u)/4u ||
       channels>std::numeric_limits<std::size_t>::max()/samples ||
       channels>std::vector<detail::WavFile>().max_size()/2 ||
       planar.size()!=channels*samples)
      throw std::invalid_argument("WAV planar shape or RIFF32 extent is invalid");
   double peak=0;
   for (double value : planar) {
      if (!std::isfinite(value)) throw std::invalid_argument("WAV samples must be finite");
      peak=std::max(peak,std::fabs(value));
   }
   const bool native=peak<1;
   const int directory_fd=::open(directory.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC);
   if (directory_fd<0) throw detail::wav_error("Cannot open WAV output directory");
   detail::WavFd held_directory(directory_fd);
   std::vector<detail::WavFile> files;
   std::vector<std::string> paths;
   files.reserve(channels*(native ? 2 : 1));
   paths.reserve(files.capacity());
   const std::string prefix=directory+(directory.back()=='/' ? "" : "/");
   for (std::size_t channel=0; channel<channels; ++channel) {
      char receiver[64];
      const int count=std::snprintf(receiver,sizeof(receiver),"R%03zu_out",channel+1);
      if (count<0 || static_cast<std::size_t>(count)>=sizeof(receiver))
         throw std::invalid_argument("WAV receiver filename is not representable");
      for (unsigned kind=0; kind<(native ? 2u : 1u); ++kind) {
         detail::WavFile file;
         file.name=std::string(receiver)+(kind ? "_native.wav" : "_normalised.wav");
         if (::fstatat(directory_fd,file.name.c_str(),&file.original,AT_SYMLINK_NOFOLLOW)==0) {
            if (!overwrite) throw std::runtime_error("WAV output already exists: "+prefix+file.name);
            if (!S_ISREG(file.original.st_mode) || file.original.st_nlink!=1)
               throw std::runtime_error("WAV overwrite requires a regular file without hardlinks: "+prefix+file.name);
            file.exists=true;
         }
         else if (errno!=ENOENT) throw detail::wav_error("Cannot inspect WAV output "+prefix+file.name);
         paths.push_back(prefix+file.name);
         files.push_back(file);
      }
   }

   detail::WavStaging staging(directory_fd,files);
   for (std::size_t i=0; i<files.size(); ++i) {
      detail::WavFile &file=files[i];
      detail::WavFd fd(detail::wav_temporary(directory_fd,file.temporary));
      const std::size_t channel=i/(native ? 2 : 1);
      detail::wav_encode(fd.get(),planar.data()+channel*samples,samples,
                         static_cast<std::uint32_t>(fs),peak,!native || i%2==0);
      if (::fsync(fd.get())<0 || ::fstat(fd.get(),&file.staged)<0)
         throw detail::wav_error("Cannot finish temporary WAV file");
      fd.close();
   }
   try {
      for (detail::WavFile &file : files) {
         if (file.exists) {
            struct stat current;
            if (::fstatat(directory_fd,file.name.c_str(),&current,AT_SYMLINK_NOFOLLOW)<0 ||
                !detail::wav_same_file(current,file.original) || !S_ISREG(current.st_mode) || current.st_nlink!=1)
               throw std::runtime_error("WAV output changed during overwrite: "+prefix+file.name);
            if (::renameat(directory_fd,file.temporary.c_str(),directory_fd,file.name.c_str())<0)
               throw detail::wav_error("Cannot publish WAV output "+prefix+file.name);
            file.temporary.clear();
         }
         else {
            if (::linkat(directory_fd,file.temporary.c_str(),directory_fd,file.name.c_str(),0)<0)
               throw detail::wav_error("Cannot publish WAV output exclusively "+prefix+file.name);
         }
         file.published=true;
      }
   }
   catch (...) {
      // Only erase this call's new inode; leave concurrent writers untouched.
      for (const detail::WavFile &file : files) {
         struct stat current;
         if (file.published && !file.exists &&
             ::fstatat(directory_fd,file.name.c_str(),&current,AT_SYMLINK_NOFOLLOW)==0 &&
             detail::wav_same_file(current,file.staged))
            ::unlinkat(directory_fd,file.name.c_str(),0);
      }
      throw;
   }
   return paths;
}

} // namespace pffdtd_post
#endif
