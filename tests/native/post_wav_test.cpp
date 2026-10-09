// Decode the emitted RIFF files independently, without an audio library.
#include <post_wav.h>
#include <cassert>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <dirent.h>

namespace {

unsigned cases=0;

void check(bool condition, const char *message)
{
   if (!condition) throw std::runtime_error(message);
}

std::uint16_t read16(const std::vector<unsigned char> &bytes, std::size_t offset)
{
   check(offset+2<=bytes.size(),"truncated 16-bit field");
   return static_cast<std::uint16_t>(bytes[offset]|(std::uint16_t(bytes[offset+1])<<8));
}

std::uint32_t read32(const std::vector<unsigned char> &bytes, std::size_t offset)
{
   check(offset+4<=bytes.size(),"truncated 32-bit field");
   std::uint32_t value=0;
   for (unsigned i=0; i<4; ++i) value|=std::uint32_t(bytes[offset+i])<<(8*i);
   return value;
}

std::vector<unsigned char> bytes_at(const std::string &name)
{
   std::ifstream input(name.c_str(),std::ios::binary);
   check(input.good(),"cannot open expected output");
   return std::vector<unsigned char>(std::istreambuf_iterator<char>(input),{});
}

struct Wave {
   unsigned rate=0;
   std::vector<std::uint32_t> bits;
};

Wave decode(const std::string &name)
{
   const std::vector<unsigned char> bytes=bytes_at(name);
   check(bytes.size()>=12,"RIFF header missing");
   check(std::memcmp(bytes.data(),"RIFF",4)==0,"wrong RIFF signature");
   check(std::memcmp(bytes.data()+8,"WAVE",4)==0,"wrong WAVE signature");
   check(read32(bytes,4)==bytes.size()-8,"incorrect RIFF size");
   Wave wave;
   bool format=false,fact=false,data=false;
   std::uint32_t frames=0;
   for (std::size_t pos=12; pos<bytes.size();) {
      check(pos+8<=bytes.size(),"truncated chunk");
      const std::uint32_t size=read32(bytes,pos+4);
      check(size<=bytes.size()-pos-8,"chunk exceeds file");
      const std::size_t body=pos+8;
      if (std::memcmp(bytes.data()+pos,"fmt ",4)==0) {
         check(!format && size==18,"invalid float fmt chunk");
         check(read16(bytes,body)==3,"not IEEE float");
         check(read16(bytes,body+2)==1,"not mono");
         wave.rate=read32(bytes,body+4);
         check(read32(bytes,body+8)==4*wave.rate,"incorrect byte rate");
         check(read16(bytes,body+12)==4 && read16(bytes,body+14)==32,"incorrect alignment or depth");
         check(read16(bytes,body+16)==0,"unexpected format extension");
         format=true;
      }
      else if (std::memcmp(bytes.data()+pos,"fact",4)==0) {
         check(!fact && size==4,"invalid fact chunk");
         frames=read32(bytes,body);
         fact=true;
      }
      else if (std::memcmp(bytes.data()+pos,"data",4)==0) {
         check(!data && size%4==0,"invalid data chunk");
         for (std::size_t i=0; i<size; i+=4) {
            const std::uint32_t bits=read32(bytes,body+i);
            float sample;
            std::memcpy(&sample,&bits,sizeof(sample));
            check(std::isfinite(sample),"nonfinite encoded sample");
            wave.bits.push_back(bits);
         }
         data=true;
      }
      else throw std::runtime_error("unexpected RIFF chunk");
      pos=body+size+(size%2);
   }
   check(format && fact && data && frames==wave.bits.size(),"missing chunk or wrong frame count");
   return wave;
}

void remove_tree(const std::string &path)
{
   DIR *directory=::opendir(path.c_str());
   if (!directory) return;
   while (dirent *entry=::readdir(directory)) {
      const std::string name(entry->d_name);
      if (name=="." || name=="..") continue;
      const std::string child=path+"/"+name;
      struct stat info;
      if (::lstat(child.c_str(),&info)==0 && S_ISDIR(info.st_mode)) remove_tree(child);
      else ::unlink(child.c_str());
   }
   ::closedir(directory);
   ::rmdir(path.c_str());
}

class TestDirectory {
public:
   std::string path;
   TestDirectory()
   {
      char name[]="/tmp/pffdtd-wav-test.XXXXXX";
      const char *created=::mkdtemp(name);
      check(created!=nullptr,"cannot create test directory");
      path=created;
   }
   ~TestDirectory() { remove_tree(path); }
   std::string child(const char *name)
   {
      const std::string result=path+"/"+name;
      check(::mkdir(result.c_str(),0700)==0,"cannot create case directory");
      return result;
   }
};

std::size_t entry_count(const std::string &path)
{
   DIR *directory=::opendir(path.c_str());
   check(directory!=nullptr,"cannot inspect case directory");
   std::size_t count=0;
   while (dirent *entry=::readdir(directory))
      if (std::strcmp(entry->d_name,".") && std::strcmp(entry->d_name,"..")) ++count;
   ::closedir(directory);
   return count;
}

void reject(const std::function<void()> &operation)
{
   bool failed=false;
   try { operation(); } catch (const std::exception &) { failed=true; }
   check(failed,"expected invalid operation to fail");
   ++cases;
}

void write_text(const std::string &path)
{
   std::ofstream output(path.c_str(),std::ios::binary);
   output<<"sentinel";
   check(output.good(),"cannot create sentinel");
}

void verify_bits(const std::vector<std::uint32_t> &actual,
                 std::initializer_list<std::uint32_t> expected)
{
   check(actual==std::vector<std::uint32_t>(expected),"incorrect sample encoding");
}

} // namespace

int main()
{
   try {
      TestDirectory root;
      {
         const std::string dir=root.child("headroom");
         const std::vector<std::string> files=pffdtd_post::export_wav(dir,{0.25,-0.5,0,0.125,0.25,-0.125},2,3,48000);
         check(files.size()==4,"wrong native file count");
         check(files[0]==dir+"/R001_out_normalised.wav" && files[1]==dir+"/R001_out_native.wav" &&
               files[2]==dir+"/R002_out_normalised.wav" && files[3]==dir+"/R002_out_native.wav","wrong receiver filenames");
         check(decode(files[0]).rate==48000,"incorrect sample rate");
         verify_bits(decode(files[0]).bits,{0x3f000000u,0xbf800000u,0});
         verify_bits(decode(files[1]).bits,{0x3e800000u,0xbf000000u,0});
         verify_bits(decode(files[2]).bits,{0x3e800000u,0x3f000000u,0xbe800000u});
         verify_bits(decode(files[3]).bits,{0x3e000000u,0x3e800000u,0xbe000000u});
         check(entry_count(dir)==4,"temporary file leaked");
         ++cases;
      }
      for (double peak : {1.0,2.0,std::numeric_limits<double>::max()}) {
         const std::string dir=root.child(peak==1 ? "unity" : peak==2 ? "above-unity" : "large-finite");
         const std::vector<std::string> files=pffdtd_post::export_wav(dir,{peak,-peak,peak/2},1,3,44100);
         check(files.size()==1 && entry_count(dir)==1,"native file written without headroom");
         verify_bits(decode(files[0]).bits,{0x3f800000u,0xbf800000u,0x3f000000u});
         ++cases;
      }
      {
         const std::string dir=root.child("silence");
         const auto files=pffdtd_post::export_wav(dir,{0.0,-0.0,0.0,0.0},2,2,1);
         check(files.size()==4,"silence must retain native files");
         for (const std::string &file : files) {
            const Wave wave=decode(file);
            check(wave.rate==1,"low positive rate rejected");
            for (std::uint32_t bits : wave.bits) check((bits&0x7fffffffu)==0,"silence is not zero");
         }
         ++cases;
      }
      {
         const std::string dir=root.child("block-tail");
         std::vector<double> values(4097,0.25);
         values.back()=-0.5;
         const auto files=pffdtd_post::export_wav(dir,values,1,values.size(),192000);
         const Wave wave=decode(files[0]);
         check(wave.bits.size()==4097 && wave.bits[4095]==0x3f000000u &&
               wave.bits.back()==0xbf800000u,"block tail missing or misnormalised");
         ++cases;
      }
      {
         const std::string dir=root.child("last-collision");
         const std::string last=dir+"/R002_out_native.wav";
         write_text(last);
         const auto sentinel=bytes_at(last);
         reject([&] { pffdtd_post::export_wav(dir,{0.25,0.5},2,1,48000); });
         check(entry_count(dir)==1 && bytes_at(last)==sentinel,"collision preflight partially published or overwrote files");
         const auto files=pffdtd_post::export_wav(dir,{0.25,0.5},2,1,48000,true);
         check(files.size()==4 && entry_count(dir)==4,"overwrite did not replace whole output set");
         verify_bits(decode(last).bits,{0x3f000000u});
         ++cases;
      }
      {
         const std::string dir=root.child("symlink");
         const std::string input=dir+"/sim_outs.h5", output=dir+"/R001_out_normalised.wav";
         write_text(input);
         check(::symlink("sim_outs.h5",output.c_str())==0,"cannot create symlink fixture");
         reject([&] { pffdtd_post::export_wav(dir,{0.5},1,1,48000,true); });
         check(bytes_at(input)==std::vector<unsigned char>({'s','e','n','t','i','n','e','l'}) && entry_count(dir)==2,
               "symlink input alias was altered");
      }
      {
         const std::string dir=root.child("hardlink");
         const std::string input=dir+"/comms_out.h5", output=dir+"/R001_out_normalised.wav";
         write_text(input);
         check(::link(input.c_str(),output.c_str())==0,"cannot create hardlink fixture");
         reject([&] { pffdtd_post::export_wav(dir,{0.5},1,1,48000,true); });
         check(bytes_at(input)==bytes_at(output) && entry_count(dir)==2,"hardlink alias was altered");
      }
      {
         const std::string dir=root.child("fifo");
         check(::mkfifo((dir+"/R001_out_normalised.wav").c_str(),0600)==0,"cannot create FIFO fixture");
         reject([&] { pffdtd_post::export_wav(dir,{0.5},1,1,48000,true); });
         check(entry_count(dir)==1,"FIFO preflight leaked output");
      }
      {
         const std::string dir=root.child("overwrite-late-alias");
         const std::string first=dir+"/R001_out_normalised.wav";
         const std::string input=dir+"/sim_outs.h5";
         write_text(first);
         write_text(input);
         check(::symlink("sim_outs.h5",(dir+"/R002_out_native.wav").c_str())==0,"cannot create late alias fixture");
         const auto sentinel=bytes_at(first);
         reject([&] { pffdtd_post::export_wav(dir,{0.25,0.5},2,1,48000,true); });
         check(entry_count(dir)==3 && bytes_at(first)==sentinel && bytes_at(input)==sentinel,
               "overwrite preflight altered early files before rejecting a late alias");
      }
      {
         const std::string dir=root.child("directory-output");
         check(::mkdir((dir+"/R001_out_normalised.wav").c_str(),0700)==0,"cannot create directory fixture");
         reject([&] { pffdtd_post::export_wav(dir,{0.5},1,1,48000,true); });
         check(entry_count(dir)==1,"directory preflight leaked output");
      }
      {
         const std::string dir=root.child("invalid");
         for (double rate : {0.0,-1.0,48000.5,1073741824.0,
              std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()})
            reject([&] { pffdtd_post::export_wav(dir,{0.5},1,1,rate); });
         reject([&] { pffdtd_post::export_wav(dir,{},0,1,48000); });
         reject([&] { pffdtd_post::export_wav(dir,{},1,0,48000); });
         reject([&] { pffdtd_post::export_wav(dir,{0.5},1,2,48000); });
         reject([&] { pffdtd_post::export_wav(dir,{0.5},1,1073741812u,48000); });
         reject([&] { pffdtd_post::export_wav(dir,{},std::numeric_limits<std::size_t>::max(),2,48000); });
         reject([&] { pffdtd_post::export_wav(dir,{std::numeric_limits<double>::infinity()},1,1,48000); });
         reject([&] { pffdtd_post::export_wav(dir,{std::numeric_limits<double>::quiet_NaN()},1,1,48000); });
         reject([&] { pffdtd_post::export_wav("",{0.5},1,1,48000); });
         reject([&] { pffdtd_post::export_wav(std::string("a\0b",3),{0.5},1,1,48000); });
         reject([&] { pffdtd_post::export_wav(dir+"/absent",{0.5},1,1,48000); });
         check(entry_count(dir)==0,"invalid parameters created output");
      }
      std::cout<<"post_wav: "<<cases<<" native export/decoder/guard cases passed\n";
      return 0;
   }
   catch (const std::exception &error) {
      std::cerr<<"post_wav test: "<<error.what()<<'\n';
      return 1;
   }
}
