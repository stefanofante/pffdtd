// Native comparison of rigid+ADE fusion against an independent rigid reduction
// and the existing node-major CPU ADE implementation. Compile with
// -frounding-math -ffp-contract=off; CUDA runtime equivalence is a separate gate.
#include <cpu_engine.h>
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void require(bool condition, const char *message)
{
   if (!condition) {
      std::fprintf(stderr,"FAIL: %s\n",message);
      std::exit(EXIT_FAILURE);
   }
}

// The FP32 CUDA stencil uses add rounded toward zero, rather than the host's
// default nearest mode. Volatile operands plus -frounding-math preserve the
// explicitly selected mode; restore it before any ADE arithmetic is executed.
static Real native_add_off(Real a, Real b)
{
#if PRECISION == 1
   const int previous=std::fegetround();
   require(previous != -1 && std::fesetround(FE_TOWARDZERO)==0,"rounding mode unavailable");
   volatile Real left=a, right=b;
   volatile Real result=left+right;
   require(std::fesetround(previous)==0,"rounding mode could not be restored");
   return result;
#else
   return a+b;
#endif
}

static Real native_fma_diag(Real a, Real b, Real c)
{
   return std::fma(a,b,c);
}

#undef ADD_O
#define ADD_O native_add_off
#undef FMA_D
#define FMA_D native_fma_diag
#include <boundary_stencil.h>
#include <boundary_map.h>

namespace {

struct Shape { int nx, ny, nz; };
enum { MATERIALS=13 };

int64_t index(const Shape &s, int x, int y, int z)
{
   return (static_cast<int64_t>(z)*s.ny+y)*s.nx+x;
}

// Independent address table and reduction tree, rather than another invocation
// of the helper being tested. Its order matches the pre-fusion CUDA stencil.
Real reference_rigid(bool fcc, const Real *u1, Real previous, int64_t ii,
                     uint16_t adjacency, int8_t neighbours, Real sl2, Real a2,
                     int64_t nx, int64_t plane)
{
   const int64_t cart_offsets[]={plane,-plane,nx,-nx,1,-1};
   const int64_t fcc_offsets[]={plane+nx,-plane-nx,nx+1,-nx-1,
                               plane+1,-plane-1,plane-nx,-plane+nx,
                               nx-1,-nx+1,plane-1,-plane+1};
   const int64_t *offsets=fcc ? fcc_offsets : cart_offsets;
   const int pairs=fcc ? 6 : 3;
   Real pair[6];
   for (int p=0; p<pairs; ++p) {
      const int first=2*p, second=first+1;
      const Real left=static_cast<Real>((adjacency>>first)&1)*u1[ii+offsets[first]];
      const Real right=static_cast<Real>((adjacency>>second)&1)*u1[ii+offsets[second]];
      pair[p]=native_add_off(left,right);
   }
   Real total;
   if (fcc) {
      const Real left=native_add_off(native_add_off(pair[0],pair[1]),pair[4]);
      const Real right=native_add_off(native_add_off(pair[2],pair[3]),pair[5]);
      total=native_add_off(left,right);
   }
   else total=native_add_off(native_add_off(pair[0],pair[1]),pair[2]);
   const Real diagonal=Real(2)-sl2*static_cast<Real>(neighbours);
   return native_fma_diag(diagonal,u1[ii],native_fma_diag(a2,total,-previous));
}

void same(const char *name, const std::vector<Real> &expected,
          const std::vector<Real> &actual, int step, int ade_mode)
{
   require(expected.size()==actual.size(),"comparison length differs");
   for (std::size_t i=0; i<expected.size(); ++i) {
      if (!std::isfinite(expected[i]) || !std::isfinite(actual[i]) ||
          std::memcmp(&expected[i],&actual[i],sizeof(Real))!=0) {
         std::fprintf(stderr,"FAIL %s FP%d ADEmode=%d step=%d index=%zu: %.17g != %.17g\n",
                      name,static_cast<int>(8*sizeof(Real)),ade_mode,step,i,
                      static_cast<double>(expected[i]),static_cast<double>(actual[i]));
         std::exit(EXIT_FAILURE);
      }
   }
}

template<bool FCC, typename MapIdx, int ADEMode>
void check_case(const Shape &shape, std::size_t nb, int lossy_mode, int material_mode,
                unsigned seed)
{
   const int branches=FCC ? 12 : 6;
   const int64_t plane=static_cast<int64_t>(shape.nx)*shape.ny;
   const int64_t count=plane*shape.nz;
   std::mt19937 random(seed);
   std::vector<int64_t> candidates;
   for (int z=1; z<shape.nz-1; ++z)
      for (int y=1; y<shape.ny-1; ++y)
         for (int x=1; x<shape.nx-1; ++x)
            candidates.push_back(index(shape,x,y,z));
   require(nb<=candidates.size(),"too many fixture boundary nodes");
   std::shuffle(candidates.begin(),candidates.end(),random);
   const std::vector<int64_t> boundary(candidates.begin(),candidates.begin()+nb);
   std::vector<int64_t> lossy;
   for (std::size_t i=0; i<nb; ++i)
      if (lossy_mode==1 || (lossy_mode==2 && i%2==0)) lossy.push_back(boundary[i]);
   // Preserve a lossy order different from both the boundary and grid order.
   std::shuffle(lossy.begin(),lossy.end(),random);
   const int64_t nbl=static_cast<int64_t>(lossy.size());
   std::vector<MapIdx> map(nb);
   require(pffdtd::build_boundary_lossy_map(boundary.data(),nb,lossy.data(),nbl,
                                           count,map.data())==pffdtd::BoundaryMapStatus::Success,
           "valid fixture map rejected");
   std::vector<unsigned char> is_boundary(static_cast<std::size_t>(count),0);
   std::vector<uint16_t> adjacency(nb);
   std::vector<int8_t> neighbours(nb);
   for (std::size_t i=0; i<nb; ++i) {
      is_boundary[static_cast<std::size_t>(boundary[i])]=1;
      const uint16_t full=static_cast<uint16_t>((1u<<branches)-1u);
      uint16_t bits=static_cast<uint16_t>(random()&full);
      if (bits==full) bits^=static_cast<uint16_t>(1u<<(i%branches));
      if (map[i]>=0 && bits==0) bits=1;
      if (map[i]<0 && i%7==0) bits=0;
      adjacency[i]=bits;
      int active=0;
      for (int b=0; b<branches; ++b) active+=(bits>>b)&1;
      neighbours[i]=static_cast<int8_t>(active);
   }
   int8_t poles[MATERIALS],mutable_poles[MATERIALS];
   Real beta[MATERIALS];
   for (int k=0; k<MATERIALS; ++k) {
      poles[k]=mutable_poles[k]=static_cast<int8_t>(k);
      beta[k]=Real(k+1)*Real(0.0625);
   }
   std::vector<MatQuad> quads(MATERIALS*MMb);
   for (int k=0; k<MATERIALS; ++k)
      for (int m=0; m<MMb; ++m) {
         MatQuad &q=quads[k*MMb+m];
         q.b=Real(m+1)*Real(0.0009765625);
         q.bd=Real(0.5);
         q.bDh=Real(m+1)*Real(0.001953125);
         q.bFh=Real(k+1)*Real(0.0009765625);
      }
   std::vector<Real> surface(static_cast<std::size_t>(nbl));
   std::vector<int8_t> materials(static_cast<std::size_t>(nbl));
   for (int64_t i=0; i<nbl; ++i) {
      materials[static_cast<std::size_t>(i)]=static_cast<int8_t>(material_mode<0 ? i%MATERIALS : material_mode);
      surface[static_cast<std::size_t>(i)]=Real(i%5+1)*Real(0.125);
   }
   std::vector<Real> ref_grid[2], fused_grid[2], ref_carry[3], fused_carry[3];
   for (int bank=0; bank<2; ++bank) {
      ref_grid[bank].resize(static_cast<std::size_t>(count));
      for (Real &pressure : ref_grid[bank])
         pressure=Real(static_cast<int>(random()%4096)-2048)*Real(0.00390625);
      fused_grid[bank]=ref_grid[bank];
   }
   // Padding sentinels verify that rigid nodes never touch lossy carry slots.
   const std::size_t carry_count=static_cast<std::size_t>(nbl)+3;
   for (int bank=0; bank<3; ++bank) {
      ref_carry[bank].resize(carry_count);
      for (std::size_t i=0; i<carry_count; ++i)
         ref_carry[bank][i]=Real((i+1)*(bank+1))*Real(0.0078125);
      fused_carry[bank]=ref_carry[bank];
   }
   const std::size_t states=static_cast<std::size_t>(nbl)*MMb;
   std::vector<Real> ref_vh(states),ref_gh(states),fused_vh(states),fused_gh(states);
   for (int64_t i=0; i<nbl; ++i)
      for (int m=0; m<MMb; ++m) {
         const Real v=Real((i+1)*(m+1))*Real(0.0009765625);
         const Real g=-Real((i+2)*(m+1))*Real(0.00048828125);
         ref_vh[static_cast<std::size_t>(i)*MMb+m]=v;
         ref_gh[static_cast<std::size_t>(i)*MMb+m]=g;
         fused_vh[static_cast<std::size_t>(m)*nbl+i]=v;
         fused_gh[static_cast<std::size_t>(m)*nbl+i]=g;
      }
   std::vector<std::size_t> threads(nb);
   for (std::size_t i=0; i<nb; ++i) threads[i]=i;
   const Real sl2=FCC ? Real(0.015625) : Real(0.0625);
   const Real a2=FCC ? Real(0.015625) : Real(0.0625);
   const Real lo2=Real(0.125);
   int write=0,read=1,carry0=0,carry1=1,carry2=2;
   for (int step=0; step<37; ++step) {
      if (FCC)
         for (int z=0; z<shape.nz; ++z)
            for (int x=0; x<shape.nx; ++x) {
               ref_grid[read][static_cast<std::size_t>(index(shape,x,shape.ny-1,z))]
                  =ref_grid[read][static_cast<std::size_t>(index(shape,x,shape.ny-2,z))];
               fused_grid[read][static_cast<std::size_t>(index(shape,x,shape.ny-1,z))]
                  =fused_grid[read][static_cast<std::size_t>(index(shape,x,shape.ny-2,z))];
            }
      // Evolve non-boundary cells identically; boundary previous pressures stay
      // intact until the reference/new stencil consumes them.
      for (std::size_t i=0; i<static_cast<std::size_t>(count); ++i) {
         if (is_boundary[i]) continue;
         const Real forcing=Real((step%7)-3)*Real(0.0009765625);
         ref_grid[write][i]=Real(1.125)*ref_grid[read][i]-ref_grid[write][i]+forcing;
         fused_grid[write][i]=Real(1.125)*fused_grid[read][i]-fused_grid[write][i]+forcing;
      }
      for (std::size_t i=0; i<nb; ++i) {
         const int64_t ii=boundary[i];
         ref_grid[write][static_cast<std::size_t>(ii)]=reference_rigid(
            FCC,ref_grid[read].data(),ref_grid[write][static_cast<std::size_t>(ii)],
            ii,adjacency[i],neighbours[i],sl2,a2,shape.nx,plane);
      }
      for (int64_t i=0; i<nbl; ++i)
         ref_carry[carry0][static_cast<std::size_t>(i)]
            =ref_grid[write][static_cast<std::size_t>(lossy[static_cast<std::size_t>(i)])];
      process_bnl_pts_fd(ref_carry[carry0].data(),ref_carry[carry2].data(),surface.data(),
                        materials.data(),nbl,mutable_poles,lo2,ref_vh.data(),ref_gh.data(),
                        quads.data(),beta);
      for (int64_t i=0; i<nbl; ++i)
         ref_grid[write][static_cast<std::size_t>(lossy[static_cast<std::size_t>(i)])]
            =ref_carry[carry0][static_cast<std::size_t>(i)];
      std::shuffle(threads.begin(),threads.end(),random);
      for (std::size_t i : threads)
         boundary_stencil_node<FCC,MapIdx,ADEMode>(fused_grid[write].data(),fused_grid[read].data(),
            fused_carry[carry0].data(),fused_carry[carry2].data(),fused_vh.data(),fused_gh.data(),
            boundary.data(),adjacency.data(),neighbours.data(),map.data(),surface.data(),
            materials.data(),beta,quads.data(),poles,sl2,a2,lo2,shape.nx,plane,nbl,i);
      for (int bank=0; bank<2; ++bank) same("grid",ref_grid[bank],fused_grid[bank],step,ADEMode);
      for (int bank=0; bank<3; ++bank) same("carry",ref_carry[bank],fused_carry[bank],step,ADEMode);
      std::vector<Real> expected_vh(states),expected_gh(states);
      for (int64_t i=0; i<nbl; ++i)
         for (int m=0; m<MMb; ++m) {
            expected_vh[static_cast<std::size_t>(m)*nbl+i]=ref_vh[static_cast<std::size_t>(i)*MMb+m];
            expected_gh[static_cast<std::size_t>(m)*nbl+i]=ref_gh[static_cast<std::size_t>(i)*MMb+m];
         }
      same("vh including inactive poles",expected_vh,fused_vh,step,ADEMode);
      same("gh including inactive poles",expected_gh,fused_gh,step,ADEMode);
      std::swap(read,write);
      const int oldest=carry2; carry2=carry1; carry1=carry0; carry0=oldest;
   }
}

} // namespace

int main()
{
   require(std::fesetround(FE_TONEAREST)==0,"nearest rounding unavailable");
   const Shape shapes[]={{3,3,3},{5,7,9},{9,11,13}};
   unsigned cases=0;
   for (const Shape &shape : shapes) {
      const std::size_t inside=static_cast<std::size_t>(shape.nx-2)*(shape.ny-2)*(shape.nz-2);
      std::vector<std::size_t> counts={0,1,std::min<std::size_t>(17,inside),
                                     std::min<std::size_t>(257,inside),inside};
      std::sort(counts.begin(),counts.end());
      counts.erase(std::unique(counts.begin(),counts.end()),counts.end());
      for (std::size_t nb : counts)
         for (int mode=0; mode<3; ++mode)
            for (int material=-1; material<MATERIALS; ++material) {
               if ((mode==0 || nb==0) && material!=-1) continue;
               if (nb==0 && mode!=0) continue;
               const unsigned seed=0xb0a0da7au+cases;
               check_case<false,int32_t,0>(shape,nb,mode,material,seed);
               check_case<false,int64_t,0>(shape,nb,mode,material,seed);
               check_case<true,int32_t,0>(shape,nb,mode,material,seed);
               check_case<true,int64_t,0>(shape,nb,mode,material,seed);
               check_case<false,int32_t,1>(shape,nb,mode,material,seed);
               check_case<false,int64_t,1>(shape,nb,mode,material,seed);
               check_case<true,int32_t,1>(shape,nb,mode,material,seed);
               check_case<true,int64_t,1>(shape,nb,mode,material,seed);
               check_case<false,int32_t,2>(shape,nb,mode,material,seed);
               check_case<false,int64_t,2>(shape,nb,mode,material,seed);
               check_case<true,int32_t,2>(shape,nb,mode,material,seed);
               check_case<true,int64_t,2>(shape,nb,mode,material,seed);
               cases+=12;
            }
   }
   std::printf("PASS: rigid+ADE FP%d, %u cases x 37 steps, Cart/FCC, all 0..12 poles, "
               "rigid/lossy/mixed nodes, int32/int64 maps, shuffled order, 2/3 rotations; "
               "generic/reload/fixed ADE; native rounding model only\n",static_cast<int>(8*sizeof(Real)),cases);
   return EXIT_SUCCESS;
}
