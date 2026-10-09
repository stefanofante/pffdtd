// Fixed-pole and scalar ADE regressions against the independent node-major CPU
// recurrence. Compile with -ffp-contract=off and PRECISION=1/2. Device equivalence
// and performance still require CUDA runtime tests on the selected hardware.
#include <cpu_engine.h>
#include <boundary_fd_specialized.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

namespace {

enum { GridPoints = 49, MaxNodes = 17, Materials = 13, Steps = 37 };

void same(const char *name, const Real *expected, const Real *actual,
          std::size_t count, int step)
{
   for (std::size_t i=0; i<count; ++i)
      if (!std::isfinite(expected[i]) || !std::isfinite(actual[i]) ||
          std::memcmp(&expected[i],&actual[i],sizeof(Real))!=0) {
         std::fprintf(stderr,"FAIL %s FP%d step=%d index=%zu: %.17g != %.17g\n",
                      name,static_cast<int>(8*sizeof(Real)),step,i,
                      static_cast<double>(expected[i]),static_cast<double>(actual[i]));
         std::exit(EXIT_FAILURE);
      }
}

void check_case(int64_t nbl, int material, bool scalar_only, unsigned seed)
{
   int8_t poles[Materials], materials[MaxNodes];
   int64_t locations[MaxNodes];
   Real surface[MaxNodes], beta[Materials];
   MatQuad quads[Materials*MMb];
   Real reference_grid[2][GridPoints], specialized_grid[2][GridPoints];
   Real reference_carry[3][MaxNodes], specialized_carry[3][MaxNodes];
   Real reference_vh[MaxNodes*MMb], reference_gh[MaxNodes*MMb];
   Real specialized_vh[MaxNodes*MMb], specialized_gh[MaxNodes*MMb];
   Real expected_vh[MaxNodes*MMb], expected_gh[MaxNodes*MMb];
   int write_grid=0, read_grid=1, carry0=0, carry1=1, carry2=2;
   std::mt19937 random(seed);
   std::array<int64_t,MaxNodes> order;
   const Real lo2=Real(0.25);

   for (int bank=0; bank<2; ++bank)
      for (int i=0; i<GridPoints; ++i)
         reference_grid[bank][i]=Real((i+1)*(bank+1))*Real(0.015625);
   std::memcpy(specialized_grid,reference_grid,sizeof(reference_grid));
   for (int bank=0; bank<3; ++bank)
      for (int nb=0; nb<MaxNodes; ++nb)
         reference_carry[bank][nb]=Real((nb+1)*(bank+1))*Real(0.0078125);
   std::memcpy(specialized_carry,reference_carry,sizeof(reference_carry));
   // Initialize unused tail storage too, so every allocated history slot is
   // checked rather than silently comparing only the active pole ranges.
   for (int i=0; i<MaxNodes*MMb; ++i) {
      reference_vh[i]=specialized_vh[i]=Real(i+1)*Real(0.000244140625);
      reference_gh[i]=specialized_gh[i]=-Real(i+2)*Real(0.0001220703125);
   }
   for (int k=0; k<Materials; ++k) {
      poles[k]=static_cast<int8_t>(k);
      beta[k]=Real(k%4+1)*Real(0.125);
      for (int m=0; m<MMb; ++m) {
         MatQuad &quad=quads[k*MMb+m];
         quad.b=Real(m+1)*Real(0.0009765625);
         quad.bd=Real(0.5);
         quad.bDh=Real(m+1)*Real(0.001953125);
         quad.bFh=Real(k+1)*Real(0.000244140625);
      }
   }
   for (int64_t nb=0; nb<nbl; ++nb) {
      order[static_cast<std::size_t>(nb)]=nb;
      locations[nb]=2+2*nb;
      materials[nb]=static_cast<int8_t>(material<0 ? nb%Materials : material);
      surface[nb]=(nb%7==0 && nb>0) ? Real(0) : Real(nb%5+1)*Real(0.125);
      for (int m=0; m<MMb; ++m) {
         const Real v=Real((nb+1)*(m+1))*Real(0.0009765625);
         const Real g=-Real((nb+2)*(m+1))*Real(0.00048828125);
         reference_vh[nb*MMb+m]=specialized_vh[m*nbl+nb]=v;
         reference_gh[nb*MMb+m]=specialized_gh[m*nbl+nb]=g;
      }
   }
   for (int step=0; step<Steps; ++step) {
      for (int i=0; i<GridPoints; ++i) {
         const Real pulse=Real(step%7-3)*Real(0.0009765625);
         reference_grid[write_grid][i]=Real(1.5)*reference_grid[read_grid][i]
                                         -reference_grid[write_grid][i]+pulse;
         specialized_grid[write_grid][i]=Real(1.5)*specialized_grid[read_grid][i]
                                           -specialized_grid[write_grid][i]+pulse;
      }
      for (int64_t nb=0; nb<nbl; ++nb)
         reference_carry[carry0][nb]=reference_grid[write_grid][locations[nb]];
      process_bnl_pts_fd(reference_carry[carry0],reference_carry[carry2],surface,
                        materials,nbl,poles,lo2,reference_vh,reference_gh,quads,beta);
      for (int64_t nb=0; nb<nbl; ++nb)
         reference_grid[write_grid][locations[nb]]=reference_carry[carry0][nb];

      std::shuffle(order.begin(),order.begin()+nbl,random);
      for (int64_t j=0; j<nbl; ++j) {
         const int64_t nb=order[static_cast<std::size_t>(j)], location=locations[nb];
         Real pressure=specialized_grid[write_grid][location];
         if (scalar_only)
            pressure=boundary_fd_scalar_value_node(pressure,specialized_carry[carry0],
                  specialized_carry[carry2],specialized_vh,specialized_gh,surface,
                  materials,beta,quads,poles,lo2,nbl,nb);
         else
            pressure=boundary_fd_specialized_value_node(pressure,specialized_carry[carry0],
                  specialized_carry[carry2],specialized_vh,specialized_gh,surface,
                  materials,beta,quads,poles,lo2,nbl,nb);
         specialized_grid[write_grid][location]=pressure;
      }

      for (int bank=0; bank<2; ++bank)
         same("grid",reference_grid[bank],specialized_grid[bank],GridPoints,step);
      for (int bank=0; bank<3; ++bank)
         same("pressure carry",reference_carry[bank],specialized_carry[bank],MaxNodes,step);
      // The unused tail has the same original sentinel layout in both forms.
      std::memcpy(expected_vh,reference_vh,sizeof(reference_vh));
      std::memcpy(expected_gh,reference_gh,sizeof(reference_gh));
      for (int64_t nb=0; nb<nbl; ++nb)
         for (int m=0; m<MMb; ++m) {
            expected_vh[m*nbl+nb]=reference_vh[nb*MMb+m];
            expected_gh[m*nbl+nb]=reference_gh[nb*MMb+m];
         }
      same("vh1 including unused poles/tail",expected_vh,specialized_vh,MaxNodes*MMb,step);
      same("gh1 including unused poles/tail",expected_gh,specialized_gh,MaxNodes*MMb,step);
      std::swap(write_grid,read_grid);
      const int oldest=carry2;
      carry2=carry1; carry1=carry0; carry0=oldest;
   }
}

void check_zero_poles_null_states()
{
   int8_t poles[1]={0}, materials[1]={0};
   Real surface[1]={Real(0.75)}, beta[1]={Real(0.25)}, previous[1]={Real(-0.125)};
   Real expected[1]={Real(0.375)}, specialized[1]={0}, scalar[1]={0};
   const Real initial=expected[0], lo2=Real(0.125);
   process_bnl_pts_fd(expected,previous,surface,materials,1,poles,lo2,
                     nullptr,nullptr,nullptr,beta);
   specialized[0]=boundary_fd_specialized_value_node(initial,specialized,previous,
                     nullptr,nullptr,surface,materials,beta,nullptr,poles,lo2,1,0);
   scalar[0]=boundary_fd_scalar_value_node(initial,scalar,previous,nullptr,nullptr,
                     surface,materials,beta,nullptr,poles,lo2,1,0);
   same("zero poles with null state storage",expected,specialized,1,0);
   same("scalar zero poles with null state storage",expected,scalar,1,0);
}

} // namespace

int main()
{
   check_zero_poles_null_states();
   int cases=0;
   for (bool scalar_only : {false,true}) {
      check_case(0,-1,scalar_only,29); ++cases;
      for (int material=0; material<Materials; ++material)
         for (int64_t nodes : {int64_t(1),int64_t(MaxNodes)}) {
            check_case(nodes,material,scalar_only,static_cast<unsigned>(material+nodes));
            ++cases;
         }
      check_case(MaxNodes,-1,scalar_only,103); ++cases;
   }
   std::printf("boundary_fd_specialized_test: PASS FP%d, %d cases x %d steps, 0..12 poles, mixed materials, specialized/scalar, all grid/carry/state/padding rotations\n",
               static_cast<int>(8*sizeof(Real)),cases,Steps);
   return EXIT_SUCCESS;
}
