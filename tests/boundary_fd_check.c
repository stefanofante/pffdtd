// Native regression for fused boundary I/O, using the existing CPU ADE reference.
// Build with PRECISION=1 and PRECISION=2; no simulation data or GPU is required.
#include <cpu_engine.h>
#include <boundary_fd.h>

enum { GRID_POINTS = 49, MAX_NODES = 17, MATERIALS = 4, STEPS = 37 };

static void check_equal(const char *name, const Real *expected, const Real *actual,
                        size_t count, int step)
{
   for (size_t i=0; i<count; i++) {
      if (!isfinite(expected[i]) || !isfinite(actual[i])) {
         fprintf(stderr,"%s is nonfinite at step %d index %zu\n",name,step,i);
         exit(EXIT_FAILURE);
      }
      if (memcmp(&expected[i], &actual[i], sizeof(Real)) != 0) {
         fprintf(stderr,"%s differs at step %d index %zu: %.17g != %.17g\n",
                 name,step,i,(double)expected[i],(double)actual[i]);
         exit(EXIT_FAILURE);
      }
   }
}

static void check_case(int64_t Nbl, int material)
{
   const int8_t Mb[MATERIALS] = {0,1,11,12};
   Real ssaf[MAX_NODES];
   int8_t mats[MAX_NODES];
   int64_t locs[MAX_NODES];
   Real beta[MATERIALS] = {0.0,0.125,0.25,0.5};
   struct MatQuad quads[MATERIALS*MMb];
   Real ref_grid[2][GRID_POINTS], fused_grid[2][GRID_POINTS];
   Real ref_carry[3][MAX_NODES], fused_carry[3][MAX_NODES];
   Real ref_vh[MAX_NODES*MMb], ref_gh[MAX_NODES*MMb];
   Real fused_vh[MAX_NODES*MMb], fused_gh[MAX_NODES*MMb];
   Real expected_vh[MAX_NODES*MMb], expected_gh[MAX_NODES*MMb];
   const Real lo2 = 0.25;
   int write_grid=0, read_grid=1;
   int carry0=0, carry1=1, carry2=2;

   // Finite nonzero histories expose bad carry rotations and padded-state writes.
   for (int bank=0; bank<2; bank++) {
      for (int i=0; i<GRID_POINTS; i++) {
         ref_grid[bank][i] = (Real)((i+1)*(bank+1))*0.015625;
      }
   }
   memcpy(fused_grid,ref_grid,sizeof(ref_grid));
   for (int bank=0; bank<3; bank++) {
      for (int nb=0; nb<MAX_NODES; nb++) {
         ref_carry[bank][nb] = (Real)((nb+1)*(bank+1))*0.0078125;
      }
   }
   memcpy(fused_carry,ref_carry,sizeof(ref_carry));
   for (int k=0; k<MATERIALS; k++) {
      for (int m=0; m<MMb; m++) {
         struct MatQuad *tm = &quads[k*MMb+m];
         tm->b = (Real)(m+1)*0.0009765625;
         tm->bd = 0.5;
         tm->bDh = (Real)(m+1)*0.001953125;
         tm->bFh = (Real)(k+1)*0.0009765625;
      }
   }
   for (int64_t nb=0; nb<Nbl; nb++) {
      locs[nb] = 2 + 2*nb;
      mats[nb] = (int8_t)((material<0) ? nb%MATERIALS : material);
      ssaf[nb] = (Real)(nb%5+1)*0.125;
      for (int m=0; m<MMb; m++) {
         Real v = (Real)((nb+1)*(m+1))*0.0009765625;
         Real g = -(Real)((nb+2)*(m+1))*0.00048828125;
         ref_vh[nb*MMb+m] = v;
         ref_gh[nb*MMb+m] = g;
         fused_vh[m*Nbl+nb] = v;
         fused_gh[m*Nbl+nb] = g;
      }
   }

   for (int step=0; step<STEPS; step++) {
      // Emulate the completed rigid stencil, while retaining two grid time levels.
      for (int i=0; i<GRID_POINTS; i++) {
         Real input = (Real)((step%7)-3)*0.0009765625;
         ref_grid[write_grid][i] = 1.5*ref_grid[read_grid][i]
                                   - ref_grid[write_grid][i] + input;
         fused_grid[write_grid][i] = 1.5*fused_grid[read_grid][i]
                                     - fused_grid[write_grid][i] + input;
      }

      // Original three passes, with the repository's independent node-major CPU solver.
      for (int64_t nb=0; nb<Nbl; nb++) {
         ref_carry[carry0][nb] = ref_grid[write_grid][locs[nb]];
      }
      int8_t mutable_Mb[MATERIALS];
      memcpy(mutable_Mb,Mb,sizeof(Mb));
      process_bnl_pts_fd(ref_carry[carry0],ref_carry[carry2],ssaf,mats,Nbl,
                        mutable_Mb,lo2,ref_vh,ref_gh,quads,beta);
      for (int64_t nb=0; nb<Nbl; nb++) {
         ref_grid[write_grid][locs[nb]] = ref_carry[carry0][nb];
      }

      // Reverse node order every other step: fused nodes must be independent.
      for (int64_t j=0; j<Nbl; j++) {
         int64_t nb = (step%2) ? Nbl-1-j : j;
         boundary_fd_grid_node(fused_grid[write_grid],fused_carry[carry0],
               fused_carry[carry2],fused_vh,fused_gh,locs,ssaf,mats,beta,quads,
               Mb,lo2,Nbl,nb);
      }

      for (int bank=0; bank<2; bank++) {
         check_equal("grid",ref_grid[bank],fused_grid[bank],GRID_POINTS,step);
      }
      for (int bank=0; bank<3; bank++) {
         check_equal("pressure carry",ref_carry[bank],fused_carry[bank],MAX_NODES,step);
      }
      for (int64_t nb=0; nb<Nbl; nb++) {
         for (int m=0; m<MMb; m++) {
            expected_vh[m*Nbl+nb] = ref_vh[nb*MMb+m];
            expected_gh[m*Nbl+nb] = ref_gh[nb*MMb+m];
         }
      }
      check_equal("vh1 including unused poles",expected_vh,fused_vh,Nbl*MMb,step);
      check_equal("gh1 including unused poles",expected_gh,fused_gh,Nbl*MMb,step);

      // Match gpu_engine.h's two-slot grid and three-slot carry rotations.
      int tmp = read_grid;
      read_grid = write_grid;
      write_grid = tmp;
      tmp = carry2;
      carry2 = carry1;
      carry1 = carry0;
      carry0 = tmp;
   }
}

int main(void)
{
   check_case(0,-1);
   for (int material=0; material<MATERIALS; material++) {
      check_case(1,material);
      check_case(MAX_NODES,material);
   }
   check_case(MAX_NODES,-1);
   printf("boundary_fd_check: PASS FP%d, 10 cases x %d steps, 0/1/11/12 poles\n",
          (int)(8*sizeof(Real)),STEPS);
   return EXIT_SUCCESS;
}
