// Independent long-double geometry oracle and native BVH/link contracts.
#include "mesh_geometry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using namespace pffdtd_prepare;
std::size_t checks=0;

void require(bool condition, const char *message)
{
   if (!condition) {
      std::fprintf(stderr,"FAIL: %s\n",message);
      std::exit(EXIT_FAILURE);
   }
   ++checks;
}

void close(double actual, long double expected, double tolerance, const char *message)
{
   if (!std::isfinite(actual) || !std::isfinite(expected) ||
       std::abs(static_cast<long double>(actual)-expected)>tolerance) {
      std::fprintf(stderr,"FAIL: %s: %.17g != %.21Lg (tol %.3g)\n",
                   message,actual,expected,tolerance);
      std::exit(EXIT_FAILURE);
   }
   ++checks;
}

template<typename F>
void invalid(F function, const char *message)
{
   bool caught=false;
   try { function(); }
   catch (const std::invalid_argument &) { caught=true; }
   require(caught,message);
}

struct V { long double x,y,z; };
V wide(Vec3 a) { return {a.x,a.y,a.z}; }
V plus(V a,V b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
V minus(V a,V b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
V scale(V a,long double b) { return {a.x*b,a.y*b,a.z*b}; }
long double scalar(V a,V b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
V vector_product(V a,V b)
{ return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x}; }
long double length(V a) { return std::sqrt(scalar(a,a)); }

V normal_reference(const Triangle &triangle)
{
   const V a=wide(triangle.v[0]),b=wide(triangle.v[1]),c=wide(triangle.v[2]);
   const V normal=vector_product(minus(b,a),minus(c,a));
   return scale(normal,1/length(normal));
}

Vec3 direction_reference(unsigned k,bool fcc)
{
   static const int cart[6][3]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
   static const int face[12][3]={{1,1,0},{-1,-1,0},{0,1,1},{0,-1,-1},
      {1,0,1},{-1,0,-1},{1,-1,0},{-1,1,0},{0,1,-1},{0,-1,1},{1,0,-1},{-1,0,1}};
   const int *d=fcc ? face[k] : cart[k];
   return {static_cast<double>(d[0]),static_cast<double>(d[1]),static_cast<double>(d[2])};
}

// Barycentric coordinates times the corresponding altitudes are signed
// distances to opposite edges. This reconstructs the tolerance model from
// vertices, independently of cached edge normals/midpoints or ray helpers.
bool inside_reference(const Triangle &triangle,V point,long double edge_eps)
{
   const V a=wide(triangle.v[0]),b=wide(triangle.v[1]),c=wide(triangle.v[2]);
   const V e0=minus(b,a),e1=minus(c,a),offset=minus(point,a);
   const long double d00=scalar(e0,e0),d01=scalar(e0,e1),d11=scalar(e1,e1),
                     d20=scalar(offset,e0),d21=scalar(offset,e1);
   const long double determinant=d00*d11-d01*d01;
   if (!(determinant>0)) return false;
   const long double lambda1=(d11*d20-d01*d21)/determinant;
   const long double lambda2=(d00*d21-d01*d20)/determinant;
   const long double lambda0=1-lambda1-lambda2;
   const long double doubled_area=length(vector_product(e0,e1));
   return lambda0*doubled_area/length(minus(c,b))>=-edge_eps &&
          lambda1*doubled_area/length(minus(a,c))>=-edge_eps &&
          lambda2*doubled_area/length(minus(b,a))>=-edge_eps;
}

long double ray_reference(const Triangle &triangle,Vec3 origin,Vec3 direction,double edge_eps)
{
   if (!(triangle.area>0)) return std::numeric_limits<long double>::infinity();
   V d=wide(direction);
   const long double len=length(d);
   if (!(len>0)) return std::numeric_limits<long double>::infinity();
   d=scale(d,1/len);
   const V normal=normal_reference(triangle);
   const long double beta=scalar(normal,d);
   if (std::abs(beta)<static_cast<long double>(1e-6))
      return std::numeric_limits<long double>::infinity();
   const long double distance=scalar(normal,minus(wide(triangle.v[0]),wide(origin)))/beta;
   if (distance<0 || !inside_reference(triangle,plus(wide(origin),scale(d,distance)),edge_eps))
      return std::numeric_limits<long double>::infinity();
   return distance;
}

struct RefHit { long double distance; bool near; };
RefHit link_reference(const Triangle &triangle,Vec3 point,Vec3 link,double h)
{
   const long double infinity=std::numeric_limits<long double>::infinity();
   if (!(triangle.area>0)) return {infinity,false};
   V d=wide(link);
   const long double len=length(d);
   if (!(len>0)) return {infinity,false};
   d=scale(d,1/len);
   const V normal=normal_reference(triangle);
   const long double beta=scalar(normal,d);
   if (std::abs(beta)<static_cast<long double>(1e-6)) return {infinity,false};
   const long double distance=scalar(normal,minus(wide(triangle.v[0]),wide(point)))/beta;
   const long double epsilon=static_cast<long double>(1e-6)*len;
   if (distance<-epsilon || distance>(1+static_cast<long double>(1e-6))*len ||
       !inside_reference(triangle,plus(wide(point),scale(d,distance)),static_cast<long double>(1e-3)*h))
      return {infinity,false};
   const bool near=std::abs(distance)<=epsilon;
   return {near ? std::abs(distance) : distance,near};
}

bool nearer_reference(long double distance,std::uint32_t ordinal,std::uint32_t index,
                      long double previous,std::uint32_t old_ordinal,std::uint32_t old_index)
{
   return distance<previous || (distance==previous &&
      (ordinal<old_ordinal || (ordinal==old_ordinal && index<old_index)));
}

NearestHit query_reference(const std::vector<Triangle> &triangles,Vec3 point,Vec3 link,double h)
{
   NearestHit result={HUGE_VAL,NO_TRIANGLE,NO_TRIANGLE,0};
   long double nearest=std::numeric_limits<long double>::infinity();
   for (std::size_t i=0;i<triangles.size();++i) {
      const RefHit hit=link_reference(triangles[i],point,link,h);
      if (!std::isfinite(hit.distance)) continue;
      result.near_boundary|=hit.near ? 1 : 0;
      if (nearer_reference(hit.distance,triangles[i].ordinal,static_cast<std::uint32_t>(i),
                           nearest,result.ordinal,result.triangle)) {
         nearest=hit.distance; result.distance=static_cast<double>(nearest);
         result.triangle=static_cast<std::uint32_t>(i); result.ordinal=triangles[i].ordinal;
      }
   }
   return result;
}

NodeBoundary classify_reference(const std::vector<Triangle> &triangles,Vec3 point,double h,bool fcc)
{
   const unsigned directions=fcc ? 12 : 6;
   NodeBoundary result={static_cast<std::uint16_t>((1U<<directions)-1),NO_TRIANGLE,
                        NO_TRIANGLE,HUGE_VAL,0,-1,0};
   long double nearest=std::numeric_limits<long double>::infinity();
   for (std::size_t i=0;i<triangles.size();++i)
      for (unsigned k=0;k<directions;++k) {
         const Vec3 base=direction_reference(k,fcc);
         const RefHit hit=link_reference(triangles[i],point,{h*base.x,h*base.y,h*base.z},h);
         if (!std::isfinite(hit.distance)) continue;
         result.adjacency=static_cast<std::uint16_t>(result.adjacency&~(1U<<k));
         result.near_boundary|=hit.near ? 1 : 0;
         if (nearer_reference(hit.distance,triangles[i].ordinal,static_cast<std::uint32_t>(i),
                              nearest,result.ordinal,result.nearest_triangle)) {
            nearest=hit.distance; result.distance=static_cast<double>(nearest);
            result.nearest_triangle=static_cast<std::uint32_t>(i); result.ordinal=triangles[i].ordinal;
         }
      }
   if (result.near_boundary) result.adjacency=0;
   if (result.nearest_triangle==NO_TRIANGLE) return result;
   const Triangle &triangle=triangles[result.nearest_triangle];
   const V normal=normal_reference(triangle);
   const long double side=scalar(minus(wide(point),wide(triangle.v[0])),normal);
   if (!result.near_boundary && result.adjacency && triangle.material>=0 && triangle.sides &&
       !(side>0 && triangle.sides==1) && !(side<0 && triangle.sides==2))
      result.material=triangle.material;
   long double saf=0;
   for (unsigned k=0;k<directions;++k)
      if (!(result.adjacency&(1U<<k))) {
         const V d=wide(direction_reference(k,fcc));
         saf+=std::abs(scalar(d,normal))/length(d);
      }
   result.saf=static_cast<double>(saf);
   return result;
}

void compare_hit(const NearestHit &actual,const NearestHit &expected)
{
   require(actual.triangle==expected.triangle && actual.ordinal==expected.ordinal,
           "BVH nearest triangle or original ordinal differs from brute force");
   require(actual.near_boundary==expected.near_boundary,"BVH link near flag differs");
   if (expected.triangle==NO_TRIANGLE) require(!std::isfinite(actual.distance),"miss has finite distance");
   else close(actual.distance,expected.distance,2e-10,"BVH link distance differs");
}

void compare_node(const NodeBoundary &actual,const NodeBoundary &expected)
{
   require(actual.adjacency==expected.adjacency,"BVH adjacency differs from barycentric brute force");
   require(actual.nearest_triangle==expected.nearest_triangle && actual.ordinal==expected.ordinal,
           "BVH node nearest triangle or ordinal differs");
   require(actual.material==expected.material && actual.near_boundary==expected.near_boundary,
           "BVH side/material/near classification differs");
   close(actual.saf,expected.saf,2e-10,"BVH physical surface factor differs");
   if (expected.nearest_triangle==NO_TRIANGLE) require(!std::isfinite(actual.distance),"empty node distance");
   else close(actual.distance,expected.distance,2e-10,"BVH node distance differs");
}

Triangle triangle(Vec3 a,Vec3 b,Vec3 c,std::uint32_t ordinal=0,int material=3,unsigned sides=3)
{
   Triangle result;
   require(precompute_triangle(result,a,b,c,material,static_cast<std::uint8_t>(sides),ordinal),
           "valid fixture triangle rejected");
   return result;
}

std::vector<Triangle> plane(unsigned sides,double span=100)
{
   return {triangle({-span,-span,0},{span,-span,0},{span,span,0},7,3,sides),
           triangle({-span,-span,0},{span,span,0},{-span,span,0},13,3,sides)};
}

void check_precompute_and_rays()
{
   const Triangle t=triangle({1,2,3},{5,2,3},{1,5,3});
   close(t.area,6,0,"triangle area"); close(t.centroid.x,7.0L/3,1e-15,"triangle centroid");
   close(t.centroid.y,3,0,"triangle centroid y"); close(t.centroid.z,3,0,"triangle centroid z");
   close(t.normal.z,1,0,"oriented unit normal");
   for (unsigned edge=0;edge<3;++edge) {
      close(length(wide(t.edge_normal[edge])),1,2e-15,"normalized edge distance plane");
      require(dot(subtract(t.centroid,t.edge_midpoint[edge]),t.edge_normal[edge])<0,
              "edge normal points outward");
   }
   const Triangle reversed=triangle(t.v[0],t.v[2],t.v[1]);
   close(reversed.area,t.area,0,"winding preserves area");
   close(reversed.normal.z,-1,0,"winding reverses normal");
   const Triangle cyclic=triangle(t.v[1],t.v[2],t.v[0]);
   close(cyclic.normal.z,t.normal.z,0,"cyclic permutation preserves normal");
   const Triangle ray=triangle({0,0,0},{2,0,0},{0,2,0});
   const Vec3 points[]={{0.5,0.5,-1},{1,1,-1},{0,0,-1},{-0.0005,0.5,-1},
      {-0.0011,0.5,-1},{1.1,1.1,-1},{-0.0005,-0.0005,-1}};
   for (Vec3 p : points) {
      const long double expected=ray_reference(ray,p,{0,0,17},0.001);
      const double actual=triangle_ray_distance(ray,p,{0,0,17},0.001);
      if (std::isfinite(expected)) close(actual,expected,1e-14,"ray barycentric edge tolerance");
      else require(!std::isfinite(actual),"outside triangle ray must miss");
   }
   require(!std::isfinite(triangle_ray_distance(ray,{0.5,0.5,0},{1,0,0},0.001)),
           "coplanar ray must miss");
   require(!std::isfinite(triangle_ray_distance(ray,{0.5,0.5,1},{0,0,1},0.001)),
           "backwards ray must miss");
   require(!std::isfinite(triangle_ray_distance(ray,{0,0.5,-1e-6},{1,0,0.5e-6},0.001)),
           "nearly coplanar ray follows cp epsilon");
   close(triangle_ray_distance(ray,{0,0.5,-1e-6},{1,0,2e-6},0.001),
         ray_reference(ray,{0,0.5,-1e-6},{1,0,2e-6},0.001),1e-12,"nonparallel small-angle ray");
   Triangle degenerate;
   require(!precompute_triangle(degenerate,{0,0,0},{1,1,1},{2,2,2},-1,0,99),
           "collinear triangle is removed");
   require(!std::isfinite(triangle_ray_distance(degenerate,{0,0,0},{1,0,0},0)),
           "degenerate triangle ray must miss");
   Triangle threshold;
   require(precompute_triangle(threshold,{0,0,0},{2,0,0},{0,2,0},0,3,1,2),
           "area equal to threshold is kept");
   require(!precompute_triangle(threshold,{0,0,0},{2,0,0},{0,2,0},0,3,1,
                                std::nextafter(2.0,3.0)),"area below threshold is removed");
   const double nan=std::numeric_limits<double>::quiet_NaN();
   invalid([&] { precompute_triangle(threshold,{nan,0,0},{1,0,0},{0,1,0},0,3,0); },"NaN vertex rejected");
   invalid([&] { precompute_triangle(threshold,{0,0,0},{1,0,0},{0,1,0},-2,3,0); },"invalid material rejected");
   invalid([&] { precompute_triangle(threshold,{0,0,0},{1,0,0},{0,1,0},0,4,0); },"invalid sides rejected");
}

void check_planes_and_near()
{
   for (bool fcc : {false,true}) {
      const unsigned directions=fcc ? 12 : 6;
      const std::uint16_t full=static_cast<std::uint16_t>((1U<<directions)-1);
      for (unsigned k=0;k<directions;++k) {
         const Vec3 actual=mesh_direction(k,fcc),expected=direction_reference(k,fcc),opposite=mesh_direction(k^1U,fcc);
         require(actual.x==expected.x && actual.y==expected.y && actual.z==expected.z,"direction solver order");
         require(actual.x==-opposite.x && actual.y==-opposite.y && actual.z==-opposite.z,"inverse direction k xor 1");
      }
      for (unsigned sides=0;sides<=3;++sides) {
         const FlatBvh bvh=build_bvh(plane(sides));
         for (int sign : {-1,1}) {
            const Vec3 point={0.25,0.35,sign*0.25};
            const NodeBoundary result=classify_node(bvh,point,1,fcc);
            std::uint16_t expected=full;
            for (unsigned k=0;k<directions;++k)
               if (sign*direction_reference(k,fcc).z<0) expected=static_cast<std::uint16_t>(expected&~(1U<<k));
            require(result.adjacency==expected && !result.near_boundary,"analytic plane blocked directions");
            close(result.distance,0.25L*(fcc ? std::sqrt(2.0L) : 1.0L),1e-14,"analytic plane physical distance");
            close(result.saf,fcc ? 2*std::sqrt(2.0L) : 1.0L,1e-14,"analytic plane SAF");
            const int material=(sides && !(sign>0 && sides==1) && !(sign<0 && sides==2)) ? 3 : -1;
            require(result.material==material,"analytic plane side material selection");
            for (unsigned k=0;k<directions;++k) {
               const Vec3 link=direction_reference(k,fcc);
               if (sign*link.z>=0) continue;
               const Vec3 other={point.x+link.x,point.y+link.y,point.z+link.z};
               const Vec3 inverse={-link.x,-link.y,-link.z};
               const NearestHit outward=query_link(bvh,point,link,1),
                                inward=query_link(bvh,other,inverse,1);
               require(outward.triangle!=NO_TRIANGLE && inward.ordinal==outward.ordinal,
                       "reciprocal crossing link preserves intersected triangle ordinal");
               close(outward.distance+inward.distance,length(wide(link)),2e-14,
                     "reciprocal intersection distances sum to physical link length");
            }
         }
         // Both signed near thresholds must retain backward-near intersections
         // and disconnect every link, independent of side/material policy.
         for (double fraction : {-1.0,-0.75,0.0,0.75,1.0}) {
            const NodeBoundary result=classify_node(bvh,{0.25,0.35,fraction*1e-6},1,fcc);
            require(result.near_boundary && result.adjacency==0 && result.material==-1,
                    "signed near threshold forces all-adjacency-zero rigid node");
         }
      }
      const std::uint16_t both=static_cast<std::uint16_t>(full&~(3U<<(fcc ? 2 : 4)));
      close(surface_factor({0,0,1},both,fcc),fcc ? std::sqrt(2.0L) : 2.0L,1e-14,
            "SAF counts both blocked opposite directions");
      close(surface_factor({0,0,1},0,fcc),fcc ? 4*std::sqrt(2.0L) : 2.0L,1e-14,
            "SAF physical projection of all blocked links");
   }
   Triangle front=triangle({-3,-3,0},{3,-3,0},{0,3,0},0,5,1);
   Triangle back=triangle(front.v[0],front.v[2],front.v[1],0,5,2);
   const FlatBvh first=build_bvh({front}),second=build_bvh({back});
   for (int side : {-1,1}) {
      const NodeBoundary a=classify_node(first,{0,0,side*0.25},1,false);
      const NodeBoundary b=classify_node(second,{0,0,side*0.25},1,false);
      require(a.adjacency==b.adjacency && a.material==b.material,"winding plus side inversion preserves wall");
      close(a.saf,b.saf,0,"winding preserves SAF");
   }
   const FlatBvh shared=build_bvh(plane(3));
   for (Vec3 point : {Vec3{0,0,-0.5},Vec3{-100,-100,-0.5}}) {
      const NearestHit hit=query_link(shared,point,{0,0,1},1);
      require(hit.ordinal==7,"shared edge/vertex hit uses stable minimum ordinal");
      compare_hit(hit,query_reference(shared.triangles,point,{0,0,1},1));
   }
   const Triangle t=triangle({-3,-3,0},{3,-3,0},{0,3,0});
   for (bool fcc : {false,true}) {
      const Vec3 link=fcc ? Vec3{0,1,1} : Vec3{0,0,1};
      const double len=std::sqrt(fcc ? 2.0 : 1.0);
      for (double fraction : {-2e-6,-0.75e-6,0.0,0.75e-6,2e-6,0.5,1.0,1+0.75e-6,1+2e-6}) {
         const LinkHit hit=triangle_link_hit(t,{0,-0.25,-fraction},link,1);
         const bool accepted=fraction>=-1e-6 && fraction<=1+1e-6;
         require(std::isfinite(hit.distance)==accepted,"link finite support and near interval");
         if (accepted) {
            require(static_cast<bool>(hit.near_boundary)==(std::abs(fraction)<=1e-6),"link near predicate");
            close(hit.distance,std::abs(fraction)*len,2e-14,"link physical signed/near distance");
         }
      }
   }
}

void check_bvh_structure(const FlatBvh &tree)
{
   if (tree.nodes.empty()) { require(tree.order.empty(),"empty tree has no active order"); return; }
   require(tree.nodes.front().escape==tree.nodes.size(),"root escape traverses complete tree");
   std::vector<unsigned> seen(tree.triangles.size(),0);
   for (std::size_t i=0;i<tree.nodes.size();++i) {
      const BvhNode &node=tree.nodes[i];
      require(node.escape>i && node.escape<=tree.nodes.size(),"stackless escape range");
      require(node.count<=4 && static_cast<std::size_t>(node.begin)+node.count<=tree.order.size(),"leaf size/range");
      for (std::uint32_t j=0;j<node.count;++j) ++seen[tree.order[node.begin+j]];
      for (std::size_t child=i;child<node.escape;++child)
         for (std::uint32_t j=0;j<tree.nodes[child].count;++j) {
            const Triangle &t=tree.triangles[tree.order[tree.nodes[child].begin+j]];
            require(node.bound_slack>=t.bound_slack,"acute tolerance bound propagated to ancestors");
            for (unsigned axis=0;axis<3;++axis)
               require(component(node.bmin,axis)<=component(t.bmin,axis) &&
                       component(node.bmax,axis)>=component(t.bmax,axis),"ancestor contains descendant triangle");
         }
   }
   for (std::size_t i=0;i<seen.size();++i)
      require(seen[i]==(tree.triangles[i].area>0 ? 1U : 0U),"each active triangle owns exactly one leaf");
}

void check_acute_and_order()
{
   const Triangle acute=triangle({0,0,0},{1,0.0001,0},{1,-0.0001,0},9);
   const Vec3 point={-5,0,-0.5},link={0,0,1};
   require(std::isfinite(link_reference(acute,point,link,1).distance),"acute fixture lies in expanded edge planes");
   const FlatBvh single=build_bvh({acute});
   compare_hit(query_link(single,point,link,1),query_reference(single.triangles,point,link,1));
   compare_node(classify_node(single,point,1,false),classify_reference(single.triangles,point,1,false));
   const Vec3 fcc_point={-5.5,0,-0.5};
   compare_node(classify_node(single,fcc_point,1,true),classify_reference(single.triangles,fcc_point,1,true));
   std::vector<Triangle> triangles={acute};
   for (unsigned i=0;i<30;++i)
      triangles.push_back(triangle({10.0+i,10,10},{11.0+i,10,10},{10.0+i,11,10},100+i));
   const FlatBvh multi=build_bvh(triangles);
   check_bvh_structure(multi);
   compare_hit(query_link(multi,point,link,1),query_reference(multi.triangles,point,link,1));
   compare_node(classify_node(multi,point,1,false),classify_reference(multi.triangles,point,1,false));
   compare_node(classify_node(multi,fcc_point,1,true),classify_reference(multi.triangles,fcc_point,1,true));
   std::mt19937 random(0x47ad120u);
   for (unsigned count : {1U,4U,5U,17U,65U}) {
      triangles.clear();
      for (unsigned i=0;i<count;++i)
         triangles.push_back(triangle({-2,-2,0},{2,-2,0},{0,2,0},count-i,static_cast<int>(i%5)));
      for (unsigned trial=0;trial<8;++trial) {
         std::shuffle(triangles.begin(),triangles.end(),random);
         const FlatBvh tree=build_bvh(triangles);
         check_bvh_structure(tree);
         const NearestHit hit=query_link(tree,{0,0,-0.25},{0,0,1},1);
         require(hit.ordinal==1,"equal-distance original ordinal wins across median/leaf/input permutations");
         require(tree.triangles[hit.triangle].ordinal==hit.ordinal,"nearest index still references input order");
         compare_hit(hit,query_reference(tree.triangles,{0,0,-0.25},{0,0,1},1));
         for (bool fcc : {false,true})
            compare_node(classify_node(tree,{0,0,-0.25},1,fcc),classify_reference(tree.triangles,{0,0,-0.25},1,fcc));
      }
   }
   Triangle degenerate;
   precompute_triangle(degenerate,{0,0,0},{0,0,0},{0,0,0},-1,0,7);
   const FlatBvh empty=build_bvh({degenerate});
   check_bvh_structure(empty);
   require(query_link(empty,{0,0,0},{1,0,0},1).triangle==NO_TRIANGLE,"degenerate-only BVH query misses");
   require(classify_node(empty,{0,0,0},1,false).adjacency==63,"empty BVH keeps all links");
   invalid([&] { build_bvh({acute,acute}); },"duplicate active ordinals rejected");
   invalid([&] { classify_node(single,{0,0,0},0,false); },"zero h rejected");
   invalid([&] { query_link(single,{0,0,0},{1,0,0},-1); },"negative h rejected");
}

void check_random_bvh()
{
   std::mt19937_64 random(0x372aa91fUL);
   std::uniform_real_distribution<double> coordinate(-4,4),step(-1.5,1.5);
   std::vector<Triangle> triangles;
   for (unsigned i=0;i<79;++i) {
      const Vec3 a={coordinate(random),coordinate(random),coordinate(random)};
      const Vec3 b={a.x+step(random),a.y+step(random),a.z+step(random)};
      const Vec3 c={a.x+step(random),a.y+step(random),a.z+step(random)};
      triangles.push_back(triangle(a,b,c,700+i,static_cast<int>(i%6)-1,i%4));
   }
   const FlatBvh tree=build_bvh(triangles);
   check_bvh_structure(tree);
   std::size_t hit_count=0;
   for (unsigned trial=0;trial<800;++trial) {
      const Vec3 point={coordinate(random),coordinate(random),coordinate(random)};
      const double h=trial%2 ? 0.375 : 1.0;
      for (bool fcc : {false,true}) {
         const NodeBoundary node=classify_node(tree,point,h,fcc);
         compare_node(node,classify_reference(tree.triangles,point,h,fcc));
         if (node.nearest_triangle!=NO_TRIANGLE) ++hit_count;
         for (unsigned k=0;k<(fcc ? 12U : 6U);++k) {
            const Vec3 base=direction_reference(k,fcc),link={h*base.x,h*base.y,h*base.z};
            compare_hit(query_link(tree,point,link,h),query_reference(tree.triangles,point,link,h));
         }
      }
      const Vec3 link={step(random),step(random),step(random)};
      compare_hit(query_link(tree,point,link,h),query_reference(tree.triangles,point,link,h));
   }
   require(hit_count>100,"random corpus contains substantial actual intersections");
}

std::size_t grid_index(int x,int y,int z) { return static_cast<std::size_t>((x*7+y)*7+z); }

void check_reciprocity()
{
   // At a finite wall edge, a near node clears tangential links too. Its
   // coplanar outside neighbor can initially remain free. This fixture exposes
   // the asymmetry and provides the conservative graph-removal oracle required
   // by scene reconciliation (tested separately from this geometry-only unit).
   const FlatBvh tree=build_bvh(plane(3,1));
   for (bool fcc : {false,true}) {
      std::vector<std::uint16_t> raw(7*7*7),reconciled;
      for (int x=0;x<7;++x) for (int y=0;y<7;++y) for (int z=0;z<7;++z)
         raw[grid_index(x,y,z)]=classify_node(tree,{double(x-3),double(y-3),double(z-3)},1,fcc).adjacency;
      reconciled=raw;
      std::size_t asymmetric=0,removed=0;
      for (int x=0;x<7;++x) for (int y=0;y<7;++y) for (int z=0;z<7;++z) {
         if (fcc && ((x+y+z)&1)) continue;
         const std::size_t here=grid_index(x,y,z);
         for (unsigned k=0;k<(fcc ? 12U : 6U);++k) {
            const Vec3 d=direction_reference(k,fcc);
            const int nx=x+static_cast<int>(d.x),ny=y+static_cast<int>(d.y),nz=z+static_cast<int>(d.z);
            if (nx<0 || nx>=7 || ny<0 || ny>=7 || nz<0 || nz>=7) continue;
            const std::size_t there=grid_index(nx,ny,nz);
            const bool a=(raw[here]&(1U<<k))!=0,b=(raw[there]&(1U<<(k^1U)))!=0;
            if (a!=b) ++asymmetric;
            if (!a || !b) {
               if (reconciled[here]&(1U<<k)) ++removed;
               reconciled[here]=static_cast<std::uint16_t>(reconciled[here]&~(1U<<k));
               reconciled[there]=static_cast<std::uint16_t>(reconciled[there]&~(1U<<(k^1U)));
            }
         }
      }
      require(asymmetric>0 && removed>0,"finite near wall exposes links requiring conservative reconciliation");
      for (int x=0;x<7;++x) for (int y=0;y<7;++y) for (int z=0;z<7;++z) {
         if (fcc && ((x+y+z)&1)) continue;
         const std::size_t here=grid_index(x,y,z);
         require((reconciled[here]&raw[here])==reconciled[here],"reconciliation never reconnects a blocked link");
         for (unsigned k=0;k<(fcc ? 12U : 6U);++k) {
            const Vec3 d=direction_reference(k,fcc);
            const int nx=x+static_cast<int>(d.x),ny=y+static_cast<int>(d.y),nz=z+static_cast<int>(d.z);
            if (nx<0 || nx>=7 || ny<0 || ny>=7 || nz<0 || nz>=7) continue;
            const std::size_t there=grid_index(nx,ny,nz);
            require(((reconciled[here]>>k)&1)==((reconciled[there]>>(k^1U))&1),
                    "conservative removal produces reciprocal internal grid links");
         }
      }
   }
}

} // namespace

int main()
{
   check_precompute_and_rays();
   check_planes_and_near();
   check_acute_and_order();
   check_random_bvh();
   check_reciprocity();
   std::printf("PASS: native mesh geometry (%zu checks), independent barycentric rays, Cart/FCC, "
               "sides/near/SAF, acute bounds, BVH order and reciprocal-removal oracle\n",checks);
   return EXIT_SUCCESS;
}
