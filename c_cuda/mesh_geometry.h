// FP64 triangle/link geometry and a deterministic, stackless flat BVH.
#ifndef PFFDTD_MESH_GEOMETRY_H
#define PFFDTD_MESH_GEOMETRY_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pffdtd_prepare {

#ifdef __CUDACC__
#define PFFDTD_MESH_INLINE __host__ __device__ __forceinline__
#else
#define PFFDTD_MESH_INLINE inline
#endif

struct Vec3 { double x,y,z; };
struct Triangle {
   Vec3 v[3],centroid,normal,edge_normal[3],edge_midpoint[3],bmin,bmax;
   double area,bound_slack;
   std::uint32_t ordinal;
   std::int32_t material;
   std::uint8_t sides;
};
struct BvhNode {
   Vec3 bmin,bmax;
   double bound_slack;
   std::uint32_t begin,count,escape;
};
struct FlatBvh {
   std::vector<Triangle> triangles;
   std::vector<BvhNode> nodes;
   std::vector<std::uint32_t> order;
};
struct LinkHit { double distance; std::uint8_t near_boundary; };
struct NearestHit {
   double distance;
   std::uint32_t triangle,ordinal;
   std::uint8_t near_boundary;
};
struct NodeBoundary {
   std::uint16_t adjacency;
   std::uint32_t nearest_triangle,ordinal;
   double distance,saf;
   std::int32_t material;
   std::uint8_t near_boundary;
};

static const std::uint32_t NO_TRIANGLE=0xffffffffU;

PFFDTD_MESH_INLINE Vec3 add(Vec3 a, Vec3 b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
PFFDTD_MESH_INLINE Vec3 subtract(Vec3 a, Vec3 b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
PFFDTD_MESH_INLINE Vec3 multiply(Vec3 a, double s) { return {a.x*s,a.y*s,a.z*s}; }
PFFDTD_MESH_INLINE Vec3 divide(Vec3 a, double s) { return {a.x/s,a.y/s,a.z/s}; }
PFFDTD_MESH_INLINE double dot(Vec3 a, Vec3 b) { return (a.x*b.x+a.y*b.y)+a.z*b.z; }
PFFDTD_MESH_INLINE Vec3 cross(Vec3 a, Vec3 b)
{ return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x}; }
PFFDTD_MESH_INLINE double component(Vec3 a, unsigned axis)
{ return axis==0 ? a.x : (axis==1 ? a.y : a.z); }
PFFDTD_MESH_INLINE Vec3 minimum(Vec3 a, Vec3 b)
{ return {::fmin(a.x,b.x),::fmin(a.y,b.y),::fmin(a.z,b.z)}; }
PFFDTD_MESH_INLINE Vec3 maximum(Vec3 a, Vec3 b)
{ return {::fmax(a.x,b.x),::fmax(a.y,b.y),::fmax(a.z,b.z)}; }
PFFDTD_MESH_INLINE double mesh_norm(Vec3 value)
{
   const double scale=::fmax(::fabs(value.x),::fmax(::fabs(value.y),::fabs(value.z)));
   if (!(scale>0)) return 0;
   const Vec3 normalized=divide(value,scale);
   return scale*::sqrt(dot(normalized,normalized));
}

// Directions are paired with their inverse, in the solver/legacy order.
PFFDTD_MESH_INLINE Vec3 mesh_direction(unsigned direction, bool fcc)
{
   if (!fcc) {
      switch (direction) {
         case 0: return {1,0,0}; case 1: return {-1,0,0};
         case 2: return {0,1,0}; case 3: return {0,-1,0};
         case 4: return {0,0,1}; default: return {0,0,-1};
      }
   }
   switch (direction) {
      case 0: return {1,1,0}; case 1: return {-1,-1,0};
      case 2: return {0,1,1}; case 3: return {0,-1,-1};
      case 4: return {1,0,1}; case 5: return {-1,0,-1};
      case 6: return {1,-1,0}; case 7: return {-1,1,0};
      case 8: return {0,1,-1}; case 9: return {0,-1,1};
      case 10: return {1,0,-1}; default: return {-1,0,1};
   }
}

// Counts every blocked link explicitly. This intentionally corrects legacy
// boolean addition for a pair with both opposite directions disconnected.
PFFDTD_MESH_INLINE double surface_factor(Vec3 normal, std::uint16_t adjacency, bool fcc)
{
   const unsigned directions=fcc ? 12 : 6;
   const double inverse_length=fcc ? 0.707106781186547524400844362104849 : 1.0;
   double saf=0;
   for (unsigned k=0; k<directions; k+=2) {
      const unsigned blocked=((adjacency&(1U<<k)) ? 0U : 1U)+
                             ((adjacency&(1U<<(k+1))) ? 0U : 1U);
      saf+=blocked*::fabs(dot(multiply(mesh_direction(k,fcc),inverse_length),normal));
   }
   return saf;
}

PFFDTD_MESH_INLINE std::int32_t boundary_material(const Triangle& triangle,
      Vec3 point, std::uint16_t adjacency, bool near_boundary)
{
   if (near_boundary || !adjacency || triangle.sides==0 || triangle.material<0) return -1;
   const double side=dot(subtract(point,triangle.centroid),triangle.normal);
   if ((side>0 && triangle.sides==1) || (side<0 && triangle.sides==2)) return -1;
   return triangle.material;
}

// Customized legacy ray test: normalized direction, no back-face culling,
// cp_eps=1e-6 and outward edge distance tests with edge_eps=1e-3*h.
PFFDTD_MESH_INLINE double triangle_ray_unit_distance(const Triangle& triangle,
      Vec3 origin, Vec3 direction, double edge_eps, double cp_eps=1e-6)
{
   if (!(triangle.area>0)) return HUGE_VAL;
   const double beta=dot(direction,triangle.normal);
   if (::fabs(beta)<cp_eps) return HUGE_VAL;
   const double distance=dot(triangle.normal,subtract(triangle.centroid,origin))/beta;
   if (!(distance>=0) || !(distance<=1.7976931348623157e308)) return HUGE_VAL;
   const Vec3 point=add(origin,multiply(direction,distance));
   for (unsigned edge=0; edge<3; ++edge)
      if (dot(subtract(point,triangle.edge_midpoint[edge]),triangle.edge_normal[edge])>edge_eps)
         return HUGE_VAL;
   return distance;
}

PFFDTD_MESH_INLINE double triangle_ray_distance(const Triangle& triangle,
      Vec3 origin, Vec3 direction, double edge_eps, double cp_eps=1e-6)
{
   const double length=mesh_norm(direction);
   if (!(length>0) || !(length<=1.7976931348623157e308)) return HUGE_VAL;
   return triangle_ray_unit_distance(triangle,origin,divide(direction,length),edge_eps,cp_eps);
}

PFFDTD_MESH_INLINE LinkHit triangle_link_unit_hit(const Triangle& triangle,
      Vec3 point, Vec3 direction, double length, double h)
{
   if (!(length>0) || !(length<=0.25*1.7976931348623157e308)) return {HUGE_VAL,0};
   if (!(triangle.area>0)) return {HUGE_VAL,0};
   const double beta=dot(direction,triangle.normal);
   if (::fabs(beta)<1e-6) return {HUGE_VAL,0};
   // The legacy ray starts one link behind the node and subtracts its length
   // from the hit. Evaluate that signed distance directly to avoid cancellation
   // at +/-near_distance; retain the same acceptance interval behind the node.
   double distance=dot(triangle.normal,subtract(triangle.centroid,point))/beta;
   const double near_distance=1e-6*length;
   if (!(distance>=-near_distance) || !(distance<=(1.0+1e-6)*length)) return {HUGE_VAL,0};
   const Vec3 intersection=add(point,multiply(direction,distance));
   for (unsigned edge=0; edge<3; ++edge)
      if (dot(subtract(intersection,triangle.edge_midpoint[edge]),triangle.edge_normal[edge])>1e-3*h)
         return {HUGE_VAL,0};
   const std::uint8_t near_boundary=::fabs(distance)<=near_distance ? 1 : 0;
   if (near_boundary) distance=::fabs(distance);
   return {distance,near_boundary};
}

PFFDTD_MESH_INLINE LinkHit triangle_link_hit(const Triangle& triangle,
      Vec3 point, Vec3 link, double h)
{
   const double length=mesh_norm(link);
   if (!(length>0) || !(length<=0.25*1.7976931348623157e308)) return {HUGE_VAL,0};
   return triangle_link_unit_hit(triangle,point,divide(link,length),length,h);
}

namespace detail {

PFFDTD_MESH_INLINE double bound_padding(double lower, double upper,
      double coordinate, double extent, double slack, double edge_eps)
{
   const double magnitude=::fmax(::fabs(coordinate),::fmax(::fabs(lower),::fabs(upper)));
   return slack*edge_eps+64.0*2.2204460492503130808472633361816e-16*
                              ::fmax(magnitude,extent);
}

PFFDTD_MESH_INLINE bool node_box_hit(const BvhNode& node, Vec3 point, double h)
{
   // All accepted Cart/FCC link intersections are inside this coordinate box.
   // The triangle-specific slack conservatively includes enlarged edge tests,
   // including acute vertices where edge_eps alone is insufficient padding.
   const double extent=(1.0+1e-6)*h;
   for (unsigned axis=0; axis<3; ++axis) {
      const double lower=component(node.bmin,axis),upper=component(node.bmax,axis),
                   coordinate=component(point,axis);
      const double padding=bound_padding(lower,upper,coordinate,extent,node.bound_slack,1e-3*h);
      if (coordinate+extent<lower-padding || coordinate-extent>upper+padding) return false;
   }
   return true;
}

PFFDTD_MESH_INLINE bool link_box_hit(const BvhNode& node, Vec3 point, Vec3 link, double h)
{
   // Conservative segment AABB avoids division by zero/near-zero directions.
   // A stackless traversal remains correct when a degenerate slack is infinite.
   const double length=mesh_norm(link),near=1e-6*length;
   const Vec3 low=minimum(point,add(point,link)),high=maximum(point,add(point,link));
   for (unsigned axis=0; axis<3; ++axis) {
      const double lower=component(node.bmin,axis),upper=component(node.bmax,axis);
      const double padding=bound_padding(lower,upper,component(point,axis),length,
                                         node.bound_slack,1e-3*h)+near;
      if (component(high,axis)<lower-padding || component(low,axis)>upper+padding) return false;
   }
   return true;
}

PFFDTD_MESH_INLINE bool nearer(double distance, std::uint32_t ordinal,
      std::uint32_t index, double previous, std::uint32_t previous_ordinal,
      std::uint32_t previous_index)
{
   return distance<previous || (distance==previous &&
      (ordinal<previous_ordinal || (ordinal==previous_ordinal && index<previous_index)));
}

inline bool finite(Vec3 value)
{ return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z); }
inline double norm(Vec3 value)
{ return std::hypot(std::hypot(value.x,value.y),value.z); }

} // namespace detail

PFFDTD_MESH_INLINE NearestHit query_link(const Triangle* triangles,
      const BvhNode* nodes, const std::uint32_t* order, std::uint32_t node_count,
      Vec3 point, Vec3 link, double h)
{
   NearestHit result={HUGE_VAL,NO_TRIANGLE,NO_TRIANGLE,0};
   for (std::uint32_t node_index=0; node_index<node_count;) {
      const BvhNode& node=nodes[node_index];
      if (!detail::link_box_hit(node,point,link,h)) { node_index=node.escape; continue; }
      for (std::uint32_t n=0; n<node.count; ++n) {
         const std::uint32_t index=order[node.begin+n];
         const Triangle& triangle=triangles[index];
         const LinkHit hit=triangle_link_hit(triangle,point,link,h);
         if (hit.distance==HUGE_VAL) continue;
         result.near_boundary|=hit.near_boundary;
         if (detail::nearer(hit.distance,triangle.ordinal,index,result.distance,result.ordinal,result.triangle))
            { result.distance=hit.distance; result.triangle=index; result.ordinal=triangle.ordinal; }
      }
      ++node_index;
   }
   return result;
}

// One traversal per node, testing all link directions after a triangle load.
// Raw buffers are immutable; each CUDA thread can own one output independently.
PFFDTD_MESH_INLINE NodeBoundary classify_node(const Triangle* triangles,
      const BvhNode* nodes, const std::uint32_t* order, std::uint32_t node_count,
      Vec3 point, double h, bool fcc)
{
   const unsigned directions=fcc ? 12 : 6;
   const double link_length=h*(fcc ? 1.4142135623730950488016887242097 : 1.0);
   const double unit_component=fcc ? 0.707106781186547524400844362104849 : 1.0;
   NodeBoundary result={static_cast<std::uint16_t>((1U<<directions)-1),NO_TRIANGLE,
                        NO_TRIANGLE,HUGE_VAL,0,-1,0};
   for (std::uint32_t node_index=0; node_index<node_count;) {
      const BvhNode& node=nodes[node_index];
      if (!detail::node_box_hit(node,point,h)) { node_index=node.escape; continue; }
      for (std::uint32_t n=0; n<node.count; ++n) {
         const std::uint32_t index=order[node.begin+n];
         const Triangle& triangle=triangles[index];
         for (unsigned k=0; k<directions; ++k) {
            const Vec3 vv=mesh_direction(k,fcc);
            const LinkHit hit=triangle_link_unit_hit(triangle,point,
               multiply(vv,unit_component),link_length,h);
            if (hit.distance==HUGE_VAL) continue;
            result.adjacency=static_cast<std::uint16_t>(result.adjacency&~(1U<<k));
            result.near_boundary|=hit.near_boundary;
            if (detail::nearer(hit.distance,triangle.ordinal,index,result.distance,result.ordinal,
                               result.nearest_triangle))
               { result.distance=hit.distance; result.nearest_triangle=index; result.ordinal=triangle.ordinal; }
         }
      }
      ++node_index;
   }
   if (result.near_boundary) result.adjacency=0;
   if (result.nearest_triangle!=NO_TRIANGLE) {
      const Triangle& triangle=triangles[result.nearest_triangle];
      result.material=boundary_material(triangle,point,result.adjacency,result.near_boundary!=0);
      result.saf=surface_factor(triangle.normal,result.adjacency,fcc);
   }
   return result;
}

#undef PFFDTD_MESH_INLINE

// Returns false for a degenerate/below-threshold triangle; invalid values throw.
// A zero-area result may remain in the triangle array to preserve source indices.
inline bool precompute_triangle(Triangle& triangle, Vec3 a, Vec3 b, Vec3 c,
      std::int32_t material, std::uint8_t sides, std::uint32_t ordinal, double min_area=0)
{
   if (!detail::finite(a) || !detail::finite(b) || !detail::finite(c) ||
       material<-1 || sides>3 || !std::isfinite(min_area) || min_area<0)
      throw std::invalid_argument("Invalid triangle vertices, material, side or area threshold");
   triangle=Triangle{};
   triangle.v[0]=a; triangle.v[1]=b; triangle.v[2]=c;
   triangle.material=material; triangle.sides=sides; triangle.ordinal=ordinal;
   triangle.bmin=minimum(minimum(a,b),c); triangle.bmax=maximum(maximum(a,b),c);
   triangle.centroid=multiply(add(add(a,b),c),1.0/3.0);
   const Vec3 edges[3]={subtract(b,a),subtract(c,b),subtract(a,c)};
   const Vec3 nor=multiply(add(add(cross(edges[0],multiply(edges[2],-1)),
      cross(edges[1],multiply(edges[0],-1))),cross(edges[2],multiply(edges[1],-1))),1.0/3.0);
   const double normal_length=detail::norm(nor);
   triangle.area=0.5*normal_length;
   if (!detail::finite(triangle.centroid) || !std::isfinite(triangle.area))
      throw std::invalid_argument("Triangle precomputation exceeds FP64 range");
   if (!(triangle.area>0) || triangle.area<min_area) { triangle.area=0; return false; }
   triangle.normal=divide(nor,normal_length);
   double lengths[3];
   for (unsigned edge=0; edge<3; ++edge) {
      lengths[edge]=detail::norm(edges[edge]);
      const Vec3 outward=cross(edges[edge],triangle.normal);
      const double outward_length=detail::norm(outward);
      if (!(outward_length>0) || !std::isfinite(outward_length))
         throw std::invalid_argument("Triangle edge normal exceeds FP64 range");
      triangle.edge_normal[edge]=divide(outward,outward_length);
      triangle.edge_midpoint[edge]=multiply(add(triangle.v[edge],triangle.v[(edge+1)%3]),0.5);
   }
   // Moving all three outward edge planes by eps scales the triangle about
   // its incenter by 1+eps/inradius. diameter/inradius bounds every vertex's
   // displacement, hence every expanded point's displacement conservatively.
   const double perimeter=lengths[0]+lengths[1]+lengths[2];
   const double diameter=std::max(lengths[0],std::max(lengths[1],lengths[2]));
   const double inradius=normal_length/perimeter;
   triangle.bound_slack=diameter/inradius;
   if (std::isnan(triangle.bound_slack) || !(triangle.bound_slack>0))
      throw std::invalid_argument("Triangle tolerance bounds are not representable");
   return true;
}

namespace detail {

struct BvhBuilder {
   FlatBvh& tree;
   explicit BvhBuilder(FlatBvh& value) : tree(value) {}
   void append(std::size_t begin, std::size_t end)
   {
      if (tree.nodes.size()>=std::numeric_limits<std::uint32_t>::max())
         throw std::invalid_argument("BVH node count exceeds uint32 range");
      const std::size_t index=tree.nodes.size();
      BvhNode node={tree.triangles[tree.order[begin]].bmin,
                    tree.triangles[tree.order[begin]].bmax,0,
                    static_cast<std::uint32_t>(begin),0,0};
      Vec3 centroid_min=tree.triangles[tree.order[begin]].centroid,centroid_max=centroid_min;
      for (std::size_t i=begin; i<end; ++i) {
         const Triangle& triangle=tree.triangles[tree.order[i]];
         node.bmin=minimum(node.bmin,triangle.bmin); node.bmax=maximum(node.bmax,triangle.bmax);
         node.bound_slack=std::max(node.bound_slack,triangle.bound_slack);
         centroid_min=minimum(centroid_min,triangle.centroid);
         centroid_max=maximum(centroid_max,triangle.centroid);
      }
      tree.nodes.push_back(node);
      if (end-begin<=4) tree.nodes[index].count=static_cast<std::uint32_t>(end-begin);
      else {
         const Vec3 extent=subtract(centroid_max,centroid_min);
         unsigned axis=0;
         if (extent.y>extent.x) axis=1;
         if (extent.z>component(extent,axis)) axis=2;
         const std::size_t middle=begin+(end-begin)/2;
         std::nth_element(tree.order.begin()+begin,tree.order.begin()+middle,tree.order.begin()+end,
            [&](std::uint32_t a, std::uint32_t b) {
               const double ca=component(tree.triangles[a].centroid,axis),
                            cb=component(tree.triangles[b].centroid,axis);
               return ca!=cb ? ca<cb : (tree.triangles[a].ordinal<tree.triangles[b].ordinal);
            });
         append(begin,middle); append(middle,end);
      }
      tree.nodes[index].escape=static_cast<std::uint32_t>(tree.nodes.size());
   }
};

} // namespace detail

inline FlatBvh build_bvh(const std::vector<Triangle>& triangles)
{
   if (triangles.size()>std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument("Triangle count exceeds uint32 range");
   FlatBvh result; result.triangles=triangles;
   std::vector<std::uint32_t> ordinals;
   for (std::size_t index=0; index<triangles.size(); ++index) {
      const Triangle& triangle=triangles[index];
      if (!std::isfinite(triangle.area) || triangle.area<0)
         throw std::invalid_argument("BVH triangle area must be finite and nonnegative");
      if (!(triangle.area>0)) continue;
      if (!std::isfinite(triangle.area) || !detail::finite(triangle.bmin) ||
          !detail::finite(triangle.bmax) || !detail::finite(triangle.centroid) ||
          !detail::finite(triangle.normal) || !(triangle.bound_slack>0) ||
          triangle.sides>3 || triangle.material<-1)
         throw std::invalid_argument("BVH requires valid precomputed triangles");
      for (unsigned k=0; k<3; ++k) {
         if (component(triangle.bmin,k)>component(triangle.bmax,k) ||
             !detail::finite(triangle.edge_normal[k]) || !detail::finite(triangle.edge_midpoint[k]) ||
             !detail::finite(triangle.v[k]))
            throw std::invalid_argument("BVH triangle bounds or edge data are invalid");
         for (unsigned axis=0; axis<3; ++axis)
            if (component(triangle.v[k],axis)<component(triangle.bmin,axis) ||
                component(triangle.v[k],axis)>component(triangle.bmax,axis))
               throw std::invalid_argument("BVH triangle bounds do not contain its vertices");
      }
      result.order.push_back(static_cast<std::uint32_t>(index));
      ordinals.push_back(triangle.ordinal);
   }
   std::sort(ordinals.begin(),ordinals.end());
   if (std::adjacent_find(ordinals.begin(),ordinals.end())!=ordinals.end())
      throw std::invalid_argument("Triangle ordinals must be unique");
   if (!result.order.empty()) detail::BvhBuilder(result).append(0,result.order.size());
   return result;
}

inline NodeBoundary classify_node(const FlatBvh& tree, Vec3 point, double h, bool fcc)
{
   if (!detail::finite(point) || !std::isfinite(h) || !(h>0) ||
       h>0.125*std::numeric_limits<double>::max())
      throw std::invalid_argument("Geometry query point or link scale is invalid");
   return classify_node(tree.triangles.data(),tree.nodes.data(),tree.order.data(),
                        static_cast<std::uint32_t>(tree.nodes.size()),point,h,fcc);
}

inline NearestHit query_link(const FlatBvh& tree, Vec3 point, Vec3 link, double h)
{
   if (!detail::finite(point) || !detail::finite(link) || !std::isfinite(h) || !(h>0))
      throw std::invalid_argument("Geometry link query is invalid");
   return query_link(tree.triangles.data(),tree.nodes.data(),tree.order.data(),
                     static_cast<std::uint32_t>(tree.nodes.size()),point,link,h);
}

} // namespace pffdtd_prepare
#endif
