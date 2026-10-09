// Numerical stages of the native JSONRoomExport -> solver HDF5 pipeline.
// Coordinates and geometry calculations use double precision on CPU and CUDA.
#ifndef PFFDTD_PREPARE_SCENE_H
#define PFFDTD_PREPARE_SCENE_H

#include "mesh_geometry.h"
#include "prepare_grid.h"

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace pffdtd_prepare {

struct NamedPoint {
   std::array<double,3> xyz;
   std::string name;
};

struct SceneMesh {
   FlatBvh geometry;
   std::array<double,3> bmin, bmax;
   std::vector<std::string> material_labels;
   std::vector<NamedPoint> sources, receivers;
   std::size_t triangles_input=0, triangles_removed=0;
};

struct MaterialDef {
   std::string label, filename;
   // Row-major normalized specific impedance branches: D*s + E + F/s.
   std::vector<double> def;
};

struct SimConstants {
   double c=0, h=0, ts=0, fs=0, l=0, l2=0, temperature=20, humidity=50;
};

struct BoundaryRecord {
   int64_t index=0;
   uint16_t adjacency=0;
   uint32_t triangle=UINT32_MAX;
   int32_t material=-1;
   double saf=0;
   bool near_boundary=false;
};

struct PrepareOptions {
   double spacing=0, fmax=1000, ppw=10, duration=1;
   double temperature=20, humidity=50, min_triangle_area=1e-6;
   int64_t steps=0;
   unsigned source_num=1, threads=0;
   bool fcc=false, rotate=true, fold=true, differentiated=true;
   int target_precision=32;
};

struct PreparedScene {
   SceneMesh mesh;
   Grid physical_grid;
   IndexTransform layout;
   SimConstants constants;
   std::vector<MaterialDef> materials;
   std::vector<BoundaryRecord> boundary;
   std::vector<int64_t> sources, receivers, reorder;
   std::vector<double> input, weights;
   int64_t steps=0;
   bool differentiated=false;
   int target_precision=32;
   std::size_t reconciled_links=0, added_boundary_nodes=0;
   std::string backend="cpu";
};

typedef std::vector<BoundaryRecord> (*Voxelizer)(const FlatBvh &,const Grid &,unsigned);

SceneMesh import_scene(const std::string &filename,double min_triangle_area=1e-6);
std::vector<MaterialDef> import_materials(const SceneMesh &,
                                        const std::map<std::string,std::string> &);
SimConstants make_constants(const PrepareOptions &);
std::vector<BoundaryRecord> voxelize_cpu(const FlatBvh &,const Grid &,unsigned threads=0);
void reconcile_boundary(PreparedScene &);
void prepare_comms(PreparedScene &,const PrepareOptions &);
void apply_layout(PreparedScene &,const PrepareOptions &);
void validate_prepared(const PreparedScene &);
PreparedScene prepare_scene(SceneMesh,const std::vector<MaterialDef> &,
                            const PrepareOptions &,Voxelizer=voxelize_cpu);

// Creates a new directory exclusively. Failed writes remove only files created
// by this call. The original native solver loader validates every saved input.
void write_prepared(const PreparedScene &,const std::string &directory,bool verbose=false);

} // namespace pffdtd_prepare
#endif
