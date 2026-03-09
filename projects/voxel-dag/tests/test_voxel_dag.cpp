#include <gtest/gtest.h>
#include "voxel_dag/MeshIngestor.hpp"
#include "voxel_dag/EikonalSolver.hpp"
#include "voxel_dag/DagCompressor.hpp"
#include <fstream>
#include <cmath>

using namespace toponav::dag;

// Test helper to create a simple cube OBJ file
void create_cube_obj(const std::string& filename) {
    std::ofstream f(filename);
    f << "v 0.0 0.0 0.0
"
      << "v 1.0 0.0 0.0
"
      << "v 1.0 1.0 0.0
"
      << "v 0.0 1.0 0.0
"
      << "v 0.0 0.0 1.0
"
      << "v 1.0 0.0 1.0
"
      << "v 1.0 1.0 1.0
"
      << "v 0.0 1.0 1.0
"
      << "f 1 2 3
f 1 3 4
"
      << "f 5 6 7
f 5 7 8
"
      << "f 1 2 6
f 1 6 5
"
      << "f 2 3 7
f 2 7 6
"
      << "f 3 4 8
f 3 8 7
"
      << "f 4 1 5
f 4 5 8
";
    f.close();
}

TEST(VoxelDagTest, MeshIngestion) {
    create_cube_obj("cube.obj");
    MeshData mesh = MeshIngestor::load_obj("cube.obj");
    
    EXPECT_EQ(mesh.vertices.size(), 8 * 3);
    EXPECT_EQ(mesh.indices.size(), 12 * 3);
    EXPECT_FLOAT_EQ(mesh.bounds.min_x, 0.0f);
    EXPECT_FLOAT_EQ(mesh.bounds.max_x, 1.0f);
}

TEST(VoxelDagTest, EikonalSolver) {
    create_cube_obj("cube.obj");
    MeshData mesh = MeshIngestor::load_obj("cube.obj");
    
    int resolution = 8;
    auto sdf = EikonalSolver::compute_dense_sdf(mesh, resolution, 0.5f);
    
    EXPECT_EQ(sdf.size(), resolution * resolution * resolution);
    
    // Check some distances (should be positive away from surface)
    for (float d : sdf) {
        EXPECT_GE(d, 0.0f);
        EXPECT_LT(d, 10.0f);
    }
}

TEST(VoxelDagTest, DagCompression) {
    int resolution = 4; // Small resolution for testing
    std::vector<float> dense_grid(resolution * resolution * resolution, 1.0f);
    
    auto dag = DagCompressor::compress_to_dag(dense_grid, resolution);
    
    // Since all voxels are 1.0, there should be significant deduplication
    EXPECT_GT(dag.size(), 0);
    EXPECT_LT(dag.size(), resolution * resolution * resolution);
}
