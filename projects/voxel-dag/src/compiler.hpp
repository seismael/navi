#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>

namespace voxeldag {

// Strictly matches the Python struct.unpack('<4sIIffffI', ...)
#pragma pack(push, 1)
struct DagHeader {
    char magic[4] = {'G', 'D', 'A', 'G'};
    uint32_t version = 1;
    uint32_t resolution;
    float bmin_x;
    float bmin_y;
    float bmin_z;
    float voxel_size;
    uint32_t node_count;
};
#pragma pack(pop)

class Compiler {
public:
    Compiler(uint32_t resolution);
    ~Compiler() = default;

    // The primary execution pipeline
    void compile(const std::string& input_path, const std::string& output_path);

private:
    uint32_t m_resolution;
    float m_voxel_size;
    float m_bmin[3];
    float m_bmax[3];

    // Mesh data (populated by loadMesh)
    std::vector<float> m_vertices;     // Flattened [x, y, z, x, y, z...]
    std::vector<uint32_t> m_indices;   // Flattened triangle indices [v0, v1, v2...]

    // Pipeline stages
    void loadMesh(const std::string& path);
    std::vector<float> computeSDF();
    std::vector<uint64_t> compressToDAG(const std::vector<float>& dense_sdf);
    void writeBinary(const std::string& path, const std::vector<uint64_t>& dag_nodes);

    // --- SDF Math (Eikonal / Fast Sweeping) ---
    static inline float solve_eikonal(float a, float b, float c, float h);
    static float point_triangle_distance_sq(
        float px, float py, float pz,
        const float* v0, const float* v1, const float* v2);

    // --- DAG Compression ---
    static uint64_t hash_node(const std::vector<uint32_t>& child_indices);
    static uint16_t float_to_half(float f);
    static uint32_t build_recursive(
        const std::vector<float>& dense_grid,
        int N, int x, int y, int z, int size,
        std::vector<uint64_t>& dag_pool,
        std::unordered_map<uint64_t, uint32_t>& unique_nodes);
};

} // namespace voxeldag
