#include "compiler.hpp"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <limits>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

namespace voxeldag {

Compiler::Compiler(uint32_t resolution) : m_resolution(resolution), m_voxel_size(0.0f) {
    m_bmin[0] = m_bmin[1] = m_bmin[2] = 0.0f;
    m_bmax[0] = m_bmax[1] = m_bmax[2] = 0.0f;
}

void Compiler::compile(const std::string& input_path, const std::string& output_path) {
    std::cout << "[1/4] Loading Mesh: " << input_path << "...\n";
    loadMesh(input_path);

    std::cout << "[2/4] Computing Signed Distance Field (Resolution: " << m_resolution << "^3)...\n";
    std::vector<float> dense_sdf = computeSDF();

    std::cout << "[3/4] Compressing to Directed Acyclic Graph...\n";
    std::vector<uint64_t> dag_nodes = compressToDAG(dense_sdf);

    std::cout << "[4/4] Exporting .gmdag binary...\n";
    writeBinary(output_path, dag_nodes);

    std::cout << "Compilation Successful: " << output_path
              << " (" << dag_nodes.size() << " nodes)\n";
}

// ═══════════════════════════════════════════════════════════════════
//  Stage 1: Mesh Loading (Assimp — supports .glb, .obj, .ply, .fbx)
// ═══════════════════════════════════════════════════════════════════

void Compiler::loadMesh(const std::string& path) {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path,
        aiProcess_Triangulate |
        aiProcess_JoinIdenticalVertices |
        aiProcess_PreTransformVertices);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        throw std::runtime_error("Assimp Error: " + std::string(importer.GetErrorString()));
    }

    // Initialize bounds to extreme values
    m_bmin[0] = m_bmin[1] = m_bmin[2] =  std::numeric_limits<float>::max();
    m_bmax[0] = m_bmax[1] = m_bmax[2] = std::numeric_limits<float>::lowest();

    m_vertices.clear();
    m_indices.clear();

    uint32_t vertex_offset = 0;

    // Extract all meshes from all nodes
    for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
        const aiMesh* mesh = scene->mMeshes[m];

        // Extract vertices and update bounding box
        for (unsigned int v = 0; v < mesh->mNumVertices; ++v) {
            float x = mesh->mVertices[v].x;
            float y = mesh->mVertices[v].y;
            float z = mesh->mVertices[v].z;

            m_vertices.push_back(x);
            m_vertices.push_back(y);
            m_vertices.push_back(z);

            if (x < m_bmin[0]) m_bmin[0] = x;
            if (y < m_bmin[1]) m_bmin[1] = y;
            if (z < m_bmin[2]) m_bmin[2] = z;
            if (x > m_bmax[0]) m_bmax[0] = x;
            if (y > m_bmax[1]) m_bmax[1] = y;
            if (z > m_bmax[2]) m_bmax[2] = z;
        }

        // Extract face indices (all triangulated)
        for (unsigned int f = 0; f < mesh->mNumFaces; ++f) {
            const aiFace& face = mesh->mFaces[f];
            for (unsigned int i = 0; i < face.mNumIndices; ++i) {
                m_indices.push_back(vertex_offset + face.mIndices[i]);
            }
        }

        vertex_offset += mesh->mNumVertices;
    }

    // Compute voxel size from the longest axis
    float extent_x = m_bmax[0] - m_bmin[0];
    float extent_y = m_bmax[1] - m_bmin[1];
    float extent_z = m_bmax[2] - m_bmin[2];
    float max_extent = std::max({extent_x, extent_y, extent_z});
    m_voxel_size = (max_extent + 2.0f) / static_cast<float>(m_resolution); // +2m padding

    float center_x = (m_bmax[0] + m_bmin[0]) / 2.0f;
    float center_y = (m_bmax[1] + m_bmin[1]) / 2.0f;
    float center_z = (m_bmax[2] + m_bmin[2]) / 2.0f;
    
    m_bmin[0] = center_x - (max_extent + 2.0f) / 2.0f;
    m_bmin[1] = center_y - (max_extent + 2.0f) / 2.0f;
    m_bmin[2] = center_z - (max_extent + 2.0f) / 2.0f;
    m_bmax[0] = center_x + (max_extent + 2.0f) / 2.0f;
    m_bmax[1] = center_y + (max_extent + 2.0f) / 2.0f;
    m_bmax[2] = center_z + (max_extent + 2.0f) / 2.0f;

    std::cout << "  Loaded " << m_vertices.size() / 3 << " vertices, "
              << m_indices.size() / 3 << " triangles\n";
    std::cout << "  Bounds: [" << m_bmin[0] << ", " << m_bmin[1] << ", " << m_bmin[2] << "] -> ["
              << m_bmax[0] << ", " << m_bmax[1] << ", " << m_bmax[2] << "]\n";
    std::cout << "  Voxel size: " << m_voxel_size << "m\n";
}

// ═══════════════════════════════════════════════════════════════════
//  Stage 2: SDF Computation (Eikonal / Fast Sweeping Method)
// ═══════════════════════════════════════════════════════════════════

float Compiler::point_triangle_distance_sq(
    float px, float py, float pz,
    const float* v0, const float* v1, const float* v2)
{
    float diff[3] = { px - v0[0], py - v0[1], pz - v0[2] };
    float edge0[3] = { v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2] };
    float edge1[3] = { v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2] };
    float a00 = edge0[0]*edge0[0] + edge0[1]*edge0[1] + edge0[2]*edge0[2];
    float a01 = edge0[0]*edge1[0] + edge0[1]*edge1[1] + edge0[2]*edge1[2];
    float a11 = edge1[0]*edge1[0] + edge1[1]*edge1[1] + edge1[2]*edge1[2];
    float b0 = -(diff[0]*edge0[0] + diff[1]*edge0[1] + diff[2]*edge0[2]);
    float b1 = -(diff[0]*edge1[0] + diff[1]*edge1[1] + diff[2]*edge1[2]);
    float det = std::abs(a00 * a11 - a01 * a01);
    float s = a01 * b1 - a11 * b0;
    float t = a01 * b0 - a00 * b1;

    if (s + t <= det) {
        if (s < 0) {
            if (t < 0) {
                if (b0 < 0) {
                    t = 0;
                    if (-b0 >= a00) s = 1; else s = -b0 / a00;
                } else {
                    s = 0;
                    if (b1 >= 0) t = 0; else if (-b1 >= a11) t = 1; else t = -b1 / a11;
                }
            } else {
                s = 0;
                if (b1 >= 0) t = 0; else if (-b1 >= a11) t = 1; else t = -b1 / a11;
            }
        } else if (t < 0) {
            t = 0;
            if (b0 >= 0) s = 0; else if (-b0 >= a00) s = 1; else s = -b0 / a00;
        } else {
            float invDet = 1.0f / det;
            s *= invDet;
            t *= invDet;
        }
    } else {
        if (s < 0) {
            float tmp0 = a01 + b0;
            float tmp1 = a11 + b1;
            if (tmp1 > tmp0) {
                float numer = tmp1 - tmp0;
                float denom = a00 - 2.0f * a01 + a11;
                if (numer >= denom) s = 1; else s = numer / denom;
                t = 1 - s;
            } else {
                s = 0;
                if (tmp1 >= 0) t = 0; else if (-b1 >= a11) t = 1; else t = -b1 / a11;
            }
        } else if (t < 0) {
            float tmp0 = a01 + b1;
            float tmp1 = a00 + b0;
            if (tmp1 > tmp0) {
                float numer = tmp1 - tmp0;
                float denom = a00 - 2.0f * a01 + a11;
                if (numer >= denom) t = 1; else t = numer / denom;
                s = 1 - t;
            } else {
                t = 0;
                if (tmp1 >= 0) s = 0; else if (-b0 >= a00) s = 1; else s = -b0 / a00;
            }
        } else {
            float numer = a11 + b1 - a01 - b0;
            if (numer <= 0) {
                s = 0;
                t = 1;
            } else {
                float denom = a00 - 2.0f * a01 + a11;
                if (numer >= denom) s = 1; else s = numer / denom;
                t = 1 - s;
            }
        }
    }

    float res_x = v0[0] + s * edge0[0] + t * edge1[0] - px;
    float res_y = v0[1] + s * edge0[1] + t * edge1[1] - py;
    float res_z = v0[2] + s * edge0[2] + t * edge1[2] - pz;
    return res_x * res_x + res_y * res_y + res_z * res_z;
}

inline float Compiler::solve_eikonal(float a, float b, float c, float h) {
    if (a > b) std::swap(a, b);
    if (b > c) std::swap(b, c);
    if (a > b) std::swap(a, b);

    float d = a + h;
    if (d <= b) return d;

    d = 0.5f * (a + b + std::sqrt(2.0f * h * h - (a - b) * (a - b)));
    if (d <= c) return d;

    float sum = a + b + c;
    float sum_sq = a * a + b * b + c * c;
    float discriminant = sum * sum - 3.0f * (sum_sq - h * h);
    if (discriminant >= 0.0f) {
        return (sum + std::sqrt(discriminant)) / 3.0f;
    }
    return d;
}

std::vector<float> Compiler::computeSDF() {
    int N = static_cast<int>(m_resolution);
    std::vector<float> grid(static_cast<size_t>(N) * N * N, std::numeric_limits<float>::max());

    float dx = (m_bmax[0] - m_bmin[0]) / N;
    float dy = (m_bmax[1] - m_bmin[1]) / N;
    float dz = (m_bmax[2] - m_bmin[2]) / N;
    float h = std::min({dx, dy, dz});

    // --- STEP 1: Initialization (seed with exact triangle distances) ---
    for (size_t i = 0; i < m_indices.size(); i += 3) {
        const float* v0 = &m_vertices[m_indices[i] * 3];
        const float* v1 = &m_vertices[m_indices[i + 1] * 3];
        const float* v2 = &m_vertices[m_indices[i + 2] * 3];

        float tri_min_x = std::min({v0[0], v1[0], v2[0]}) - h;
        float tri_max_x = std::max({v0[0], v1[0], v2[0]}) + h;
        float tri_min_y = std::min({v0[1], v1[1], v2[1]}) - h;
        float tri_max_y = std::max({v0[1], v1[1], v2[1]}) + h;
        float tri_min_z = std::min({v0[2], v1[2], v2[2]}) - h;
        float tri_max_z = std::max({v0[2], v1[2], v2[2]}) + h;

        int ix_start = std::max(0, (int)((tri_min_x - m_bmin[0]) / dx));
        int ix_end   = std::min(N - 1, (int)((tri_max_x - m_bmin[0]) / dx));
        int iy_start = std::max(0, (int)((tri_min_y - m_bmin[1]) / dy));
        int iy_end   = std::min(N - 1, (int)((tri_max_y - m_bmin[1]) / dy));
        int iz_start = std::max(0, (int)((tri_min_z - m_bmin[2]) / dz));
        int iz_end   = std::min(N - 1, (int)((tri_max_z - m_bmin[2]) / dz));

        for (int z = iz_start; z <= iz_end; ++z) {
            float pz = m_bmin[2] + z * dz;
            for (int y = iy_start; y <= iy_end; ++y) {
                float py = m_bmin[1] + y * dy;
                for (int x = ix_start; x <= ix_end; ++x) {
                    float px = m_bmin[0] + x * dx;

                    float d2 = point_triangle_distance_sq(px, py, pz, v0, v1, v2);
                    float d = std::sqrt(d2);

                    int idx = z * N * N + y * N + x;
                    if (d < grid[idx]) {
                        grid[idx] = d;
                    }
                }
            }
        }
    }

    // --- STEP 2: Fast Sweeping Passes (8 diagonal directions) ---
    int sweep_dirs[8][3] = {
        {1, 1, 1}, {-1, 1, 1}, {1, -1, 1}, {-1, -1, 1},
        {1, 1, -1}, {-1, 1, -1}, {1, -1, -1}, {-1, -1, -1}
    };

    for (int sweep = 0; sweep < 8; ++sweep) {
        int ix_s = (sweep_dirs[sweep][0] > 0) ? 0 : N - 1;
        int ix_e = (sweep_dirs[sweep][0] > 0) ? N : -1;
        int ix_d = sweep_dirs[sweep][0];

        int iy_s = (sweep_dirs[sweep][1] > 0) ? 0 : N - 1;
        int iy_e = (sweep_dirs[sweep][1] > 0) ? N : -1;
        int iy_d = sweep_dirs[sweep][1];

        int iz_s = (sweep_dirs[sweep][2] > 0) ? 0 : N - 1;
        int iz_e = (sweep_dirs[sweep][2] > 0) ? N : -1;
        int iz_d = sweep_dirs[sweep][2];

        for (int z = iz_s; z != iz_e; z += iz_d) {
            for (int y = iy_s; y != iy_e; y += iy_d) {
                for (int x = ix_s; x != ix_e; x += ix_d) {
                    int idx = z * N * N + y * N + x;

                    float a = (x - ix_d >= 0 && x - ix_d < N) ? grid[z * N * N + y * N + (x - ix_d)] : std::numeric_limits<float>::max();
                    float b = (y - iy_d >= 0 && y - iy_d < N) ? grid[z * N * N + (y - iy_d) * N + x] : std::numeric_limits<float>::max();
                    float c = (z - iz_d >= 0 && z - iz_d < N) ? grid[(z - iz_d) * N * N + y * N + x] : std::numeric_limits<float>::max();

                    float current = grid[idx];
                    float updated = solve_eikonal(a, b, c, h);

                    if (updated < current) {
                        grid[idx] = updated;
                    }
                }
            }
        }
    }

    return grid;
}

// ═══════════════════════════════════════════════════════════════════
//  Stage 3: DAG Compression (SVO + Hash-based Deduplication)
// ═══════════════════════════════════════════════════════════════════

uint64_t Compiler::hash_node(const std::vector<uint32_t>& child_indices) {
    uint64_t h = 0xCBF29CE484222325ULL; // FNV offset basis
    for (uint32_t idx : child_indices) {
        h ^= static_cast<uint64_t>(idx);
        h *= 0x100000001B3ULL; // FNV prime
    }
    return h;
}

uint16_t Compiler::float_to_half(float f) {
    uint32_t i;
    std::memcpy(&i, &f, sizeof(float));
    uint32_t s = (i >> 16) & 0x00008000;
    uint32_t e = ((i >> 23) & 0x000000ff) - (127 - 15);
    uint32_t m = i & 0x007fffff;

    if (e <= 0) {
        if (static_cast<int>(e) < -10) return static_cast<uint16_t>(s);
        m = (m | 0x00800000) >> (1 - e);
        return static_cast<uint16_t>(s | (m >> 13));
    } else if (e == 0xff - (127 - 15)) {
        if (m == 0) return static_cast<uint16_t>(s | 0x7c00);
        return static_cast<uint16_t>(s | 0x7c00 | (m >> 13) | (m ? 1 : 0));
    } else {
        if (e > 30) return static_cast<uint16_t>(s | 0x7c00);
        return static_cast<uint16_t>(s | (e << 10) | (m >> 13));
    }
}

uint32_t Compiler::build_recursive(
    const std::vector<float>& dense_grid,
    int N, int x, int y, int z, int size,
    std::vector<uint64_t>& dag_pool,
    std::unordered_map<uint64_t, uint32_t>& unique_nodes)
{
    if (size == 1) {
        // LEAF NODE (Type 1)
        float dist = dense_grid[z * N * N + y * N + x];
        uint16_t semantic_id = 0;
        uint16_t dist_f16 = float_to_half(dist);

        uint64_t node = (1ULL << 63); // Type 1 flag
        node |= (static_cast<uint64_t>(semantic_id) << 16);
        node |= static_cast<uint64_t>(dist_f16);

        if (unique_nodes.find(node) != unique_nodes.end()) {
            return unique_nodes[node];
        }

        uint32_t new_ptr = static_cast<uint32_t>(dag_pool.size());
        dag_pool.push_back(node);
        unique_nodes[node] = new_ptr;
        return new_ptr;
    }

    int half = size / 2;
    uint8_t child_mask = 0;
    std::vector<uint32_t> children;

    for (int i = 0; i < 8; ++i) {
        int ox = (i & 1) ? half : 0;
        int oy = (i & 2) ? half : 0;
        int oz = (i & 4) ? half : 0;

        uint32_t child_ptr = build_recursive(dense_grid, N, x + ox, y + oy, z + oz, half, dag_pool, unique_nodes);
        children.push_back(child_ptr);
        child_mask |= (1 << i);
    }

    // INTERNAL NODE (Type 0)
    uint64_t h = hash_node(children);
    if (unique_nodes.find(h) != unique_nodes.end()) {
        return unique_nodes[h];
    }

    uint32_t node_ptr = static_cast<uint32_t>(dag_pool.size());
    dag_pool.push_back(0); // Placeholder for the node

    uint32_t start_ptr = static_cast<uint32_t>(dag_pool.size());
    for (uint32_t c : children) {
        dag_pool.push_back(static_cast<uint64_t>(c));
    }

    uint64_t internal_node = 0;
    internal_node |= (static_cast<uint64_t>(child_mask) << 55);
    internal_node |= static_cast<uint64_t>(start_ptr);

    dag_pool[node_ptr] = internal_node;
    unique_nodes[h] = node_ptr;
    return node_ptr;
}

std::vector<uint64_t> Compiler::compressToDAG(const std::vector<float>& dense_sdf) {
    std::vector<uint64_t> dag_pool;
    dag_pool.push_back(0); // Reserve index 0 for the root node copy

    std::unordered_map<uint64_t, uint32_t> unique_nodes;

    uint32_t root_ptr = build_recursive(dense_sdf, static_cast<int>(m_resolution), 0, 0, 0,
                    static_cast<int>(m_resolution), dag_pool, unique_nodes);

    // Copy the root node to index 0 so that CUDA sphere-tracing can simply start at index 0
    dag_pool[0] = dag_pool[root_ptr];

    return dag_pool;
}

// ═══════════════════════════════════════════════════════════════════
//  Stage 4: Binary Serialization (.gmdag)
// ═══════════════════════════════════════════════════════════════════

void Compiler::writeBinary(const std::string& path, const std::vector<uint64_t>& dag_nodes) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open output file: " + path);
    }

    DagHeader header;
    header.resolution = m_resolution;
    header.bmin_x = m_bmin[0];
    header.bmin_y = m_bmin[1];
    header.bmin_z = m_bmin[2];
    header.voxel_size = m_voxel_size;
    header.node_count = static_cast<uint32_t>(dag_nodes.size());

    out.write(reinterpret_cast<const char*>(&header), sizeof(DagHeader));
    out.write(reinterpret_cast<const char*>(dag_nodes.data()), dag_nodes.size() * sizeof(uint64_t));
    out.close();
}

} // namespace voxeldag
