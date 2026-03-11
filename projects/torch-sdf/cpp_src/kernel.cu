#include "kernel.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>

namespace toponav {
namespace cuda {

// --- Device Helper: Stackless DAG Query ---
__device__ inline void query_dag_stackless(
    const uint64_t* __restrict__ dag_memory,
    float px, float py, float pz,
    const float bbox_min[3], const float bbox_max[3], int resolution,
    float& out_dist, int32_t& out_semantic) 
{
    if (px < bbox_min[0] || px > bbox_max[0] ||
        py < bbox_min[1] || py > bbox_max[1] ||
        pz < bbox_min[2] || pz > bbox_max[2]) {
        out_dist = kOutsideDomainDistance;
        out_semantic = 0;
        return;
    }

    uint32_t current_ptr = 0;
    float cur_bmin[3] = {bbox_min[0], bbox_min[1], bbox_min[2]};
    float cur_bmax[3] = {bbox_max[0], bbox_max[1], bbox_max[2]};
    
    // Iterative descent
    for (int depth = 0; depth < 32; ++depth) {
        uint64_t node = dag_memory[current_ptr];
        
        if ((node >> 63) == 1) {
            uint16_t dist_bits = static_cast<uint16_t>(node & 0xFFFF);
            out_dist = __half2float(*reinterpret_cast<__half*>(&dist_bits));
            out_semantic = static_cast<int32_t>((node >> 16) & 0xFFFF);
            return;
        }

        // Calculate midpoints
        float mx = (cur_bmin[0] + cur_bmax[0]) * 0.5f;
        float my = (cur_bmin[1] + cur_bmax[1]) * 0.5f;
        float mz = (cur_bmin[2] + cur_bmax[2]) * 0.5f;

        // Determine octant
        int octant_idx = (px >= mx ? 1 : 0) | (py >= my ? 2 : 0) | (pz >= mz ? 4 : 0);

        // Update bounds for next level
        if (px >= mx) cur_bmin[0] = mx; else cur_bmax[0] = mx;
        if (py >= my) cur_bmin[1] = my; else cur_bmax[1] = my;
        if (pz >= mz) cur_bmin[2] = mz; else cur_bmax[2] = mz;

        uint8_t mask = static_cast<uint8_t>((node >> 55) & 0xFF);
        if ((mask & (1 << octant_idx)) == 0) {
            out_dist = kOutsideDomainDistance;
            out_semantic = 0;
            return;
        }

        uint32_t child_base = static_cast<uint32_t>(node & 0xFFFFFFFF);
        uint32_t bitmask_prior = mask & ((1 << octant_idx) - 1);
        uint32_t offset = __popc(bitmask_prior);

        // Fetch the actual pointer to the child node from the pointer array
        current_ptr = static_cast<uint32_t>(dag_memory[child_base + offset]);    }
}

// --- The Primary Raycasting Kernel ---
__global__ void sphere_trace_kernel(
    const uint64_t* __restrict__ dag_memory,
    const float* __restrict__ origins,
    const float* __restrict__ dirs,
    float* __restrict__ out_distances,
    int32_t* __restrict__ out_semantics,
    int num_rays, int max_steps, float max_distance,
    float bmin_x, float bmin_y, float bmin_z,
    float bmax_x, float bmax_y, float bmax_z,
    int resolution) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rays) return;

    float ox = origins[idx * 3 + 0];
    float oy = origins[idx * 3 + 1];
    float oz = origins[idx * 3 + 2];

    float dx = dirs[idx * 3 + 0];
    float dy = dirs[idx * 3 + 1];
    float dz = dirs[idx * 3 + 2];

    float current_t = 0.0f;
    float dist = 0.0f;
    int32_t semantic = 0;

    const float float_bmin[3] = {bmin_x, bmin_y, bmin_z};
    const float float_bmax[3] = {bmax_x, bmax_y, bmax_z};

    for (int step = 0; step < max_steps; ++step) {
        float px = ox + current_t * dx;
        float py = oy + current_t * dy;
        float pz = oz + current_t * dz;

        query_dag_stackless(dag_memory, px, py, pz, float_bmin, float_bmax, resolution, dist, semantic);

        if (dist < kHitEpsilon) {
            out_distances[idx] = current_t;
            out_semantics[idx] = semantic;
            return;
        }
        current_t += dist;
        
        if (current_t > max_distance) break;
    }

    out_distances[idx] = current_t;
    out_semantics[idx] = 0;
}

void launch_sphere_trace_kernel(
    const uint64_t* dag_memory, const float* origins, const float* dirs,
    float* out_distances, int32_t* out_semantics,
    int num_rays, int max_steps, float max_distance,
    const float bbox_min[3], const float bbox_max[3], int resolution)
{
    int threads = 256;
    int blocks = (num_rays + threads - 1) / threads;

    sphere_trace_kernel<<<blocks, threads>>>(
        dag_memory, origins, dirs, out_distances, out_semantics, num_rays, max_steps, max_distance,
        bbox_min[0], bbox_min[1], bbox_min[2],
        bbox_max[0], bbox_max[1], bbox_max[2],
        resolution
    );
}

} // namespace cuda
} // namespace toponav
