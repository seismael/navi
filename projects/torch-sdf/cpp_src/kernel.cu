#include "kernel.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>

namespace toponav {
namespace cuda {

constexpr float kVoidAdvanceEpsilon = 1e-5f;

__device__ inline bool point_in_bounds(
    float px,
    float py,
    float pz,
    const float bmin[3],
    const float bmax[3])
{
    return px >= bmin[0] && px <= bmax[0] &&
           py >= bmin[1] && py <= bmax[1] &&
           pz >= bmin[2] && pz <= bmax[2];
}

__device__ inline bool point_in_bounds_strict(
    float px,
    float py,
    float pz,
    const float bmin[3],
    const float bmax[3])
{
    return px > bmin[0] && px < bmax[0] &&
           py > bmin[1] && py < bmax[1] &&
           pz > bmin[2] && pz < bmax[2];
}

__device__ inline uint64_t load_dag_word(
    const uint64_t* __restrict__ dag_memory,
    const uint64_t* __restrict__ dag_cache,
    uint32_t ptr)
{
    return (ptr < kDagCacheSize) ? dag_cache[ptr] : dag_memory[ptr];
}

__device__ inline float decode_leaf_distance(uint64_t node)
{
    uint16_t dist_bits = static_cast<uint16_t>(node & 0xFFFF);
    return __half2float(*reinterpret_cast<__half*>(&dist_bits));
}

__device__ inline int32_t decode_leaf_semantic(uint64_t node)
{
    return static_cast<int32_t>((node >> 16) & 0xFFFF);
}

__device__ inline float distance_to_bounds_exit(
    float px,
    float py,
    float pz,
    float dx,
    float dy,
    float dz,
    const float bmin[3],
    const float bmax[3])
{
    float exit_t = kOutsideDomainDistance;

    if (dx > kDirectionNormEpsilon) {
        exit_t = fminf(exit_t, (bmax[0] - px) / dx);
    } else if (dx < -kDirectionNormEpsilon) {
        exit_t = fminf(exit_t, (bmin[0] - px) / dx);
    }

    if (dy > kDirectionNormEpsilon) {
        exit_t = fminf(exit_t, (bmax[1] - py) / dy);
    } else if (dy < -kDirectionNormEpsilon) {
        exit_t = fminf(exit_t, (bmin[1] - py) / dy);
    }

    if (dz > kDirectionNormEpsilon) {
        exit_t = fminf(exit_t, (bmax[2] - pz) / dz);
    } else if (dz < -kDirectionNormEpsilon) {
        exit_t = fminf(exit_t, (bmin[2] - pz) / dz);
    }

    return fmaxf(exit_t, kVoidAdvanceEpsilon);
}

// --- Device Helper: Stackless DAG Query ---
__device__ inline void query_dag_stackless(
    const uint64_t* __restrict__ dag_memory,
    const uint64_t* __restrict__ dag_cache,
    float px, float py, float pz,
    float dx, float dy, float dz,
    const float bbox_min[3], const float bbox_max[3], int resolution,
    uint32_t& cached_ptr,
    float cached_bmin[3],
    float cached_bmax[3],
    bool& cache_valid,
    float void_bmin[3],
    float void_bmax[3],
    bool& void_cache_valid,
    float leaf_bmin[3],
    float leaf_bmax[3],
    float& cached_leaf_dist,
    int32_t& cached_leaf_semantic,
    bool& leaf_cache_valid,
    float& out_dist, int32_t& out_semantic) 
{
    if (!point_in_bounds(px, py, pz, bbox_min, bbox_max)) {
        out_dist = kOutsideDomainDistance;
        out_semantic = 0;
        return;
    }

    if (leaf_cache_valid && point_in_bounds(px, py, pz, leaf_bmin, leaf_bmax)) {
        out_dist = cached_leaf_dist;
        out_semantic = cached_leaf_semantic;
        return;
    }

    const bool use_cached_prefix = cache_valid && point_in_bounds(px, py, pz, cached_bmin, cached_bmax);
    uint32_t current_ptr = use_cached_prefix ? cached_ptr : 0;
    float cur_bmin[3] = {
        use_cached_prefix ? cached_bmin[0] : bbox_min[0],
        use_cached_prefix ? cached_bmin[1] : bbox_min[1],
        use_cached_prefix ? cached_bmin[2] : bbox_min[2],
    };
    float cur_bmax[3] = {
        use_cached_prefix ? cached_bmax[0] : bbox_max[0],
        use_cached_prefix ? cached_bmax[1] : bbox_max[1],
        use_cached_prefix ? cached_bmax[2] : bbox_max[2],
    };
    int depth = use_cached_prefix ? kTraversalCacheDepth : 0;
    (void)resolution;
    
    // Iterative descent
    for (; depth < 32; ++depth) {
        uint64_t node = load_dag_word(dag_memory, dag_cache, current_ptr);
        
        if ((node >> 63) == 1) {
            leaf_bmin[0] = cur_bmin[0];
            leaf_bmin[1] = cur_bmin[1];
            leaf_bmin[2] = cur_bmin[2];
            leaf_bmax[0] = cur_bmax[0];
            leaf_bmax[1] = cur_bmax[1];
            leaf_bmax[2] = cur_bmax[2];
            cached_leaf_dist = decode_leaf_distance(node);
            cached_leaf_semantic = decode_leaf_semantic(node);
            leaf_cache_valid = true;
            out_dist = cached_leaf_dist;
            out_semantic = cached_leaf_semantic;
            return;
        }

        if (!use_cached_prefix && depth == kTraversalCacheDepth) {
            cached_ptr = current_ptr;
            cached_bmin[0] = cur_bmin[0];
            cached_bmin[1] = cur_bmin[1];
            cached_bmin[2] = cur_bmin[2];
            cached_bmax[0] = cur_bmax[0];
            cached_bmax[1] = cur_bmax[1];
            cached_bmax[2] = cur_bmax[2];
            cache_valid = true;
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
            void_bmin[0] = cur_bmin[0];
            void_bmin[1] = cur_bmin[1];
            void_bmin[2] = cur_bmin[2];
            void_bmax[0] = cur_bmax[0];
            void_bmax[1] = cur_bmax[1];
            void_bmax[2] = cur_bmax[2];
            void_cache_valid = true;
            out_dist = distance_to_bounds_exit(px, py, pz, dx, dy, dz, void_bmin, void_bmax);
            out_semantic = 0;
            return;
        }

        uint32_t child_base = static_cast<uint32_t>(node & 0xFFFFFFFF);
        uint32_t bitmask_prior = mask & ((1 << octant_idx) - 1);
        uint32_t offset = __popc(bitmask_prior);

        // Fetch the actual pointer to the child node from the pointer array
        uint32_t child_ptr_idx = child_base + offset;
        current_ptr = static_cast<uint32_t>(load_dag_word(dag_memory, dag_cache, child_ptr_idx));
    }
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
    __shared__ uint64_t dag_cache[kDagCacheSize];

    // Cooperative load of the DAG top-levels into shared memory
    for (int i = threadIdx.x; i < kDagCacheSize; i += blockDim.x) {
        dag_cache[i] = dag_memory[i];
    }
    __syncthreads();

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
    uint32_t cached_ptr = 0;
    float cached_bmin[3] = {bmin_x, bmin_y, bmin_z};
    float cached_bmax[3] = {bmax_x, bmax_y, bmax_z};
    bool cache_valid = false;
    float void_bmin[3] = {bmin_x, bmin_y, bmin_z};
    float void_bmax[3] = {bmax_x, bmax_y, bmax_z};
    bool void_cache_valid = false;
    float leaf_bmin[3] = {bmin_x, bmin_y, bmin_z};
    float leaf_bmax[3] = {bmax_x, bmax_y, bmax_z};
    float cached_leaf_dist = 0.0f;
    int32_t cached_leaf_semantic = 0;
    bool leaf_cache_valid = false;

    const float float_bmin[3] = {bmin_x, bmin_y, bmin_z};
    const float float_bmax[3] = {bmax_x, bmax_y, bmax_z};

    for (int step = 0; step < max_steps; ++step) {
        float px = ox + current_t * dx;
        float py = oy + current_t * dy;
        float pz = oz + current_t * dz;

        if (void_cache_valid && point_in_bounds_strict(px, py, pz, void_bmin, void_bmax)) {
            dist = distance_to_bounds_exit(px, py, pz, dx, dy, dz, void_bmin, void_bmax);
            semantic = 0;
        } else {
            query_dag_stackless(
                dag_memory,
                dag_cache,
                px,
                py,
                pz,
                dx,
                dy,
                dz,
                float_bmin,
                float_bmax,
                resolution,
                cached_ptr,
                cached_bmin,
                cached_bmax,
                cache_valid,
                void_bmin,
                void_bmax,
                void_cache_valid,
                leaf_bmin,
                leaf_bmax,
                cached_leaf_dist,
                cached_leaf_semantic,
                leaf_cache_valid,
                dist,
                semantic);
        }

        if (semantic != 0 && dist < kHitEpsilon) {
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
