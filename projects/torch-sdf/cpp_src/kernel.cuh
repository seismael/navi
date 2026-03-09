#pragma once
#include <cstdint>

namespace toponav {
namespace cuda {

/**
 * @brief Launches the CUDA grid for batched SDF sphere-tracing.
 * @param dag_memory    Read-only pointer to the 64-bit DAG array.
 * @param origins       Pointer to [N, 3] ray origins (float32).
 * @param dirs          Pointer to [N, 3] ray directions (float32).
 * @param out_distances Pointer to [N] output distances (float32).
 * @param out_semantics Pointer to [N] output semantics (int32).
 * @param num_rays      Total number of rays to process (Batch * RaysPerActor).
 * @param max_steps     Maximum sphere-tracing iterations before termination.
 * @param bbox_min      Absolute spatial minimum [x, y, z].
 * @param bbox_max      Absolute spatial maximum [x, y, z].
 * @param resolution    The DAG resolution (e.g., 2048).
 */
void launch_sphere_trace_kernel(
    const uint64_t* dag_memory,
    const float* origins,
    const float* dirs,
    float* out_distances,
    int32_t* out_semantics,
    int num_rays,
    int max_steps,
    const float bbox_min[3],
    const float bbox_max[3],
    int resolution
);

} // namespace cuda
} // namespace toponav
