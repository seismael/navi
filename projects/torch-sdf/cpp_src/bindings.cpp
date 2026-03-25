#include <torch/extension.h>
#include <cmath>
#include <vector>
#include "kernel.cuh"

namespace toponav {

namespace {

void validate_bounds(const std::vector<float>& bbox_min, const std::vector<float>& bbox_max) {
    TORCH_CHECK(bbox_min.size() == 3, "bbox_min must contain exactly 3 floats");
    TORCH_CHECK(bbox_max.size() == 3, "bbox_max must contain exactly 3 floats");
    for (size_t idx = 0; idx < 3; ++idx) {
        TORCH_CHECK(std::isfinite(bbox_min[idx]), "bbox_min must contain only finite floats");
        TORCH_CHECK(std::isfinite(bbox_max[idx]), "bbox_max must contain only finite floats");
        TORCH_CHECK(
            bbox_min[idx] < bbox_max[idx],
            "bbox_min entries must be strictly less than bbox_max entries"
        );
    }
}

void validate_direction_norms(const torch::Tensor& dirs) {
    const torch::Tensor norms = torch::sqrt(torch::sum(dirs * dirs, -1));
    TORCH_CHECK(torch::isfinite(norms).all().item<bool>(), "dirs must contain only finite direction vectors");
    const float max_error = torch::abs(norms - 1.0f).amax().item<float>();
    TORCH_CHECK(
        max_error <= cuda::kDirectionNormEpsilon,
        "dirs must be normalized within tolerance ",
        cuda::kDirectionNormEpsilon,
        "; max error was ",
        max_error
    );
}

// Lightweight finite-only check: ensures no NaN/Inf without a GPU→CPU pipeline drain.
void validate_direction_finite(const torch::Tensor& dirs) {
    TORCH_CHECK(torch::isfinite(dirs).all().item<bool>(), "dirs must contain only finite values");
}

}  // namespace

void cast_rays_forward(
    torch::Tensor dag_tensor,
    torch::Tensor origins,
    torch::Tensor dirs,
    torch::Tensor out_distances,
    torch::Tensor out_semantics,
    int max_steps,
    float max_distance,
    std::vector<float> bbox_min,
    std::vector<float> bbox_max,
    int resolution,
    bool skip_direction_validation) 
{
    // --- Interface Contract Enforcements ---
    TORCH_CHECK(dag_tensor.is_cuda(), "dag_tensor must be a CUDA tensor");
    TORCH_CHECK(origins.is_cuda(), "origins must be a CUDA tensor");
    TORCH_CHECK(dirs.is_cuda(), "dirs must be a CUDA tensor");
    TORCH_CHECK(out_distances.is_cuda(), "out_distances must be a CUDA tensor");
    TORCH_CHECK(out_semantics.is_cuda(), "out_semantics must be a CUDA tensor");
    
    TORCH_CHECK(origins.is_contiguous(), "origins must be contiguous");
    TORCH_CHECK(dirs.is_contiguous(), "dirs must be contiguous");
    TORCH_CHECK(out_distances.is_contiguous(), "out_distances must be contiguous");
    TORCH_CHECK(out_semantics.is_contiguous(), "out_semantics must be contiguous");

    TORCH_CHECK(dag_tensor.scalar_type() == torch::kInt64, "dag_tensor must be int64");
    TORCH_CHECK(origins.scalar_type() == torch::kFloat32, "origins must be float32");
    TORCH_CHECK(dirs.scalar_type() == torch::kFloat32, "dirs must be float32");
    TORCH_CHECK(out_distances.scalar_type() == torch::kFloat32, "out_distances must be float32");
    TORCH_CHECK(out_semantics.scalar_type() == torch::kInt32, "out_semantics must be int32");
    
    TORCH_CHECK(origins.dim() == 3 && origins.size(2) == 3, "origins must be [Batch, Rays, 3]");
    TORCH_CHECK(dirs.dim() == 3 && dirs.size(2) == 3, "dirs must be [Batch, Rays, 3]");
    TORCH_CHECK(dirs.sizes() == origins.sizes(), "dirs must match origins shape");
    TORCH_CHECK(out_distances.dim() == 2, "out_distances must be [Batch, Rays]");
    TORCH_CHECK(out_semantics.dim() == 2, "out_semantics must be [Batch, Rays]");
    TORCH_CHECK(out_distances.size(0) == origins.size(0) && out_distances.size(1) == origins.size(1),
        "out_distances must match the [Batch, Rays] shape of origins");
    TORCH_CHECK(out_semantics.size(0) == origins.size(0) && out_semantics.size(1) == origins.size(1),
        "out_semantics must match the [Batch, Rays] shape of origins");
    TORCH_CHECK(max_steps > 0, "max_steps must be a positive integer");
    TORCH_CHECK(std::isfinite(max_distance) && max_distance > 0.0f, "max_distance must be a finite positive float");
    TORCH_CHECK(resolution > 0, "resolution must be a positive integer");
    validate_bounds(bbox_min, bbox_max);
    if (!skip_direction_validation) {
        validate_direction_norms(dirs);
    }
    
    int num_rays = origins.size(0) * origins.size(1);

    // --- Pointer Extraction ---
    const uint64_t* dag_ptr = reinterpret_cast<const uint64_t*>(dag_tensor.data_ptr<int64_t>());
    const float* origins_ptr = origins.data_ptr<float>();
    const float* dirs_ptr = dirs.data_ptr<float>();
    float* out_dist_ptr = out_distances.data_ptr<float>();
    int32_t* out_sem_ptr = out_semantics.data_ptr<int32_t>();

    // --- Launch CUDA Kernel ---
    {
        pybind11::gil_scoped_release release;
        cuda::launch_sphere_trace_kernel(
            dag_ptr, origins_ptr, dirs_ptr, 
            out_dist_ptr, out_sem_ptr, 
            num_rays, max_steps, max_distance,
            bbox_min.data(), bbox_max.data(), resolution
        );
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cast_rays", &cast_rays_forward, "TopoNav bounded stackless sphere tracing",
        py::arg("dag_tensor"), py::arg("origins"), py::arg("dirs"),
        py::arg("out_distances"), py::arg("out_semantics"),
        py::arg("max_steps"), py::arg("max_distance"),
        py::arg("bbox_min"), py::arg("bbox_max"),
        py::arg("resolution"),
        py::arg("skip_direction_validation") = false);
}

} // namespace toponav
