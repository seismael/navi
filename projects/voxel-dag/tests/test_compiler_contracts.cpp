#define private public
#include "compiler.hpp"
#undef private

#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_canonical_node_hash_is_deterministic() {
    const std::vector<uint32_t> payload{1U, 2U, 3U, 4U};
    const uint64_t first = voxeldag::canonical_node_hash(payload, 0);
    const uint64_t second = voxeldag::canonical_node_hash(payload, 0);
    const uint64_t alternate_seed = voxeldag::canonical_node_hash(payload, 7);

    expect(first == second, "canonical_node_hash must be deterministic for identical payloads");
    expect(first != alternate_seed, "canonical_node_hash must honor the configured seed");
}

void test_deduplicate_child_layouts_uses_structural_fallback() {
    const std::vector<std::vector<uint32_t>> layouts{
        {1U, 2U, 3U, 4U},
        {9U, 8U, 7U, 6U},
        {1U, 2U, 3U, 4U},
    };

    const voxeldag::NodeHashFn constant_hash =
        [](const std::vector<uint32_t>&, uint64_t) -> uint64_t { return 1ULL; };

    const auto [unique_layouts, remap] =
        voxeldag::deduplicate_child_layouts(layouts, 0, constant_hash);

    expect(unique_layouts.size() == 2U, "structural fallback must prevent false merges on hash collision");
    expect(remap.size() == layouts.size(), "dedup remap must preserve input cardinality");
    expect(remap[0] == 0U && remap[1] == 1U && remap[2] == 0U,
           "dedup remap must reuse only structurally identical layouts");
}

void test_compress_to_dag_recursively_deduplicates_identical_subtrees() {
    voxeldag::Compiler compiler(4U);
    const std::vector<float> dense_grid(4U * 4U * 4U, 1.0f);
    const std::vector<uint64_t> dag = compiler.compressToDAG(dense_grid);

    expect(!dag.empty(), "compressToDAG must emit at least the root node");

    const uint64_t root = dag[0];
    expect((root >> 63) == 0U, "root copy at dag[0] must be an internal node for a 4^3 grid");

    const uint8_t root_mask = static_cast<uint8_t>((root >> 55) & 0xFFULL);
    expect(root_mask == 0xFFU, "fully populated dense grids must expose all eight root children");

    const uint32_t root_child_table = static_cast<uint32_t>(root & 0xFFFFFFFFULL);
    std::set<uint64_t> unique_root_children;
    for (uint32_t idx = 0; idx < 8U; ++idx) {
        unique_root_children.insert(dag[root_child_table + idx]);
    }
    expect(unique_root_children.size() == 1U, "recursive dedup must collapse identical root subtrees to one child pointer");

    const uint32_t child_ptr = static_cast<uint32_t>(*unique_root_children.begin());
    const uint64_t child_node = dag[child_ptr];
    expect((child_node >> 63) == 0U, "deduplicated root child must remain an internal node");

    const uint32_t leaf_child_table = static_cast<uint32_t>(child_node & 0xFFFFFFFFULL);
    std::set<uint64_t> unique_leaf_children;
    for (uint32_t idx = 0; idx < 8U; ++idx) {
        unique_leaf_children.insert(dag[leaf_child_table + idx]);
    }
    expect(unique_leaf_children.size() == 1U, "recursive dedup must also collapse identical leaf-level children");
}

} // namespace

int main() {
    try {
        test_canonical_node_hash_is_deterministic();
        test_deduplicate_child_layouts_uses_structural_fallback();
        test_compress_to_dag_recursively_deduplicates_identical_subtrees();
    } catch (const std::exception& ex) {
        std::cerr << "voxel-dag contract test failure: " << ex.what() << '\n';
        return 1;
    }

    std::cout << "voxel-dag contract tests passed\n";
    return 0;
}