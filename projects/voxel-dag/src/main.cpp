#include <iostream>
#include <string>
#include "cxxopts.hpp"
#include "compiler.hpp"

int main(int argc, char** argv) {
    cxxopts::Options options("voxel-dag", "Universal Asset Compiler for EgoSphere S2R Pipeline");

    options.add_options()
        ("i,input", "Input 3D Mesh (.glb, .obj, .ply)", cxxopts::value<std::string>())
        ("o,output", "Output Binary (.gmdag)", cxxopts::value<std::string>())
        ("r,resolution", "Voxel Grid Resolution (e.g., 256, 1024, 2048)", cxxopts::value<uint32_t>()->default_value("2048"))
        ("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help") || !result.count("input") || !result.count("output")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    try {
        std::string input_path = result["input"].as<std::string>();
        std::string output_path = result["output"].as<std::string>();
        uint32_t resolution = result["resolution"].as<uint32_t>();

        voxeldag::Compiler compiler(resolution);
        compiler.compile(input_path, output_path);

    } catch (const std::exception& e) {
        std::cerr << "[FATAL ERROR] " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
