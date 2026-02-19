"""World generators — pluggable strategy for producing voxel chunks."""

from __future__ import annotations

__all__: list[str] = [
    "AbstractWorldGenerator",
    "ArenaGenerator",
    "CityGenerator",
    "FileGenerator",
    "MazeGenerator",
    "Open3DVoxelGenerator",
    "RoomsGenerator",
]

from navi_section_manager.generators.arena import ArenaGenerator
from navi_section_manager.generators.base import AbstractWorldGenerator
from navi_section_manager.generators.city import CityGenerator
from navi_section_manager.generators.file_loader import FileGenerator
from navi_section_manager.generators.maze import MazeGenerator
from navi_section_manager.generators.open3d_voxel import Open3DVoxelGenerator
from navi_section_manager.generators.rooms import RoomsGenerator
