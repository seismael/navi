"""CLI for GMDAG 3D Inspector.

Commands:
    view   — Open interactive 3D viewer
    info   — Print file metadata
    export — Extract mesh and save to file
    corpus — List all .gmdag files in a directory
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from navi_inspector.config import InspectorConfig
from navi_inspector.gmdag_io import GmdagAsset, gmdag_info, load_gmdag

__all__: list[str] = ["app", "inspector_shortcut"]

app = typer.Typer(
    name="navi-inspector",
    help="GMDAG 3D Inspector — interactive viewer and mesh exporter for .gmdag files.",
)
console = Console()


@app.command()
def view(
    gmdag_path: Path = typer.Argument(  # noqa: B008
        ..., help="Path to a .gmdag file", exists=True, readable=True
    ),
    resolution: int = typer.Option(  # noqa: B008
        128, "--resolution", "-r", help="Initial extraction resolution (32–512)"
    ),
    level: float = typer.Option(  # noqa: B008
        -999.0, "--level", "-l", help="Isosurface level (auto-detect if omitted)"
    ),
) -> None:
    """Open an interactive 3D viewer for a .gmdag file."""
    cfg = InspectorConfig()
    resolved_level = None if level == -999.0 else level
    console.print(f"[bold cyan]Loading[/] {gmdag_path.name}\u2026")
    asset = load_gmdag(gmdag_path)
    _print_summary(asset)

    console.print(f"[bold cyan]Extracting mesh[/] at {resolution}\u00b3\u2026")

    from navi_inspector.viewer import launch_viewer

    launch_viewer(asset, initial_resolution=resolution, level=resolved_level, config=cfg)


@app.command()
def info(
    gmdag_path: Path = typer.Argument(  # noqa: B008
        ..., help="Path to a .gmdag file", exists=True, readable=True
    ),
) -> None:
    """Print .gmdag file metadata."""
    meta = gmdag_info(gmdag_path)

    table = Table(title=f"GMDAG Info: {Path(meta['path']).name}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Path", meta["path"])
    table.add_row("Version", str(meta["version"]))
    table.add_row("Resolution", str(meta["resolution"]))
    table.add_row("Voxel Size", f"{meta['voxel_size']:.6f} m")
    table.add_row(
        "BBox Min",
        f"({meta['bbox_min'][0]:.2f}, {meta['bbox_min'][1]:.2f}, {meta['bbox_min'][2]:.2f})",
    )
    table.add_row(
        "BBox Max",
        f"({meta['bbox_max'][0]:.2f}, {meta['bbox_max'][1]:.2f}, {meta['bbox_max'][2]:.2f})",
    )
    table.add_row("Node Count", f"{meta['node_count']:,}")
    table.add_row("File Size", f"{meta['file_size_mb']:.2f} MB ({meta['file_size_bytes']:,} B)")

    extent = meta["bbox_max"][0] - meta["bbox_min"][0]
    table.add_row("World Extent", f"{extent:.2f} m")

    console.print(table)


@app.command()
def export(
    gmdag_path: Path = typer.Argument(  # noqa: B008
        ..., help="Path to a .gmdag file", exists=True, readable=True
    ),
    output_path: Path = typer.Argument(  # noqa: B008
        ..., help="Output mesh file path"
    ),
    resolution: int = typer.Option(  # noqa: B008
        0, "--resolution", "-r",
        help="Extraction resolution (0 = use export default from config)"
    ),
    level: float = typer.Option(  # noqa: B008
        -999.0, "--level", "-l", help="Isosurface level (auto-detect if omitted)"
    ),
    fmt: str = typer.Option(  # noqa: B008
        "ply", "--format", "-f", help="Output format: ply, obj, stl"
    ),
    simplify: float = typer.Option(  # noqa: B008
        1.0, "--simplify", "-s",
        help="Simplification ratio 0.0–1.0 (1.0 = no simplification)"
    ),
) -> None:
    """Extract mesh from .gmdag and export to file."""
    from navi_inspector.dag_extractor import extract_sdf_grid
    from navi_inspector.mesh_builder import build_mesh, export_mesh

    cfg = InspectorConfig()
    target_res = resolution if resolution > 0 else cfg.export_resolution

    console.print(f"[bold cyan]Loading[/] {gmdag_path.name}…")
    asset = load_gmdag(gmdag_path)
    _print_summary(asset)

    console.print(f"[bold cyan]Extracting SDF grid[/] at {target_res}³…")
    sdf = extract_sdf_grid(asset, target_resolution=target_res)
    console.print(f"  Grid shape: {sdf.grid.shape}")

    finite_mask = sdf.grid < 1000.0
    console.print(
        f"  Filled voxels: {int(finite_mask.sum()):,} / {sdf.grid.size:,} "
        f"({100.0 * finite_mask.sum() / sdf.grid.size:.1f}%)"
    )

    resolved_level = None if level == -999.0 else level
    console.print(f"[bold cyan]Running marching cubes[/] level={resolved_level}\u2026")
    mesh = build_mesh(sdf, level=resolved_level, simplify_ratio=simplify)
    console.print(
        f"  Mesh: {mesh.n_points:,} vertices, {mesh.n_faces_strict:,} faces"
    )

    saved = export_mesh(mesh, output_path, fmt=fmt)
    console.print(f"[bold green]Saved[/] → {saved}")


@app.command()
def corpus(
    corpus_dir: Path = typer.Argument(  # noqa: B008
        Path("artifacts/gmdag/corpus"),
        help="Directory to scan for .gmdag files",
    ),
) -> None:
    """List all .gmdag files in a directory with metadata summary."""
    if not corpus_dir.exists():
        console.print(f"[red]Directory not found:[/] {corpus_dir}")
        raise typer.Exit(code=1)

    files = sorted(corpus_dir.rglob("*.gmdag"))
    if not files:
        console.print(f"[yellow]No .gmdag files found in[/] {corpus_dir}")
        return

    table = Table(title=f"GMDAG Corpus — {corpus_dir}")
    table.add_column("#", style="dim")
    table.add_column("File", style="cyan")
    table.add_column("Res", justify="right")
    table.add_column("Nodes", justify="right")
    table.add_column("Size (MB)", justify="right")

    total_size = 0
    for idx, fp in enumerate(files, 1):
        try:
            meta = gmdag_info(fp)
            size_mb = meta["file_size_mb"]
            total_size += meta["file_size_bytes"]
            table.add_row(
                str(idx),
                str(fp.relative_to(corpus_dir)),
                str(meta["resolution"]),
                f"{meta['node_count']:,}",
                f"{size_mb:.2f}",
            )
        except (RuntimeError, FileNotFoundError) as exc:
            table.add_row(str(idx), str(fp.relative_to(corpus_dir)), "ERR", "", str(exc))

    console.print(table)
    console.print(
        f"\n[bold]{len(files)}[/] files, "
        f"[bold]{total_size / (1024 * 1024):.1f}[/] MB total"
    )


def _print_summary(asset: GmdagAsset) -> None:
    """Print a brief asset summary for CLI feedback."""
    console.print(
        f"  Resolution: {asset.resolution}, "
        f"Voxel: {asset.voxel_size:.4f} m, "
        f"Nodes: {len(asset.nodes):,}"
    )


def inspector_shortcut() -> None:
    """Shortcut entry point for 'uv run inspector'."""
    app()
