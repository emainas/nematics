#!/usr/bin/env python3
"""Render two real 5CB molecules with the repository's vector/center definitions."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ATOMS_PER_MOLECULE = 38
MOLECULAR_CENTER_ATOMS = (2, 6, 14)
MOLECULAR_AXIS_ATOMS = (10, 9)
POLAR_CENTER_ATOMS = (3, 4)
POLAR_VECTOR_ATOMS = (3, 4)


@dataclass(frozen=True)
class ArcFrame:
    positions: np.ndarray
    elements: tuple[str, ...]
    bonds: tuple[tuple[int, ...], ...]
    box_angstrom: np.ndarray


@dataclass(frozen=True)
class PairSelection:
    first: int
    second: int
    distance_angstrom: float
    molecular_axis_dot: float
    polar_vector_dot: float


def minimum_image(displacement: np.ndarray, box: np.ndarray) -> np.ndarray:
    return displacement - np.rint(displacement / box) * box


def unit(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1.0e-12:
        raise ValueError("Cannot normalize a zero-length vector")
    return vector / length


def read_first_arc_frame(path: Path) -> ArcFrame:
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().split()
        if not header:
            raise ValueError(f"Empty ARC trajectory: {path}")
        natoms = int(header[0])
        if natoms % ATOMS_PER_MOLECULE:
            raise ValueError(f"ARC atom count is not divisible by {ATOMS_PER_MOLECULE}: {path}")
        box_fields = handle.readline().split()
        if len(box_fields) < 6:
            raise ValueError(f"Invalid ARC box line: {path}")
        box = np.asarray([float(value) for value in box_fields[:3]], dtype=float)
        angles = np.asarray([float(value) for value in box_fields[3:6]], dtype=float)
        if not np.allclose(angles, 90.0, atol=1.0e-6):
            raise ValueError("This visualization currently requires an orthorhombic box")

        positions = np.empty((natoms // ATOMS_PER_MOLECULE, ATOMS_PER_MOLECULE, 3))
        elements: list[str] = []
        bonds: list[tuple[int, ...]] = []
        for expected_id in range(1, natoms + 1):
            fields = handle.readline().split()
            if len(fields) < 6 or int(fields[0]) != expected_id:
                raise ValueError(f"Invalid ARC atom record {expected_id} in {path}")
            molecule = (expected_id - 1) // ATOMS_PER_MOLECULE
            local_atom = (expected_id - 1) % ATOMS_PER_MOLECULE
            positions[molecule, local_atom] = [float(value) for value in fields[2:5]]
            if molecule == 0:
                elements.append(fields[1].title())
                local_neighbors = tuple(
                    (int(value) - 1) % ATOMS_PER_MOLECULE + 1 for value in fields[6:]
                )
                bonds.append(local_neighbors)
    return ArcFrame(
        positions=positions,
        elements=tuple(elements),
        bonds=tuple(bonds),
        box_angstrom=box,
    )


def unwrap_molecule(positions: np.ndarray, box: np.ndarray) -> np.ndarray:
    reference = positions[MOLECULAR_CENTER_ATOMS[0] - 1]
    return reference + minimum_image(positions - reference, box)


def molecular_center(positions: np.ndarray) -> np.ndarray:
    indices = np.asarray(MOLECULAR_CENTER_ATOMS) - 1
    return np.mean(positions[indices], axis=0)


def dipole_center(positions: np.ndarray) -> np.ndarray:
    indices = np.asarray(POLAR_CENTER_ATOMS) - 1
    return np.mean(positions[indices], axis=0)


def molecular_axis(positions: np.ndarray) -> np.ndarray:
    start, end = (atom - 1 for atom in MOLECULAR_AXIS_ATOMS)
    return unit(positions[end] - positions[start])


def polar_vector(positions: np.ndarray) -> np.ndarray:
    start, end = (atom - 1 for atom in POLAR_VECTOR_ATOMS)
    return unit(positions[end] - positions[start])


def select_close_pair(frame: ArcFrame) -> PairSelection:
    molecules = np.asarray(
        [unwrap_molecule(positions, frame.box_angstrom) for positions in frame.positions]
    )
    centers = np.asarray([molecular_center(positions) for positions in molecules])
    molecular_axes = np.asarray([molecular_axis(positions) for positions in molecules])
    polar_vectors = np.asarray([polar_vector(positions) for positions in molecules])

    best: tuple[float, PairSelection] | None = None
    for first in range(len(molecules) - 1):
        displacements = minimum_image(centers[first + 1 :] - centers[first], frame.box_angstrom)
        distances = np.linalg.norm(displacements, axis=1)
        candidates = np.where((distances >= 4.5) & (distances <= 8.0))[0]
        for relative in candidates:
            second = first + 1 + int(relative)
            axis_dot = float(np.dot(molecular_axes[first], molecular_axes[second]))
            polar_dot = float(np.dot(polar_vectors[first], polar_vectors[second]))
            # Prefer a close, nematically aligned, antipolar pair so both vector
            # definitions are visually and scientifically informative.
            if abs(axis_dot) < 0.70 or polar_dot > -0.35:
                continue
            distance = float(distances[relative])
            score = distance + 1.5 * (polar_dot + 1.0) + 0.5 * (1.0 - abs(axis_dot))
            selection = PairSelection(first, second, distance, axis_dot, polar_dot)
            if best is None or score < best[0]:
                best = (score, selection)
    if best is None:
        raise RuntimeError("Could not find a suitable close 5CB pair in the selected frame")
    return best[1]


def prepare_pair(frame: ArcFrame, pair: PairSelection) -> tuple[np.ndarray, np.ndarray]:
    first = unwrap_molecule(frame.positions[pair.first], frame.box_angstrom)
    second = unwrap_molecule(frame.positions[pair.second], frame.box_angstrom)
    first_center = molecular_center(first)
    second_center = molecular_center(second)
    nearest_displacement = minimum_image(second_center - first_center, frame.box_angstrom)
    second += first_center + nearest_displacement - second_center
    combined = np.vstack((first, second))

    # Rotate into a stable PCA view so the long molecular dimensions occupy
    # the image plane while retaining the exact geometry and vector directions.
    centered = combined - np.mean(combined, axis=0)
    covariance = centered.T @ centered
    _, eigenvectors = np.linalg.eigh(covariance)
    basis = eigenvectors[:, ::-1]
    if np.linalg.det(basis) < 0.0:
        basis[:, 2] *= -1.0
    rotated = centered @ basis
    if np.mean(rotated[:ATOMS_PER_MOLECULE, 0]) > np.mean(rotated[ATOMS_PER_MOLECULE:, 0]):
        rotated[:, 0] *= -1.0
        rotated[:, 2] *= -1.0
    return rotated[:ATOMS_PER_MOLECULE], rotated[ATOMS_PER_MOLECULE:]


def write_pdb(path: Path, molecules: tuple[np.ndarray, np.ndarray], frame: ArcFrame) -> None:
    lines = [
        "REMARK 5CB two-molecule vector-definition visualization",
        "REMARK molecular axis: local atom 10 -> 9",
        "REMARK polar vector: local atom 3 -> 4",
        "REMARK molecular center: average local atoms 2, 6, 14",
        "REMARK dipole center: midpoint local atoms 3, 4",
    ]
    for molecule_index, positions in enumerate(molecules, start=1):
        chain = "A" if molecule_index == 1 else "B"
        offset = (molecule_index - 1) * ATOMS_PER_MOLECULE
        for local_atom, (element, xyz) in enumerate(zip(frame.elements, positions), start=1):
            serial = offset + local_atom
            atom_name = f"{element}{local_atom:02d}"[-4:]
            lines.append(
                f"HETATM{serial:5d} {atom_name:<4s} CB5 {chain}{molecule_index:4d}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {element:>2s}"
            )
    for molecule_index in (1, 2):
        offset = (molecule_index - 1) * ATOMS_PER_MOLECULE
        for local_atom, neighbors in enumerate(frame.bonds, start=1):
            for neighbor in neighbors:
                if neighbor > local_atom:
                    lines.append(f"CONECT{offset + local_atom:5d}{offset + neighbor:5d}")
    lines.extend(("END", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def tcl_point(point: np.ndarray) -> str:
    return "{" + " ".join(f"{value:.8f}" for value in point) + "}"


def write_vmd_scene(
    path: Path,
    pdb_path: Path,
    tga_path: Path,
    molecules: tuple[np.ndarray, np.ndarray],
) -> None:
    graphics_commands: list[str] = []
    for positions in molecules:
        mol_center = molecular_center(positions)
        cn_center = dipole_center(positions)
        mol_axis = molecular_axis(positions)
        cn_axis = polar_vector(positions)
        mol_start = mol_center - 4.5 * mol_axis
        mol_end = mol_center + 4.5 * mol_axis
        cn_start = cn_center - 1.6 * cn_axis
        cn_end = cn_center + 2.8 * cn_axis
        graphics_commands.extend(
            [
                f"draw_arrow $molid {tcl_point(mol_start)} {tcl_point(mol_end)} orange 0.28 0.62",
                f"draw_arrow $molid {tcl_point(cn_start)} {tcl_point(cn_end)} red 0.25 0.55",
                f"draw_point $molid {tcl_point(mol_center)} green 0.48",
                f"draw_point $molid {tcl_point(cn_center)} magenta 0.43",
            ]
        )
    script = f"""# Generated 5CB vector-definition scene.
set pdb_file {{{pdb_path.resolve()}}}
set output_tga {{{tga_path.resolve()}}}
set molid [mol new $pdb_file type pdb waitfor all]

color Display Background white
color change rgb orange 1.00 0.42 0.00
color change rgb red 0.92 0.05 0.05
color change rgb green 0.05 0.72 0.20
color change rgb magenta 0.72 0.10 0.82

mol delrep 0 $molid
mol representation Licorice 0.18 18 18
mol color ColorID 2
mol selection all
mol material AOShiny
mol addrep $molid

proc draw_arrow {{molid start end color shaft_radius head_radius}} {{
    set delta [vecsub $end $start]
    set neck [vecadd $start [vecscale 0.78 $delta]]
    graphics $molid material AOShiny
    graphics $molid color $color
    graphics $molid cylinder $start $neck radius $shaft_radius resolution 28 filled yes
    graphics $molid cone $neck $end radius $head_radius resolution 28
}}

proc draw_point {{molid point color radius}} {{
    graphics $molid material AOShiny
    graphics $molid color $color
    graphics $molid sphere $point radius $radius resolution 28
}}

{chr(10).join(graphics_commands)}

display projection Orthographic
display depthcue off
display shadows on
display ambientocclusion on
display aoambient 0.70
display aodirect 0.35
axes location Off
display resetview
scale by 1.55
render TachyonInternal $output_tga
quit
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script, encoding="utf-8")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = (
        Path("/usr/share/fonts/dejavu") / name,
        Path("/usr/share/fonts/truetype/dejavu") / name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default(size=size)


def annotate_render(tga_path: Path, png_path: Path, pair: PairSelection) -> None:
    molecular = Image.open(tga_path).convert("RGB")
    width, height = molecular.size
    top = 110
    bottom = 245
    canvas = Image.new("RGB", (width, height + top + bottom), "white")
    canvas.paste(molecular, (0, top))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (width / 2, 34),
        "5CB molecular and polar definitions - 304 K, frame 1",
        fill="black",
        font=font(36, bold=True),
        anchor="ma",
    )
    draw.text(
        (width / 2, 78),
        f"Actual close pair: molecules {pair.first + 1} and {pair.second + 1}; "
        f"center separation {pair.distance_angstrom / 10.0:.3f} nm",
        fill=(55, 55, 55),
        font=font(23),
        anchor="ma",
    )

    legend_y = top + height + 45
    x_positions = (95, width // 2 + 20)
    entries = (
        (x_positions[0], legend_y, "arrow", (255, 107, 0), "Molecular axis: unit(r9 - r10), direction 10 -> 9"),
        (x_positions[1], legend_y, "arrow", (235, 13, 13), "Polar CN vector: unit(r4 - r3), direction C3 -> N4"),
        (x_positions[0], legend_y + 78, "sphere", (13, 184, 51), "Molecular center: (r2 + r6 + r14) / 3"),
        (x_positions[1], legend_y + 78, "sphere", (184, 26, 209), "CN dipole center: (r3 + r4) / 2"),
    )
    for x, y, kind, color, label in entries:
        if kind == "arrow":
            draw.line((x, y, x + 72, y), fill=color, width=12)
            draw.polygon(((x + 72, y - 18), (x + 108, y), (x + 72, y + 18)), fill=color)
        else:
            draw.ellipse((x + 20, y - 22, x + 64, y + 22), fill=color, outline="black", width=2)
        draw.text((x + 125, y), label, fill="black", font=font(22), anchor="lm")

    draw.text(
        (width / 2, top + height + bottom - 28),
        f"u_mol(1) dot u_mol(2) = {pair.molecular_axis_dot:+.3f}    "
        f"u_CN(1) dot u_CN(2) = {pair.polar_vector_dot:+.3f}",
        fill=(55, 55, 55),
        font=font(21),
        anchor="ms",
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(png_path, dpi=(300, 300))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path(
            "/hickory/proj/cdpoonlab/users/emainas/"
            "tinker9-5cb-x1000-pt2.7/004e_d-md-304/5cb-x1000.arc"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data-processed/5cb/amoeba-pt2.7/004e_d-md-304/"
            "vector-definition-visualization"
        ),
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("reports/5cb_vector_definitions/two_molecule_vector_definitions.png"),
    )
    parser.add_argument("--vmd", default="vmd", help="VMD executable available on PATH")
    args = parser.parse_args()

    frame = read_first_arc_frame(args.trajectory)
    pair = select_close_pair(frame)
    molecules = prepare_pair(frame, pair)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = args.output_dir / "5cb_close_pair_304K_frame1.pdb"
    scene_path = args.output_dir / "view_5cb_vector_definitions.vmd"
    tga_path = args.output_dir / "5cb_vector_definitions_raw.tga"
    write_pdb(pdb_path, molecules, frame)
    write_vmd_scene(scene_path, pdb_path, tga_path, molecules)

    vmd = shutil.which(args.vmd)
    if vmd is None:
        raise RuntimeError(f"VMD executable not found: {args.vmd}")
    tga_path.unlink(missing_ok=True)
    subprocess.run(
        [
            vmd,
            "-dispdev",
            "text",
            "-size",
            "1800",
            "1200",
            "-eofexit",
            "-e",
            str(scene_path.resolve()),
        ],
        check=True,
    )
    if not tga_path.is_file():
        raise RuntimeError("VMD completed without producing the Tachyon render")
    annotate_render(tga_path, args.output_png, pair)
    print(f"Selected molecules {pair.first + 1} and {pair.second + 1}")
    print(f"Center separation: {pair.distance_angstrom / 10.0:.6f} nm")
    print(f"Molecular-axis dot product: {pair.molecular_axis_dot:.6f}")
    print(f"Polar-vector dot product: {pair.polar_vector_dot:.6f}")
    print(f"Wrote {pdb_path}")
    print(f"Wrote {scene_path}")
    print(f"Wrote {args.output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
