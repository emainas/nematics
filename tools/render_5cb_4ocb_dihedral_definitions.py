#!/usr/bin/env python3
"""Render the two adjacent 5CB and 4OCB dihedral definitions side by side."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class MoleculeFrame:
    positions: np.ndarray
    elements: tuple[str, ...]
    bonds: tuple[tuple[int, ...], ...]
    box_angstrom: np.ndarray


@dataclass(frozen=True)
class SystemDefinition:
    name: str
    residue: str
    atoms_per_molecule: int
    nitrile_atom: int
    path_atoms: tuple[int, int, int, int, int]
    first_atoms: tuple[int, int, int, int]
    second_atoms: tuple[int, int, int, int]
    first_name: str
    second_name: str
    trajectory: Path
    first_dihedral_table: Path
    second_dihedral_table: Path


@dataclass(frozen=True)
class CharacteristicConformer:
    index: int
    positions: np.ndarray
    angle: float
    pmf_minimum: float


REPO_ROOT = Path(__file__).resolve().parents[1]


FIVE_CB = SystemDefinition(
    name="5CB",
    residue="CB5",
    atoms_per_molecule=38,
    nitrile_atom=4,
    path_atoms=(8, 10, 15, 16, 17),
    first_atoms=(8, 10, 15, 16),
    second_atoms=(10, 15, 16, 17),
    first_name="Ring junction",
    second_name="First-chain torsion",
    trajectory=Path(
        "/hickory/proj/cdpoonlab/users/emainas/"
        "tinker9-5cb-x1000-pt2.7/004e_d-md-304/5cb-x1000.arc"
    ),
    first_dihedral_table=REPO_ROOT
    / "data-processed/5cb/amoeba-pt2.7/004e_d-md-304/dihedral/ring-junction/dihedral.dat",
    second_dihedral_table=REPO_ROOT
    / "data-processed/5cb/amoeba-pt2.7/004e_d-md-304/dihedral/first-chain/dihedral.dat",
)

FOUR_OCB = SystemDefinition(
    name="4OCB",
    residue="OCB",
    atoms_per_molecule=36,
    nitrile_atom=2,
    path_atoms=(10, 11, 28, 15, 16),
    first_atoms=(10, 11, 28, 15),
    second_atoms=(16, 15, 28, 11),
    first_name="Ring junction",
    second_name="Ether / butoxy torsion",
    trajectory=Path(
        "/hickory/proj/cdpoonlab/users/emainas/"
        "tinker9-4ocb-x1000-pt2.2/009a_d-md-304/4ocb-x1000.arc"
    ),
    first_dihedral_table=REPO_ROOT
    / "data-processed/4ocb/amoeba-pt2.2/009a_d-md-304/dihedral/ring-junction/dihedral.dat",
    second_dihedral_table=REPO_ROOT
    / "data-processed/4ocb/amoeba-pt2.2/009a_d-md-304/dihedral/ether/dihedral.dat",
)


def minimum_image(displacement: np.ndarray, box: np.ndarray) -> np.ndarray:
    return displacement - np.rint(displacement / box) * box


def read_first_frame(definition: SystemDefinition) -> MoleculeFrame:
    with definition.trajectory.open(encoding="utf-8") as handle:
        header = handle.readline().split()
        if not header:
            raise ValueError(f"Empty ARC trajectory: {definition.trajectory}")
        natoms = int(header[0])
        if natoms % definition.atoms_per_molecule:
            raise ValueError("ARC atom count is incompatible with atoms_per_molecule")
        box_fields = handle.readline().split()
        box = np.asarray([float(value) for value in box_fields[:3]])
        angles = np.asarray([float(value) for value in box_fields[3:6]])
        if not np.allclose(angles, 90.0, atol=1.0e-6):
            raise ValueError("This renderer currently requires an orthorhombic box")

        molecule_count = natoms // definition.atoms_per_molecule
        positions = np.empty((molecule_count, definition.atoms_per_molecule, 3))
        elements: list[str] = []
        bonds: list[tuple[int, ...]] = []
        for expected_id in range(1, natoms + 1):
            fields = handle.readline().split()
            if len(fields) < 6 or int(fields[0]) != expected_id:
                raise ValueError(f"Invalid ARC atom record {expected_id}")
            molecule = (expected_id - 1) // definition.atoms_per_molecule
            local_atom = (expected_id - 1) % definition.atoms_per_molecule
            positions[molecule, local_atom] = [float(value) for value in fields[2:5]]
            if molecule == 0:
                elements.append(fields[1].title())
                bonds.append(
                    tuple(
                        (int(value) - 1) % definition.atoms_per_molecule + 1
                        for value in fields[6:]
                    )
                )
    return MoleculeFrame(positions, tuple(elements), tuple(bonds), box)


def unwrap_molecule(positions: np.ndarray, box: np.ndarray) -> np.ndarray:
    reference = positions[0]
    return reference + minimum_image(positions - reference, box)


def dihedral_degrees(positions: np.ndarray, atoms: tuple[int, int, int, int]) -> float:
    p0, p1, p2, p3 = (positions[atom - 1] for atom in atoms)
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    angle = math.degrees(math.atan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))
    return angle % 360.0


def select_pmf_conformer(
    frame: MoleculeFrame,
    atoms: tuple[int, int, int, int],
    table_path: Path,
) -> CharacteristicConformer:
    angles = np.loadtxt(table_path, comments="#", usecols=range(1, 1001))
    wrapped = np.mod(angles, 360.0)
    counts, edges = np.histogram(wrapped.ravel(), bins=np.linspace(0.0, 360.0, 361))
    mode_bin = int(np.argmax(counts))
    pmf_minimum = 0.5 * (edges[mode_bin] + edges[mode_bin + 1])

    frame_one = wrapped[0]
    circular_distance = np.abs((frame_one - pmf_minimum + 180.0) % 360.0 - 180.0)
    index = int(np.argmin(circular_distance))
    positions = unwrap_molecule(frame.positions[index], frame.box_angstrom)
    rendered_angle = dihedral_degrees(positions, atoms)
    return CharacteristicConformer(index, positions, rendered_angle, pmf_minimum)


def orient_for_display(
    selected: CharacteristicConformer,
    frame: MoleculeFrame,
    definition: SystemDefinition,
    atoms: tuple[int, int, int, int],
) -> CharacteristicConformer:
    positions = selected.positions.copy()
    heavy_indices = [i for i, element in enumerate(frame.elements) if element != "H"]
    center = np.mean(positions[heavy_indices], axis=0)
    centered = positions - center

    p0, p1, p2, p3 = (centered[atom - 1] for atom in atoms)
    x_axis = p2 - p1
    x_axis /= np.linalg.norm(x_axis)
    normal_one = np.cross(p1 - p0, p2 - p1)
    normal_two = np.cross(p2 - p1, p3 - p2)
    normal_one /= np.linalg.norm(normal_one)
    normal_two /= np.linalg.norm(normal_two)

    # View along the bisector of the two plane normals. Both surfaces therefore
    # remain visible for a perpendicular torsion, while 0/180-degree torsions
    # appear as a single flat pair of triangles.
    if np.dot(normal_one, normal_two) < 0.0:
        normal_two *= -1.0
    z_axis = normal_one + normal_two
    if np.linalg.norm(z_axis) < 1.0e-8:
        z_axis = normal_one
    z_axis -= np.dot(z_axis, x_axis) * x_axis
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    basis = np.column_stack((x_axis, y_axis, z_axis))
    rotated = centered @ basis

    # Keep the nitrile end on the right in both panels.
    nitrile_x = rotated[definition.nitrile_atom - 1, 0]
    chain_x = rotated[definition.path_atoms[-1] - 1, 0]
    if nitrile_x < chain_x:
        rotated[:, 0] *= -1.0
        rotated[:, 2] *= -1.0
    return CharacteristicConformer(
        selected.index, rotated, selected.angle, selected.pmf_minimum
    )


def write_pdb(
    path: Path,
    systems: tuple[
        tuple[SystemDefinition, MoleculeFrame, CharacteristicConformer, CharacteristicConformer],
        tuple[SystemDefinition, MoleculeFrame, CharacteristicConformer, CharacteristicConformer],
    ],
) -> None:
    lines = [
        "REMARK 5CB and 4OCB adjacent-dihedral definition visualization",
        "REMARK 5CB: 8-10-15-16 and 10-15-16-17",
        "REMARK 4OCB: 10-11-28-15 and 16-15-28-11",
        "REMARK Coordinates are real 304 K frame-1 conformers, rigidly rotated and translated",
    ]
    panels = (
        ((systems[0][0], systems[0][1], systems[0][2]), (-12.0, 9.5, 0.0)),
        ((systems[1][0], systems[1][1], systems[1][2]), (12.0, 9.5, 0.0)),
        ((systems[0][0], systems[0][1], systems[0][3]), (-12.0, -9.5, 0.0)),
        ((systems[1][0], systems[1][1], systems[1][3]), (12.0, -9.5, 0.0)),
    )
    serial_offset = 0
    for panel, ((definition, frame, selected), offset) in enumerate(panels, start=1):
        positions = selected.positions + np.asarray(offset)
        chain = "ABCD"[panel - 1]
        for local_atom, (element, xyz) in enumerate(
            zip(frame.elements, positions), start=1
        ):
            serial = serial_offset + local_atom
            atom_name = f"{element}{local_atom:02d}"[-4:]
            lines.append(
                f"HETATM{serial:5d} {atom_name:<4s} {definition.residue:>3s} "
                f"{chain}{panel:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                f"  1.00  0.00          {element:>2s}"
            )
        for local_atom, neighbors in enumerate(frame.bonds, start=1):
            for neighbor in neighbors:
                if neighbor > local_atom:
                    lines.append(
                        f"CONECT{serial_offset + local_atom:5d}{serial_offset + neighbor:5d}"
                    )
        serial_offset += definition.atoms_per_molecule
    lines.extend(("END", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def tcl_point(point: np.ndarray) -> str:
    return "{" + " ".join(f"{value:.8f}" for value in point) + "}"


def write_vmd_scene(
    path: Path,
    pdb_path: Path,
    tga_path: Path,
    systems: tuple[
        tuple[SystemDefinition, MoleculeFrame, CharacteristicConformer, CharacteristicConformer],
        tuple[SystemDefinition, MoleculeFrame, CharacteristicConformer, CharacteristicConformer],
    ],
) -> None:
    commands: list[str] = []
    reports: list[str] = []
    panels = (
        ((systems[0][0], systems[0][1], systems[0][2]), (-12.0, 9.5, 0.0), systems[0][0].first_atoms, "5CB 8-10-15-16"),
        ((systems[1][0], systems[1][1], systems[1][2]), (12.0, 9.5, 0.0), systems[1][0].first_atoms, "4OCB 10-11-28-15"),
        ((systems[0][0], systems[0][1], systems[0][3]), (-12.0, -9.5, 0.0), systems[0][0].second_atoms, "5CB 10-15-16-17"),
        ((systems[1][0], systems[1][1], systems[1][3]), (12.0, -9.5, 0.0), systems[1][0].second_atoms, "4OCB 16-15-28-11"),
    )
    serial_offset = 0
    for (definition, _, selected), offset, atoms, label in panels:
        positions = selected.positions + np.asarray(offset)
        points = [positions[atom - 1] for atom in atoms]
        hinge_center = 0.5 * (points[1] + points[2])
        plane_one_outer = hinge_center + 2.2 * (points[0] - hinge_center)
        plane_two_outer = hinge_center + 2.2 * (points[3] - hinge_center)
        color = "yellow" if len(reports) < 2 else "red"
        commands.extend(
            [
                f"draw_plane $molid {tcl_point(plane_one_outer)} {tcl_point(points[1])} {tcl_point(points[2])} blue",
                f"draw_plane $molid {tcl_point(points[1])} {tcl_point(points[2])} {tcl_point(plane_two_outer)} green",
                f"draw_hinge $molid {tcl_point(points[1])} {tcl_point(points[2])}",
            ]
        )
        commands.extend(
            f"draw_atom $molid {tcl_point(point)} {color} 0.38" for point in points
        )
        serials = " ".join(str(serial_offset + atom) for atom in atoms)
        reports.append(f'report_dihedral $molid "{label}" {{{serials}}}')
        serial_offset += definition.atoms_per_molecule

    script = f"""# Generated two-tier 5CB/4OCB dihedral-definition scene.
set pdb_file {{{pdb_path.resolve()}}}
set output_tga {{{tga_path.resolve()}}}
set molid [mol new $pdb_file type pdb waitfor all]

color Display Background white
color change rgb yellow 1.00 0.78 0.05
color change rgb red 0.92 0.06 0.04
color change rgb blue 0.12 0.50 0.95
color change rgb green 0.05 0.72 0.26

mol delrep 0 $molid
mol representation Licorice 0.17 18 18
mol color ColorID 2
mol selection all
mol material AOShiny
mol addrep $molid

proc draw_plane {{molid a b c color}} {{
    graphics $molid color $color
    graphics $molid material Transparent
    graphics $molid triangle $a $b $c
}}

proc draw_hinge {{molid a b}} {{
    graphics $molid color black
    graphics $molid material AOShiny
    graphics $molid cylinder $a $b radius 0.22 resolution 28 filled yes
}}

proc draw_atom {{molid point color radius}} {{
    graphics $molid color $color
    graphics $molid material AOShiny
    graphics $molid sphere $point radius $radius resolution 28
}}

proc report_dihedral {{molid label serials}} {{
    set indices {{}}
    foreach serial $serials {{
        set selection [atomselect $molid "serial $serial"]
        lappend indices [lindex [$selection get index] 0]
        $selection delete
    }}
    set angle [measure dihed $indices molid $molid frame 0]
    set wrapped [expr {{fmod($angle + 360.0, 360.0)}}]
    puts [format "%s: signed %.6f deg; wrapped %.6f deg" $label $angle $wrapped]
}}

{chr(10).join(commands)}

{chr(10).join(reports)}

display projection Orthographic
display depthcue off
display shadows on
display ambientocclusion on
display aoambient 0.72
display aodirect 0.32
axes location Off
display resetview
scale by 2.45
render TachyonInternal $output_tga
quit
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script, encoding="utf-8")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for directory in (Path("/usr/share/fonts/dejavu"), Path("/usr/share/fonts/truetype/dejavu")):
        candidate = directory / filename
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default(size=size)


def plane_relation(angle: float) -> str:
    wrapped = angle % 360.0
    if min(wrapped, 360.0 - wrapped) <= 20.0:
        return "coplanar near 0 deg"
    if abs(wrapped - 180.0) <= 20.0:
        return "coplanar near 180 deg"
    if min(abs(wrapped - 90.0), abs(wrapped - 270.0)) <= 20.0:
        return "about 90 deg out of plane"
    return "intermediate plane angle"


def plane_separation(angle: float) -> float:
    folded = angle % 180.0
    return min(folded, 180.0 - folded)


def draw_end_on_inset(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    angle: float,
) -> None:
    x, y = center
    radius = 66.0
    draw.ellipse(
        (x - radius - 12, y - radius - 12, x + radius + 12, y + radius + 12),
        fill="white",
        outline=(190, 190, 190),
        width=2,
    )
    separation = plane_separation(angle)
    theta = math.radians(separation)
    draw.line((x - radius, y, x + radius, y), fill=(31, 128, 242), width=13)
    dx, dy = radius * math.cos(theta), radius * math.sin(theta)
    draw.line((x - dx, y + dy, x + dx, y - dy), fill=(13, 184, 66), width=7)
    draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="black")
    draw.text(
        (x, y + radius + 25),
        f"end-on: {separation:.1f} deg",
        fill=(55, 55, 55),
        font=font(17),
        anchor="ma",
    )


def annotate(
    tga_path: Path,
    png_path: Path,
    five_junction: CharacteristicConformer,
    four_junction: CharacteristicConformer,
    five_chain: CharacteristicConformer,
    four_ether: CharacteristicConformer,
) -> None:
    render = Image.open(tga_path).convert("RGB")
    width, height = render.size
    top, bottom = 105, 150
    canvas = Image.new("RGB", (width, height + top + bottom), "white")
    canvas.paste(render, (0, top))
    draw = ImageDraw.Draw(canvas)

    draw.text(
        (width / 2, 34),
        "Dihedral planes at characteristic 304 K PMF minima",
        fill="black",
        font=font(38, bold=True),
        anchor="ma",
    )
    draw.line((width // 2, top + 15, width // 2, top + height - 15), fill=(210, 210, 210), width=3)
    draw.line((35, top + height // 2, width - 35, top + height // 2), fill=(210, 210, 210), width=3)

    panel_labels = (
        (width * 0.25, top + 32, "ma", "Tier 1 - 5CB ring junction", "8-10-15-16", five_junction),
        (width * 0.75, top + 32, "ma", "Tier 1 - 4OCB ring junction", "10-11-28-15", four_junction),
        (width * 0.48, top + height / 2 + 32, "ra", "Tier 2 - 5CB first-chain torsion", "10-15-16-17", five_chain),
        (width * 0.98, top + height / 2 + 32, "ra", "Tier 2 - 4OCB ether / butoxy torsion", "16-15-28-11", four_ether),
    )
    for x, y, anchor, heading, atoms, conformer in panel_labels:
        draw.text(
            (x, y),
            heading,
            fill="black",
            font=font(25, bold=True),
            anchor=anchor,
        )
        draw.text(
            (x, y + 35),
            f"{atoms}   | {conformer.pmf_minimum:.1f} deg - {plane_relation(conformer.pmf_minimum)}",
            fill=(65, 65, 65),
            font=font(20),
            anchor=anchor,
        )

    inset_y_top = top + height / 2 - 115
    inset_y_bottom = top + height - 115
    draw_end_on_inset(draw, (width * 0.44, inset_y_top), five_junction.pmf_minimum)
    draw_end_on_inset(draw, (width * 0.94, inset_y_top), four_junction.pmf_minimum)
    draw_end_on_inset(draw, (width * 0.44, inset_y_bottom), five_chain.pmf_minimum)
    draw_end_on_inset(draw, (width * 0.94, inset_y_bottom), four_ether.pmf_minimum)

    footer_y = top + height + 60
    draw.polygon(((105, footer_y + 20), (150, footer_y - 20), (195, footer_y + 20)), fill=(31, 128, 242))
    draw.text((215, footer_y), "plane 1-2-3", fill="black", font=font(20), anchor="lm")
    draw.polygon(((535, footer_y + 20), (580, footer_y - 20), (625, footer_y + 20)), fill=(13, 184, 66))
    draw.text((645, footer_y), "plane 2-3-4", fill="black", font=font(20), anchor="lm")
    draw.ellipse((1055, footer_y - 22, 1099, footer_y + 22), fill=(255, 199, 13), outline="black", width=2)
    draw.text((1120, footer_y), "junction atoms", fill="black", font=font(20), anchor="lm")
    draw.ellipse((1490, footer_y - 22, 1534, footer_y + 22), fill=(235, 15, 10), outline="black", width=2)
    draw.text((1555, footer_y), "chain / ether atoms", fill="black", font=font(20), anchor="lm")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(png_path, dpi=(300, 300))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data-processed/comparisons/5cb-4ocb-dihedral-definitions"),
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("reports/dihedral_definition_comparison/5cb_4ocb_two_dihedrals_two_tier.png"),
    )
    parser.add_argument("--vmd", default="vmd")
    args = parser.parse_args()

    prepared = []
    for definition in (FIVE_CB, FOUR_OCB):
        frame = read_first_frame(definition)
        first = select_pmf_conformer(
            frame, definition.first_atoms, definition.first_dihedral_table
        )
        second = select_pmf_conformer(
            frame, definition.second_atoms, definition.second_dihedral_table
        )
        first = orient_for_display(first, frame, definition, definition.first_atoms)
        second = orient_for_display(second, frame, definition, definition.second_atoms)
        prepared.append((definition, frame, first, second))
    systems = (prepared[0], prepared[1])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = args.output_dir / "5cb_4ocb_304K_frame1_dihedrals_two_tier.pdb"
    scene_path = args.output_dir / "view_5cb_4ocb_two_dihedrals_two_tier.vmd"
    tga_path = args.output_dir / "5cb_4ocb_two_dihedrals_two_tier_raw.tga"
    write_pdb(pdb_path, systems)
    write_vmd_scene(scene_path, pdb_path, tga_path, systems)

    vmd = shutil.which(args.vmd)
    if vmd is None:
        raise RuntimeError(f"VMD executable not found: {args.vmd}")
    tga_path.unlink(missing_ok=True)
    subprocess.run(
        [vmd, "-dispdev", "text", "-size", "2000", "1300", "-eofexit", "-e", str(scene_path.resolve())],
        check=True,
    )
    if not tga_path.is_file():
        raise RuntimeError("VMD completed without producing the Tachyon render")
    five_junction, five_chain = systems[0][2], systems[0][3]
    four_junction, four_ether = systems[1][2], systems[1][3]
    annotate(
        tga_path,
        args.output_png,
        five_junction,
        four_junction,
        five_chain,
        four_ether,
    )

    for label, conformer in (
        ("5CB junction", five_junction),
        ("5CB first-chain", five_chain),
        ("4OCB junction", four_junction),
        ("4OCB ether", four_ether),
    ):
        print(
            f"{label}: PMF minimum {conformer.pmf_minimum:.3f} deg; "
            f"frame 1 molecule {conformer.index + 1}; shown {conformer.angle:.3f} deg"
        )
    print(f"Wrote {pdb_path}")
    print(f"Wrote {scene_path}")
    print(f"Wrote {args.output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
