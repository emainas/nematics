#!/usr/bin/env python3
"""Calculate and visualize CB7CB induced chiral asymmetry with cpptraj.

The local atom numbering below is taken from the 65-atom Tinker CB7CB
topology.  For each requested molecule/residue, cpptraj constructs the raw
vectors

    L    = N(7) - C(18)
    L'   = N(8) - C(4)
    dLL' = center(biphenyl 2) - center(biphenyl 1)

and evaluates c = (unit(L) x unit(L')) dot unit(dLL').  The two biphenyl
centers use their 12 aromatic carbon atoms.  Since all atoms in each center
mask are carbon, cpptraj's mass-weighted center is also exactly the geometric
center.

The generated cpptraj input is suitable for a single PDB or a multi-frame
trajectory.  A separate one-frame visualization is emitted for each selected
residue.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


ATOMS_PER_MOLECULE = 65

# Local (one-based) atom numbers in one CB7CB molecule.
L_BASE = 18
L_TIP = 7
LP_BASE = 4
LP_TIP = 8

BIPHENYL_1_CARBONS = (1, 2, 9, 10, 13, 14, 17, 18, 21, 22, 25, 26)
BIPHENYL_2_CARBONS = (4, 5, 11, 12, 15, 16, 19, 20, 23, 24, 27, 28)


def parse_residue_spec(spec: str) -> list[int]:
    """Parse comma-separated residue numbers and inclusive ranges."""
    residues: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, stop_text = item.split("-", 1)
            start = int(start_text)
            stop = int(stop_text)
            if stop < start:
                raise ValueError(f"Invalid descending residue range: {item}")
            residues.update(range(start, stop + 1))
        else:
            residues.add(int(item))
    if not residues or min(residues) < 1:
        raise ValueError("Residues must be positive integers")
    return sorted(residues)


def global_atom(residue: int, local_atom: int) -> int:
    return (residue - 1) * ATOMS_PER_MOLECULE + local_atom


def atom_mask(residue: int, local_atoms: tuple[int, ...]) -> str:
    return "@" + ",".join(str(global_atom(residue, atom)) for atom in local_atoms)


def cpptraj_quote(path: Path) -> str:
    # cpptraj accepts double-quoted paths. Embedded quotes are not supported.
    text = str(path)
    if '"' in text:
        raise ValueError(f"A double quote is not supported in cpptraj paths: {path}")
    return f'"{text}"'


def dataset_prefix(residue: int) -> str:
    return f"r{residue:04d}"


def write_analysis_input(
    path: Path,
    topology: Path,
    trajectory: Path,
    residues: list[int],
    out_dir: Path,
    start: int,
    stop: int | None,
    stride: int,
) -> None:
    trajin = f"trajin {cpptraj_quote(trajectory)} {start}"
    if stop is not None:
        trajin += f" {stop} {stride}"
    elif stride != 1:
        raise ValueError("--stride requires --stop")

    lines = [f"parm {cpptraj_quote(topology)}", trajin]
    for residue in residues:
        prefix = dataset_prefix(residue)
        raw_path = out_dir / f"residue_{residue:04d}_vectors_raw.dat"
        lines.extend(
            [
                (
                    f"vector {prefix}_L mask @{global_atom(residue, L_BASE)} "
                    f"@{global_atom(residue, L_TIP)} out {cpptraj_quote(raw_path)} magnitude"
                ),
                (
                    f"vector {prefix}_Lp mask @{global_atom(residue, LP_BASE)} "
                    f"@{global_atom(residue, LP_TIP)} out {cpptraj_quote(raw_path)} magnitude"
                ),
                (
                    f"vector {prefix}_dLL mask {atom_mask(residue, BIPHENYL_1_CARBONS)} "
                    f"{atom_mask(residue, BIPHENYL_2_CARBONS)} "
                    f"out {cpptraj_quote(raw_path)} magnitude"
                ),
            ]
        )

    lines.append("run")

    # 'norm' modifies the input datasets. The raw-vector files above are
    # written during 'run', before any normalization occurs.
    for residue in residues:
        prefix = dataset_prefix(residue)
        math_path = out_dir / f"residue_{residue:04d}_chirality.dat"
        lines.extend(
            [
                (
                    f"vectormath vec1 {prefix}_L vec2 {prefix}_Lp crossproduct norm "
                    f"name {prefix}_LcrossLp out {cpptraj_quote(math_path)}"
                ),
                (
                    f"vectormath vec1 {prefix}_dLL vec2 {prefix}_dLL dotproduct norm "
                    f"name {prefix}_dLL_normcheck out {cpptraj_quote(math_path)}"
                ),
                (
                    f"vectormath vec1 {prefix}_LcrossLp vec2 {prefix}_dLL dotproduct "
                    f"name {prefix}_c out {cpptraj_quote(math_path)}"
                ),
            ]
        )
    lines.append("runanalysis")
    for residue in residues:
        prefix = dataset_prefix(residue)
        scalar_path = out_dir / f"residue_{residue:04d}_c.dat"
        lines.append(
            f"writedata {cpptraj_quote(scalar_path)} {prefix}_c prec 16.8"
        )
    lines.extend(["quit", ""])
    path.write_text("\n".join(lines))


def write_visualization_input(
    path: Path,
    topology: Path,
    trajectory: Path,
    residue: int,
    frame: int,
    out_dir: Path,
) -> tuple[Path, Path, Path]:
    prefix = dataset_prefix(residue)
    molecule_pdb = out_dir / f"residue_{residue:04d}_molecule.pdb"
    vectors_pdb = out_dir / f"residue_{residue:04d}_vectors_cpptraj.pdb"
    vectors_parm = out_dir / f"residue_{residue:04d}_vectors_cpptraj.parm7"
    viz_math = out_dir / f"residue_{residue:04d}_visualized_chirality.dat"
    viz_scalar = out_dir / f"residue_{residue:04d}_visualized_c.dat"
    lines = [
        f"parm {cpptraj_quote(topology)}",
        f"trajin {cpptraj_quote(trajectory)} {frame} {frame}",
        (
            f"vector {prefix}_L mask @{global_atom(residue, L_BASE)} "
            f"@{global_atom(residue, L_TIP)}"
        ),
        (
            f"vector {prefix}_Lp mask @{global_atom(residue, LP_BASE)} "
            f"@{global_atom(residue, LP_TIP)}"
        ),
        (
            f"vector {prefix}_dLL mask {atom_mask(residue, BIPHENYL_1_CARBONS)} "
            f"{atom_mask(residue, BIPHENYL_2_CARBONS)}"
        ),
        f"strip !:{residue}",
        f"trajout {cpptraj_quote(molecule_pdb)} pdb conectmode all",
        "run",
        (
            f"writedata {cpptraj_quote(vectors_pdb)} {prefix}_L {prefix}_Lp {prefix}_dLL "
            f"vectraj trajfmt pdb parmout {cpptraj_quote(vectors_parm)}"
        ),
        (
            f"vectormath vec1 {prefix}_L vec2 {prefix}_Lp crossproduct norm "
            f"name {prefix}_LcrossLp out {cpptraj_quote(viz_math)}"
        ),
        (
            f"vectormath vec1 {prefix}_dLL vec2 {prefix}_dLL dotproduct norm "
            f"name {prefix}_dLL_normcheck out {cpptraj_quote(viz_math)}"
        ),
        (
            f"vectormath vec1 {prefix}_LcrossLp vec2 {prefix}_dLL dotproduct "
            f"name {prefix}_c out {cpptraj_quote(viz_math)}"
        ),
        "runanalysis",
        f"writedata {cpptraj_quote(viz_scalar)} {prefix}_c prec 16.8",
        "quit",
        "",
    ]
    path.write_text("\n".join(lines))
    return molecule_pdb, vectors_pdb, viz_scalar


def run_cpptraj(executable: str, input_path: Path, log_path: Path) -> None:
    with log_path.open("w") as log:
        subprocess.run(
            [executable, "-i", str(input_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )


def parse_vector_endpoints(path: Path) -> list[tuple[float, float, float]]:
    endpoints: list[tuple[float, float, float]] = []
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            endpoints.append(
                (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            )
    if len(endpoints) != 6:
        raise ValueError(f"Expected six cpptraj vector endpoints in {path}, found {len(endpoints)}")
    return endpoints


def parse_chirality(path: Path) -> float:
    for line in path.read_text().splitlines():
        if not line.startswith("#") and line.strip():
            fields = line.split()
            # The precise scalar file contains frame index followed by c.
            return float(fields[-1])
    raise ValueError(f"No chirality data found in {path}")


def pseudo_atom_line(
    serial: int,
    name: str,
    resname: str,
    resid: int,
    xyz: tuple[float, float, float],
    element: str,
    beta: float,
) -> str:
    x, y, z = xyz
    return (
        f"HETATM{serial:5d} {name:>4s} {resname:>3s} V{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{beta:6.2f}          {element:>2s}"
    )


def write_combined_visualization(
    molecule_pdb: Path,
    vectors_pdb: Path,
    out_path: Path,
    chirality: float,
) -> None:
    molecule_lines = molecule_pdb.read_text().splitlines()
    headers = [line for line in molecule_lines if line.startswith(("TITLE ", "CRYST1"))]
    atoms = [line for line in molecule_lines if line.startswith(("ATOM  ", "HETATM"))]
    conect = [line for line in molecule_lines if line.startswith("CONECT")]
    endpoints = parse_vector_endpoints(vectors_pdb)

    if not atoms:
        raise ValueError(f"No molecular atoms found in {molecule_pdb}")
    first_vector_serial = len(atoms) + 1
    vector_specs = [
        ("LORG", "LV1", 901, "O", 10.0),
        ("LTIP", "LV1", 901, "O", 10.0),
        ("PORG", "LV2", 902, "N", 20.0),
        ("PTIP", "LV2", 902, "N", 20.0),
        ("DORG", "DLL", 903, "S", 30.0),
        ("DTIP", "DLL", 903, "S", 30.0),
    ]
    pseudo_atoms = []
    for i, (name, resname, resid, element, beta) in enumerate(vector_specs):
        pseudo_atoms.append(
            pseudo_atom_line(
                first_vector_serial + i,
                name,
                resname,
                resid,
                endpoints[i],
                element,
                beta,
            )
        )

    vector_bonds = [
        f"CONECT{first_vector_serial:5d}{first_vector_serial + 1:5d}",
        f"CONECT{first_vector_serial + 2:5d}{first_vector_serial + 3:5d}",
        f"CONECT{first_vector_serial + 4:5d}{first_vector_serial + 5:5d}",
    ]
    output = [
        "TITLE     CB7CB WITH RAW INDUCED-CHIRALITY VECTORS",
        f"REMARK 650 C = (UNIT(L) CROSS UNIT(LP)) DOT UNIT(DLL) = {chirality:.8f}",
        "REMARK 651 LV1=L; LV2=L-PRIME; DLL=CENTER-TO-CENTER VECTOR",
    ]
    output.extend(line for line in headers if line.startswith("CRYST1"))
    output.extend(atoms)
    output.extend(pseudo_atoms)
    output.extend(conect)
    output.extend(vector_bonds)
    output.extend(["END", ""])
    out_path.write_text("\n".join(output))


def write_vmd_script(path: Path, visualization_pdb: Path, chirality: float) -> None:
    # The pseudo-atoms are retained in the PDB so it is independently useful.
    # This VMD helper renders the same endpoint pairs as clear arrows.
    text = f'''# VMD helper for raw CB7CB induced-chirality vectors.
set pdb_file [file join [file dirname [info script]] "{visualization_pdb.name}"]
set molid [mol new $pdb_file type pdb waitfor all]

mol delrep 0 $molid
mol representation Licorice 0.16 12 12
mol color Element
mol selection "resname CB7"
mol material Opaque
mol addrep $molid

proc atom_xyz {{molid selection}} {{
    set sel [atomselect $molid $selection]
    set xyz [lindex [$sel get {{x y z}}] 0]
    $sel delete
    return $xyz
}}

proc draw_arrow {{molid start end color}} {{
    set delta [vecsub $end $start]
    set neck [vecadd $start [vecscale 0.82 $delta]]
    graphics $molid color $color
    graphics $molid cylinder $start $neck radius 0.20 resolution 20 filled yes
    graphics $molid cone $neck $end radius 0.42 resolution 20
}}

proc draw_angle_arc {{molid origin from_vector to_vector radius color c_value}} {{
    set u [vecnorm $from_vector]
    set v [vecnorm $to_vector]
    set cosine [vecdot $u $v]
    if {{$cosine > 1.0}} {{ set cosine 1.0 }}
    if {{$cosine < -1.0}} {{ set cosine -1.0 }}

    set theta [expr {{acos($cosine)}}]
    set theta_deg [expr {{$theta * 180.0 / acos(-1.0)}}]
    set perpendicular [vecsub $v [vecscale $cosine $u]]
    set perpendicular_length [veclength $perpendicular]

    if {{$perpendicular_length > 1.0e-8}} {{
        set basis [vecscale [expr {{1.0 / $perpendicular_length}}] $perpendicular]
        set nsteps 48
        set previous [vecadd $origin [vecscale $radius $u]]
        graphics $molid color $color
        for {{set i 1}} {{$i <= $nsteps}} {{incr i}} {{
            set phi [expr {{$theta * double($i) / double($nsteps)}}]
            set direction [vecadd [vecscale [expr {{cos($phi)}}] $u] \
                                      [vecscale [expr {{sin($phi)}}] $basis]]
            set point [vecadd $origin [vecscale $radius $direction]]
            graphics $molid cylinder $previous $point radius 0.065 resolution 12 filled yes
            set previous $point
        }}
        set half_theta [expr {{0.5 * $theta}}]
        set middle_direction [vecadd [vecscale [expr {{cos($half_theta)}}] $u] \
                                          [vecscale [expr {{sin($half_theta)}}] $basis]]
    }} else {{
        set middle_direction $u
    }}

    set label_position [vecadd $origin \
        [vecscale [expr {{1.22 * $radius}}] [vecnorm $middle_direction]]]
    graphics $molid color white
    graphics $molid text $label_position \
        [format "theta=%.2f deg; c=%s" $theta_deg $c_value] size 1.2 thickness 2.0
    return $theta_deg
}}

set l_origin [atom_xyz $molid "resname LV1 and name LORG"]
set l_tip [atom_xyz $molid "resname LV1 and name LTIP"]
set lp_origin [atom_xyz $molid "resname LV2 and name PORG"]
set lp_tip [atom_xyz $molid "resname LV2 and name PTIP"]
set d_origin [atom_xyz $molid "resname DLL and name DORG"]
set d_tip [atom_xyz $molid "resname DLL and name DTIP"]

# The red display vector is d-length times (unit(L) cross unit(L')).
# This preserves the cross-product magnitude sin(theta) at a visible scale.
set l_hat [vecnorm [vecsub $l_tip $l_origin]]
set lp_hat [vecnorm [vecsub $lp_tip $lp_origin]]
set d_vec [vecsub $d_tip $d_origin]
set l_cross_lp [veccross $l_hat $lp_hat]
set cross_tip [vecadd $d_origin [vecscale [veclength $d_vec] $l_cross_lp]]

# Draw theta between the LxL' direction and dLL'. The same label reports the
# cpptraj scalar triple product c = (unit(L) x unit(L')) dot unit(dLL').
set arc_radius [expr {{0.30 * [veclength [vecsub $cross_tip $d_origin]]}}]
set theta_deg [draw_angle_arc $molid $d_origin $l_cross_lp $d_vec \
                   $arc_radius red "{chirality:.8f}"]

draw_arrow $molid $l_origin $l_tip orange
draw_arrow $molid $lp_origin $lp_tip orange
draw_arrow $molid $d_origin $d_tip green
draw_arrow $molid $d_origin $cross_tip red

set vecsel [atomselect $molid "resname LV1 LV2 DLL"]
$vecsel set beta 0.0
$vecsel delete

display projection Orthographic
axes location Off
display resetview
puts [format "orange=L and L-prime, green=dLL-prime, red=L cross L-prime; theta=%.2f deg; c={chirality:.8f}" $theta_deg]
'''
    path.write_text(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate CB7CB induced chiral asymmetry with cpptraj."
    )
    parser.add_argument("--topology", type=Path, required=True, help="Topology-readable PDB")
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=None,
        help="Trajectory input (defaults to --topology for a one-frame PDB)",
    )
    parser.add_argument(
        "--residues",
        default="1",
        help="Molecule/residue numbers, e.g. 1 or 1,3-5 (default: 1)",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=1, help="First analysis frame")
    parser.add_argument("--stop", type=int, default=None, help="Last analysis frame")
    parser.add_argument("--stride", type=int, default=1, help="Analysis frame stride")
    parser.add_argument(
        "--visualize-frame", type=int, default=1, help="Frame used for visualization PDB"
    )
    parser.add_argument(
        "--cpptraj", default="cpptraj", help="cpptraj executable (default: cpptraj on PATH)"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    topology = args.topology.resolve()
    trajectory = (args.trajectory or args.topology).resolve()
    out_dir = args.out_dir.resolve()
    residues = parse_residue_spec(args.residues)

    if args.start < 1 or args.visualize_frame < 1 or args.stride < 1:
        raise SystemExit("Frame numbers and stride must be positive")
    if not topology.is_file():
        raise SystemExit(f"Topology not found: {topology}")
    if not trajectory.is_file():
        raise SystemExit(f"Trajectory not found: {trajectory}")
    if shutil.which(args.cpptraj) is None:
        raise SystemExit(
            f"cpptraj executable not found: {args.cpptraj!r}. Load Amber first (ml amber/26)."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_input = out_dir / "induced_chirality.cpptraj.in"
    write_analysis_input(
        analysis_input,
        topology,
        trajectory,
        residues,
        out_dir,
        args.start,
        args.stop,
        args.stride,
    )
    run_cpptraj(args.cpptraj, analysis_input, out_dir / "induced_chirality.cpptraj.log")

    for residue in residues:
        residue_dir = out_dir / f"residue_{residue:04d}_visualization"
        residue_dir.mkdir(parents=True, exist_ok=True)
        viz_input = residue_dir / "visualization.cpptraj.in"
        molecule_pdb, vectors_pdb, viz_math = write_visualization_input(
            viz_input,
            topology,
            trajectory,
            residue,
            args.visualize_frame,
            residue_dir,
        )
        run_cpptraj(args.cpptraj, viz_input, residue_dir / "visualization.cpptraj.log")
        chirality = parse_chirality(viz_math)
        combined_pdb = residue_dir / "cb7cb_with_raw_vectors.pdb"
        write_combined_visualization(molecule_pdb, vectors_pdb, combined_pdb, chirality)
        write_vmd_script(residue_dir / "view_vectors.vmd", combined_pdb, chirality)
        print(f"Residue {residue}: c={chirality:.8f}")
        print(f"  visualization: {combined_pdb}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
