#!/usr/bin/env python3
"""Render transparent VMD sprites for the real 5CB pair used in the cylinder diagram."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


SOURCE = Path(
    "data-processed/5cb/amoeba-pt2.7/004e_d-md-304/"
    "vector-definition-visualization/5cb_close_pair_304K_frame1.pdb"
)
OUTPUT_DIR = Path(
    "data-processed/5cb/amoeba-pt2.7/004e_d-md-304/"
    "corr-cylinder-geometry"
)


def split_pdb(source: Path, chain: str, output: Path) -> None:
    lines = source.read_text(encoding="utf-8").splitlines()
    atom_lines = [
        line
        for line in lines
        if line.startswith(("ATOM  ", "HETATM")) and line[21:22] == chain
    ]
    serials = {int(line[6:11]) for line in atom_lines}
    conect_lines: list[str] = []
    for line in lines:
        if not line.startswith("CONECT"):
            continue
        fields = line.split()
        connected = [int(value) for value in fields[1:]]
        if connected and all(value in serials for value in connected):
            conect_lines.append(line)
    output.write_text(
        "\n".join(
            [f"REMARK Real 5CB molecule from {source}", *atom_lines, *conect_lines, "END", ""]
        ),
        encoding="utf-8",
    )


def write_scene(pdb: Path, tga: Path, script: Path, y_rotation: float) -> None:
    script.write_text(
        f"""set pdb_file {{{pdb.resolve()}}}
set output_tga {{{tga.resolve()}}}
set molid [mol new $pdb_file type pdb waitfor all]
color Display Background white
mol delrep 0 $molid
mol representation Licorice 0.24 24 24
mol color ColorID 2
mol selection all
mol material AOShiny
mol addrep $molid
display projection Orthographic
display depthcue off
display shadows off
display ambientocclusion on
display aoambient 0.72
display aodirect 0.38
axes location Off
display resetview
rotate x by -12
rotate y by {y_rotation:.1f}
scale by 1.55
render TachyonInternal $output_tga
quit
""",
        encoding="utf-8",
    )


def transparent_crop(tga: Path, png: Path) -> None:
    image = Image.open(tga).convert("RGB")
    rgb = np.asarray(image, dtype=np.uint8)
    darkness = 255 - np.min(rgb, axis=2)
    alpha = np.clip((darkness.astype(float) - 2.0) * 3.2, 0, 255).astype(np.uint8)
    rgba = np.dstack((rgb, alpha))
    result = Image.fromarray(rgba, mode="RGBA")
    box = result.getchannel("A").getbbox()
    if box is None:
        raise RuntimeError(f"VMD render contains no non-white pixels: {tga}")
    padding = 18
    box = (
        max(0, box[0] - padding),
        max(0, box[1] - padding),
        min(result.width, box[2] + padding),
        min(result.height, box[3] + padding),
    )
    result.crop(box).save(png)


def main() -> int:
    vmd = shutil.which("vmd")
    if vmd is None:
        raise RuntimeError("vmd is not on PATH; load the vmd/1.9.4 module first")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for chain, rotation in (("A", -18.0), ("B", 18.0)):
        stem = f"5cb_molecule_{chain.lower()}"
        pdb = OUTPUT_DIR / f"{stem}.pdb"
        tga = OUTPUT_DIR / f"{stem}_vmd.tga"
        script = OUTPUT_DIR / f"render_{stem}.vmd"
        png = OUTPUT_DIR / f"{stem}_vmd.png"
        split_pdb(SOURCE, chain, pdb)
        write_scene(pdb, tga, script, rotation)
        subprocess.run(
            [
                vmd,
                "-dispdev",
                "text",
                "-size",
                "850",
                "1150",
                "-eofexit",
                "-e",
                str(script.resolve()),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        transparent_crop(tga, png)
        print(png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
