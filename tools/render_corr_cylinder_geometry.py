#!/usr/bin/env python3
"""Render the geometry used by the director-resolved parallel correlation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d


OUTPUT = Path("reports/corr/cylinder_geometry_definition.png")
SPRITE_DIR = Path(
    "data-processed/5cb/amoeba-pt2.7/004e_d-md-304/corr-cylinder-geometry"
)
MOLECULE_A = SPRITE_DIR / "5cb_molecule_a_vmd.png"
MOLECULE_B = SPRITE_DIR / "5cb_molecule_b_vmd.png"


def arrow3d(
    ax,
    start,
    end,
    color,
    linewidth=3.0,
    mutation=16,
    label=None,
    label_fraction=0.54,
    label_offset=(0.0, 0.0, 0.0),
):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    delta = end - start
    ax.quiver(
        *start,
        *delta,
        color=color,
        linewidth=linewidth,
        arrow_length_ratio=0.10,
        normalize=False,
        zorder=15,
    )
    if label:
        midpoint = start + label_fraction * delta + np.asarray(label_offset, dtype=float)
        ax.text(
            *midpoint,
            label,
            color=color,
            fontsize=14,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
            zorder=20,
        )


def draw_molecule(ax, center, direction, label):
    center = np.asarray(center, dtype=float)
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)

    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(direction, helper)) > 0.92:
        helper = np.array([0.0, 1.0, 0.0])
    side = np.cross(direction, helper)
    side /= np.linalg.norm(side)

    offsets = np.linspace(-0.62, 0.62, 9)
    points = np.array(
        [center + s * direction + 0.07 * (-1) ** k * side for k, s in enumerate(offsets)]
    )
    ax.plot(points[:, 0], points[:, 1], points[:, 2], color="#505050", lw=4.2, zorder=8)
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=42,
        color="#707070",
        edgecolor="#303030",
        linewidth=0.7,
        depthshade=True,
        zorder=9,
    )
    ax.text(
        *(center + 0.72 * direction),
        label,
        color="#303030",
        fontsize=13,
        fontweight="bold",
    )


def add_vmd_sprite(ax, center, path, angle_degrees, zoom):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing VMD sprite {path}; run tools/render_vmd_molecule_sprites.py first"
        )
    molecule = Image.open(path).convert("RGBA").rotate(
        angle_degrees,
        resample=Image.Resampling.BICUBIC,
        expand=True,
    )
    projected_x, projected_y, _ = proj3d.proj_transform(*center, ax.get_proj())
    artist = AnnotationBbox(
        OffsetImage(np.asarray(molecule), zoom=zoom),
        (projected_x, projected_y),
        xycoords="data",
        frameon=False,
        pad=0.0,
        zorder=8,
    )
    ax.add_artist(artist)


def draw_cylinder(ax, radius, zmin, zmax):
    theta = np.linspace(0.0, 2.0 * np.pi, 100)
    z = np.linspace(zmin, zmax, 2)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x = radius * np.cos(theta_grid)
    y = radius * np.sin(theta_grid)
    ax.plot_surface(
        x,
        y,
        z_grid,
        color="#8ecae6",
        alpha=0.11,
        linewidth=0,
        shade=False,
        zorder=0,
    )
    for zcap in (zmin, zmax):
        ax.plot(
            radius * np.cos(theta),
            radius * np.sin(theta),
            np.full_like(theta, zcap),
            color="#4292c6",
            alpha=0.7,
            lw=1.8,
        )
    for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
        ax.plot(
            [radius * np.cos(angle)] * 2,
            [radius * np.sin(angle)] * 2,
            [zmin, zmax],
            color="#4292c6",
            alpha=0.30,
            lw=0.9,
        )


def draw_axial_bin(ax, radius, center_z, width):
    theta = np.linspace(0.0, 2.0 * np.pi, 72)
    lower = center_z - 0.5 * width
    upper = center_z + 0.5 * width
    vertices = []
    for index in range(len(theta) - 1):
        vertices.append(
            [
                (radius * np.cos(theta[index]), radius * np.sin(theta[index]), lower),
                (radius * np.cos(theta[index + 1]), radius * np.sin(theta[index + 1]), lower),
                (radius * np.cos(theta[index + 1]), radius * np.sin(theta[index + 1]), upper),
                (radius * np.cos(theta[index]), radius * np.sin(theta[index]), upper),
            ]
        )
    wall = Poly3DCollection(
        vertices,
        facecolor="#ffd166",
        edgecolor="none",
        alpha=0.15,
        zorder=1,
    )
    ax.add_collection3d(wall)
    for level in (lower, upper):
        ax.plot(
            radius * np.cos(theta),
            radius * np.sin(theta),
            np.full_like(theta, level),
            color="#d89b00",
            lw=2.0,
            alpha=0.9,
        )
    return lower, upper


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(13.8, 8.4), facecolor="white")
    ax = figure.add_axes([0.02, 0.05, 0.72, 0.88], projection="3d")
    notes = figure.add_axes([0.74, 0.08, 0.245, 0.82])
    notes.axis("off")

    radius = 1.75
    zmin, zmax = -2.6, 3.15
    ri = np.array([0.0, 0.0, 0.0])
    rj = np.array([1.10, 0.78, 1.82])
    projection = np.array([0.0, 0.0, rj[2]])
    bin_width = 0.55

    draw_cylinder(ax, radius, zmin, zmax)
    lower, upper = draw_axial_bin(ax, radius, rj[2], bin_width)

    ax.scatter(*ri, s=155, color="#009E73", edgecolor="white", linewidth=1.4, zorder=12)
    ax.scatter(*rj, s=155, color="#CC79A7", edgecolor="white", linewidth=1.4, zorder=12)
    label_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 1.2}
    ax.text(
        *(ri + np.array([-0.52, -0.12, -0.30])),
        r"$\mathbf{R}_i^{\rm corr}$",
        fontsize=15,
        color="#007a59",
        bbox=label_box,
        zorder=20,
    )
    ax.text(
        *(rj + np.array([0.18, 0.10, 0.16])),
        r"$\mathbf{R}_j^{\rm corr}$",
        fontsize=15,
        color="#9b3d74",
        bbox=label_box,
        zorder=20,
    )

    arrow3d(
        ax,
        ri,
        rj,
        "#7B2CBF",
        4.0,
        label=r"$\mathbf{r}_{ij}$",
        label_fraction=0.64,
        label_offset=(0.18, -0.18, -0.02),
    )
    arrow3d(
        ax,
        ri,
        projection,
        "#0072B2",
        3.6,
        label=r"$z_{ij}\mathbf{n}(t)$",
        label_fraction=0.53,
        label_offset=(-0.48, -0.12, 0.02),
    )
    arrow3d(
        ax,
        projection,
        rj,
        "#E69F00",
        3.6,
        label=r"$\mathbf{r}_{\perp,ij}$",
        label_fraction=0.48,
        label_offset=(0.02, -0.28, 0.12),
    )
    arrow3d(ax, [0, 0, zmin - 0.15], [0, 0, zmax + 0.45], "#111111", 3.0)
    ax.text(0.08, 0.05, zmax + 0.45, r"$\mathbf{n}(t)$", fontsize=16, fontweight="bold")

    # Cylinder radius and axial-bin thickness.
    ax.plot([0, radius], [0, 0], [zmin, zmin], color="#0072B2", lw=2.6)
    ax.scatter([0, radius], [0, 0], [zmin, zmin], s=20, color="#0072B2")
    ax.text(0.78, -0.08, zmin - 0.23, r"radius $R$", color="#0072B2", fontsize=14)

    bracket_x, bracket_y = -1.92, 0.0
    ax.plot([bracket_x, bracket_x], [bracket_y, bracket_y], [lower, upper], color="#A96F00", lw=3)
    ax.plot([bracket_x - 0.08, bracket_x + 0.08], [0, 0], [lower, lower], color="#A96F00", lw=2)
    ax.plot([bracket_x - 0.08, bracket_x + 0.08], [0, 0], [upper, upper], color="#A96F00", lw=2)
    ax.text(bracket_x - 0.20, 0.0, rj[2], r"$\Delta z$", color="#A96F00", fontsize=15)

    # Right-angle marker at the projection point.
    marker = 0.14
    unit_perp = (rj - projection) / np.linalg.norm(rj - projection)
    p1 = projection + marker * unit_perp
    p2 = p1 + np.array([0, 0, marker])
    p3 = projection + np.array([0, 0, marker])
    ax.plot(
        [p1[0], p2[0], p3[0]],
        [p1[1], p2[1], p3[1]],
        [p1[2], p2[2], p3[2]],
        color="#333333",
        lw=1.5,
    )

    ax.set_xlim(-2.25, 2.25)
    ax.set_ylim(-2.25, 2.25)
    ax.set_zlim(zmin - 0.35, zmax + 0.65)
    ax.set_box_aspect((1.0, 1.0, 1.45), zoom=1.22)
    ax.view_init(elev=20, azim=-52)
    ax.set_axis_off()

    # The exact 304 K conformers previously rendered by VMD replace the generic
    # bead-chain cartoons. They are transparent sprites centered on the same
    # correlation points used by the vector decomposition.
    add_vmd_sprite(ax, ri, MOLECULE_A, 72.0, 0.155)
    add_vmd_sprite(ax, rj, MOLECULE_B, 72.0, 0.150)
    molecule_label_box = {
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.76,
        "pad": 1.0,
    }
    ax.text(
        *(ri + np.array([0.18, 0.10, 0.46])),
        "molecule i",
        fontsize=13,
        fontweight="bold",
        color="#303030",
        bbox=molecule_label_box,
        zorder=20,
    )
    ax.text(
        *(rj + np.array([0.10, 0.08, 0.54])),
        "molecule j",
        fontsize=13,
        fontweight="bold",
        color="#303030",
        bbox=molecule_label_box,
        zorder=20,
    )

    notes.text(
        0.0,
        1.00,
        "Parallel-correlation geometry",
        fontsize=19,
        fontweight="bold",
        va="top",
        color="#222222",
    )
    notes.text(
        0.0,
        0.89,
        r"$\mathbf{r}_{ij}=\mathbf{R}_j^{\rm corr}-\mathbf{R}_i^{\rm corr}$",
        fontsize=16,
        color="#7B2CBF",
    )
    notes.text(
        0.0,
        0.81,
        r"$z_{ij}=\mathbf{r}_{ij}\!\cdot\!\mathbf{n}(t)$",
        fontsize=16,
        color="#0072B2",
    )
    notes.text(
        0.0,
        0.73,
        r"$r_{\perp,ij}=\sqrt{r_{ij}^{2}-z_{ij}^{2}}$",
        fontsize=16,
        color="#E69F00",
    )

    notes.text(0.0, 0.60, "Pair contributes when", fontsize=16, fontweight="bold")
    notes.text(0.05, 0.52, r"$r_{\perp,ij}<R$", fontsize=17, color="#0072B2")
    notes.text(
        0.05,
        0.44,
        r"$|z_{ij}|\in$ axial bin $k$",
        fontsize=17,
        color="#A96F00",
    )
    notes.text(
        0.05,
        0.36,
        r"bin width $=\Delta z$",
        fontsize=17,
        color="#A96F00",
    )

    notes.text(
        0.0,
        0.25,
        "What the surfaces mean",
        fontsize=16,
        fontweight="bold",
        va="top",
    )
    notes.text(
        0.0,
        0.16,
        "Blue wall: cylinder of radius R\n"
        "Yellow slice: selected |z| bin\n"
        "Cylinder axis passes through molecule i",
        fontsize=13.5,
        linespacing=1.45,
        color="#444444",
        va="top",
    )
    notes.text(
        0.0,
        -0.02,
        "Actual 304 K VMD conformers; schematic placement",
        fontsize=11.5,
        style="italic",
        color="#666666",
        va="bottom",
    )

    figure.suptitle(
        "Director-resolved pair geometry used in the parallel histogram",
        x=0.47,
        y=0.975,
        fontsize=22,
        fontweight="bold",
    )
    figure.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
