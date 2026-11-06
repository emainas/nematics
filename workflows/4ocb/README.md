# 4OCB AMOEBA report workflow

This workflow reproduces the numerical sections of the 5CB report for the 11
read-only 4OCB AMOEBA trajectories. It intentionally excludes GAFF, Alekos
overlays, and dipole-specific/Kirkwood analysis.

The production series is 288, 296, 304, 312, 320, 324, 326, 328, 330, 332,
and 334 K. Every run contains 1,000 frames of 1,000 molecules with 36 atoms
per molecule. Thus each one-body distribution contains 1,000,000
molecule-frames per temperature.

## Chemistry used

- end-to-end axis: local atom 18 (terminal butyl carbon) to atom 2 (nitrile N)
- biphenyl-core axis: local atom 11 (butoxy-side ipso) to atom 5
  (nitrile-side ipso)
- inertia axis: all 36 atoms, smallest principal moment
- ether torsion: local atoms 16-15-28-11, `C-C-O-C_ar`, central bond 15-28
- ring junction: local atoms 10-11-28-15, `C_ring-C_ipso-O-C`, central
  aryl--O bond 11-28

Cpptraj applies `autoimage` before every vector, COM, RDF, or torsion
calculation. Pair correlations use the existing nematics implementation: 27
periodic replicas, 1/2/3 nm parallel cylinders, a 0.2 nm perpendicular slab,
and 200 direct bins over 0--5 nm.

## Setup and validation

```bash
uv venv .venv-nematics
source .venv-nematics/bin/activate
uv pip install -e ".[dev,viz]"
command -v cpptraj
nematics datasets --check
```

Load the AmberTools module available on the machine if `cpptraj` is not
already on `PATH`, or pass its executable explicitly with
`--cpptraj /path/to/cpptraj`. On the current cluster session it resolves to
`/proj/pierilab/software/amber/ambertools25/bin/cpptraj`; the documented
`amber/26` module is not available here.

Before production, run every stage over one frame. Use a separate smoke root
so these files cannot be mistaken for the final products:

```bash
python workflows/4ocb/analyze_4ocb_report.py order \
  --start 1 --stop 1 --workers 4 \
  --out-data data-processed/4ocb/smoke/order \
  --out-plot reports/4ocb_smoke/order.png

python workflows/4ocb/analyze_4ocb_report.py rdf \
  --start 1 --stop 1 --workers 4 \
  --out-data data-processed/4ocb/smoke/com-rdf \
  --out-plot reports/4ocb_smoke/com-rdf.png

python workflows/4ocb/analyze_4ocb_report.py paircorr \
  --start 1 --stop 1 --workers 4 \
  --order-root data-processed/4ocb/smoke/order \
  --out-data data-processed/4ocb/smoke/director-paircorr \
  --out-report reports/4ocb_smoke/paircorr

python workflows/4ocb/analyze_4ocb_report.py dihedral \
  --start 1 --stop 1 --workers 4 \
  --out-data data-processed/4ocb/smoke/junction-dihedral \
  --out-report reports/4ocb_smoke/dihedral

python workflows/4ocb/analyze_4ocb_report.py ring-dihedral \
  --start 1 --stop 1 --workers 4 \
  --out-data data-processed/4ocb/smoke/ring-junction-dihedral \
  --out-report reports/4ocb_smoke/ring-junction-dihedral
```

## Full production sequence

The stages cache their expensive cpptraj tables and numerical products. Do not
use a force option unless intentionally replacing a corresponding cache.

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/nematics-mpl \
python workflows/4ocb/analyze_4ocb_report.py order --workers 4

MPLBACKEND=Agg MPLCONFIGDIR=/tmp/nematics-mpl \
python workflows/4ocb/analyze_4ocb_report.py rdf --workers 4

MPLBACKEND=Agg MPLCONFIGDIR=/tmp/nematics-mpl \
python workflows/4ocb/analyze_4ocb_report.py paircorr --workers 4

MPLBACKEND=Agg MPLCONFIGDIR=/tmp/nematics-mpl \
python workflows/4ocb/analyze_4ocb_report.py dihedral --workers 4

MPLBACKEND=Agg MPLCONFIGDIR=/tmp/nematics-mpl \
python workflows/4ocb/analyze_4ocb_report.py ring-dihedral --workers 4

python workflows/4ocb/build_4ocb_report.py
```

The `paircorr` stage depends on the `order` products. The other stages are
independent. All generated cpptraj inputs and logs remain beside their data
products for provenance.

The five analysis commands supply the report sections directly:

- `order`: three molecular-axis definitions, framewise Q tensors, directors,
  S values, the temperature summary, and the order figure.
- `rdf`: mass-weighted molecular COM RDF tables and the 11-panel RDF figure.
- `paircorr`: first-rank end-to-end/core and second-rank
  end-to-end/core/inertia tables and four 11-panel figures.
- `dihedral`: all 1,000,000 C-C-O-C(ar) ether-torsion samples per
  temperature, one-degree histograms, PMFs, and figures.
- `ring-dihedral`: the separate aryl--O-centered ring-plane torsion with the
  same sample count and retained products.

The one-frame smoke products are validation artifacts only. Do not quote
their order parameters, RDFs, pair correlations, or PMFs as ensemble results.

The final command writes the concise Avenir-styled report to
`reports/4ocb_results_report/4OCB_results_report_v1.docx`, using the 5CB v2
document as its layout reference. It requires `pandoc` and refuses to build if
any production summary is missing.

For local VMD, download either scene plus the shared PDB into one directory:

```text
tools/view_4ocb_ether_dihedral.vmd
tools/view_4ocb_ring_junction_dihedral.vmd
data-processed/4ocb/amoeba-pt2.2/visualization/4ocb_molecule_frame.pdb
```

The orange and cyan triangles are the two dihedral planes; the yellow bond is
the central torsional hinge. Both scenes were validated with VMD 1.9.4 in
text mode, and run normally in the local VMD GUI with `vmd -e <scene.vmd>`.
