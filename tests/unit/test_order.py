from pathlib import Path

from nematics.order import _default_gro_dirs


def test_default_gro_directories_match_source_layout():
    root = Path("/raw/5cb")

    assert _default_gro_dirs(root) == [
        root / "gro",
        root / "gro2",
        root / "gro3",
        root / "gro4",
    ]
