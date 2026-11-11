#!/usr/bin/env python3
"""Apply Avenir Book to all non-equation Word runs in a generated DOCX."""

from __future__ import annotations

import argparse
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PR = "http://schemas.openxmlformats.org/package/2006/relationships"
ET.register_namespace("w", W)


def qn(local: str) -> str:
    return f"{{{W}}}{local}"


def set_rfonts(element: ET.Element, font: str) -> None:
    for attr in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
        element.attrib.pop(qn(attr), None)
    for attr in ("ascii", "hAnsi", "eastAsia", "cs"):
        element.set(qn(attr), font)


def ensure_run_font(run: ET.Element, font: str) -> None:
    run_properties = run.find(qn("rPr"))
    if run_properties is None:
        run_properties = ET.Element(qn("rPr"))
        run.insert(0, run_properties)
    fonts = run_properties.find(qn("rFonts"))
    if fonts is None:
        fonts = ET.Element(qn("rFonts"))
        run_properties.insert(0, fonts)
    set_rfonts(fonts, font)


def update_xml(name: str, payload: bytes, font: str) -> bytes:
    root = ET.fromstring(payload)

    if name == "word/styles.xml":
        for fonts in root.iter(qn("rFonts")):
            set_rfonts(fonts, font)

        doc_defaults = root.find(qn("docDefaults"))
        if doc_defaults is None:
            doc_defaults = ET.Element(qn("docDefaults"))
            root.insert(0, doc_defaults)
        rpr_default = doc_defaults.find(qn("rPrDefault"))
        if rpr_default is None:
            rpr_default = ET.SubElement(doc_defaults, qn("rPrDefault"))
        rpr = rpr_default.find(qn("rPr"))
        if rpr is None:
            rpr = ET.SubElement(rpr_default, qn("rPr"))
        fonts = rpr.find(qn("rFonts"))
        if fonts is None:
            fonts = ET.SubElement(rpr, qn("rFonts"))
        set_rfonts(fonts, font)
    else:
        # Word equations use m:r rather than w:r, so this leaves native OMML
        # equation typography untouched while styling prose, captions, tables,
        # headings, and paths consistently.
        for run in root.iter(qn("r")):
            ensure_run_font(run, font)

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def apply_font(path: Path, font: str) -> None:
    xml_parts = {
        "word/document.xml",
        "word/styles.xml",
        "word/footnotes.xml",
        "word/endnotes.xml",
    }

    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.", suffix=".docx", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)

    try:
        with ZipFile(path, "r") as source:
            items = source.infolist()
            payloads = {item.filename: source.read(item.filename) for item in items}

        document_name = "word/document.xml"
        relationships_name = "word/_rels/document.xml.rels"
        document_root = ET.fromstring(payloads[document_name])
        used_relationships = {
            value
            for element in document_root.iter()
            for attribute, value in element.attrib.items()
            if attribute in {f"{{{R}}}embed", f"{{{R}}}link"}
        }
        relationships_root = ET.fromstring(payloads[relationships_name])
        for relationship in list(relationships_root):
            if (
                relationship.get("Type", "").endswith("/image")
                and relationship.get("Id") not in used_relationships
            ):
                relationships_root.remove(relationship)
        payloads[relationships_name] = ET.tostring(
            relationships_root, encoding="utf-8", xml_declaration=True
        )

        referenced_media: set[str] = set()
        for name, payload in payloads.items():
            if not name.endswith(".rels"):
                continue
            relationship_file = ET.fromstring(payload)
            for relationship in relationship_file:
                if relationship.get("Type", "").endswith("/image"):
                    target = relationship.get("Target", "")
                    if target.startswith("media/"):
                        referenced_media.add(f"word/{target}")
        unused_media = {
            name
            for name in payloads
            if name.startswith("word/media/") and name not in referenced_media
        }

        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as destination:
            for item in items:
                if item.filename in unused_media:
                    continue
                payload = payloads[item.filename]
                if item.filename in xml_parts or item.filename.startswith(
                    ("word/header", "word/footer")
                ):
                    payload = update_xml(item.filename, payload, font)
                destination.writestr(item, payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("docx", type=Path)
    parser.add_argument("--font", default="Avenir Book")
    args = parser.parse_args()
    apply_font(args.docx, args.font)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
