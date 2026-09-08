#!/usr/bin/env python3
"""
Convert a 3MF file into:
  1) <stem>_core3mf.3mf - minimal 3MF Core Spec only: no Production/BeamLattice/
     Material/Slice/SecureContent extensions, no p:UUID, standard double-quoted
     XML declaration, coordinates rounded to a fixed decimal count.
  2) <stem>.stl          - binary STL of the same geometry.

Requires: numpy

Usage:
    python3 convert_3mf.py model.3mf
    python3 convert_3mf.py model.3mf --decimals 6 --outdir ./converted
"""
import argparse
import re
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np


def parse_mesh(xml: str):
    n_objects = xml.count("<object ")
    if n_objects != 1:
        print(f"WARNING: expected 1 <object>, found {n_objects}. "
              f"This script only handles single-object tiles.", file=sys.stderr)

    v0, v1 = xml.index("<vertices>"), xml.index("</vertices>")
    vblock = xml[v0:v1]
    xs = np.array(re.findall(r'x="([^"]*)"', vblock), dtype=np.float64)
    ys = np.array(re.findall(r'y="([^"]*)"', vblock), dtype=np.float64)
    zs = np.array(re.findall(r'z="([^"]*)"', vblock), dtype=np.float64)
    verts = np.stack([xs, ys, zs], axis=1)

    t0, t1 = xml.index("<triangles>"), xml.index("</triangles>")
    tblock = xml[t0:t1]
    a = np.array(re.findall(r'v1="(\d+)"', tblock), dtype=np.int64)
    b = np.array(re.findall(r'v2="(\d+)"', tblock), dtype=np.int64)
    c = np.array(re.findall(r'v3="(\d+)"', tblock), dtype=np.int64)
    tris = np.stack([a, b, c], axis=1)

    return verts, tris


def fmt_num(x: float, decimals: int) -> str:
    v = round(float(x), decimals)
    if v == 0:
        v = 0.0  # avoid "-0.000000"
    return f"{v:.{decimals}f}"


def write_core_3mf(path: Path, verts: np.ndarray, tris: np.ndarray, decimals: int) -> str:
    vlines = "".join(
        f'<vertex x="{fmt_num(x, decimals)}" y="{fmt_num(y, decimals)}" z="{fmt_num(z, decimals)}"/>'
        for x, y, z in verts
    )
    tlines = "".join(
        f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in tris
    )

    xml_out = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
        '<resources><object id="1" type="model"><mesh>'
        f'<vertices>{vlines}</vertices>'
        f'<triangles>{tlines}</triangles>'
        '</mesh></object></resources>'
        '<build><item objectid="1"/></build>'
        '</model>'
    )

    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '</Types>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" '
        'Target="/3D/3dmodel.model" Id="rel0"/></Relationships>'
    )

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("3D/3dmodel.model", xml_out)

    return xml_out


def write_stl(path: Path, verts: np.ndarray, tris: np.ndarray):
    p0, p1, p2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    n = n / norm

    ntri = tris.shape[0]
    facet_dtype = np.dtype([
        ("normal", "<f4", (3,)),
        ("v0", "<f4", (3,)),
        ("v1", "<f4", (3,)),
        ("v2", "<f4", (3,)),
        ("attr", "<u2"),
    ])
    facets = np.zeros(ntri, dtype=facet_dtype)
    facets["normal"] = n
    facets["v0"], facets["v1"], facets["v2"] = p0, p1, p2

    with open(path, "wb") as f:
        header = b"Binary STL - converted from 3MF"
        f.write(header + b"\x00" * (80 - len(header)))
        f.write(struct.pack("<I", ntri))
        f.write(facets.tobytes())


def main():
    ap = argparse.ArgumentParser(description="Convert a 3MF into a Photon-Workshop-safe core 3MF + STL")
    ap.add_argument("input", type=Path, help="input .3mf file")
    ap.add_argument("--outdir", type=Path, default=None, help="output directory (default: alongside input)")
    ap.add_argument("--decimals", type=int, default=6, help="coordinate decimal places to keep (default 6)")
    args = ap.parse_args()

    outdir = args.outdir or args.input.parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    with zipfile.ZipFile(args.input) as z:
        xml = z.read("3D/3dmodel.model").decode("utf-8")

    verts, tris = parse_mesh(xml)
    print(f"parsed: {verts.shape[0]} vertices, {tris.shape[0]} triangles")

    core_path = outdir / f"{stem}_core3mf.3mf"
    stl_path = outdir / f"{stem}.stl"

    xml_out = write_core_3mf(core_path, verts, tris, args.decimals)
    write_stl(stl_path, verts, tris)

    assert xml_out.count("<vertex ") == verts.shape[0], "vertex count mismatch after write"
    assert xml_out.count("<triangle ") == tris.shape[0], "triangle count mismatch after write"

    print(f"core 3MF written: {core_path}  ({core_path.stat().st_size/1e6:.1f} MB)")
    print(f"STL written:      {stl_path}  ({stl_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
