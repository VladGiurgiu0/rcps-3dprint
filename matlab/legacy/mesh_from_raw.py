#!/usr/bin/env python3
"""mesh_from_raw.py

CLI used by RCPS_v2.m.

Positional args:
  1) raw_file  : float32 binary dump written from MATLAB with
                 fwrite(permute(single(F), [2 1 3]), 'single')
  2) meta_file : 4 lines
       nx ny nz
       voxSize_mm
       origin_x origin_y origin_z
       isoLevel
  3) out_stl   : output STL path

Methods (--method):
  pyvista backends: contour | marching_cubes | flying_edges
  numpy backends  : skimage | pymcubes
  iso2mesh backend: iso2mesh  (pyiso2mesh)

Notes:
  - Volume axes are interpreted as (x,y,z) == (nx,ny,nz).
  - Mesh vertices are output in physical mm coordinates.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def read_meta(meta_path: Path):
    lines = [ln.strip() for ln in meta_path.read_text().splitlines() if ln.strip()]
    if len(lines) < 4:
        raise ValueError(f"Meta file must have >=4 non-empty lines: {meta_path}")

    nx, ny, nz = map(int, lines[0].split())
    vox = float(lines[1])
    ox, oy, oz = map(float, lines[2].split())
    iso = float(lines[3])

    return nx, ny, nz, vox, np.array([ox, oy, oz], dtype=float), iso


def read_raw(raw_path: Path, nx: int, ny: int, nz: int, raw_layout: str) -> np.ndarray:
    data = np.fromfile(raw_path, dtype=np.float32)
    n_expect = nx * ny * nz
    if data.size != n_expect:
        raise ValueError(f"RAW size mismatch: got {data.size} floats, expected {n_expect} (nx*ny*nz).")

    if raw_layout == 'yxz':
        # MATLAB wrote permute(F,[2 1 3]) in column-major order.
        F_yxz = data.reshape((ny, nx, nz), order='F')
        F = np.transpose(F_yxz, (1, 0, 2))
        return F

    if raw_layout == 'xyz':
        # MATLAB wrote F directly via fwrite(F,'single') in column-major order.
        return data.reshape((nx, ny, nz), order='F')

    raise ValueError(f"Unknown raw_layout: {raw_layout}")




def mesh_pyvista(F: np.ndarray, vox: float, origin: np.ndarray, iso: float, method: str):
    import pyvista as pv

    nx, ny, nz = F.shape

    grid = pv.ImageData(
        dimensions=(nx, ny, nz),
        spacing=(vox, vox, vox),
        origin=(float(origin[0]), float(origin[1]), float(origin[2])),
    )

    # VTK point ordering expects Fortran-flattened scalars for ImageData
    grid.point_data['F'] = F.ravel(order='F')

    surf = grid.contour(
        isosurfaces=[float(iso)],
        scalars='F',
        method=method,
        progress_bar=True,
    )

    #surf = surf.triangulate().clean(tolerance=0.0, absolute=True)
    surf = surf.triangulate()   # no clean here (MeshFix will do the heavy lifting if needed)

    verts = np.asarray(surf.points, dtype=float)
    faces = np.asarray(surf.faces.reshape((-1, 4))[:, 1:4], dtype=np.int64)

    return verts, faces


def mesh_skimage(F: np.ndarray, vox: float, origin: np.ndarray, iso: float):
    from skimage import measure

    verts, faces, _normals, _values = measure.marching_cubes(
        volume=F,
        level=float(iso),
        spacing=(vox, vox, vox),
    )

    verts = verts + origin.reshape((1, 3))
    faces = faces.astype(np.int64)
    return verts.astype(float), faces


def mesh_pymcubes(F: np.ndarray, vox: float, origin: np.ndarray, iso: float):
    import mcubes  # provided by PyMCubes

    verts, faces = mcubes.marching_cubes(F, float(iso))
    verts = verts * float(vox) + origin.reshape((1, 3))
    faces = faces.astype(np.int64)
    return verts.astype(float), faces


def mesh_pyiso2mesh(F: np.ndarray, vox: float, origin: np.ndarray, iso: float,
                    angbound_deg: float, radbound: float, distbound: float, maxnode: int):
    # --- FIX: make beads be "inside" for iso2mesh thresholding ---
    F   = -F
    iso = -iso
    
    # pyiso2mesh installs as `iso2mesh`
    from iso2mesh.core import vol2restrictedtri

    mask = (F >= float(iso))
    idx = np.argwhere(mask)
    if idx.size == 0:
        raise ValueError('iso2mesh: empty interior (no voxels with F>=isoLevel).')
    
    imgp = np.pad(F.astype(np.float32, copy=False), pad_width=1, mode='constant', constant_values=-1.0)
    cent = idx.mean(axis=0).astype(float) + 0.5 + 1.0
    
    thres = float(iso)
    brad = 2.0 * float(np.sum(np.array(imgp.shape, dtype=float) ** 2))

    # vol2restrictedtri expects MATLAB-like inputs; face indices are 1-based
    node, elem = vol2restrictedtri(
        imgp,
        thres,
        cent.tolist(),
        brad,
        float(angbound_deg),
        float(radbound),
        float(distbound),
        int(maxnode),
    )

    node = np.asarray(node, dtype=float)
    elem = np.asarray(elem, dtype=np.int64)
    if elem.shape[1] > 3:
        elem = elem[:, :3]

    faces = elem - 1

    # Same mapping as MATLAB RCPS_v2 iso2mesh branch:
    origin_corner_p = origin - 0.5 * float(vox) - 1.0 * float(vox)
    verts = origin_corner_p.reshape((1, 3)) + node * float(vox)

    return verts.astype(float), faces


def build_argparser():
    ap = argparse.ArgumentParser(
        description='Mesh a scalar field dumped from MATLAB and write STL.'
    )
    ap.add_argument('raw_file', type=Path)
    ap.add_argument('meta_file', type=Path)
    ap.add_argument('out_stl', type=Path)

    ap.add_argument('--method', required=True,
                    choices=['flying_edges', 'marching_cubes', 'contour', 'skimage', 'pymcubes', 'iso2mesh'])

    # --- ADD: RAW layout to make MATLAB<->Python axis conventions explicit ---
    ap.add_argument('--raw_layout', default='yxz', choices=['yxz', 'xyz'],
                    help="RAW layout. 'yxz' matches current MATLAB fwrite(permute(F,[2 1 3])). "
                         "'xyz' assumes MATLAB wrote fwrite(F) directly.")

    # iso2mesh knobs (ignored by other methods)
    ap.add_argument('--angbound_deg', type=float, default=25.0)
    ap.add_argument('--radbound', type=float, default=1.0)
    ap.add_argument('--distbound', type=float, default=0.1)
    ap.add_argument('--maxnode', type=int, default=2000000)

    # write 3mf
    ap.add_argument('--out_3mf', type=Path, default=None,
                help='Optional output 3MF path (writes repaired mesh).')

    return ap

def export_trimesh_stl(vertices: np.ndarray, faces: np.ndarray, out_stl: Path,
                       vox_mm: float,
                       merge_tol_mm: float | None = None,
                       deg_height_mm: float | None = None,
                       do_meshfix: bool = True,
                       meshfix_joincomp: bool = True,
                       out_3mf: Path | None = None):
    import trimesh
    from trimesh.exchange.stl import export_stl

    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)

    # --- defaults tied to grid spacing ---
    if merge_tol_mm is None:
        merge_tol_mm = max(1e-6, 1e-4 * float(vox_mm))
    if deg_height_mm is None:
        deg_height_mm = max(1e-6, 1e-3 * float(vox_mm))

    # --- remove duplicate faces (order-independent) ---
    Fkey = np.sort(F, axis=1)
    _, keep = np.unique(Fkey, axis=0, return_index=True)
    F = F[np.sort(keep)]

    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)

    # --- merge near-duplicate vertices via decimal rounding ---
    # trimesh.merge_vertices uses digit rounding; convert tol to digits.
    digits = int(np.ceil(-np.log10(merge_tol_mm))) if merge_tol_mm > 0 else None
    if digits is not None:
        mesh.merge_vertices(digits_vertex=digits)

    # --- remove degenerate triangles based on minimum height ---
    ok = trimesh.triangles.nondegenerate(mesh.triangles, height=float(deg_height_mm))
    if ok is not None and ok.size == mesh.faces.shape[0]:
        mesh.update_faces(ok)
        mesh.remove_unreferenced_vertices()

    # --- remove any remaining duplicates created by merging ---
    #mesh.remove_duplicate_faces()
    #mesh.remove_unreferenced_vertices()

    # --- optional MeshFix repair (safe API; do NOT drop smallest components) ---
    if do_meshfix:
        try:
            from pymeshfix import MeshFix
            mf = MeshFix(mesh.vertices, mesh.faces)
            # IMPORTANT: keep all components (porous bead packs may be multi-component)
            mf.repair(joincomp=meshfix_joincomp, remove_smallest_components=False)
            mesh = trimesh.Trimesh(vertices=np.asarray(mf.points),
                                   faces=np.asarray(mf.faces, dtype=np.int64),
                                   process=False)
            print("MeshFix repair applied.", flush=True)
        except Exception as e:
            print(f"MeshFix skipped/failed: {type(e).__name__}: {e}", flush=True)

    # --- normals/winding consistency ---
    trimesh.repair.fix_normals(mesh, multibody=True)

    # global outward orientation check
    fc = mesh.triangles_center
    fn = mesh.face_normals
    c  = mesh.centroid
    s = np.mean(np.einsum('ij,ij->i', fn, fc - c))
    if np.isfinite(s) and (s < 0):
        mesh.invert()

    # --- diagnostics (after final geometry is established) ---
    A = 0.5 * np.linalg.norm(np.cross(mesh.triangles[:, 1] - mesh.triangles[:, 0],
                                      mesh.triangles[:, 2] - mesh.triangles[:, 0]), axis=1)
    area_tol = 1e-12
    degA = int(np.count_nonzero(A < area_tol))

    E = mesh.edges_unique_inverse
    cntE = np.bincount(E)
    n_boundary_edges = int(np.count_nonzero(cntE == 1))
    n_nonmanifold_edges = int(np.count_nonzero(cntE > 2))

    print(f"Degenerate faces (A<{area_tol:g}): {degA}", flush=True)
    print(f"Boundary edges: {n_boundary_edges}", flush=True)
    print(f"Non-manifold edges: {n_nonmanifold_edges}", flush=True)
    print(f"watertight: {bool(mesh.is_watertight)}", flush=True)

    # --- single final write ---
    #out_stl.write_bytes(export_stl(mesh))

    # --- optional 3MF write (same final mesh) ---
    if out_3mf is not None:
        from trimesh.exchange.threemf import export_3MF
        out_3mf.parent.mkdir(parents=True, exist_ok=True)
        out_3mf.write_bytes(export_3MF(mesh))
        print(f"wrote: {out_3mf}", flush=True)



def main():
    args = build_argparser().parse_args()

    nx, ny, nz, vox, origin, iso = read_meta(args.meta_file)
    F = read_raw(args.raw_file, nx, ny, nz, raw_layout=args.raw_layout)

    method = args.method.lower()

    if method in ('flying_edges', 'marching_cubes', 'contour'):
        verts, faces = mesh_pyvista(F, vox, origin, iso, method=method)

    elif method == 'skimage':
        verts, faces = mesh_skimage(F, vox, origin, iso)

    elif method == 'pymcubes':
        verts, faces = mesh_pymcubes(F, vox, origin, iso)

    elif method == 'iso2mesh':
        verts, faces = mesh_pyiso2mesh(
            F, vox, origin, iso,
            angbound_deg=args.angbound_deg,
            radbound=args.radbound,
            distbound=args.distbound,
            maxnode=args.maxnode,
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    args.out_stl.parent.mkdir(parents=True, exist_ok=True)

    export_trimesh_stl(verts, faces, args.out_stl,
                       vox_mm=vox,
                       merge_tol_mm=None,         # use defaults based on vox unless you add CLI flags
                       deg_height_mm=None,
                       do_meshfix=True,           # set False for speed if upstream mesh is clean
                       meshfix_joincomp=True,
                       out_3mf=args.out_3mf)
    
    print(f"method: {method}")
    print(f"grid: nx ny nz = {nx} {ny} {nz}")

    print(f"voxSize_mm: {vox:.15g}")
    print(f"origin_mm: {origin[0]:.15g} {origin[1]:.15g} {origin[2]:.15g}")
    print(f"isoLevel: {iso:.15g}")
    print(f"mesh: Nv={verts.shape[0]} Nf={faces.shape[0]}")
    print(f"wrote: {args.out_stl}")


if __name__ == '__main__':
    main()
