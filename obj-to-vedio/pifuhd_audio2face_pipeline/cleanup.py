from __future__ import annotations
import argparse
import numpy as np
from pathlib import Path
from typing import Any
import trimesh
try:
    import open3d as o3d
    _HAS_O3D = True
except ImportError:
    o3d = None
    _HAS_O3D = False
from config import CLEANUP, OUTPUTS_DIR
from utils.logger import get_logger
from utils.mesh_io import load_obj, save_obj
log = get_logger(__name__)

def clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    log.info('=== Mesh Cleanup ===')
    if not _HAS_O3D:
        log.warning('open3d غير مثبت — استخدام trimesh فقط لملء الثقوب والتبسيط. للجودة الأفضل: pip install open3d')
    log.info(f'  Input : {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces')
    if CLEANUP['remove_duplicate_verts']:
        mesh = _remove_duplicates(mesh)
    if CLEANUP['fix_normals']:
        mesh = _fix_normals(mesh)
    mesh = _remove_degenerate(mesh)
    if CLEANUP['fill_holes']:
        mesh = _fill_holes(mesh)
    mesh = _decimate(mesh)
    log.info(f'  Output: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces')
    log.info('=== Cleanup complete ===')
    return mesh

def _remove_duplicates(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    thresh = CLEANUP['duplicate_threshold']
    before = len(mesh.vertices)
    digits = max(1, int(round(-np.log10(max(float(thresh), 1e-15)))))
    mesh.merge_vertices(digits_vertex=digits)
    after = len(mesh.vertices)
    log.debug(f'  Duplicate removal: {before:,} → {after:,} vertices (removed {before - after:,})')
    return mesh

def _fix_normals(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    log.debug('  Normals fixed')
    return mesh

def _remove_degenerate(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mask = mesh.area_faces > 1e-12
    before = len(mesh.faces)
    mesh.update_faces(mask)
    after = len(mesh.faces)
    if before != after:
        log.debug(f'  Removed {before - after:,} degenerate faces')
    return mesh

def _fill_holes(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    max_size = CLEANUP['max_hole_size']
    if not _HAS_O3D:
        trimesh.repair.fill_holes(mesh)
        log.debug('  Holes filled via trimesh (open3d not installed)')
        return mesh
    try:
        o3d_mesh = _trimesh_to_o3d(mesh)
        if hasattr(o3d_mesh, 'fill_holes'):
            o3d_mesh = o3d_mesh.fill_holes(hole_size=max_size)
            mesh = _o3d_to_trimesh(o3d_mesh)
            log.debug('  Holes filled via Open3D')
        else:
            raise AttributeError('fill_holes not available')
    except Exception as exc:
        log.debug(f'  Open3D hole fill unavailable ({exc}), using trimesh fallback')
        trimesh.repair.fill_holes(mesh)
        log.debug('  Holes filled via trimesh')
    return mesh

def _decimate(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    target = CLEANUP['decimate_face_count']
    if len(mesh.faces) <= target:
        log.debug(f'  Decimation skipped (already ≤ {target:,} faces)')
        return mesh
    log.debug(f'  Decimating {len(mesh.faces):,} → ~{target:,} faces …')
    if not _HAS_O3D:
        ratio = target / len(mesh.faces)
        mesh = mesh.simplify_quadric_decimation(percent=ratio)
        log.debug(f'  Decimation (trimesh only) → {len(mesh.faces):,} faces')
        return mesh
    try:
        o3d_mesh = _trimesh_to_o3d(mesh)
        o3d_dec = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=target)
        mesh = _o3d_to_trimesh(o3d_dec)
        log.debug(f'  Decimation done → {len(mesh.faces):,} faces')
    except Exception as exc:
        log.warning(f'  Open3D decimation failed ({exc}), using trimesh fallback')
        ratio = target / len(mesh.faces)
        mesh = mesh.simplify_quadric_decimation(percent=ratio)
        log.debug(f'  Trimesh decimation → {len(mesh.faces):,} faces')
    return mesh

def _trimesh_to_o3d(mesh: trimesh.Trimesh) -> Any:
    assert o3d is not None
    o = o3d.geometry.TriangleMesh()
    o.vertices = o3d.utility.Vector3dVector(mesh.vertices.astype(np.float64))
    o.triangles = o3d.utility.Vector3iVector(mesh.faces.astype(np.int32))
    o.compute_vertex_normals()
    return o

def _o3d_to_trimesh(o: Any) -> trimesh.Trimesh:
    verts = np.asarray(o.vertices, dtype=np.float64)
    faces = np.asarray(o.triangles, dtype=np.int32)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)

def _cli():
    parser = argparse.ArgumentParser(description='Stage 2: Mesh Cleanup')
    parser.add_argument('input', help='Input OBJ path')
    parser.add_argument('--output', default=str(OUTPUTS_DIR / 'cleaned.obj'))
    args = parser.parse_args()
    mesh = load_obj(args.input)
    cleaned = clean_mesh(mesh)
    save_obj(cleaned, args.output)
    log.info(f'Saved cleaned mesh → {args.output}')
if __name__ == '__main__':
    _cli()
