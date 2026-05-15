from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import trimesh
from trimesh import Trimesh
from utils.logger import get_logger
log = get_logger(__name__)

def load_obj(obj_path: str | Path, texture_path: Optional[str | Path]=None) -> Trimesh:
    obj_path = Path(obj_path)
    log.info(f'Loading OBJ: {obj_path}')
    if not obj_path.exists():
        raise FileNotFoundError(f'OBJ file not found: {obj_path}')
    scene_or_mesh = trimesh.load(str(obj_path), process=False, force='mesh')
    if isinstance(scene_or_mesh, trimesh.Scene):
        log.debug('Loaded as Scene — merging geometries')
        mesh = trimesh.util.concatenate(list(scene_or_mesh.geometry.values()))
    else:
        mesh = scene_or_mesh
    log.info(f'Loaded mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces')
    if texture_path is not None:
        mesh = _apply_texture(mesh, Path(texture_path))
    _validate(mesh)
    return mesh

def _apply_texture(mesh: Trimesh, tex_path: Path) -> Trimesh:
    if not tex_path.exists():
        log.warning(f'Texture not found: {tex_path} — skipping')
        return mesh
    try:
        from PIL import Image
        img = np.array(Image.open(tex_path).convert('RGBA'))
        mat = trimesh.visual.texture.SimpleMaterial(image=img)
        if mesh.visual.uv is not None:
            mesh.visual = trimesh.visual.TextureVisuals(uv=mesh.visual.uv, material=mat)
            log.info(f'Texture applied from: {tex_path}')
        else:
            log.warning('Mesh has no UV coords — texture skipped')
    except Exception as exc:
        log.warning(f'Could not apply texture: {exc}')
    return mesh

def _validate(mesh: Trimesh) -> None:
    issues = []
    if not mesh.is_watertight:
        issues.append('mesh is NOT watertight (has holes)')
    if not mesh.is_winding_consistent:
        issues.append('winding is inconsistent (bad normals)')
    if mesh.is_empty:
        raise ValueError('Loaded mesh is empty!')
    dup_verts = len(mesh.vertices) - len(np.unique(mesh.vertices, axis=0))
    if dup_verts > 0:
        issues.append(f'{dup_verts:,} duplicate vertices detected')
    degenerate = np.sum(mesh.area_faces == 0)
    if degenerate:
        issues.append(f'{degenerate:,} degenerate (zero-area) faces')
    if issues:
        for i in issues:
            log.warning(f'  ⚠ {i}')
    else:
        log.info('Mesh validation: OK')

def save_obj(mesh: Trimesh, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))
    log.info(f'Saved OBJ → {path}')
