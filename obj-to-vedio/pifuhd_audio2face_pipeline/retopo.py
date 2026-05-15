from __future__ import annotations
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import trimesh
import scipy.spatial as spatial
from config import FACE, RETOPO, OUTPUTS_DIR
from utils.logger import get_logger
from utils.mesh_io import load_obj, save_obj
log = get_logger(__name__)
Landmarks = Dict[str, np.ndarray]

def detect_face_region(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, Landmarks]:
    log.info('=== Face Detection ===')
    face_mask = _head_mask(mesh)
    face_mesh = _submesh_from_mask(mesh, face_mask)
    log.info(f'  Face region: {len(face_mesh.vertices):,} vertices, {len(face_mesh.faces):,} faces')
    landmarks = _detect_landmarks(face_mesh)
    log.info(f'  Landmarks detected: {len(landmarks)}')
    return (face_mesh, landmarks)

def retopologize(face_mesh: trimesh.Trimesh, landmarks: Landmarks) -> trimesh.Trimesh:
    log.info('=== Retopology ===')
    target_verts = RETOPO['target_verts']
    retopo_mesh = _uniform_remesh(face_mesh, target_verts)
    retopo_mesh = _enforce_symmetry(retopo_mesh)
    retopo_mesh = _sharpen_loops(retopo_mesh, landmarks)
    log.info(f'  Retopo result: {len(retopo_mesh.vertices):,} verts, {len(retopo_mesh.faces):,} faces')
    return retopo_mesh

def _head_mask(mesh: trimesh.Trimesh) -> np.ndarray:
    verts = mesh.vertices
    (y_min, y_max) = (verts[:, 1].min(), verts[:, 1].max())
    height = y_max - y_min
    head_y_thresh = y_min + height * FACE['head_y_fraction']
    mask = verts[:, 1] >= head_y_thresh
    log.debug(f'  Y threshold for head: {head_y_thresh:.3f}  ({mask.sum():,} / {len(verts):,} vertices)')
    return mask

def _detect_landmarks(face_mesh: trimesh.Trimesh) -> Landmarks:
    backend = FACE['landmark_backend']
    if backend == 'mediapipe':
        lm = _landmarks_mediapipe(face_mesh)
        if lm:
            return lm
        log.warning('  MediaPipe failed — falling back to heuristic')
    elif backend == 'insightface':
        lm = _landmarks_insightface(face_mesh)
        if lm:
            return lm
        log.warning('  InsightFace failed — falling back to heuristic')
    return _landmarks_heuristic(face_mesh)

def _landmarks_mediapipe(face_mesh: trimesh.Trimesh) -> Optional[Landmarks]:
    try:
        import mediapipe as mp
        import cv2
        img = _render_face_silhouette(face_mesh, size=512)
        if img is None:
            return None
        mp_face = mp.solutions.face_mesh
        with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as fm:
            results = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            log.debug('  MediaPipe: no face detected in projection')
            return None
        raw = results.multi_face_landmarks[0].landmark
        (h, w) = img.shape[:2]
        verts = face_mesh.vertices
        bb_min = verts.min(axis=0)
        bb_max = verts.max(axis=0)

        def unproject(lm):
            (nx, ny, nz) = (lm.x, lm.y, lm.z)
            x = bb_min[0] + nx * (bb_max[0] - bb_min[0])
            y = bb_max[1] - ny * (bb_max[1] - bb_min[1])
            z = bb_min[2] + (nz + 0.5) * (bb_max[2] - bb_min[2])
            return np.array([x, y, z])
        kd = spatial.cKDTree(verts)
        idx = {i: kd.query(unproject(raw[i]))[1] for i in range(len(raw))}
        MP = _mp_indices()
        landmarks: Landmarks = {}
        for (name, mi) in MP.items():
            if mi < len(raw):
                landmarks[name] = verts[idx[mi]]
        log.info(f'  MediaPipe: {len(landmarks)} landmarks extracted')
        return landmarks
    except ImportError:
        log.debug('  mediapipe not installed')
        return None
    except Exception as exc:
        log.debug(f'  MediaPipe error: {exc}')
        return None

def _landmarks_insightface(face_mesh: trimesh.Trimesh) -> Optional[Landmarks]:
    try:
        import insightface
        from insightface.app import FaceAnalysis
        img = _render_face_silhouette(face_mesh, size=512)
        if img is None:
            return None
        app = FaceAnalysis(allowed_modules=['detection', 'landmark_3d_68'])
        app.prepare(ctx_id=0, det_size=(512, 512))
        faces = app.get(img)
        if not faces:
            return None
        pts = faces[0].landmark_3d_68
        verts = face_mesh.vertices
        kd = spatial.cKDTree(verts)
        INS = _insightface_indices()
        landmarks: Landmarks = {}
        for (name, ii) in INS.items():
            if ii < len(pts):
                pt = pts[ii]
                (_, vi) = kd.query(pt)
                landmarks[name] = verts[vi]
        log.info(f'  InsightFace: {len(landmarks)} landmarks')
        return landmarks
    except ImportError:
        log.debug('  insightface not installed')
        return None
    except Exception as exc:
        log.debug(f'  InsightFace error: {exc}')
        return None

def _landmarks_heuristic(face_mesh: trimesh.Trimesh) -> Landmarks:
    v = face_mesh.vertices
    bb = (v.min(axis=0), v.max(axis=0))
    (lo, hi) = bb
    cx = (lo[0] + hi[0]) / 2
    fy = hi[1]

    def near(target, radius=0.015):
        (_, idx) = spatial.cKDTree(v).query(target)
        return v[idx]
    scale = hi - lo
    lm: Landmarks = {'left_eye_center': near([cx - scale[0] * 0.18, fy - scale[1] * 0.3, hi[2] * 0.9]), 'right_eye_center': near([cx + scale[0] * 0.18, fy - scale[1] * 0.3, hi[2] * 0.9]), 'left_eye_inner': near([cx - scale[0] * 0.1, fy - scale[1] * 0.3, hi[2] * 0.9]), 'right_eye_inner': near([cx + scale[0] * 0.1, fy - scale[1] * 0.3, hi[2] * 0.9]), 'left_eye_outer': near([cx - scale[0] * 0.26, fy - scale[1] * 0.3, hi[2] * 0.88]), 'right_eye_outer': near([cx + scale[0] * 0.26, fy - scale[1] * 0.3, hi[2] * 0.88]), 'nose_tip': near([cx, fy - scale[1] * 0.5, hi[2]]), 'nose_base': near([cx, fy - scale[1] * 0.58, hi[2] * 0.92]), 'mouth_center': near([cx, fy - scale[1] * 0.7, hi[2] * 0.92]), 'mouth_left': near([cx - scale[0] * 0.14, fy - scale[1] * 0.7, hi[2] * 0.9]), 'mouth_right': near([cx + scale[0] * 0.14, fy - scale[1] * 0.7, hi[2] * 0.9]), 'mouth_top': near([cx, fy - scale[1] * 0.66, hi[2] * 0.93]), 'mouth_bottom': near([cx, fy - scale[1] * 0.74, hi[2] * 0.91]), 'chin': near([cx, fy - scale[1] * 0.9, hi[2] * 0.88]), 'left_brow_center': near([cx - scale[0] * 0.18, fy - scale[1] * 0.22, hi[2] * 0.89]), 'right_brow_center': near([cx + scale[0] * 0.18, fy - scale[1] * 0.22, hi[2] * 0.89]), 'left_cheek': near([cx - scale[0] * 0.28, fy - scale[1] * 0.55, hi[2] * 0.88]), 'right_cheek': near([cx + scale[0] * 0.28, fy - scale[1] * 0.55, hi[2] * 0.88])}
    log.info(f'  Heuristic: {len(lm)} landmarks estimated')
    return lm

def _uniform_remesh(mesh: trimesh.Trimesh, target_verts: int) -> trimesh.Trimesh:
    try:
        import open3d as o3d
        o = o3d.geometry.TriangleMesh()
        o.vertices = o3d.utility.Vector3dVector(mesh.vertices.astype(np.float64))
        o.triangles = o3d.utility.Vector3iVector(mesh.faces.astype(np.int32))
        o.compute_vertex_normals()
        bbox = mesh.bounding_box.extents
        voxel = float(np.mean(bbox)) / (target_verts ** (1 / 3) * 1.5)
        pcd = o.sample_points_uniformly(number_of_points=target_verts * 4)
        pcd.estimate_normals()
        (o_new, _) = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        o_new = o_new.simplify_quadric_decimation(target_number_of_triangles=target_verts * 2)
        verts = np.asarray(o_new.vertices, dtype=np.float64)
        faces = np.asarray(o_new.triangles, dtype=np.int32)
        result = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
        log.debug(f'  Remeshed → {len(result.vertices):,} verts')
        return result
    except Exception as exc:
        log.warning(f'  Open3D remesh failed ({exc}), using trimesh fallback')
        return trimesh.remesh.subdivide_to_size(mesh, max_edge=np.mean(mesh.edges_unique_length) * 0.7)

def _enforce_symmetry(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    axis = RETOPO['symmetry_axis']
    verts = mesh.vertices.copy()
    kd = spatial.cKDTree(verts)
    thresh = np.mean(mesh.edges_unique_length) * 2.0
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
    for (i, v) in enumerate(verts):
        mirrored = v.copy()
        mirrored[axis_idx] *= -1
        (dist, j) = kd.query(mirrored)
        if dist < thresh and j != i:
            avg = (v + verts[j]) / 2
            avg_mirrored = avg.copy()
            avg_mirrored[axis_idx] *= -1
            verts[i] = avg
            verts[j] = avg_mirrored
    mesh.vertices = verts
    return mesh

def _sharpen_loops(mesh: trimesh.Trimesh, landmarks: Landmarks) -> trimesh.Trimesh:
    verts = mesh.vertices.copy()
    kd = spatial.cKDTree(verts)
    loop_targets = {'eyes': ['left_eye_center', 'right_eye_center'], 'mouth': ['mouth_center']}
    n_loops = {'eyes': RETOPO['edge_loops_eyes'], 'mouth': RETOPO['edge_loops_mouth']}
    radii = {'eyes': 0.025, 'mouth': 0.04}
    for (region, lm_names) in loop_targets.items():
        for lm_name in lm_names:
            if lm_name not in landmarks:
                continue
            centre = landmarks[lm_name]
            n = n_loops[region]
            r_max = radii[region]
            idxs = kd.query_ball_point(centre, r_max)
            for idx in idxs:
                v = verts[idx]
                d = np.linalg.norm(v - centre)
                if d < 1e-06:
                    continue
                t = d / r_max
                snap_r = np.round(t * n) / n * r_max
                snap_r = max(snap_r, r_max / n)
                direction = (v - centre) / d
                verts[idx] = centre + direction * snap_r
    mesh.vertices = verts
    return mesh

def _render_face_silhouette(mesh: trimesh.Trimesh, size: int=512):
    try:
        import pyrender
        import cv2
        scene = pyrender.Scene()
        tm = pyrender.Mesh.from_trimesh(mesh)
        scene.add(tm)
        bb = mesh.bounding_box.bounds
        c = (bb[0] + bb[1]) / 2
        cam = pyrender.OrthographicCamera(xmag=0.25, ymag=0.25)
        T = np.eye(4)
        T[:3, 3] = [c[0], c[1], c[2] + 0.5]
        scene.add(cam, pose=T)
        light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2.0)
        scene.add(light, pose=T)
        r = pyrender.OffscreenRenderer(size, size)
        (img, _) = r.render(scene)
        r.delete()
        return img.astype(np.uint8)
    except Exception as exc:
        log.debug(f'  Render failed: {exc}')
        return None

def _mp_indices() -> dict:
    return {'left_eye_center': 159, 'right_eye_center': 386, 'left_eye_inner': 133, 'right_eye_inner': 362, 'left_eye_outer': 33, 'right_eye_outer': 263, 'nose_tip': 4, 'nose_base': 195, 'mouth_center': 13, 'mouth_left': 61, 'mouth_right': 291, 'mouth_top': 0, 'mouth_bottom': 17, 'chin': 152, 'left_brow_center': 70, 'right_brow_center': 300, 'left_cheek': 234, 'right_cheek': 454}

def _insightface_indices() -> dict:
    return {'left_eye_center': 42, 'right_eye_center': 45, 'nose_tip': 30, 'mouth_left': 48, 'mouth_right': 54, 'mouth_top': 51, 'mouth_bottom': 57, 'chin': 8, 'left_brow_center': 19, 'right_brow_center': 24}

def _submesh_from_mask(mesh: trimesh.Trimesh, vertex_mask: np.ndarray) -> trimesh.Trimesh:
    vert_indices = np.where(vertex_mask)[0]
    vert_set = set(vert_indices)
    face_mask = np.array([f[0] in vert_set and f[1] in vert_set and (f[2] in vert_set) for f in mesh.faces])
    sub = mesh.submesh([np.where(face_mask)[0]], append=True)
    return sub

def _cli():
    parser = argparse.ArgumentParser(description='Stage 3-4: Face Detection & Retopology')
    parser.add_argument('input', help='Input (cleaned) OBJ')
    parser.add_argument('--output-face', default=str(OUTPUTS_DIR / 'face.obj'))
    parser.add_argument('--output-retopo', default=str(OUTPUTS_DIR / 'face_retopo.obj'))
    args = parser.parse_args()
    from utils.mesh_io import load_obj, save_obj
    mesh = load_obj(args.input)
    (face_mesh, lm) = detect_face_region(mesh)
    retopo_mesh = retopologize(face_mesh, lm)
    save_obj(face_mesh, args.output_face)
    save_obj(retopo_mesh, args.output_retopo)
    log.info('Retopology complete.')
if __name__ == '__main__':
    _cli()
