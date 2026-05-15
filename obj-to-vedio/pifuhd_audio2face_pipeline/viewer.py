from __future__ import annotations
import argparse
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from config import BLENDSHAPE_NAMES, OUTPUTS_DIR
from utils.logger import get_logger
log = get_logger(__name__)

def launch_viewer(mesh_path: str | Path, blendshape_dir: str | Path, lipsync_curve: Optional[str | Path]=None, webcam: bool=False):
    log.info('=== Launching Viewer ===')
    (mesh, bs) = _load_assets(mesh_path, blendshape_dir)
    backend = _detect_backend()
    log.info(f'  Render backend: {backend}')
    if backend == 'pyrender':
        _viewer_pyrender(mesh, bs, lipsync_curve, webcam)
    elif backend == 'open3d':
        _viewer_open3d(mesh, bs, lipsync_curve, webcam)
    else:
        log.error('No 3-D render backend found. Install pyrender or open3d.')
        sys.exit(1)

def _load_assets(mesh_path, blendshape_dir):
    import trimesh
    from blendshapes import load_blendshapes
    mesh = trimesh.load(str(mesh_path), force='mesh', process=False)
    bs = load_blendshapes(blendshape_dir)
    log.info(f'  Mesh: {len(mesh.vertices):,} verts | Blendshapes: {len(bs)}')
    return (mesh, bs)

def _detect_backend() -> str:
    try:
        import pyrender
        return 'pyrender'
    except ImportError:
        pass
    try:
        import open3d
        return 'open3d'
    except ImportError:
        pass
    return 'none'

def _viewer_pyrender(mesh, bs, lipsync_curve, webcam):
    import pyrender
    import trimesh
    scene = pyrender.Scene(bg_color=[0.15, 0.15, 0.15, 1.0])
    basis_v = mesh.vertices.copy()
    weights = {n: 0.0 for n in BLENDSHAPE_NAMES}
    pr_mesh = pyrender.Mesh.from_trimesh(mesh)
    node = scene.add(pr_mesh)
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    bb = mesh.bounding_box.bounds
    c = (bb[0] + bb[1]) / 2
    dist = np.linalg.norm(bb[1] - bb[0]) * 1.5
    cam_T = np.eye(4)
    cam_T[:3, 3] = [c[0], c[1], c[2] + dist]
    scene.add(cam, pose=cam_T)
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0)
    scene.add(light, pose=cam_T)
    state = {'weights': weights, 'running': True, 'anim_cursor': 0, 'anim_curve': None}
    if lipsync_curve:
        from lipsync import load_curve
        (fps, curve) = load_curve(lipsync_curve)
        state['anim_curve'] = curve
        log.info(f'  Lipsync curve loaded: {len(curve)} frames @ {fps} fps')
    cam_thread = None
    if webcam:
        cam_thread = threading.Thread(target=_webcam_driver, args=(state,), daemon=True)
        cam_thread.start()
    kb_thread = threading.Thread(target=_keyboard_driver, args=(state,), daemon=True)
    kb_thread.start()
    viewer = pyrender.Viewer(scene, use_direct_lighting=True, run_in_thread=True, window_title='PIFuHD → Audio2Face Viewer')
    fps_target = 30
    dt = 1.0 / fps_target
    log.info('  Viewer running. Press Ctrl+C to quit.')
    try:
        while viewer.is_active and state['running']:
            t0 = time.time()
            if state['anim_curve']:
                ci = state['anim_cursor'] % len(state['anim_curve'])
                state['weights'].update(state['anim_curve'][ci])
                state['anim_cursor'] += 1
            blended = _blend_verts(basis_v, bs, state['weights'])
            mesh.vertices = blended
            new_pr = pyrender.Mesh.from_trimesh(mesh)
            with viewer.render_lock:
                scene.remove_node(node)
                node = scene.add(new_pr)
            elapsed = time.time() - t0
            time.sleep(max(0, dt - elapsed))
    except KeyboardInterrupt:
        pass
    state['running'] = False
    viewer.close_external()
    log.info('  Viewer closed.')

def _viewer_open3d(mesh, bs, lipsync_curve, webcam):
    import open3d as o3d
    verts = mesh.vertices.copy()
    faces = mesh.faces.copy()
    basis_v = verts.copy()
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    o3d_mesh.compute_vertex_normals()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window('PIFuHD → Audio2Face (Open3D)', width=1024, height=768)
    vis.add_geometry(o3d_mesh)
    weights = {n: 0.0 for n in BLENDSHAPE_NAMES}
    state = {'weights': weights, 'running': True, 'bs_idx': 0}

    def on_space(vis):
        names = list(bs.keys())
        old_n = names[state['bs_idx'] % len(names)]
        weights[old_n] = 0.0
        state['bs_idx'] = (state['bs_idx'] + 1) % len(names)
        new_n = names[state['bs_idx']]
        weights[new_n] = 1.0
        log.info(f'  Blendshape: {new_n}')
        _update_o3d(o3d_mesh, basis_v, bs, weights, vis)

    def on_r(vis):
        for k in weights:
            weights[k] = 0.0
        log.info('  Reset blendshapes')
        _update_o3d(o3d_mesh, basis_v, bs, weights, vis)
    vis.register_key_callback(ord(' '), on_space)
    vis.register_key_callback(ord('R'), on_r)
    if webcam:
        t = threading.Thread(target=_webcam_driver, args=(state,), daemon=True)
        t.start()
    log.info('  Open3D viewer: SPACE = next blendshape, R = reset, Q = quit')
    vis.run()
    vis.destroy_window()
    state['running'] = False

def _update_o3d(o3d_mesh, basis_v, bs, weights, vis):
    import open3d as o3d
    blended = _blend_verts(basis_v, bs, weights)
    o3d_mesh.vertices = o3d.utility.Vector3dVector(blended.astype(np.float64))
    o3d_mesh.compute_vertex_normals()
    vis.update_geometry(o3d_mesh)
    vis.poll_events()
    vis.update_renderer()

def _blend_verts(basis: np.ndarray, bs: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    result = basis.copy()
    for (name, w) in weights.items():
        if w == 0.0 or name not in bs:
            continue
        delta = bs[name] - basis
        result += w * delta
    return result

def _keyboard_driver(state: dict):
    print('\n── Blendshape Control ──────────────────────────────────────')
    print('  Commands:  bs <NAME> <0-1>  |  reset  |  list  |  quit')
    print('  Example:   bs JawOpen 0.8')
    print('─────────────────────────────────────────────────────────────\n')
    while state['running']:
        try:
            line = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            state['running'] = False
            break
        parts = line.split()
        if not parts:
            continue
        cmd = parts[0].lower()
        if cmd == 'quit':
            state['running'] = False
        elif cmd == 'reset':
            for k in state['weights']:
                state['weights'][k] = 0.0
            print('  All blendshapes reset.')
        elif cmd == 'list':
            for n in BLENDSHAPE_NAMES:
                print(f"  {n}: {state['weights'].get(n, 0.0):.2f}")
        elif cmd == 'bs' and len(parts) == 3:
            name = parts[1]
            try:
                w = float(parts[2])
            except ValueError:
                print('  Weight must be a number 0.0 – 1.0')
                continue
            if name not in BLENDSHAPE_NAMES:
                print(f'  Unknown blendshape: {name}')
                continue
            state['weights'][name] = np.clip(w, 0.0, 1.0)
            print(f'  Set {name} = {w:.2f}')
        else:
            print('  Unknown command.')

def _webcam_driver(state: dict):
    try:
        import cv2
        import mediapipe as mp
    except ImportError:
        log.error('opencv-python or mediapipe not installed — webcam mode disabled')
        return
    log.info('  Webcam tracking started')
    mp_face = mp.solutions.face_mesh
    cap = cv2.VideoCapture(0)
    with mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as fm:
        while state['running']:
            (ok, frame) = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = fm.process(rgb)
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                weights = _landmarks_to_weights(lm)
                state['weights'].update(weights)
                _draw_overlay(frame, lm, weights)
            cv2.imshow('Webcam Tracking', frame)
            if cv2.waitKey(1) & 255 == ord('q'):
                state['running'] = False
                break
    cap.release()
    cv2.destroyAllWindows()
    log.info('  Webcam tracking stopped')

def _landmarks_to_weights(lm) -> Dict[str, float]:

    def pt(i):
        return np.array([lm[i].x, lm[i].y, lm[i].z])
    weights: Dict[str, float] = {}
    upper_lip = pt(13)
    lower_lip = pt(14)
    jaw_dist = np.linalg.norm(upper_lip - lower_lip)
    weights['JawOpen'] = float(np.clip(jaw_dist * 25.0, 0, 1))
    left_open = np.linalg.norm(pt(159) - pt(145))
    right_open = np.linalg.norm(pt(386) - pt(374))
    left_w = np.linalg.norm(pt(33) - pt(133))
    right_w = np.linalg.norm(pt(362) - pt(263))
    left_ratio = left_open / (left_w + 1e-06)
    right_ratio = right_open / (right_w + 1e-06)
    weights['EyeBlinkLeft'] = float(np.clip(1.0 - left_ratio * 5, 0, 1))
    weights['EyeBlinkRight'] = float(np.clip(1.0 - right_ratio * 5, 0, 1))
    mouth_w = np.linalg.norm(pt(61) - pt(291))
    face_w = np.linalg.norm(pt(234) - pt(454))
    smile_v = float(np.clip((mouth_w / (face_w + 1e-06) - 0.35) * 5, 0, 1))
    weights['MouthSmileLeft'] = smile_v
    weights['MouthSmileRight'] = smile_v
    pucker = float(np.clip(0.4 - mouth_w / (face_w + 1e-06), 0, 1) * 4)
    weights['MouthPucker'] = pucker
    left_brow = pt(70)[1]
    left_eye_y = pt(159)[1]
    brow_raise = float(np.clip((left_eye_y - left_brow - 0.03) * 20, 0, 1))
    weights['BrowInnerUp'] = brow_raise
    weights['BrowDown_L'] = float(np.clip(-brow_raise + 0.3, 0, 1))
    weights['BrowDown_R'] = weights['BrowDown_L']
    jaw_w = weights['JawOpen']
    if jaw_w > 0.7:
        weights['Viseme_AA'] = (jaw_w - 0.7) / 0.3
    elif jaw_w > 0.4:
        weights['Viseme_OH'] = (jaw_w - 0.4) / 0.3
    weights['Viseme_MP'] = float(np.clip(1.0 - jaw_w * 4, 0, 1))
    weights['Viseme_EE'] = smile_v * (1.0 - jaw_w)
    return weights

def _draw_overlay(frame, lm, weights: Dict[str, float]):
    import cv2
    (h, w) = frame.shape[:2]
    active = [(k, v) for (k, v) in weights.items() if v > 0.1]
    active.sort(key=lambda x: -x[1])
    for (i, (k, v)) in enumerate(active[:6]):
        bar_len = int(v * 150)
        y = 20 + i * 22
        cv2.rectangle(frame, (10, y - 12), (10 + bar_len, y + 4), (0, 200, 100), -1)
        cv2.putText(frame, f'{k}: {v:.2f}', (170, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

def _cli():
    parser = argparse.ArgumentParser(description='Stage 7: Realtime Viewer')
    parser.add_argument('mesh', help='Retopologised face OBJ')
    parser.add_argument('blendshape_dir', help='Directory with .npy blendshape files')
    parser.add_argument('--lipsync', help='JSON lipsync curve to play back')
    parser.add_argument('--webcam', action='store_true', help='Drive blendshapes from webcam facial tracking')
    args = parser.parse_args()
    launch_viewer(mesh_path=args.mesh, blendshape_dir=args.blendshape_dir, lipsync_curve=args.lipsync, webcam=args.webcam)
if __name__ == '__main__':
    _cli()
