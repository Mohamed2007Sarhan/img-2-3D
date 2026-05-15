from __future__ import annotations
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import trimesh
import scipy.spatial as spatial
from scipy.ndimage import gaussian_filter1d
from config import BLENDSHAPE, BLENDSHAPE_NAMES, OUTPUTS_DIR
from utils.logger import get_logger
log = get_logger(__name__)
BlendshapeDict = Dict[str, np.ndarray]

def generate_blendshapes(mesh: trimesh.Trimesh, landmarks: Dict[str, np.ndarray]) -> BlendshapeDict:
    log.info('=== Blendshape Generation ===')
    verts = mesh.vertices.copy()
    bs = BlendshapeGenerator(verts, landmarks, mesh)
    shapes: BlendshapeDict = {}
    for name in BLENDSHAPE_NAMES:
        try:
            displaced = bs.build(name)
            shapes[name] = displaced
            log.debug(f'  {name}: max displacement = {np.max(np.linalg.norm(displaced - verts, axis=1)):.4f}')
        except Exception as exc:
            log.warning(f'  {name} failed: {exc} — using basis pose')
            shapes[name] = verts.copy()
    log.info(f'=== Generated {len(shapes)} blendshapes ===')
    return shapes

def save_blendshapes(shapes: BlendshapeDict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for (name, verts) in shapes.items():
        np.save(out_dir / f'{name}.npy', verts)
    log.info(f'Blendshapes saved to {out_dir}')

def load_blendshapes(out_dir: Path) -> BlendshapeDict:
    out_dir = Path(out_dir)
    shapes = {}
    for f in sorted(out_dir.glob('*.npy')):
        shapes[f.stem] = np.load(f)
    log.info(f'Loaded {len(shapes)} blendshapes from {out_dir}')
    return shapes

class BlendshapeGenerator:

    def __init__(self, verts: np.ndarray, landmarks: Dict[str, np.ndarray], mesh: trimesh.Trimesh):
        self.verts = verts.copy()
        self.lm = landmarks
        self.mesh = mesh
        self.kd = spatial.cKDTree(verts)
        self.bb_size = verts.max(axis=0) - verts.min(axis=0)
        self._cfg = BLENDSHAPE
        self.vnormals = mesh.vertex_normals.copy()

    def build(self, name: str) -> np.ndarray:
        builders = {'JawOpen': self._jaw_open, 'MouthSmileLeft': self._mouth_smile_left, 'MouthSmileRight': self._mouth_smile_right, 'EyeBlinkLeft': self._blink_left, 'EyeBlinkRight': self._blink_right, 'MouthPucker': self._mouth_pucker, 'MouthFunnel': self._mouth_funnel, 'BrowInnerUp': self._brow_inner_up, 'BrowDown_L': self._brow_down_left, 'BrowDown_R': self._brow_down_right, 'CheekPuff': self._cheek_puff, 'Viseme_AA': self._viseme_aa, 'Viseme_OH': self._viseme_oh, 'Viseme_EE': self._viseme_ee, 'Viseme_FV': self._viseme_fv, 'Viseme_MP': self._viseme_mp}
        if name not in builders:
            raise ValueError(f'Unknown blendshape: {name}')
        return builders[name]()

    def _push_region(self, centre: np.ndarray, radius: float, delta: np.ndarray, falloff: float=2.0) -> np.ndarray:
        displaced = self.verts.copy()
        idxs = self.kd.query_ball_point(centre, radius)
        for i in idxs:
            d = np.linalg.norm(self.verts[i] - centre)
            w = np.exp(-falloff * (d / radius) ** 2)
            displaced[i] += delta * w
        return displaced

    def _lm(self, *names: str) -> np.ndarray:
        for n in names:
            if n in self.lm:
                return self.lm[n]
        return self.verts.mean(axis=0)

    def _scale(self, key: str) -> float:
        return self._cfg.get(key, 0.02)

    def _jaw_open(self) -> np.ndarray:
        centre = self._lm('mouth_bottom', 'mouth_center')
        amt = self._scale('jaw_open_amount')
        return self._push_region(centre, radius=amt * 3, delta=np.array([0, -amt, 0]))

    def _mouth_smile_left(self) -> np.ndarray:
        centre = self._lm('mouth_left')
        amt = self._scale('smile_amount')
        return self._push_region(centre, radius=amt * 2.5, delta=np.array([-amt * 0.5, amt, amt * 0.3]))

    def _mouth_smile_right(self) -> np.ndarray:
        centre = self._lm('mouth_right')
        amt = self._scale('smile_amount')
        return self._push_region(centre, radius=amt * 2.5, delta=np.array([amt * 0.5, amt, amt * 0.3]))

    def _blink_left(self) -> np.ndarray:
        centre = self._lm('left_eye_center')
        amt = self._scale('blink_amount')
        displaced = self.verts.copy()
        idxs = self.kd.query_ball_point(centre, amt * 3)
        for i in idxs:
            v = self.verts[i]
            if v[1] > centre[1]:
                d = np.linalg.norm(v - centre)
                w = np.exp(-2 * (d / (amt * 3)) ** 2)
                displaced[i] += np.array([0, -amt, 0]) * w
        return displaced

    def _blink_right(self) -> np.ndarray:
        centre = self._lm('right_eye_center')
        amt = self._scale('blink_amount')
        displaced = self.verts.copy()
        idxs = self.kd.query_ball_point(centre, amt * 3)
        for i in idxs:
            v = self.verts[i]
            if v[1] > centre[1]:
                d = np.linalg.norm(v - centre)
                w = np.exp(-2 * (d / (amt * 3)) ** 2)
                displaced[i] += np.array([0, -amt, 0]) * w
        return displaced

    def _mouth_pucker(self) -> np.ndarray:
        centre = self._lm('mouth_center')
        amt = self._scale('pucker_amount')
        displaced = self.verts.copy()
        idxs = self.kd.query_ball_point(centre, amt * 3)
        for i in idxs:
            v = self.verts[i]
            d = np.linalg.norm(v - centre)
            w = np.exp(-2 * (d / (amt * 3)) ** 2)
            dx = -(v[0] - centre[0]) * 0.6
            dz = amt * 0.8
            displaced[i] += np.array([dx, 0, dz]) * w
        return displaced

    def _mouth_funnel(self) -> np.ndarray:
        centre = self._lm('mouth_center')
        amt = self._scale('funnel_amount')
        displaced = self.verts.copy()
        idxs = self.kd.query_ball_point(centre, amt * 3)
        for i in idxs:
            v = self.verts[i]
            d = np.linalg.norm(v - centre)
            w = np.exp(-2 * (d / (amt * 3)) ** 2)
            dir_out = v - centre
            if np.linalg.norm(dir_out) > 1e-06:
                dir_out /= np.linalg.norm(dir_out)
            displaced[i] += dir_out * amt * 0.4 * w
            displaced[i] += np.array([0, 0, amt * 0.3]) * w
        return displaced

    def _brow_inner_up(self) -> np.ndarray:
        lc = self._lm('left_brow_center')
        rc = self._lm('right_brow_center')
        amt = self._scale('brow_up_amount')
        inner_l = (lc + self._lm('nose_tip')) / 2
        inner_r = (rc + self._lm('nose_tip')) / 2
        d1 = self._push_region(inner_l, radius=amt * 2, delta=np.array([0, amt, amt * 0.2]))
        d2 = self._push_region(inner_r, radius=amt * 2, delta=np.array([0, amt, amt * 0.2]))
        return (d1 + d2) / 2 + self.verts / 2 - self.verts / 2

    def _brow_down_left(self) -> np.ndarray:
        centre = self._lm('left_brow_center')
        amt = self._scale('brow_down_amount')
        return self._push_region(centre, radius=amt * 3, delta=np.array([0, -amt, amt * 0.1]))

    def _brow_down_right(self) -> np.ndarray:
        centre = self._lm('right_brow_center')
        amt = self._scale('brow_down_amount')
        return self._push_region(centre, radius=amt * 3, delta=np.array([0, -amt, amt * 0.1]))

    def _cheek_puff(self) -> np.ndarray:
        lc = self._lm('left_cheek')
        rc = self._lm('right_cheek')
        amt = self._scale('cheek_puff_amount')
        dl = self._push_region(lc, radius=amt * 3, delta=np.array([-amt * 0.5, 0, amt]))
        dr = self._push_region(rc, radius=amt * 3, delta=np.array([amt * 0.5, 0, amt]))
        result = self.verts.copy()
        idxl = self.kd.query_ball_point(lc, amt * 3)
        idxr = self.kd.query_ball_point(rc, amt * 3)
        for i in idxl:
            result[i] = dl[i]
        for i in idxr:
            result[i] = dr[i]
        return result

    def _viseme_aa(self) -> np.ndarray:
        top = self._lm('mouth_top')
        bottom = self._lm('mouth_bottom')
        centre = self._lm('mouth_center')
        amt = self._scale('viseme_aa_amount')
        displaced = self.verts.copy()
        for (centre_pt, delta) in [(top, np.array([0, amt * 0.4, 0])), (bottom, np.array([0, -amt, 0]))]:
            idxs = self.kd.query_ball_point(centre_pt, amt * 2)
            for i in idxs:
                d = np.linalg.norm(self.verts[i] - centre_pt)
                w = np.exp(-2 * (d / (amt * 2)) ** 2)
                displaced[i] += delta * w
        return displaced

    def _viseme_oh(self) -> np.ndarray:
        centre = self._lm('mouth_center')
        amt = self._scale('viseme_oh_amount')
        displaced = self.verts.copy()
        idxs = self.kd.query_ball_point(centre, amt * 2.5)
        for i in idxs:
            v = self.verts[i]
            d = np.linalg.norm(v - centre)
            w = np.exp(-2 * (d / (amt * 2.5)) ** 2)
            sign = 1 if v[1] > centre[1] else -1
            displaced[i] += np.array([0, sign * amt * 0.5, amt * 0.2]) * w
        return displaced

    def _viseme_ee(self) -> np.ndarray:
        ml = self._lm('mouth_left')
        mr = self._lm('mouth_right')
        amt = self._scale('viseme_ee_amount')
        result = self.verts.copy()
        for (corner, sign) in [(ml, -1), (mr, 1)]:
            idxs = self.kd.query_ball_point(corner, amt * 2)
            for i in idxs:
                d = np.linalg.norm(self.verts[i] - corner)
                w = np.exp(-2 * (d / (amt * 2)) ** 2)
                result[i] += np.array([sign * amt, amt * 0.2, 0]) * w
        return result

    def _viseme_fv(self) -> np.ndarray:
        bottom = self._lm('mouth_bottom', 'mouth_center')
        amt = self._scale('viseme_fv_amount')
        return self._push_region(bottom, radius=amt * 2, delta=np.array([0, amt * 0.5, -amt * 0.3]))

    def _viseme_mp(self) -> np.ndarray:
        centre = self._lm('mouth_center')
        amt = self._scale('viseme_mp_amount')
        displaced = self.verts.copy()
        idxs = self.kd.query_ball_point(centre, amt * 3)
        for i in idxs:
            v = self.verts[i]
            d = np.linalg.norm(v - centre)
            w = np.exp(-2 * (d / (amt * 3)) ** 2)
            sign = 1 if v[1] > centre[1] else -1
            displaced[i] += np.array([0, -sign * amt * 0.4, 0]) * w
        return displaced

def _cli():
    parser = argparse.ArgumentParser(description='Stage 5: Blendshape Generation')
    parser.add_argument('input', help='Retopologised face OBJ')
    parser.add_argument('--out-dir', default=str(OUTPUTS_DIR / 'blendshapes'))
    args = parser.parse_args()
    from utils.mesh_io import load_obj
    from retopo import detect_face_region
    (mesh, landmarks) = detect_face_region(load_obj(args.input))
    shapes = generate_blendshapes(mesh, landmarks)
    save_blendshapes(shapes, Path(args.out_dir))
if __name__ == '__main__':
    _cli()
