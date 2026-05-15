from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from config import OUTPUTS_DIR
from utils.logger import get_logger
log = get_logger(__name__)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='PIFuHD -> NVIDIA Audio2Face pipeline', formatter_class=argparse.RawDescriptionHelpFormatter, epilog='example: py main.py assets/h.obj -t assets/h.png --blender "C:\\Program Files\\Blender Foundation\\Blender 4.2\\blender.exe"')
    p.add_argument('obj', help='Path to PIFuHD OBJ mesh')
    p.add_argument('--texture', '-t', help='Optional texture PNG/JPG')
    p.add_argument('--wav', '-a', help='Optional WAV for lip sync generation')
    p.add_argument('--skip-cleanup', action='store_true')
    p.add_argument('--skip-retopo', action='store_true')
    p.add_argument('--skip-blendshapes', action='store_true')
    p.add_argument('--skip-export', action='store_true')
    p.add_argument('--skip-to', default=None, choices=['cleanup', 'retopo', 'blendshapes', 'export', 'viewer'], help='Skip directly to this stage (uses previous outputs)')
    p.add_argument('--formats', nargs='+', default=['fbx', 'usd'], choices=['fbx', 'usd'], help='Export formats')
    p.add_argument('--blender', default='blender', help='Path to Blender executable')
    p.add_argument('--viewer', action='store_true', help='Launch realtime viewer after pipeline')
    p.add_argument('--webcam', action='store_true', help='Enable webcam facial tracking in viewer')
    p.add_argument('--out-dir', default=str(OUTPUTS_DIR), help='Output directory')
    return p

def run_pipeline(args):
    import trimesh
    from utils.mesh_io import load_obj, save_obj
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_total = time.time()
    cleaned_path = out / '01_cleaned.obj'
    face_path = out / '02_face.obj'
    retopo_path = out / '03_face_retopo.obj'
    bs_dir = out / 'blendshapes'
    lipsync_path = out / 'lipsync_curve.json'
    skip_to = args.skip_to
    if skip_to not in ('cleanup', 'retopo', 'blendshapes', 'export', 'viewer'):
        log.info('━━━ Stage 1: Load OBJ ━━━')
        mesh = load_obj(args.obj, args.texture)
    else:
        mesh = None
    if skip_to in ('retopo', 'blendshapes', 'export', 'viewer') or args.skip_cleanup:
        log.info('  [skip] Cleanup — loading previous output')
        mesh = load_obj(cleaned_path) if cleaned_path.exists() else mesh
    else:
        log.info('━━━ Stage 2: Mesh Cleanup ━━━')
        from cleanup import clean_mesh
        mesh = clean_mesh(mesh)
        save_obj(mesh, cleaned_path)
        log.info(f'  → {cleaned_path}')
    if skip_to in ('blendshapes', 'export', 'viewer') or args.skip_retopo:
        log.info('  [skip] Retopology — loading previous output')
        face_mesh = load_obj(retopo_path) if retopo_path.exists() else mesh
        landmarks = {}
    else:
        log.info('━━━ Stage 3: Face Detection ━━━')
        from retopo import detect_face_region, retopologize
        (face_mesh, landmarks) = detect_face_region(mesh)
        save_obj(face_mesh, face_path)
        log.info('━━━ Stage 4: Retopology ━━━')
        face_mesh = retopologize(face_mesh, landmarks)
        save_obj(face_mesh, retopo_path)
        log.info(f'  → {retopo_path}')
    if skip_to in ('export', 'viewer') or args.skip_blendshapes:
        log.info(f'  [skip] Blendshapes — using existing files in {bs_dir}')
    else:
        log.info('━━━ Stage 5: Blendshape Generation ━━━')
        from blendshapes import generate_blendshapes, save_blendshapes
        if not landmarks:
            from retopo import detect_face_region
            (face_mesh, landmarks) = detect_face_region(face_mesh)
        shapes = generate_blendshapes(face_mesh, landmarks)
        save_blendshapes(shapes, bs_dir)
        log.info(f'  → {bs_dir}')
    if skip_to == 'viewer' or args.skip_export:
        log.info('  [skip] Export')
    else:
        log.info('━━━ Stage 6: Export (FBX / USD) ━━━')
        from export import export_for_audio2face, AUDIO2FACE_NOTES
        results = export_for_audio2face(mesh_path=retopo_path, blendshape_dir=bs_dir, output_dir=out, formats=args.formats, blender_exec=args.blender)
        for (fmt, p) in results.items():
            log.info(f'  [{fmt.upper()}] → {p}')
        print(AUDIO2FACE_NOTES)
    if args.wav:
        log.info('━━━ Stage 8: Lip Sync ━━━')
        from lipsync import process_wav, save_curve
        curve = process_wav(args.wav)
        save_curve(curve, lipsync_path)
        log.info(f'  → {lipsync_path}')
    if args.viewer or skip_to == 'viewer':
        log.info('━━━ Stage 7: Launching Viewer ━━━')
        from viewer import launch_viewer
        launch_viewer(mesh_path=retopo_path, blendshape_dir=bs_dir, lipsync_curve=lipsync_path if lipsync_path.exists() else None, webcam=args.webcam)
    elapsed = time.time() - t_total
    log.info(f'━━━ Pipeline complete in {elapsed:.1f}s ━━━')
    log.info(f'    Outputs in: {out}')

def main():
    parser = build_parser()
    args = parser.parse_args()
    if not Path(args.obj).exists():
        log.error(f'OBJ file not found: {args.obj}')
        sys.exit(1)
    log.info('======== PIFuHD -> Audio2Face pipeline ========')
    log.info(f'  Input OBJ : {args.obj}')
    if args.texture:
        log.info(f'  Texture   : {args.texture}')
    if args.wav:
        log.info(f'  Audio WAV : {args.wav}')
    log.info(f'  Output dir: {args.out_dir}')
    try:
        run_pipeline(args)
    except KeyboardInterrupt:
        log.warning('Interrupted by user.')
        sys.exit(0)
    except Exception as exc:
        log.exception(f'Pipeline failed: {exc}')
        sys.exit(1)
if __name__ == '__main__':
    main()
