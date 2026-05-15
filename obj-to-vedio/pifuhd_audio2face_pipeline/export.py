from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from config import BLENDSHAPE_NAMES, EXPORT, OUTPUTS_DIR
from utils.logger import get_logger
log = get_logger(__name__)

def _blender_version_key(folder_name: str) -> tuple[int, int]:
    m = re.search('(\\d+)\\.(\\d+)', folder_name)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return (0, 0)

def _discover_blender_windows() -> Optional[Path]:
    roots: list[Path] = []
    for env in ('ProgramFiles', 'ProgramFiles(x86)'):
        base = os.environ.get(env)
        if base:
            roots.append(Path(base) / 'Blender Foundation')
    la = os.environ.get('LOCALAPPDATA')
    if la:
        roots.append(Path(la) / 'Programs')
    candidates: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        try:
            for sub in root.iterdir():
                if not sub.is_dir():
                    continue
                name_l = sub.name.lower()
                if 'blender' not in name_l and root.name != 'Programs':
                    continue
                exe = sub / 'blender.exe'
                if exe.is_file():
                    candidates.append(exe)
        except OSError:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda p: _blender_version_key(p.parent.name), reverse=True)
    return candidates[0]

def _try_blender_path(raw: Optional[str]) -> Optional[Path]:
    if not raw:
        return None
    s = raw.strip().strip('"')
    p = Path(s)
    if p.is_file():
        return p
    if (p / 'blender.exe').is_file():
        return p / 'blender.exe'
    if os.name == 'nt' and (p / 'Blender.exe').is_file():
        return p / 'Blender.exe'
    which = shutil.which(s)
    if which:
        return Path(which)
    return None

def resolve_blender_executable(blender_exec: str='blender') -> Optional[Path]:
    for candidate in (blender_exec, os.environ.get('BLENDER_EXE'), os.environ.get('BLENDER_PATH')):
        hit = _try_blender_path(candidate)
        if hit is not None:
            return hit
    if os.name == 'nt':
        return _discover_blender_windows()
    return None

def _blender_missing_instructions() -> str:
    return 'لم يُعثر على Blender.\n\nعلى Windows: ثبّت Blender من https://www.blender.org/download/ ثم إما:\n  • أضف مجلد التثبيت (المجلد الذي فيه blender.exe) إلى PATH، أو\n  • مرّر المسار الكامل:  --blender "C:\\Program Files\\Blender Foundation\\Blender 4.2\\blender.exe"\n  • أو عيّن متغير بيئة:  set BLENDER_EXE=المسار\\blender.exe\n\nملاحظة: حزمة pip اسمها `bpy` لا تدعم معظم إصدارات Windows؛ استخدم البرنامج الرسمي.'

def export_for_audio2face(mesh_path: str | Path, blendshape_dir: str | Path, output_dir: str | Path=OUTPUTS_DIR, formats: list[str] | None=None, blender_exec: str='blender') -> Dict[str, Path]:
    if formats is None:
        formats = ['fbx', 'usd']
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    script = _build_blender_script(mesh_path=Path(mesh_path), blendshape_dir=Path(blendshape_dir), output_dir=output_dir, formats=formats)
    script_path = output_dir / '_blender_export_script.py'
    script_path.write_text(script, encoding='utf-8')
    log.info(f'Blender export script written → {script_path}')
    result_paths = _run_blender(blender_exec, script_path, output_dir)
    return result_paths

def _build_blender_script(mesh_path: Path, blendshape_dir: Path, output_dir: Path, formats: list[str]) -> str:
    fbx_out = str(output_dir / 'head_audio2face.fbx')
    usd_out = str(output_dir / 'head_audio2face.usdc')
    mesh_name = EXPORT['mesh_name']
    armature_name = EXPORT['armature_name']
    root_bone = EXPORT['root_bone_name']
    bs_names_json = json.dumps(BLENDSHAPE_NAMES)
    script = f'''\n"""\nAuto-generated Blender headless script.\nRun via:  blender --background --python <this_file>\n"""\nimport bpy\nimport bmesh\nimport numpy as np\nimport json\nfrom pathlib import Path\nfrom mathutils import Matrix, Vector\n\nprint("=== Audio2Face Blender Export Script ===")\n\n# ── Constants ────────────────────────────────────────────────────────────────\nMESH_PATH       = r"{mesh_path}"\nBS_DIR          = Path(r"{blendshape_dir}")\nFBX_OUT         = r"{fbx_out}"\nUSD_OUT         = r"{usd_out}"\nMESH_NAME       = "{mesh_name}"\nARMATURE_NAME   = "{armature_name}"\nROOT_BONE_NAME  = "{root_bone}"\nBS_NAMES        = {bs_names_json}\nDO_FBX          = {'fbx' in formats}\nDO_USD          = {'usd' in formats}\n\n# ── 1. Clean default scene ────────────────────────────────────────────────────\nbpy.ops.object.select_all(action="SELECT")\nbpy.ops.object.delete(use_global=False)\n\n# ── 2. Import OBJ ────────────────────────────────────────────────────────────\nprint(f"Importing OBJ: {{MESH_PATH}}")\nif MESH_PATH.endswith(".obj"):\n    bpy.ops.wm.obj_import(filepath=MESH_PATH)\nelse:\n    raise RuntimeError(f"Unsupported mesh format: {{MESH_PATH}}")\n\n# Get the imported mesh object\nmesh_obj = None\nfor obj in bpy.context.scene.objects:\n    if obj.type == "MESH":\n        mesh_obj = obj\n        break\n\nif mesh_obj is None:\n    raise RuntimeError("No mesh object found after import!")\n\nmesh_obj.name      = MESH_NAME\nmesh_obj.data.name = MESH_NAME + "_Data"\nbpy.context.view_layer.objects.active = mesh_obj\nprint(f"Mesh: {{len(mesh_obj.data.vertices)}} verts, {{len(mesh_obj.data.polygons)}} faces")\n\n# ── 3. Inject Shape Keys ──────────────────────────────────────────────────────\nprint("Injecting blendshapes as shape keys ...")\n\n# Basis key\nbpy.ops.object.shape_key_add(from_mix=False)\nmesh_obj.data.shape_keys.key_blocks[0].name = "Basis"\n\nbasis_verts = np.array([v.co[:] for v in mesh_obj.data.vertices])\n\nfor bs_name in BS_NAMES:\n    npy_path = BS_DIR / f"{{bs_name}}.npy"\n    if not npy_path.exists():\n        print(f"  WARNING: {{bs_name}}.npy not found — skipping")\n        continue\n\n    displaced = np.load(str(npy_path))\n    if displaced.shape[0] != len(mesh_obj.data.vertices):\n        print(f"  WARNING: {{bs_name}} vertex count mismatch — skipping")\n        continue\n\n    bpy.ops.object.shape_key_add(from_mix=False)\n    sk = mesh_obj.data.shape_keys.key_blocks[-1]\n    sk.name  = bs_name\n    sk.value = 0.0\n\n    # Apply displacements\n    for i, coord in enumerate(displaced):\n        sk.data[i].co = coord.tolist()\n\n    print(f"  Shape key added: {{bs_name}}")\n\nprint(f"Total shape keys: {{len(mesh_obj.data.shape_keys.key_blocks)}}")\n\n# ── 4. Create Head Armature ───────────────────────────────────────────────────\nprint("Creating head armature ...")\n\nbpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))\narm_obj      = bpy.context.active_object\narm_obj.name = ARMATURE_NAME\narm          = arm_obj.data\narm.name     = ARMATURE_NAME + "_Data"\n\n# Rename default bone to root\nbone = arm.edit_bones[0]\nbone.name   = ROOT_BONE_NAME\nbb           = mesh_obj.bound_box\nmin_y = min(v[1] for v in bb)\nmax_y = max(v[1] for v in bb)\nbone.head   = (0, min_y, 0)\nbone.tail   = (0, max_y, 0)\n\nbpy.ops.object.mode_set(mode="OBJECT")\n\n# Parent mesh to armature\nmesh_obj.select_set(True)\narm_obj.select_set(True)\nbpy.context.view_layer.objects.active = arm_obj\nbpy.ops.object.parent_set(type="ARMATURE_AUTO")\n\nprint("Armature created and mesh parented.")\n\n# ── 5. Audio2Face metadata (custom properties) ────────────────────────────────\nmesh_obj["audio2face_rig"]   = True\nmesh_obj["blendshape_count"] = len(BS_NAMES)\nmesh_obj["pipeline"]         = "pifuhd_audio2face"\n\n# ── 6. Export FBX ─────────────────────────────────────────────────────────────\nif DO_FBX:\n    print(f"Exporting FBX → {{FBX_OUT}}")\n    bpy.ops.object.select_all(action="SELECT")\n    bpy.ops.export_scene.fbx(\n        filepath            = FBX_OUT,\n        use_selection       = False,\n        use_mesh_modifiers  = True,\n        add_leaf_bones      = False,\n        bake_anim           = False,\n        mesh_smooth_type    = "FACE",\n        use_tspace          = True,\n        use_custom_props    = True,\n        path_mode           = "COPY",\n        embed_textures      = True,\n    )\n    print("FBX export done.")\n\n# ── 7. Export USD ─────────────────────────────────────────────────────────────\nif DO_USD:\n    print(f"Exporting USD → {{USD_OUT}}")\n    try:\n        bpy.ops.wm.usd_export(\n            filepath               = USD_OUT,\n            export_animation       = False,\n            export_hair            = False,\n            export_uvmaps          = True,\n            export_normals         = True,\n            export_materials       = True,\n            use_instancing         = False,\n            evaluation_mode        = "RENDER",\n        )\n        print("USD export done.")\n    except Exception as e:\n        print(f"USD export failed: {{e}} (usd_export may need Blender 4.0+)")\n\nprint("=== Blender export script complete ===")\n'''
    return script

def _run_blender(blender_exec: str, script_path: Path, output_dir: Path) -> Dict[str, Path]:
    resolved = resolve_blender_executable(blender_exec)
    if resolved is None:
        log.warning('لم يُعثر على blender في PATH أو المسارات الاعتيادية؛ محاولة استيراد bpy من بيئة بايثون الحالية (نادر على Windows)…')
        _run_script_direct(script_path)
    else:
        if str(resolved) != blender_exec.strip().strip('"'):
            log.info(f'استخدام Blender: {resolved}')
        cmd = [str(resolved), '--background', '--python', str(script_path)]
        log.info(f"Running Blender: {' '.join(cmd)}")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except FileNotFoundError as exc:
            raise RuntimeError(f'تعذر تشغيل الملف: {resolved}\n' + _blender_missing_instructions()) from exc
        if proc.returncode != 0:
            log.error(f'Blender stderr:\n{proc.stderr[-3000:]}')
            raise RuntimeError(f'Blender exited with code {proc.returncode}')
        log.info('Blender finished successfully')
        for line in proc.stdout.split('\n'):
            if line.strip():
                log.debug(f'  [blender] {line}')
    result: Dict[str, Path] = {}
    for (fmt, ext) in [('fbx', '.fbx'), ('usd', '.usdc')]:
        p = output_dir / f'head_audio2face{ext}'
        if p.exists():
            result[fmt] = p
            log.info(f'  Output [{fmt.upper()}]: {p}')
    return result

def _run_script_direct(script_path: Path) -> None:
    try:
        import bpy
        exec(script_path.read_text(encoding='utf-8'))
    except ImportError:
        raise RuntimeError(_blender_missing_instructions())
AUDIO2FACE_NOTES = '\nAudio2Face Compatibility Notes\n===============================\n1. Import the exported .fbx or .usdc into NVIDIA Audio2Face.\n2. In Audio2Face → Setup → BlendShape Solver:\n   - Select the mesh: "Head_Mesh"\n   - The tool auto-detects shape keys matching ARKit names.\n3. Blendshape names follow the ARKit 52-blendshape convention.\n   Supported by this pipeline:\n   {names}\n4. For best results, use the FBX export — Audio2Face 2023.2+ supports\n   both FBX and USD.\n5. Ensure "Head_Armature / Head" bone exists for head-pose control.\n6. To drive lip sync: Audio2Face → Audio → Load WAV, then\n   click "Generate" — it maps to the Viseme_* shape keys automatically.\n'.format(names='\n   '.join((f'- {n}' for n in BLENDSHAPE_NAMES)))

def _cli():
    parser = argparse.ArgumentParser(description='Stage 6: Audio2Face Export')
    parser.add_argument('mesh', help='Retopologised face OBJ')
    parser.add_argument('blendshape_dir', help='Directory with .npy blendshape files')
    parser.add_argument('--output-dir', default=str(OUTPUTS_DIR))
    parser.add_argument('--formats', nargs='+', default=['fbx', 'usd'], choices=['fbx', 'usd'])
    parser.add_argument('--blender', default='blender', help='Path to Blender executable')
    args = parser.parse_args()
    result = export_for_audio2face(mesh_path=args.mesh, blendshape_dir=args.blendshape_dir, output_dir=args.output_dir, formats=args.formats, blender_exec=args.blender)
    log.info('Export complete:')
    for (fmt, path) in result.items():
        log.info(f'  {fmt.upper()}: {path}')
    print(AUDIO2FACE_NOTES)
if __name__ == '__main__':
    _cli()
