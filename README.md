#  mesh + Audio2Face glue

Monorepo-ish layout on my machine: PiFuHD lives under `pifuhd/`, the retarget/export stack under `obj-to-vedio/` (yeah the folder name is typo'd, renaming paths would break scripts so it stays). Root helpers `meshforge_cli.py` / `meshforge_gui.py` drive both without cd'ing into each repo every time.

## What runs where

| Path | Role |
|------|------|
| `pifuhd/pifuhd/` | image → OBJ (`python -m apps.simple_test ...`, needs their checkpoints + deps) |
| `obj-to-vedio/pifuhd_audio2face_pipeline/` | OBJ cleanup → face retopo → blendshapes → Blender export for Audio2Face |
| `meshforge_cli.py` | wraps the two above from `Frontend/` |
| `meshforge_gui.py` | tkinter UI on top of the same calls |

## Pipeline quickstart

```powershell
cd obj-to-vedio\pifuhd_audio2face_pipeline
py -m pip install -r requirements.txt
py main.py assets\whatever.obj --texture assets\whatever.png --blender "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"
```

Blender: if `blender` isn't on PATH the code tries common Windows install dirs; still fails → set `BLENDER_EXE` to full `blender.exe` path or pass `--blender`. Don't bother with `pip install bpy` on win32, there's usually no wheel.

## MeshForge CLI

```powershell
cd C:\Users\...\Desktop\Frontend
py meshforge_cli.py --help
py meshforge_cli.py full -i path\to\img.jpg -a path\to\voice.wav
```

`pifuhd` subcommand feeds the **parent folder** of the image to PiFuHD (it batches the whole directory). Single-image runs: put the file alone in a folder or accept extra meshes being picked up.

## OBJ texture warning

Some OBJs have vertex colors but no UVs; trimesh then logs `ColorVisuals ... no uv` and skips `--texture`. Fix upstream mesh or bake UVs in Blender/Maya/etc.

## Logs

Pipeline file logging goes to `pifuhd_audio2face_pipeline/outputs/pipeline.log` relative to whatever the cwd is when the process starts — if that annoys you, change `utils/logger.py` paths.

## Versions

Developed against py3.10 + trimesh stack; Blender 4.x for FBX/USD export. YMMV on other combos.

— last tidy pass stripped docstrings/comments across the pipeline + meshforge files via ast unparse, so don't expect pretty vertical banners in source anymore.
