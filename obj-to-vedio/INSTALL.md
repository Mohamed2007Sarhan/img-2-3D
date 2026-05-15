# PIFuHD → NVIDIA Audio2Face Pipeline
## Installation & Usage Guide

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python    | 3.11    | 3.11        |
| RAM       | 8 GB    | 16 GB       |
| VRAM      | —       | 6 GB (optional CUDA) |
| Blender   | 4.0     | 4.1+        |
| OS        | Win/Mac/Linux | Ubuntu 22.04 / Win 11 |

---

## Step 1 — Clone / Copy Project

```bash
git clone <your-repo> pifuhd_audio2face
cd pifuhd_audio2face
```

---

## Step 2 — Create Python Environment

```bash
python3.11 -m venv .venv

# Activate
source .venv/bin/activate          # Linux/Mac
.venv\Scripts\activate             # Windows
```

---

## Step 3 — Install Python Dependencies

```bash
pip install --upgrade pip

# Core dependencies
pip install numpy scipy trimesh open3d pyvista

# Computer vision
pip install opencv-python mediapipe

# Audio
pip install librosa soundfile

# Live microphone (optional)
pip install pyaudio        # may need portaudio: sudo apt install portaudio19-dev

# 3D rendering / viewer
pip install pyrender pyglet Pillow

# Logging
pip install loguru tqdm

# Blender standalone Python (alternative to system Blender)
pip install bpy            # Blender 4.1 embedded Python
```

### Optional: GPU support (CUDA)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# wav2vec2-based phoneme extraction:
pip install transformers
```

### Optional: InsightFace (better landmark detection)

```bash
pip install insightface onnxruntime-gpu   # or onnxruntime for CPU
```

---

## Step 4 — Install Blender

### Option A — System Blender (recommended for export)

Download from https://www.blender.org/download/ (version 4.0+).

Add to PATH, then verify:
```bash
blender --version
```

### Option B — Standalone bpy (pip)

```bash
pip install bpy
```
Note: Standalone bpy has limited addon support; FBX/USD export may differ.

---

## Step 5 — Verify Installation

```bash
python -c "import trimesh, open3d, mediapipe, librosa; print('All OK')"
```

---

## Step 6 — Run the Pipeline

### Full pipeline (recommended):

```bash
python main.py assets/human.obj --texture assets/texture.png
```

### With audio lip sync + viewer:

```bash
python main.py assets/human.obj \
    --texture assets/texture.png \
    --wav assets/speech.wav \
    --viewer
```

### With webcam facial tracking:

```bash
python main.py assets/human.obj --skip-to viewer --webcam
```

### Step-by-step (each stage separately):

```bash
# Stage 2: Cleanup only
python cleanup.py assets/human.obj --output outputs/cleaned.obj

# Stage 3-4: Face detection + retopology
python retopo.py outputs/cleaned.obj \
    --output-face outputs/face.obj \
    --output-retopo outputs/face_retopo.obj

# Stage 5: Blendshapes
python blendshapes.py outputs/face_retopo.obj \
    --out-dir outputs/blendshapes

# Stage 6: Export FBX + USD
python export.py outputs/face_retopo.obj outputs/blendshapes \
    --formats fbx usd \
    --blender /path/to/blender

# Stage 7: Viewer
python viewer.py outputs/face_retopo.obj outputs/blendshapes

# Stage 8: Lip sync from WAV
python lipsync.py assets/speech.wav --output outputs/lipsync_curve.json
```

---

## Step 7 — Import into NVIDIA Audio2Face

1. Open **NVIDIA Audio2Face 2023.2+**
2. File → Import → FBX (or USD)
   - Select: `outputs/head_audio2face.fbx`
3. In **Setup panel** → Blendshape Solver:
   - Click **Auto-detect** — it will find all ARKit shape keys
4. Load your audio: **Audio panel** → Load WAV → Generate
5. The Viseme_* and jaw blendshapes will animate automatically

### Supported shape keys (ARKit subset):

```
JawOpen         MouthSmileLeft   MouthSmileRight
EyeBlinkLeft    EyeBlinkRight    MouthPucker
MouthFunnel     BrowInnerUp      BrowDown_L
BrowDown_R      CheekPuff
Viseme_AA       Viseme_OH        Viseme_EE
Viseme_FV       Viseme_MP
```

---

## Viewer Controls

| Key / Action | Effect |
|--------------|--------|
| Mouse drag   | Rotate camera |
| Scroll       | Zoom |
| `bs NAME W`  | Set blendshape weight (0–1) |
| `reset`      | Zero all blendshapes |
| `list`       | Show all blendshape values |
| `SPACE`      | Cycle to next blendshape (Open3D mode) |
| `R`          | Reset (Open3D mode) |
| `Q`          | Quit |

---

## Troubleshooting

### "Blender not found"
```bash
# Windows
set PATH=%PATH%;C:\Program Files\Blender Foundation\Blender 4.1

# Linux
export PATH=/opt/blender-4.1:$PATH

# Or specify directly:
python export.py mesh.obj bs_dir --blender /path/to/blender
```

### "pyrender display error" (headless server)
```bash
export DISPLAY=:0           # if X11 available
# or use EGL:
export PYOPENGL_PLATFORM=egl
```

### "No face detected"
- Check that `config.FACE["head_y_fraction"]` matches your mesh proportions
- Try `--skip-to retopo` and check `outputs/02_face.obj`
- Set `landmark_backend = "heuristic"` in config.py

### MediaPipe not detecting face
- Ensure the mesh face is oriented toward +Z (camera-facing)
- The pyrender projection assumes front-facing geometry

---

## Project Structure

```
pifuhd_audio2face/
├── main.py              ← Full pipeline orchestrator
├── cleanup.py           ← Stage 2: Mesh cleanup
├── retopo.py            ← Stage 3-4: Face detection + retopology
├── blendshapes.py       ← Stage 5: Blendshape generation
├── export.py            ← Stage 6: FBX / USD export via Blender
├── viewer.py            ← Stage 7: Realtime 3D viewer
├── lipsync.py           ← Stage 8: Audio → viseme animation
├── config.py            ← All configuration constants
├── requirements.txt     ← Python dependencies
├── INSTALL.md           ← This file
├── utils/
│   ├── __init__.py
│   ├── logger.py        ← Loguru-based logging
│   ├── gpu.py           ← CUDA detection
│   ├── mesh_io.py       ← OBJ import/export + validation
│   └── blender_helpers.py ← Eye rig, teeth, tongue, head pose
├── assets/              ← Place input OBJ + textures + WAV here
└── outputs/             ← All pipeline outputs written here
    ├── 01_cleaned.obj
    ├── 02_face.obj
    ├── 03_face_retopo.obj
    ├── blendshapes/     ← .npy blendshape arrays
    ├── head_audio2face.fbx
    ├── head_audio2face.usdc
    ├── lipsync_curve.json
    └── pipeline.log
```
