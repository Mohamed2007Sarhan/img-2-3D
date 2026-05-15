
"""
Auto-generated Blender headless script.
Run via:  blender --background --python <this_file>
"""
import bpy
import bmesh
import numpy as np
import json
from pathlib import Path
from mathutils import Matrix, Vector

print("=== Audio2Face Blender Export Script ===")

# ── Constants ────────────────────────────────────────────────────────────────
MESH_PATH       = r"C:\Users\MohamedSarhan\Desktop\Frontend\obj-to-vedio\pifuhd_audio2face_pipeline\outputs\03_face_retopo.obj"
BS_DIR          = Path(r"C:\Users\MohamedSarhan\Desktop\Frontend\obj-to-vedio\pifuhd_audio2face_pipeline\outputs\blendshapes")
FBX_OUT         = r"C:\Users\MohamedSarhan\Desktop\Frontend\obj-to-vedio\pifuhd_audio2face_pipeline\outputs\head_audio2face.fbx"
USD_OUT         = r"C:\Users\MohamedSarhan\Desktop\Frontend\obj-to-vedio\pifuhd_audio2face_pipeline\outputs\head_audio2face.usdc"
MESH_NAME       = "Head_Mesh"
ARMATURE_NAME   = "Head_Armature"
ROOT_BONE_NAME  = "Head"
BS_NAMES        = ["JawOpen", "MouthSmileLeft", "MouthSmileRight", "EyeBlinkLeft", "EyeBlinkRight", "MouthPucker", "MouthFunnel", "BrowInnerUp", "BrowDown_L", "BrowDown_R", "CheekPuff", "Viseme_AA", "Viseme_OH", "Viseme_EE", "Viseme_FV", "Viseme_MP"]
DO_FBX          = True
DO_USD          = True

# ── 1. Clean default scene ────────────────────────────────────────────────────
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

# ── 2. Import OBJ ────────────────────────────────────────────────────────────
print(f"Importing OBJ: {MESH_PATH}")
if MESH_PATH.endswith(".obj"):
    bpy.ops.wm.obj_import(filepath=MESH_PATH)
else:
    raise RuntimeError(f"Unsupported mesh format: {MESH_PATH}")

# Get the imported mesh object
mesh_obj = None
for obj in bpy.context.scene.objects:
    if obj.type == "MESH":
        mesh_obj = obj
        break

if mesh_obj is None:
    raise RuntimeError("No mesh object found after import!")

mesh_obj.name      = MESH_NAME
mesh_obj.data.name = MESH_NAME + "_Data"
bpy.context.view_layer.objects.active = mesh_obj
print(f"Mesh: {len(mesh_obj.data.vertices)} verts, {len(mesh_obj.data.polygons)} faces")

# ── 3. Inject Shape Keys ──────────────────────────────────────────────────────
print("Injecting blendshapes as shape keys ...")

# Basis key
bpy.ops.object.shape_key_add(from_mix=False)
mesh_obj.data.shape_keys.key_blocks[0].name = "Basis"

basis_verts = np.array([v.co[:] for v in mesh_obj.data.vertices])

for bs_name in BS_NAMES:
    npy_path = BS_DIR / f"{bs_name}.npy"
    if not npy_path.exists():
        print(f"  WARNING: {bs_name}.npy not found — skipping")
        continue

    displaced = np.load(str(npy_path))
    if displaced.shape[0] != len(mesh_obj.data.vertices):
        print(f"  WARNING: {bs_name} vertex count mismatch — skipping")
        continue

    bpy.ops.object.shape_key_add(from_mix=False)
    sk = mesh_obj.data.shape_keys.key_blocks[-1]
    sk.name  = bs_name
    sk.value = 0.0

    # Apply displacements
    for i, coord in enumerate(displaced):
        sk.data[i].co = coord.tolist()

    print(f"  Shape key added: {bs_name}")

print(f"Total shape keys: {len(mesh_obj.data.shape_keys.key_blocks)}")

# ── 4. Create Head Armature ───────────────────────────────────────────────────
print("Creating head armature ...")

bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
arm_obj      = bpy.context.active_object
arm_obj.name = ARMATURE_NAME
arm          = arm_obj.data
arm.name     = ARMATURE_NAME + "_Data"

# Rename default bone to root
bone = arm.edit_bones[0]
bone.name   = ROOT_BONE_NAME
bb           = mesh_obj.bound_box
min_y = min(v[1] for v in bb)
max_y = max(v[1] for v in bb)
bone.head   = (0, min_y, 0)
bone.tail   = (0, max_y, 0)

bpy.ops.object.mode_set(mode="OBJECT")

# Parent mesh to armature
mesh_obj.select_set(True)
arm_obj.select_set(True)
bpy.context.view_layer.objects.active = arm_obj
bpy.ops.object.parent_set(type="ARMATURE_AUTO")

print("Armature created and mesh parented.")

# ── 5. Audio2Face metadata (custom properties) ────────────────────────────────
mesh_obj["audio2face_rig"]   = True
mesh_obj["blendshape_count"] = len(BS_NAMES)
mesh_obj["pipeline"]         = "pifuhd_audio2face"

# ── 6. Export FBX ─────────────────────────────────────────────────────────────
if DO_FBX:
    print(f"Exporting FBX → {FBX_OUT}")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.fbx(
        filepath            = FBX_OUT,
        use_selection       = False,
        use_mesh_modifiers  = True,
        add_leaf_bones      = False,
        bake_anim           = False,
        mesh_smooth_type    = "FACE",
        use_tspace          = True,
        use_custom_props    = True,
        path_mode           = "COPY",
        embed_textures      = True,
    )
    print("FBX export done.")

# ── 7. Export USD ─────────────────────────────────────────────────────────────
if DO_USD:
    print(f"Exporting USD → {USD_OUT}")
    try:
        bpy.ops.wm.usd_export(
            filepath               = USD_OUT,
            export_animation       = False,
            export_hair            = False,
            export_uvmaps          = True,
            export_normals         = True,
            export_materials       = True,
            use_instancing         = False,
            evaluation_mode        = "RENDER",
        )
        print("USD export done.")
    except Exception as e:
        print(f"USD export failed: {e} (usd_export may need Blender 4.0+)")

print("=== Blender export script complete ===")
