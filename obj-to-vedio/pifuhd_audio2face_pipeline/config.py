import os
from pathlib import Path
BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / 'assets'
OUTPUTS_DIR = BASE_DIR / 'outputs'
CLEANUP = dict(remove_duplicate_verts=True, duplicate_threshold=1e-06, fix_normals=True, fill_holes=True, max_hole_size=500, decimate_face_count=15000, decimate_preserve_border=True)
FACE = dict(head_y_fraction=0.72, chin_search_fraction=0.4, landmark_backend='mediapipe', procedural_landmarks=468)
RETOPO = dict(target_verts=4096, edge_loops_eyes=6, edge_loops_mouth=8, symmetry_axis='x')
BLENDSHAPE = dict(jaw_open_amount=0.04, smile_amount=0.025, pucker_amount=0.018, funnel_amount=0.016, blink_amount=0.008, brow_up_amount=0.012, brow_down_amount=0.01, cheek_puff_amount=0.015, viseme_aa_amount=0.035, viseme_oh_amount=0.028, viseme_ee_amount=0.022, viseme_fv_amount=0.014, viseme_mp_amount=0.006)
BLENDSHAPE_NAMES = ['JawOpen', 'MouthSmileLeft', 'MouthSmileRight', 'EyeBlinkLeft', 'EyeBlinkRight', 'MouthPucker', 'MouthFunnel', 'BrowInnerUp', 'BrowDown_L', 'BrowDown_R', 'CheekPuff', 'Viseme_AA', 'Viseme_OH', 'Viseme_EE', 'Viseme_FV', 'Viseme_MP']
EXPORT = dict(fbx_version='FBX202000', usd_extension='.usdc', mesh_name='Head_Mesh', armature_name='Head_Armature', root_bone_name='Head')
LIPSYNC = dict(sample_rate=16000, hop_length=512, n_mfcc=13, phoneme_model='heuristic', smooth_window=3, fps=30)
PHONEME_VISEME_MAP = {'AA': 'Viseme_AA', 'AE': 'Viseme_AA', 'AH': 'Viseme_AA', 'AO': 'Viseme_OH', 'AW': 'Viseme_OH', 'AY': 'Viseme_AA', 'EH': 'Viseme_EE', 'ER': 'Viseme_EE', 'EY': 'Viseme_EE', 'IH': 'Viseme_EE', 'IY': 'Viseme_EE', 'OW': 'Viseme_OH', 'OY': 'Viseme_OH', 'UH': 'Viseme_OH', 'UW': 'Viseme_MP', 'B': 'Viseme_MP', 'P': 'Viseme_MP', 'M': 'Viseme_MP', 'F': 'Viseme_FV', 'V': 'Viseme_FV', 'SIL': 'Viseme_MP'}
GPU = dict(prefer_cuda=True)
LOG_LEVEL = 'INFO'
