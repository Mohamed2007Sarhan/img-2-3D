from __future__ import annotations

def _require_bpy():
    try:
        import bpy
        return bpy
    except ImportError:
        raise RuntimeError("utils/blender_helpers.py must be executed inside Blender's Python interpreter (bpy is not available standalone).")

def build_eye_rig(armature_name: str='Head_Armature', left_eye_pos: tuple=(-0.033, 0.0, 0.06), right_eye_pos: tuple=(0.033, 0.0, 0.06)) -> None:
    bpy = _require_bpy()
    import mathutils
    arm_obj = bpy.data.objects.get(armature_name)
    if arm_obj is None:
        raise ValueError(f"Armature '{armature_name}' not found in scene.")
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')
    arm = arm_obj.data
    head_bone = arm.edit_bones.get('Head')
    if head_bone is None:
        head_bone = arm.edit_bones[0]

    def add_bone(name, head, tail, parent=None, deform=True):
        b = arm.edit_bones.new(name)
        b.head = mathutils.Vector(head)
        b.tail = mathutils.Vector(tail)
        b.use_deform = deform
        if parent:
            b.parent = parent
            b.use_connect = False
        return b
    eye_root = add_bone('EyeRoot', (0, 0, left_eye_pos[2]), (0, 0.01, left_eye_pos[2]), parent=head_bone, deform=False)
    eye_l = add_bone('EyeLeft', left_eye_pos, (left_eye_pos[0], left_eye_pos[1] + 0.02, left_eye_pos[2]), parent=eye_root)
    eye_r = add_bone('EyeRight', right_eye_pos, (right_eye_pos[0], right_eye_pos[1] + 0.02, right_eye_pos[2]), parent=eye_root)
    target_l = add_bone('EyeTarget_L', (left_eye_pos[0], -0.3, left_eye_pos[2]), (left_eye_pos[0], -0.28, left_eye_pos[2]), deform=False)
    target_r = add_bone('EyeTarget_R', (right_eye_pos[0], -0.3, right_eye_pos[2]), (right_eye_pos[0], -0.28, right_eye_pos[2]), deform=False)
    bpy.ops.object.mode_set(mode='POSE')
    for (bone_name, target_name) in [('EyeLeft', 'EyeTarget_L'), ('EyeRight', 'EyeTarget_R')]:
        pb = arm_obj.pose.bones[bone_name]
        c = pb.constraints.new('IK')
        c.target = arm_obj
        c.subtarget = target_name
        c.chain_count = 1
    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"Eye rig added to '{armature_name}'")

def separate_teeth(mesh_obj_name: str, y_threshold: float=-0.04, z_range: tuple=(-0.02, 0.02)) -> None:
    bpy = _require_bpy()
    obj = bpy.data.objects.get(mesh_obj_name)
    if obj is None:
        raise ValueError(f"Object '{mesh_obj_name}' not found.")
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    import bmesh as bm_mod
    bm = bm_mod.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    for v in bm.verts:
        in_z = z_range[0] < v.co.z < z_range[1]
        in_y = v.co.y < y_threshold
        v.select = in_z and in_y
    bm_mod.update_edit_mesh(obj.data)
    bpy.ops.mesh.separate(type='SELECTED')
    bpy.ops.object.mode_set(mode='OBJECT')
    for o in bpy.context.selected_objects:
        if o.name != mesh_obj_name:
            o.name = 'Teeth_Mesh'
            o.data.name = 'Teeth_Mesh_Data'
            print(f"Separated teeth → '{o.name}'")
            break

def create_tongue_mesh(parent_armature: str='Head_Armature', mouth_center: tuple=(0.0, -0.01, -0.01), tongue_length: float=0.04) -> None:
    bpy = _require_bpy()
    import mathutils
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=12, ring_count=8, location=mouth_center)
    tongue = bpy.context.active_object
    tongue.name = 'Tongue_Mesh'
    tongue.data.name = 'Tongue_Mesh_Data'
    tongue.scale = (tongue_length * 0.4, tongue_length, tongue_length * 0.25)
    bpy.ops.object.transform_apply(scale=True)
    arm_obj = bpy.data.objects.get(parent_armature)
    if arm_obj:
        bpy.context.view_layer.objects.active = arm_obj
        bpy.ops.object.mode_set(mode='EDIT')
        arm = arm_obj.data
        head_bone = arm.edit_bones.get('Head') or arm.edit_bones[0]
        root = arm.edit_bones.new('TongueRoot')
        root.head = mathutils.Vector(mouth_center)
        root.tail = mathutils.Vector((mouth_center[0], mouth_center[1] - tongue_length * 0.5, mouth_center[2]))
        root.parent = head_bone
        root.use_connect = False
        tip = arm.edit_bones.new('TongueTip')
        tip.head = root.tail.copy()
        tip.tail = mathutils.Vector((mouth_center[0], mouth_center[1] - tongue_length, mouth_center[2]))
        tip.parent = root
        tip.use_connect = True
        bpy.ops.object.mode_set(mode='OBJECT')
        tongue.select_set(True)
        arm_obj.select_set(True)
        bpy.context.view_layer.objects.active = arm_obj
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    print('Tongue mesh and rig created.')

def add_head_pose_controller(armature_name: str='Head_Armature', root_bone_name: str='Head') -> None:
    bpy = _require_bpy()
    import mathutils
    arm_obj = bpy.data.objects.get(armature_name)
    if arm_obj is None:
        raise ValueError(f"Armature '{armature_name}' not found.")
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')
    arm = arm_obj.data
    head_eb = arm.edit_bones.get(root_bone_name)
    if head_eb is None:
        raise ValueError(f"Bone '{root_bone_name}' not found in armature.")
    ctrl = arm.edit_bones.new('HeadCtrl')
    ctrl.head = mathutils.Vector((0, -0.3, float(head_eb.head.z)))
    ctrl.tail = mathutils.Vector((0, -0.28, float(head_eb.head.z)))
    ctrl.use_deform = False
    bpy.ops.object.mode_set(mode='POSE')
    head_pb = arm_obj.pose.bones[root_bone_name]
    c = head_pb.constraints.new('COPY_ROTATION')
    c.target = arm_obj
    c.subtarget = 'HeadCtrl'
    c.use_x = True
    c.use_y = True
    c.use_z = True
    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"Head pose controller 'HeadCtrl' added → drives '{root_bone_name}'")
