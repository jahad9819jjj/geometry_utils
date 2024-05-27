import bpy
from mathutils import Vector
import os

def render_nocs_map(mesh_path):
    # メッシュをインポート
    bpy.ops.wm.obj_import(filepath=mesh_path)
    imported_object = bpy.context.selected_objects[0]

    # レンダリング設定
    bpy.context.scene.render.filepath = os.path.splitext(mesh_path)[0] + '_nocs_map.png'
    bpy.context.scene.render.film_transparent = True

    # # カメラの位置を調整
    cam_ob = bpy.context.scene.camera
    # cam_ob.location = (0, -10, 0)  # カメラを遠ざける（Y軸の負の方向に移動）
    # cam_ob.rotation_euler = (0, 0, 0)  # カメラの回転を初期化
    # cam_ob.location[0] += 50
    cam_ob.location[1] -= 100
    cam_ob.location[2] += 50
    
    direction = imported_object.location - cam_ob.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_ob.rotation_euler = rot_quat.to_euler()


    # インポートしたメッシュオブジェクトに対して処理を行う
    if imported_object.type == 'MESH':
        setup_vertex_colors(imported_object)
        setup_material(imported_object)
        export_obj_with_nocs(imported_object, os.path.splitext(mesh_path)[0] + '_nocs.obj')

    # レンダリング実行
    bpy.ops.render.render(write_still=True)

    # インポートしたメッシュオブジェクトを削除
    bpy.ops.object.delete()

def setup_vertex_colors(obj):
    vcol_layer = obj.data.vertex_colors.new()

    # メッシュの境界ボックスを計算
    bbox = obj.bound_box
    max_coord = Vector(bbox[0])
    min_coord = Vector(bbox[0])
    for coord in bbox:
        max_coord.x = max(max_coord.x, coord[0])
        max_coord.y = max(max_coord.y, coord[1])
        max_coord.z = max(max_coord.z, coord[2])
        min_coord.x = min(min_coord.x, coord[0])
        min_coord.y = min(min_coord.y, coord[1])
        min_coord.z = min(min_coord.z, coord[2])

    # 頂点座標をスケーリング
    for loop_index, loop in enumerate(obj.data.loops):
        vert_coord = obj.data.vertices[loop.vertex_index].co
        scaled_coord = Vector((
            (vert_coord.x - min_coord.x) / (max_coord.x - min_coord.x),
            (vert_coord.y - min_coord.y) / (max_coord.y - min_coord.y),
            (vert_coord.z - min_coord.z) / (max_coord.z - min_coord.z)
        )) * 2 - Vector([1, 1, 1])
        color = scaled_coord * 0.5 + Vector([0.5, 0.5, 0.5])
        vcol_layer.data[loop_index].color = (color.x, color.y, color.z, 1.0)

    obj.data.vertex_colors.active = vcol_layer
    obj.data.update()

def setup_material(obj):
    mat = bpy.data.materials.new('coord_color')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    vertex_color_node = nodes.new(type='ShaderNodeVertexColor')
    vertex_color_node.layer_name = 'Col'

    output_node = nodes.get('Material Output')

    links.new(vertex_color_node.outputs['Color'], output_node.inputs['Surface'])

    obj.data.materials.clear()
    obj.data.materials.append(mat)
    obj.active_material = mat

def export_obj_with_nocs(obj, output_path):
    # 頂点カラーを頂点グループに変換
    vertex_group = obj.vertex_groups.new(name="NOCS")
    for i, vert in enumerate(obj.data.vertices):
        vertex_group.add([i], obj.data.vertex_colors["Col"].data[i].color[0], 'REPLACE')

    # OBJファイルにエクスポート
    bpy.ops.wm.obj_export(filepath=output_path,
                         export_selected_objects=True,
                         export_vertex_groups=True,
                         export_colors=True)

if __name__ == "__main__":
    mesh_path = '/path/to/obj'  
    render_nocs_map(mesh_path)