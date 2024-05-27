import bpy
from mathutils import Vector

def render_nocs_map():
    bpy.context.scene.render.filepath = './nocs_map_cube.png'
    bpy.context.scene.render.film_transparent = True  # アルファモードの設定を変更

    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            setup_vertex_colors(obj)
            setup_material(obj)

    bpy.ops.render.render(write_still=True)

def setup_vertex_colors(obj):
    vcol_layer = obj.data.vertex_colors.new()
    scale = 0.5

    for loop_index, loop in enumerate(obj.data.loops):
        vert_coord = obj.data.vertices[loop.vertex_index].co
        color = scale * vert_coord + Vector([0.5, 0.5, 0.5])
        vcol_layer.data[loop_index].color = (color.x, color.y, color.z, 1.0)  # アルファ値を1.0に設定

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

if __name__ == "__main__":
    render_nocs_map()