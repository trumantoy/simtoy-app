from trimesh.creation import cylinder
from trimesh.visual import texture
import wgpu
import pygfx as gfx
from pygfx.renderers.wgpu import *
from pygfx.objects import WorldObject
from pygfx.materials import Material
from pygfx.utils.transform import AffineTransform
import pylinalg as la
from importlib.resources import files


import numpy as np
import io
from PIL import Image
from cairosvg import svg2png


class OriginMaterial(Material):
    """最简点材质：统一颜色与尺寸（screen-space）。"""

    uniform_type = dict(
        Material.uniform_type,
        color="4xf4",
        size="f4",
    )

    def __init__(self, *, color='black', size=5.0, **kwargs):
        super().__init__(**kwargs)
        self.color = gfx.Color('orange')
        self.size = float(size)

    @property
    def color(self):
        return tuple(self.uniform_buffer.data["color"])

    @color.setter
    def color(self, rgba):
        self.uniform_buffer.data["color"] = rgba
        self.uniform_buffer.update_full()

    @property
    def size(self):
        return float(self.uniform_buffer.data["size"])

    @size.setter
    def size(self, v):
        self.uniform_buffer.data["size"] = float(v)
        self.uniform_buffer.update_full()

    def _wgpu_get_pick_info(self, pick_value):
        from pygfx.utils import unpack_bitfield
        values = unpack_bitfield(pick_value, wobject_id=20, index=26, x=9, y=9)
        return {
            "vertex_index": values["index"],
            "point_coord": (values["x"] - 256.0, values["y"] - 256.0),
        }


@register_wgpu_render_function(gfx.WorldObject, OriginMaterial)
class OriginShader(BaseShader):
    type = "render"

    def __init__(self, wobject):
        super().__init__(wobject)

    def get_bindings(self, wobject, shared):
        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", wobject.material.uniform_buffer),
            Binding("s_positions", "buffer/read_only_storage", wobject.geometry.positions, "VERTEX"),
        ]
        bindings = {i: b for i, b in enumerate(bindings)}
        self.define_bindings(0, bindings)
        return {0: bindings}

    def get_pipeline_info(self, wobject, shared):
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.none,
        }

    def get_render_info(self, wobject, shared):
        offset, size = wobject.geometry.positions.draw_range
        offset, size = offset * 6, size * 6
        return {
            "indices": (size, 1, offset, 0),
            "render_mask": RenderMask.all,
        }

    def get_code(self):
        return """
            {$ include 'pygfx.std.wgsl' $}

            // 顶点着色器输入结构体
            struct VertexInput {
                @builtin(vertex_index) index : u32,  // 当前处理的顶点索引（0, 1, 2, 3, 4, 5...）
            };

            @vertex
            fn vs_main(in: VertexInput) -> Varyings {
                // 获取逻辑屏幕的半尺寸，用于NDC到屏幕坐标的转换
                let screen_factor: vec2<f32> = u_stdinfo.logical_size.xy / 2.0;
                
                // 将顶点索引转换为有符号整数
                let index = i32(in.index);
                
                // 计算逻辑：每个"点"展开为6个顶点（两个三角形组成正方形）
                // node_index: 这是第几个逻辑点（0, 1, 2...）
                // vertex_index: 该点内的第几个顶点（0,1,2,3,4,5）
                let node_index = index / 6;      // 整数除法，得到点索引
                let vertex_index = index % 6;    // 取余，得到点内顶点索引
                
                // 从存储缓冲区读取第node_index个点的模型空间位置
                let pos_m = load_s_positions(node_index);
                
                // 模型空间 -> 世界空间：应用对象的变换矩阵
                let pos_w = u_wobject.world_transform * vec4<f32>(pos_m.xyz, 1.0);
                
                // 世界空间 -> 相机空间：应用相机视图矩阵
                let pos_c = u_stdinfo.cam_transform * pos_w;
                
                // 相机空间 -> 裁剪空间：应用投影矩阵
                let pos_n = u_stdinfo.projection_transform * pos_c;
                
                // 裁剪空间 -> 逻辑屏幕坐标
                // 1. 透视除法：pos_n.xy / pos_n.w 得到NDC坐标 [-1,1]
                // 2. 映射到 [0,1]：+ 1.0
                // 3. 缩放到屏幕尺寸：* screen_factor
                let pos_s = (pos_n.xy / pos_n.w + 1.0) * screen_factor;
                
                // 计算点的大小（半径，逻辑像素）
                let half = 5.0 * u_material.size;
                
                // 定义6个顶点的相对偏移，形成两个三角形组成的正方形
                // 顶点顺序：左下、右下、左上、左上、右下、右上
                // 这样绘制两个三角形：左下-右下-左上 和 左上-右下-右上
                var deltas = array<vec2<f32>, 6>(
                    vec2<f32>(-1.0, -1.0),  // 左下
                    vec2<f32>( 1.0, -1.0),  // 右下  
                    vec2<f32>(-1.0,  1.0),  // 左上
                    vec2<f32>(-1.0,  1.0),  // 左上（重复，第二个三角形的起点）
                    vec2<f32>( 1.0, -1.0),  // 右下（重复，第二个三角形的第二个点）
                    vec2<f32>( 1.0,  1.0),  // 右上
                );
                
                // 根据当前顶点索引，获取对应的偏移量并缩放到点的大小
                let delta = deltas[vertex_index] * half;
                
                // 将偏移量应用到屏幕坐标，得到当前顶点的最终屏幕位置
                let the_pos_s = pos_s + delta;
                
                // 屏幕坐标 -> NDC坐标（保持与中心点相同的深度）
                // 1. 屏幕坐标归一化到 [0,1]：the_pos_s / screen_factor
                // 2. 映射到 [-1,1]：- 1.0
                // 3. 恢复裁剪空间的w分量：* pos_n.w
                // 4. z分量置零，让原点永远在最前面
                let the_pos_n = vec4<f32>((the_pos_s / screen_factor - 1.0) * pos_n.w, 0, pos_n.w);
                
                // 构建传递给片段着色器的插值变量
                var varyings: Varyings;
                
                // 设置裁剪空间位置（GPU需要这个进行光栅化）
                varyings.position = vec4<f32>(the_pos_n);
                
                // 设置世界空间位置（用于后续计算，如光照、深度等）
                varyings.world_pos = vec3<f32>(ndc_to_world_pos(the_pos_n));
                
                // 设置点内坐标（相对于点中心的偏移，物理像素单位）
                // 用于片段着色器中计算SDF和描边
                let l2p = u_stdinfo.physical_size.x / u_stdinfo.logical_size.x;  // 逻辑到物理像素比例
                varyings.pointcoord_p = vec2<f32>(delta * l2p);  // 转换为物理像素
                
                // 设置点的物理像素尺寸
                varyings.size_p = f32(u_material.size * l2p);
                varyings.pick_idx = u32(node_index);
                return varyings;
            }

            @fragment
            fn fs_main(varyings: Varyings) -> FragmentOutput {
                // 构建片段着色器输出
                var out: FragmentOutput;
                
                // 计算圆形SDF：到边缘的有向距离
                // length(pointcoord) 计算到中心的欧几里得距离
                // 0.5 * size_p 是圆的半径
                // 正值：点在圆外部，负值：点在圆内部，0值：点在圆边界上
                var dist_to_edge = length(varyings.pointcoord_p) - 0.5 * varyings.size_p;
                
                // 描边逻辑：判断当前片段是否在描边范围内
                var l2p = u_stdinfo.physical_size.x / u_stdinfo.logical_size.x;
                var edge_width = 0.75 * l2p;  // 描边宽度（物理像素）
                
                // 判断片段类型
                var is_edge = abs(dist_to_edge) < edge_width;    // 是否在描边范围内
                var is_inside = dist_to_edge < 0.0;              // 是否在点内部
                
                // 根据片段位置设置颜色
                var final_color: vec4<f32>;
                if (is_edge) {
                    // 描边颜色：黑色
                    final_color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
                } else if (is_inside) {
                    // 点内部：使用材质颜色
                    let col = u_material.color;
                    // 将sRGB颜色转换为物理线性空间，保持alpha不变
                    final_color = vec4<f32>(srgb2physical(col.rgb), col.a);
                } else {
                    final_color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                }
                
                let pointcoord: vec2<f32> = varyings.pointcoord_p / l2p;

                // 设置最终输出颜色
                out.color = final_color;

                $$ if write_pick
                out.pick = (
                    pick_pack(u32(u_wobject.id), 20) +
                    pick_pack(varyings.pick_idx, 26) +
                    pick_pack(u32(pointcoord.x + 256.0), 9) +
                    pick_pack(u32(pointcoord.y + 256.0), 9)
                );
                $$ endif
                return out;
            }
        """


class AxisMaterial(Material):
    """坐标轴材质：支持颜色和线宽设置"""
    uniform_type = dict(
        Material.uniform_type,
        color="4xf4",
        size='f4'
    )

    def __init__(self, *, color,size, **kwargs):
        super().__init__(** kwargs)
        self.color = gfx.Color(color)
        self.size = size

    @property
    def color(self):
        return tuple(self.uniform_buffer.data["color"])

    @color.setter
    def color(self, rgba):
        self.uniform_buffer.data["color"] = rgba
        self.uniform_buffer.update_full()

    @property
    def size(self):
        return tuple(self.uniform_buffer.data["color"])

    @size.setter
    def size(self, value):
        self.uniform_buffer.data["size"] = value
        self.uniform_buffer.update_full()

    def _wgpu_get_pick_info(self, pick_value):
        from pygfx.utils import unpack_bitfield
        values = unpack_bitfield(pick_value, wobject_id=20, index=26, x=9, y=9)
        return {
            "vertex_index": values["index"],
            "point_coord": (values["x"] - 256.0, values["y"] - 256.0),
        }

@register_wgpu_render_function(WorldObject, AxisMaterial)
class AxisShader(BaseShader):
    type = "render"

    def __init__(self, wobject):
        super().__init__(wobject)

    def get_bindings(self, wobject, shared):
        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", wobject.material.uniform_buffer),
            Binding("s_positions", "buffer/read_only_storage", wobject.geometry.positions, "VERTEX"),
            Binding("s_indices", "buffer/read_only_storage", wobject.geometry.indices, "VERTEX")
        ]
        bindings = {i: b for i, b in enumerate(bindings)}
        self.define_bindings(0, bindings)
        return {0: bindings}

    def get_pipeline_info(self, wobject, shared):
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.none,
        }

    def get_render_info(self, wobject, shared):
        geometry = wobject.geometry
        material = wobject.material

        offset, size = geometry.indices.draw_range
        offset, size = 6 * offset, 6 * size

        return {
            "indices": (size, 1, offset, 0),
            "render_mask": RenderMask.all,
        }

    def get_code(self):
        return """
            {$ include 'pygfx.std.wgsl' $}

            struct VertexInput {
                @builtin(vertex_index) index : u32,
            };

            @vertex
            fn vs_main(in: VertexInput) -> Varyings {
                let mvp = u_stdinfo.projection_transform * u_stdinfo.cam_transform * u_wobject.world_transform;
                var origin_screen = mvp * vec4<f32>(0,0,0,1);
                var x_screen = mvp * vec4<f32>(0.01,0,0,1); 
                var y_screen = mvp * vec4<f32>(0,0.01,0,1); 
                var z_screen = mvp * vec4<f32>(0,0,0.01,1); 

                var x_direction = (vec3<f32>(x_screen.xyz / x_screen.w) - vec3<f32>(origin_screen.xyz / origin_screen.w)); 
                var y_direction = (vec3<f32>(y_screen.xyz / y_screen.w) - vec3<f32>(origin_screen.xyz / origin_screen.w));
                var z_direction = (vec3<f32>(z_screen.xyz / z_screen.w) - vec3<f32>(origin_screen.xyz / origin_screen.w));

                var size_1_radius = max(length(x_direction),max(length(y_direction),length(z_direction)));
                var scale = u_material.size / u_stdinfo.logical_size.y / size_1_radius;
                
                let index = i32(in.index);
                let face_index = index / 6;
                var sub_index = index % 6;
                
                let vii = load_s_indices(face_index);
                let i0 = i32(vii[sub_index]);
                var pos_n = mvp * vec4<f32>(load_s_positions(i0) * scale ,1.0);
                pos_n.z = 0;

                var varyings: Varyings;
                varyings.position = vec4<f32>(pos_n);
                
                let l2p = u_stdinfo.physical_size.x / u_stdinfo.logical_size.x;  // 逻辑到物理像素比例
                varyings.pointcoord_p = vec2<f32>(load_s_positions(i0).xy * l2p);  // 转换为物理像素
                
                // 设置点的物理像素尺寸
                varyings.pick_idx = u32(i0);
                return varyings;
            }

            @fragment
            fn fs_main(varyings: Varyings) -> FragmentOutput {
                var out: FragmentOutput; 
                out.color = vec4<f32>(u_material.color);

                var l2p = u_stdinfo.physical_size.x / u_stdinfo.logical_size.x;
                let pointcoord: vec2<f32> = varyings.pointcoord_p / l2p;

                $$ if write_pick
                out.pick = (
                    pick_pack(u32(u_wobject.id), 20) +
                    pick_pack(varyings.pick_idx, 26) +
                    pick_pack(u32(pointcoord.x + 256.0), 9) +
                    pick_pack(u32(pointcoord.y + 256.0), 9)
                );
                $$ endif
                return out;
            }
        """


class TranformHelper(gfx.WorldObject):
    def __init__(self, geometry=None, material=None, *args, **kwargs):
        super().__init__(geometry, material, *args, **kwargs)

        self._ref = None
        self._object_to_control = self 
        self._camera = None
        self._ndc_to_screen = None

        self._create_elements()
        self.add_event_handler(self._process_event,"pointer_down","pointer_move","pointer_up","wheel")
    
    def set_tranform_visible(self,visible):
        self.axis_x.visible = visible
        self.axis_y.visible = visible
        self.arrow_x.visible = visible
        self.arrow_y.visible = visible
        self.origin.visible = visible
        self.plane_xy.visible = visible

    def _create_elements(self):
        axis_length = 0.01
        axis_size = 0.0002
        arrow_length = 0.002
        arrow_size = 0.001

        # 原点
        self.origin = gfx.Points(gfx.Geometry(positions=[(0,0,0)]), OriginMaterial(pick_write=True))
        self.add(self.origin)

        # X轴 (红色)
        geom = gfx.cylinder_geometry(radius_bottom=axis_size,radius_top=axis_size,height=axis_length)
        geom.positions.data[:] = la.vec_transform_quat(geom.positions.data, la.quat_from_euler((0, np.pi/2, 0))) + (axis_length / 2,0,0)
        mat = AxisMaterial(color='red',size=200,pick_write=True)
        self.axis_x = gfx.WorldObject(geom,mat)
        
        # 添加轴端点箭头
        geom = gfx.cone_geometry(arrow_size,arrow_length)
        self.arrow_x = gfx.Mesh(geom, AxisMaterial(color='red',size=200,pick_write=True))
        self.arrow_x.geometry.positions.data[:] = la.vec_transform_quat(geom.positions.data, la.quat_from_euler((0, np.pi/2, 0))) + (axis_length,0,0) 
        self.add(self.axis_x)
        self.add(self.arrow_x)

        # Y轴 (绿色)
        geom = gfx.cylinder_geometry(radius_bottom=axis_size,radius_top=axis_size,height=axis_length)
        geom.positions.data[:] = la.vec_transform_quat(geom.positions.data, la.quat_from_euler((-np.pi/2,0, 0))) + (0,axis_length / 2,0)
        mat = AxisMaterial(color='green',size=200,pick_write=True)
        self.axis_y = gfx.WorldObject(geom,mat)

        # 添加轴端点箭头
        geom = gfx.cone_geometry(arrow_size,arrow_length)
        self.arrow_y = gfx.Mesh(geom, AxisMaterial(color='green',size=200,pick_write=True))
        self.arrow_y.geometry.positions.data[:] = la.vec_transform_quat(geom.positions.data, la.quat_from_euler((-np.pi/2,0, 0))) + (0,axis_length,0)
        self.add(self.axis_y)
        self.add(self.arrow_y)

        plane_geo = gfx.plane_geometry(0.002, 0.002)
        plane_geo.positions.data[:] = plane_geo.positions.data + (0.001,0.001,0)
        self.plane_xy = gfx.Mesh(plane_geo,AxisMaterial(color='dodgerblue',size=200,pick_write=True))
        self.add(self.plane_xy)

        self._translate_children = self.axis_x, self.axis_y, self.arrow_x, self.arrow_y, self.plane_xy

        self.axis_x.dim = self.arrow_x.dim = 0
        self.axis_y.dim = self.arrow_y.dim = 1
        self.plane_xy.dim = (0,1)

    def _update_ndc_screen_transform(self):
        # Note: screen origin is at top left corner of NDC with Y-axis pointing down
        x_dim, y_dim = self._camera.logical_size
        screen_space = AffineTransform()
        screen_space.position = (-1, 1, 0)
        screen_space.scale = (2 / x_dim, -2 / y_dim, 1)
        self._ndc_to_screen = screen_space.inverse_matrix
        self._screen_to_ndc = screen_space.matrix

    def _process_event(self, event):
        self._update_ndc_screen_transform()
        self._update_directions()

        type = event.type

        if type == "pointer_down":
            if event.button != 3 or event.modifiers:
                return
            self._ref = None
            # NOTE: I imagine that if multiple tools are active, they
            # each ask for picking info, causing multiple buffer reads
            # for the same location. However, with the new event system
            # this is probably not a problem, when wobjects receive events.
            ob = event.target
            if ob not in self.children:
                return
            # Depending on the object under the pointer, we scale/translate/rotate
            if ob in self._translate_children:
                self._handle_start("translate", event, ob)
            else:
                self.set_tranform_visible(True)
            # elif ob in self._scale_children:
            #     self._handle_start("scale", event, ob)
            # elif ob in self._rotate_children:
            #     self._handle_start("rotate", event, ob)
            # Highlight the object
            self.set_pointer_capture(event.pointer_id, event.root)

        elif type == "pointer_up":
            if not self._ref:
                return
            if self._ref["dim"] is None and self._ref["maxdist"] < 3:
                # self.toggle_mode()  # clicked on the center sphere
                pass
            self._ref = None
            # De-highlight the object
            # self._highlight()

        elif type == "pointer_move":
            if not self._ref:
                return
            # Get how far we've moved from starting point - we have a dead zone
            dist = (
                (event.x - self._ref["event_pos"][0]) ** 2
                + (event.y - self._ref["event_pos"][1]) ** 2
            ) ** 0.5
            self._ref["maxdist"] = max(self._ref["maxdist"], dist)
            # Delegate to the correct handler
            if self._ref["maxdist"] < 3:
                pass
            elif self._ref["kind"] == "translate":
                self._handle_translate_move(event)

    def _handle_start(self, kind, event, ob: WorldObject):
        this_pos = self._object_to_control.world.position
        ob_pos = ob.world.position
        self._ref = {
            "kind": kind,
            "event_pos": np.array((event.x, event.y)),
            "dim": ob.dim,
            "maxdist": 0,
            # Transform at time of start
            "pos": self._object_to_control.world.position,
            "scale": self._object_to_control.world.scale,
            "rot": self._object_to_control.world.rotation,
            "world_pos": ob_pos,
            "world_offset": ob_pos - this_pos,
            "ndc_pos": la.vec_transform(ob_pos, self._camera.camera_matrix),
            # Gizmo direction state at start-time of drag
            "flips": np.sign(self.world.scale),
            "world_directions": self._world_directions.copy(),
            "ndc_directions": self._ndc_directions.copy(),
            "screen_directions": self._screen_directions.copy(),
        }

    def _handle_translate_move(self, event):
        """Translate action, either using a translate1 or translate2 handle."""

        world_to_screen = self._ndc_to_screen @ self._camera.camera_matrix
        screen_to_world = la.mat_inverse(world_to_screen)

        if isinstance(self._ref["dim"], int):
            travel_directions = (self._ref["dim"],)
        else:
            travel_directions = self._ref["dim"]

        screen_travel = np.array(
            (
                event.x - self._ref["event_pos"][0],
                event.y - self._ref["event_pos"][1],
            )
        )

        # units dragged along gizmo axes
        screen_directions = self._ref["screen_directions"][travel_directions, :]
        if len(screen_directions) == 1:
            # translate 1D: only count movement along translation axis
            units_traveled = get_scale_factor(screen_directions[..., :2], screen_travel)
        else:
            # translate 2D: change basis from screen to gizmo axes
            screen_to_axes = la.mat_inverse(screen_directions[..., :2].T)
            units_traveled = screen_to_axes @ screen_travel

        # pixel units to world units
        # Note: location of translation matters because perspective cameras have
        # shear, i.e., we need to account for start
        start = la.vec_transform(self._ref["world_pos"], world_to_screen)
        end = start + screen_directions.T @ units_traveled
        end_world = la.vec_transform(end, screen_to_world)
        world_units_traveled = end_world - self._ref["world_pos"]

        self._object_to_control.world.position = self._ref["pos"] + world_units_traveled

    def _update_directions(self):
        """
        Calculate how much 1 unit of translation in the draggable space (aka
        mode) translates the object in world and screen space.

        """

        # work out the transforms between the spaces
        camera = self._camera

        local_to_world = self._object_to_control.world.matrix
        local_to_ndc = camera.camera_matrix @ local_to_world
        local_to_screen = self._ndc_to_screen @ local_to_ndc

        # points referring to local coordinate axes and origin
        local_points = np.zeros((4, 3))
        local_points[1:, :] = np.eye(3)

        # express unit vectors and origin in the various frames
        world_points = la.vec_transform(local_points, local_to_world)
        ndc_points = la.vec_transform(local_points, local_to_ndc)
        screen_points = la.vec_transform(local_points, local_to_screen)

        # store the directions for future use
        self._world_directions = world_points[1:] - world_points[0]
        self._ndc_directions = ndc_points[1:] - ndc_points[0]
        self._screen_directions = screen_points[1:] - screen_points[0]
    
def get_scale_factor(vec1, vec2):
    """
    Vector project vec2 onto vec1. Aka, figure out how long vec2
    is when measured along vec1.

    This is used, for example, to work out how many units the cursor has
    traveled along a given direction.
    """

    # Note: implementing it like this saves a couple square-roots from
    # normalizing
    return np.sum(vec2 * vec1, axis=-1) / np.sum(vec1**2, axis=-1)


def deg_to_rad(degrees):
    return degrees / 360 * (2 * np.pi)


class Text(TranformHelper):
    def __init__(self,text):
        super().__init__()
        text = gfx.Text(text=text,font_size=0.01,material=gfx.TextMaterial(pick_write=True))
        self.add(text)

class Bitmap(TranformHelper):
    def __init__(self):
        super().__init__()
        im = (np.indices((10, 10)).sum(axis=0) % 2).astype(np.float32)
        tex = gfx.Texture(im*255,dim=2)
        obj = gfx.Mesh(gfx.plane_geometry(0.02,0.02),gfx.MeshBasicMaterial(pick_write=True,map=gfx.TextureMap(tex,filter='nearest')))
        self.add(obj)

class Vectorgraph(TranformHelper):
    def __init__(self):
        super().__init__()
        # 一个简单的 SVG 示例：带边框的圆和文字
        svg = '''<svg width="1024" height="1024" xmlns="http://www.w3.org/2000/svg">
            <circle cx="512" cy="512" r="512" stroke="#ffffff" stroke-width="10"/>
            </svg>'''

        # 使用 CairoSVG 在内存中栅格化为 PNG 字节
        data = svg2png(bytestring=svg.encode("utf-8"),output_width=1024,output_height=1024)
        img = Image.open(io.BytesIO(data)).convert("RGBA")
        arr = np.array(img)

        # 创建纹理贴到平面
        tex = gfx.Texture(arr,dim=2)
        geom = gfx.plane_geometry(0.02, 0.02)
        mat = gfx.MeshBasicMaterial(pick_write=True,map=gfx.TextureMap(tex, filter='linear'))
        obj = gfx.Mesh(geom,mat)
        self.add(obj)
        

class Engravtor(gfx.WorldObject):
    def __init__(self,env_map = None):
        super().__init__()
        path = files("simtoy.data.engravtor") / "engravtor.gltf"
        self.scene : gfx.Scene = gfx.load_gltf(path).scene
        self.scene.traverse(lambda o: setattr(o,'cast_shadow',True) or setattr(o,'receive_shadow',True),True)
        # self.scene.traverse(lambda o: o.material is not None and setattr(o.material,'env_map',env_map),True)

        tool : gfx.WorldObject = self.scene.children[0]
        self.add(tool)

        self.target_area : gfx.PerspectiveCamera = next(tool.iter(lambda o: o.name == '工作区-内'))

        camera : gfx.PerspectiveCamera = next(tool.iter(lambda o: o.name == '摄像头'))
        camera.show_pos(self.target_area.world.position,up=[0,0,1])
        camera.local.scale = 1

        persp_camera : gfx.PerspectiveCamera = next(tool.iter(lambda o: o.name == '观察点'))
        persp_camera.show_pos(self.target_area.world.position,up=[0,0,1],depth=1.0)
        self.persp_camera = persp_camera

        # ortho_camera : gfx.OrthographicCamera = next(gltf.scene.iter(lambda o: o.name == '正交相机'))
        # ortho_camera.show_pos(target.world.position,up=[0,0,1])

        # self.controller = gfx.OrbitController()
        # self.controller.add_camera(persp_camera)
        # self.controller.add_camera(ortho_camera)

    def step(self,dt):
        # self.persp_camera.world.z = self.target_area.world.z
        pass

    def get_view_focus(self):
        return self.camera.local.position,self.target_area.local.position

    def get_consumables(self):
        return ['摄像头画面','木板-110x110x1','木板-110x110x10']

    def set_consumable(self,name):
        target : gfx.WorldObject = next(self.scene.iter(lambda o: o.name == name))
        target.material.pick_write = True
        target.cast_shadow = True
        target.receive_shadow=True
        target.local.position = self.target_area.local.position
        aabb = target.get_bounding_box()
        target_height = (aabb[1][2] - aabb[0][2])
        target_height_offset = target_height / 2
        target.local.z += target_height_offset
        self.target_area.add(target)
    
        target.add_event_handler(lambda e: e.button == 3 and self.unselect_all(target),'pointer_down')

    def unselect_all(self,parent : gfx.WorldObject):
        for obj in parent.children:
            obj : TranformHelper
            obj.set_tranform_visible(False)

    def get_viewport(self):
        return [self.persp_camera]

    def get_hot_items(self):
        def text():
            target = self.target_area.children[0]
            aabb = target.get_bounding_box()
            target_height = (aabb[1][2] - aabb[0][2])
            
            element = Text('Text')
            element_height_offset = target_height / 2 
            element.local.z += element_height_offset
            element._camera = self.persp_camera
            element.set_tranform_visible(False)
            target.add(element)

            # element.add_event_handler(lambda e: e.button == 3 and self.unselect_all(target),'pointer_down')

        def bitmap():
            target = self.target_area.children[0]
            aabb = target.get_bounding_box()
            target_height = (aabb[1][2] - aabb[0][2])
            
            element = Bitmap()
            element_height_offset = target_height / 2
            element.local.z += element_height_offset
            element._camera = self.persp_camera 
            element.set_tranform_visible(False)
            target.add(element)

        def vectorgraph():
            target = self.target_area.children[0]
            aabb = target.get_bounding_box()
            target_height = (aabb[1][2] - aabb[0][2])
            
            element = Vectorgraph()
            element_height_offset = target_height / 2
            element.local.z += element_height_offset
            element._camera = self.persp_camera
            element.set_tranform_visible(False)
            target.add(element)

        return [('文本',text,'format-text-bold'),('位图',bitmap,'image-x-generic-symbolic'),('矢量图',vectorgraph,None)]

    def get_actbar(self):
        return []