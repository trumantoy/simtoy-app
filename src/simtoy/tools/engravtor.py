import wgpu
import pygfx as gfx
from pygfx.renderers.wgpu import *
from pygfx.objects import WorldObject
from pygfx.materials import Material
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
                let half = 1.0 * u_material.size;
                
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
                let dist_to_edge = length(varyings.pointcoord_p) - 0.5 * varyings.size_p;
                
                // 描边逻辑：判断当前片段是否在描边范围内
                let l2p = u_stdinfo.physical_size.x / u_stdinfo.logical_size.x;
                let edge_width = 0.75 * l2p;  // 描边宽度（物理像素）
                
                // 判断片段类型
                let is_edge = abs(dist_to_edge) < edge_width;    // 是否在描边范围内
                let is_inside = dist_to_edge < 0.0;              // 是否在点内部
                
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
                    // 点外部：完全透明
                    final_color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                }
                
                // 设置最终输出颜色
                out.color = final_color;
                return out;
            }
        """


class TranformHelper(gfx.Mesh):
    def __init__(self, geometry=None, material=None, *args, **kwargs):
        super().__init__(geometry, material, *args, **kwargs)

        point = gfx.Points(gfx.Geometry(positions=[(0,0,0)]),OriginMaterial())
        self.add(point)

        # geom = gfx.Geometry(positions=[(0,0,0),(0,0,1)])
        # material = gfx.LineMaterial(thickness=1, color='red')
        # x_axis = gfx.Line(geom,material)
        # self.add(x_axis)

        pass

class Text(TranformHelper):
    def __init__(self,text):
        text = gfx.Text(text=text,font_size=0.01,material=gfx.TextMaterial())
        aabb = text.get_bounding_box()
        super().__init__(gfx.plane_geometry(aabb[1][0]-aabb[0][0], aabb[1][1]-aabb[0][1]))
        self.add(text)
    
class Bitmap(TranformHelper):
    def __init__(self):
        im = (np.indices((10, 10)).sum(axis=0) % 2).astype(np.float32)
        tex = gfx.Texture(im*255,dim=2)
        super().__init__(gfx.plane_geometry(0.02,0.02),gfx.MeshBasicMaterial(pick_write=True,map=gfx.TextureMap(tex,filter='nearest')))

class Vectorgraph(TranformHelper):
    def __init__(self):
        # 一个简单的 SVG 示例：带边框的圆和文字
        svg = '''<svg width="512" height="512" xmlns="http://www.w3.org/2000/svg">
            <circle cx="255.5" cy="255.5" r="255" stroke="#ffffff" stroke-width="10"/>
            </svg>'''

        # 使用 CairoSVG 在内存中栅格化为 PNG 字节
        data = svg2png(bytestring=svg.encode("utf-8"),output_width=512,output_height=512)
        img = Image.open(io.BytesIO(data)).convert("RGBA")
        arr = np.array(img)

        # 创建纹理贴到平面
        tex = gfx.Texture(arr,dim=2)
        geom = gfx.plane_geometry(0.02, 0.02)
        mat = gfx.MeshBasicMaterial(map=gfx.TextureMap(tex, filter='linear'))
        super().__init__(geom, mat)


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
        target.cast_shadow = True
        target.receive_shadow=True
        target.local.position = self.target_area.local.position
        aabb = target.get_bounding_box()
        target_height = (aabb[1][2] - aabb[0][2])
        target_height_offset = target_height / 2
        target.local.z += target_height_offset
        self.target_area.add(target)

    def get_viewport(self):
        return [self.persp_camera]

    def get_hot_items(self):
        def text():
            target = self.target_area.children[0]
            aabb = target.get_bounding_box()
            target_height = (aabb[1][2] - aabb[0][2])
            
            element = Text('Text')
            element_height_offset = target_height / 2 + 0.0001
            element.local.z += element_height_offset
            target.add(element)
        def bitmap():
            target = self.target_area.children[0]
            aabb = target.get_bounding_box()
            target_height = (aabb[1][2] - aabb[0][2])
            
            element = Bitmap()
            element_height_offset = target_height / 2 + 0.0001
            element.local.z += element_height_offset
            element.local.x -= 0.05
            target.add(element)

        def vectorgraph():
            target = self.target_area.children[0]
            aabb = target.get_bounding_box()
            target_height = (aabb[1][2] - aabb[0][2])
            
            element = Vectorgraph()
            element_height_offset = target_height / 2 + 0.0001
            element.local.z += element_height_offset
            element.local.x += 0.05
            target.add(element)

        return [('文本',text,'format-text-bold'),('位图',bitmap,'image-x-generic-symbolic'),('矢量图',vectorgraph,None)]

    def get_actbar(self):
        return []