import wgpu 
import wgpu.gui
import wgpu.gui.offscreen
import pygfx as gfx
# import pybullet as bullet
# import pybullet_data as bdata
from pygfx.renderers.wgpu import *
from pygfx.objects import WorldObject
from pygfx.materials import Material

import math as m
import numpy as np

import imageio.v3 as iio
from importlib.resources import files

class SkyBox(gfx.Background):
    def __init__(self):
        self.model_path = ''
        px = files("simtoy.data.builtin").joinpath("px.png")
        nx = files("simtoy.data.builtin").joinpath("nx.png")
        py = files("simtoy.data.builtin").joinpath("py.png")
        ny = files("simtoy.data.builtin").joinpath("ny.png")
        pz = files("simtoy.data.builtin").joinpath("pz.png")
        nz = files("simtoy.data.builtin").joinpath("nz.png")

        posx = iio.imread(px)
        negx = iio.imread(nx)
        posy = iio.imread(py)
        negy = iio.imread(ny)
        posz = iio.imread(pz)
        negz = iio.imread(nz)
        pictures = np.stack([posx, negx, posy, negy, posz, negz], axis=0)

        len,h,w,ch = pictures.shape
        tex = gfx.Texture(np.stack(pictures, axis=0), dim=2, size=(w, h, 6), generate_mipmaps=True)
        super().__init__(None, gfx.BackgroundSkyboxMaterial(map=tex))
        pass

    def step(self,dt):
        self.local.euler_x = 1.57
        pass

class Ground(gfx.Mesh):
    def __init__(self):
        checker_blue = files("simtoy.data.builtin") / "checker_blue.png"
        
        im = iio.imread(checker_blue).astype("float32") / 255
        material = gfx.MeshPhongMaterial(map=gfx.Texture(im, dim=2,generate_mipmaps=True),pick_write=True)
        geom = gfx.plane_geometry(100, 100, 1)
        geom.texcoords.data[:, :] *= 100/2
        super().__init__(geom,material)
        self.local.z = -0.501
        pass
    
    def step(self,dt):
        pass

class MyMaterial(Material):
    uniform_type = dict(
        gfx.Material.uniform_type,
        height="f4",
    )

    def __init__(self,*,height = 1.0,**kwargs):
        super().__init__(**kwargs)

        self.uniform_buffer.data["height"] = height

@register_wgpu_render_function(WorldObject, MyMaterial)
class CustomShader(BaseShader):
    type = "render"

    def get_bindings(self, wobject : WorldObject, shared):
        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", wobject.material.uniform_buffer),
            Binding("s_positions", "buffer/read_only_storage", wobject.geometry.positions)
        ]
        bindings = {i:b for i, b in enumerate(bindings)}
        self.define_bindings(0, bindings)
        return {
            0: bindings,
        }

    def get_pipeline_info(self, wobject, shared):
        return {
            "primitive_topology": wgpu.PrimitiveTopology.point_list,
            "cull_mode": wgpu.CullMode.none,
        }

    def get_render_info(self, wobject, shared):
        size = wobject.geometry.positions.data.shape
        return {
            "indices": (size[0], 1),
            "render_mask": RenderMask.all,
        }

    def get_code(self):
        return '''{$ include 'pygfx.std.wgsl' $}
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> Varyings {
    let u_mvp = u_stdinfo.projection_transform * u_stdinfo.cam_transform * u_wobject.world_transform;
    let pos = load_s_positions(i32(i));
    let ndc_pos = u_mvp * vec4<f32>(pos.xyz, 1.0);

    let screen_factor = u_stdinfo.logical_size.xy / 2.0;
    let screen_pos_ndc = ndc_pos.xy + 10 * pos.xy / screen_factor;
    
    var varyings: Varyings;
    varyings.position = vec4<f32>(ndc_pos.xy, ndc_pos.z, ndc_pos.w);
    varyings.world_pos = vec4<f32>(pos,1.0);
    return varyings;
}

@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput { 
    let height = u_material.height;
    var z = (varyings.world_pos.z - -0.5) / height;

    let h = f32(1 - z) * 120 / 360;
    let s = 1.0;
    var v = 1.0;
    
    if (z > 1.0) {
        v = 0.0;
    }

    let i = i32(h * 6);
    let f = h * 6 - f32(i);
    let p = v * (1 - s);
    let q = v * (1 - s * f);
    let t = v * (1 - s * (1 - f));
    var r: f32;
    var g: f32;
    var b: f32;

    if (i % 6 == 0) {
        r = v; g = t; b = p;
    } else if (i % 6 == 1) {
        r = q; g = v; b = p;
    } else if (i % 6 == 2) {
        r = p; g = v; b = t;
    } else if (i % 6 == 3) {
        r = p; g = q; b = v;
    } else if (i % 6 == 4) {
        r = t; g = p; b = v;
    } else if (i % 6 == 5) {
        r = v; g = p; b = q;
    }

    var out: FragmentOutput;
    out.color = vec4<f32>(r, g, b, 1.0);
    return out;
}
'''

class PointCloud(gfx.Mesh):
    name = '点云'
    
    def __init__(self):
        super().__init__(gfx.box_geometry(1,1,1),gfx.MeshBasicMaterial(opacity=0.2))

        positions = np.random.uniform(-0.5, 0.5, (100000, 3)).astype(np.float32)
        geometry = gfx.Geometry(positions=positions)
        material = MyMaterial()
        self.points = gfx.Points(geometry,material)
        self.add(self.points)
        self.set_bounding_box_visible(True)

    def step(self,dt):
        pass

    def set_from_file(self,filepath):
        import laspy
        las = laspy.read(filepath)  # 修改为你的文件路径
        x = las.x - np.min(las.x)
        y = las.y - np.min(las.y)
        z = las.z - np.min(las.z)
        max_range = max(x.max()-x.min(), y.max()-y.min(), z.max()-z.min())
        scale = 1 / max_range
        positions = np.column_stack([x,y,z]).astype(np.float32) * scale
        height = np.max(positions[:,2])
        positions = positions - 0.5
        self.remove(self.points)
        geometry = gfx.Geometry(positions=positions)
        material = MyMaterial()
        self.points = gfx.Points(geometry,material)
        self.add(self.points)

        self.set_height(height)

    def set_bounding_box_visible(self, visible : bool):
        self.material.opacity = 0.2 if visible else 0.0
    
    def set_height(self,height):
        self.geometry = gfx.box_geometry(1,1,height)
        self.geometry.positions.data[:,2] += -(1 - height) / 2

        self.points.material.uniform_buffer.data['height'] = height
        self.points.material.uniform_buffer.update_full()

    def get_stored_items(self):
        return [
            Triangle,
            Box,
            Sphere,
            Cylidar,
        ]
    
    @staticmethod
    def ray_box_intersection(O, D, min_point, max_point):
        """
        计算射线与长方体的交点
        :param O: 射线起点 (Ox, Oy, Oz)
        :param D: 射线方向向量 (Dx, Dy, Dz)
        :param min_point: 长方体的最小顶点 (x_min, y_min, z_min)
        :param max_point: 长方体的最大顶点 (x_max, y_max, z_max)
        :return: 交点坐标 (x, y, z)，若无交点返回 None
        """
        # 计算射线与长方体各面的交点参数 t
        tmin = (min_point - O) / D
        tmax = (max_point - O) / D

        # 确保 tmin 小于 tmax
        t1 = np.minimum(tmin, tmax)
        t2 = np.maximum(tmin, tmax)

        # 计算射线与长方体相交的最小和最大参数
        t_enter = np.max(t1)
        t_exit = np.min(t2)

        # 检查是否有交点
        if t_enter > t_exit or t_exit < 0:
            return None

        # 计算交点坐标
        if t_enter >= 0:
            intersection = O + t_enter * D
        else:
            intersection = O + t_exit * D

        return intersection

    def pick(self,origin,direction):
        intersections = list()
        for entity in self.children:
            aabb = entity.get_world_bounding_box()
            if aabb is None: continue
            hit_pos = self.ray_box_intersection(origin,direction,aabb[0],aabb[1])
            if hit_pos is None: continue
            if entity == self.points: entity = None
            fraction = np.linalg.norm(direction)
            intersections.append((entity,hit_pos,origin,direction / fraction,fraction))
        if not intersections: return None
        intersections.sort(key=lambda x: np.linalg.norm(x[1] - origin))
        return intersections[0]

class Triangle(gfx.Mesh):
    name = '三角形'
    def __init__(self):
         # 定义三角形的顶点坐标
        positions = np.array([
            [0, 0, 0],  # 顶点1
            [1, 0, 0],  # 顶点2
            [0, 1, 0]   # 顶点3
        ], dtype=np.float32)

        # 定义三角形的索引
        indices = np.array([[0, 1, 2]], dtype=np.uint32)

        # 创建几何对象
        geometry = gfx.Geometry(positions=positions, indices=indices)
        super().__init__(geometry,gfx.MeshPhongMaterial())
        pass
    
    def step(self,dt):
        pass

class Box(gfx.Mesh):
    name = '四面体'

    def __init__(self):
        super().__init__(gfx.box_geometry(0.1,0.1,0.1),gfx.MeshBasicMaterial(opacity=0.2))
        self.box = gfx.Mesh(gfx.box_geometry(0.1,0.1,0.1),gfx.MeshBasicMaterial(color='#87CEEB'))
        self.add(self.box)
    
    def step(self,dt):
        pass

    def set_bounding_box_visible(self, visible : bool):
        self.material.opacity = 0.2 if visible else 0.0

class Sphere(gfx.Mesh):
    name = '球体'
    def __init__(self):
        super().__init__(gfx.sphere_geometry(0.1),gfx.MeshPhongMaterial())
        pass
    
    def step(self,dt):
        pass

class Cylidar(gfx.Mesh):
    name = '柱体'
    def __init__(self):
        super().__init__(gfx.cylinder_geometry(0.1,0.1),gfx.MeshPhongMaterial())
        pass
    
    def step(self,dt):
        pass


