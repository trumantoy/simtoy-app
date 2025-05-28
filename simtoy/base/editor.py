import wgpu 
import wgpu.gui
import wgpu.gui.offscreen
import pygfx as gfx
# import pybullet as bullet

import math as m
import numpy as np
import pylinalg as la

from .scene import *
from ..objects.builtin import * 

class Editor(Scene):
    def __init__(self):
        super().__init__()
        grid0 = gfx.Grid(
            gfx.box_geometry(),
            gfx.GridMaterial(
                major_step=1,
                minor_step=0.1,
                thickness_space="world",
                axis_thickness=0.1,
                major_thickness=0.02,
                minor_thickness=0.002,
                infinite=True,
            ),
            orientation="xy",
        )
        self.add(grid0)

        self.ground = Ground()
        self.ground.receive_shadow = True
        # self.ground.visible = False
        self.add(self.ground)

        self.skybox = SkyBox()
        self.add(self.skybox)
        
        ambient = gfx.AmbientLight()
        self.add(ambient)

        light = gfx.DirectionalLight()
        light.world.position = (50, -50, 100)
        light.shadow.camera.width = light.shadow.camera.height = 50
        light.cast_shadow = True
        self.add(light)


    @staticmethod
    def ray_plane_intersection(O, D, P, N):
        """
        计算射线与平面的交点
        :param O: 射线起点 (Ox, Oy, Oz)
        :param D: 射线方向向量 (Dx, Dy, Dz)
        :param P: 平面上的一点 (Px, Py, Pz)
        :param N: 平面的法向量 (Nx, Ny, Nz)
        :return: 交点坐标 (x, y, z)，若无交点返回 None
        """
        # 计算射线起点到平面上一点的向量
        OP = [P[0] - O[0], P[1] - O[1], P[2] - O[2]]

        # 计算射线方向向量与平面法向量的点积
        D_dot_N = D[0] * N[0] + D[1] * N[1] + D[2] * N[2]

        # 如果点积为0，则射线与平面平行，无交点
        if D_dot_N == 0:
            return None

        # 计算交点参数t
        t = (N[0] * OP[0] + N[1] * OP[1] + N[2] * OP[2]) / D_dot_N

        # 计算交点坐标
        intersection = [O[0] + t * D[0], O[1] + t * D[1], O[2] + t * D[2]]

        return intersection
    
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
    
    pick_info = (np.array([0,0,0]),None)
    def pick(self,pixel_x,pixel_y,owner = None):
        origin = self.active_camera.local.position
        rot = self.active_camera.local.euler
        screen_width,screen_height = self.canvas.get_physical_size()
        pixel_center_x = screen_width / 2
        pixel_center_y = screen_height / 2 
        centered_x = pixel_x - pixel_center_x
        centered_y = pixel_center_y - pixel_y
        ndc_x = centered_x / pixel_center_x
        ndc_y = centered_y / pixel_center_y

        if self.active_camera == self.ortho_camera:
            ndc_pos = [ndc_x,ndc_y,0] + la.vec_transform(self.active_camera.world.position, self.active_camera.camera_matrix)
            origin = la.vec_unproject(ndc_pos[:2], self.active_camera.camera_matrix)
            ray_direction = origin.copy()
            ray_direction[2] = -1
        else:
            ray_direction = la.vec_unproject([ndc_x,ndc_y], self.active_camera.projection_matrix) * 1000
            ray_direction = la.vec_transform_quat(ray_direction,la.quat_from_euler(rot))

        intersections = []
        for entity in (owner if owner else self).children:
            if 'simtoy.entities' not in type(entity).__module__: continue
            if 'Ground' == type(entity).__name__: continue

            aabb = entity.get_world_bounding_box()
            if aabb is None: continue
            hit_pos = self.ray_box_intersection(origin,ray_direction/np.linalg.norm(ray_direction),aabb[0],aabb[1])
            if hit_pos is None: continue
            intersections.append((entity,hit_pos,origin,ray_direction,np.linalg.norm(ray_direction)))
        if not intersections: return None
        intersections.sort(key=lambda x: np.linalg.norm(x[1] - origin))
        return intersections[0]

    def get_stored_items(self):
        return [PointCloud]
    