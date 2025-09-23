from typing import Literal, TypedDict
import pygfx as gfx
# import pybullet as bullet

import numpy as np
import pylinalg as la
from ..tools.builtin import *

class Editor(gfx.Scene):
    def __init__(self):
        super().__init__()
        self.steps = list()

        self.toolbar = list()
        self.actionbar = list()

        ortho_camera = gfx.OrthographicCamera()
        persp_camera = gfx.PerspectiveCamera()
        persp_camera.local.position = ortho_camera.local.position = [0,-0.8,0.2]
        ortho_camera.show_pos([0,0,0],up=[0,0,1])
        persp_camera.show_pos([0,0,0],up=[0,0,1])
        self.persp_camera = persp_camera
        self.ortho_camera = ortho_camera
                
        # grid0 = gfx.Grid(
        #     gfx.box_geometry(),
        #     gfx.GridMaterial(
        #         major_step=1,
        #         minor_step=0.1,
        #         thickness_space="world",
        #         axis_thickness=0,
        #         major_thickness=0.005,
        #         minor_thickness=0.001,
        #         infinite=True,
        #     ),
        #     orientation="xy",
        # )
        # self.add(grid0)

        # self.skybox = SkyBox()
        # self.add(self.skybox)

        # self.env_map = self.skybox.material.map

        # self.ground = Ground()
        # self.ground.receive_shadow = True
        # self.ground.local.z -= 0.001
        # self.ground.material.env_map = self.env_map

        # self.add(self.ground)

        ambient = gfx.AmbientLight()
        self.add(ambient)

        light = light = gfx.DirectionalLight(cast_shadow = True)
        light.local.position = (0.2, -1, 0.3)
        light.shadow.camera.width = light.shadow.camera.height = 1
        self.add(light)
        
        
    def step(self,dt=1/240):
        if self.steps:
            self.steps[0]()
            self.steps.pop(0)

        for entity in self.children:
            if 'step' not in dir(entity): continue 
            entity.step(dt)


    def get_viewport(self):
        return [self.persp_camera,self.ortho_camera]
        
    def switch_view_focus(self,origin,target):
        for camera in self.view_controller.cameras:
            camera : gfx.PerspectiveCamera
            camera.local.position = origin
            camera.show_pos(target)
        pass

    def restore_view_focus():
        pass