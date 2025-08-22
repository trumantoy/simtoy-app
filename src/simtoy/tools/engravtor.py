import pygfx as gfx
from importlib.resources import files

import numpy as np

class Engravtor(gfx.WorldObject):
    def __init__(self,ao_map = None):
        super().__init__()
        path = files("simtoy.data.engravtor") / "engravtor.gltf"
        self.scene : gfx.Scene = gfx.load_gltf(path).scene
        self.scene.traverse(lambda o: setattr(o,'cast_shadow',True) or setattr(o,'receive_shadow',True),True)
        
        # def f(o):
        #     if 'material' in vars(o):
        #         setattr(o.material,'ao_map',ao_map)

        # self.scene.traverse(f,True)
        
        tool : gfx.WorldObject = self.scene.children[0]
        self.add(tool)

        self.origin_obj : gfx.WorldObject = next(tool.iter(lambda o: o.name == '目标点'))
        
        camera : gfx.PerspectiveCamera = next(tool.iter(lambda o: o.name == '摄像头'))
        camera.show_pos(self.origin_obj.world.position,up=[0,0,1])

        persp_camera : gfx.PerspectiveCamera = next(tool.iter(lambda o: o.name == '观察点'))
        persp_camera.depth_range = None
        persp_camera.show_pos(self.origin_obj.local.position,up=[0,0,1],depth=0.7690542914429888)

        # ortho_camera : gfx.OrthographicCamera = next(gltf.scene.iter(lambda o: o.name == '正交相机'))
        # ortho_camera.show_pos(origin_obj.world.position,up=[0,0,1])

        self.controller = gfx.OrbitController()
        self.controller.add_camera(persp_camera)
        # self.controller.add_camera(ortho_camera)


    def step(self,dt):
        # self.light.local.position = self.controller.cameras[0].local.position
        pass

    def get_view_focus(self):
        return self.controller.cameras[0].local.position,self.origin_obj.local.position

    def get_consumables(self):
        return ['木板-100x100x1']

    def set_consumable(self,name):
        wood : gfx.WorldObject = next(self.scene.iter(lambda o: o.name == '木板-100x100x1'))
        wood.cast_shadow = True
        wood.receive_shadow=True
        wood.local.position = self.origin_obj.local.position
        wood.local.z += 0.0005
        self.add(wood)