import wgpu
import pygfx as gfx
# import pybullet as bullet

import numpy as np
import pylinalg as la

class Scene(gfx.Scene):
    steps  = list()
    
    def __init__(self):
        super().__init__()
        # bid = bullet.connect(bullet.DIRECT)
        # bullet.setPhysicsEngineParameter(erp=1,contactERP=1,frictionERP=1)
        # bullet.setGravity(0, 0, -9.81)
        pass

    def step(self,dt=1/240):
        if self.steps: 
            if not self.steps[0](): return
            self.steps.pop(0)

        for entity in self.children:
            if 'step' not in dir(entity): continue 
            entity.step(dt)

        # bullet.stepSimulation()

    # def ray_intersect(self,origin,direction):
    #     intersection = None
    #     for entity,properties in self.children.items():
    #         id,pos,rot = properties
    #         if 'ray_intersect' in dir(entity):
    #             origin = origin - np.array(pos)
    #             direction = Rotation.from_euler('xyz',rot).inv().apply(direction)
    #             intersection = entity.ray_intersect(origin,direction)

    #         if not intersection: continue
    #         link_id,friction,position,normal = intersection
    #         position = position + np.array(pos)
    #         normal = Rotation.from_euler('xyz',rot).apply(normal)
    #         intersection = entity,link_id,friction,position,normal
    #     return intersection
    