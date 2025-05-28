import numpy as np
import math as m
import traceback as tb
import imageio.v3 as iio
from scipy.spatial.transform import Rotation

class Plane:
    def __init__(self):
        self.model_path = ''
        self.width = 1.7
        self.height = 1
        self.center = np.array([0,0,0])
        half_width = self.width / 2
        half_height = self.height / 2
        depth = 0

        self.texture = iio.imread(R"C:\Users\SLTru\Downloads\textures\画板.jpg")

        self.points = [
            self.center + np.array([-half_width,half_height,depth]), 
            self.center + np.array([half_width,half_height,depth]),
            self.center + np.array([-half_width,-half_height,depth]), 
            self.center + np.array([half_width,-half_height,depth]),
        ]

        self.uvs = [
            (0. ,0.),(1. ,0.),(0. ,1.),(1. ,1.)
        ]

        self.indices = [
            (0,1,2),(3,2,1)
        ]

        self.normals = [
            (0,0,1),(0,0,1)
        ]
        pass
    
    def step(self,dt):
        pass

    def ray_intersect(self,origin,direction):
        normal = np.array([0,0,1])
        direction = np.array(direction)
        denom = np.dot(normal,direction)
        fraction = np.dot(normal,self.center - origin) / denom
        position = origin + fraction * direction
        
        half_width = self.width / 2
        half_height = self.height / 2
        if position[0] < -half_width or half_width < position[0] or \
           position[1] < -half_height or half_height < position[1]:
           return None

        if fraction < 0: return None

        link_id = None
        for i,c in enumerate(self.points):
            radius = 0.02
            o = np.array(origin)
            d = np.array(direction)
            oc = o - np.array(c)
            a = np.dot(d,d)
            b = 2 * np.dot(oc, d)
            c = np.dot(oc,oc) - radius * radius
            discriminant = b * b - 4 * a * c
            if discriminant > 0: link_id = i

        return link_id,fraction,position,normal
    