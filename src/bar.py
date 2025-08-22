import numpy as np
import pygfx as gfx

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk, Gio

from scipy.spatial import Delaunay
from simtoy import *
from panel import *

@Gtk.Template(filename='ui/actionbar.ui')
class Actionbar (Gtk.ScrolledWindow):
    __gtype_name__ = "Actionbar"
    
    def __init__(self):
        provider = Gtk.CssProvider.new()
        provider.load_from_path('ui/actionbar.css')
        Gtk.StyleContext.add_provider_for_display(self.get_display(),provider,Gtk.STYLE_PROVIDER_PRIORITY_USER)

@Gtk.Template(filename='ui/viewbar.ui')
class Viewbar (Gtk.ScrolledWindow):
    __gtype_name__ = "Viewbar"

    view_mode : Gtk.Button = Gtk.Template.Child('view_mode')

    def __init__(self):
        provider = Gtk.CssProvider.new()
        provider.load_from_path('ui/viewbar.css')
        Gtk.StyleContext.add_provider_for_display(self.get_display(),provider,Gtk.STYLE_PROVIDER_PRIORITY_USER)

    def set_editor(self,view_controller : gfx.OrbitController):
        self.view_controller = view_controller

    @Gtk.Template.Callback()
    def on_top_clicked(self,button):
        perspective,orthographic = self.view_controller.cameras
        perspective : gfx.PerspectiveCamera
        orthographic : gfx.OrthographicCamera        
        extent = perspective.height
        factor = 0.5 / m.tan(0.5 * m.radians(perspective.fov))
        distance = extent * factor
        origin = perspective.local.position
        direction = perspective.local.forward
        target = origin + direction * distance
        perspective.local.position = target + np.array([0,0,1]) * distance
        orthographic.local.position = target + np.array([0,0,1]) * distance
        perspective.local.euler = np.array([0,0,perspective.local.euler_z])
        orthographic.local.euler = np.array([0,0,orthographic.local.euler_z])

    @Gtk.Template.Callback()
    def on_bottom_clicked(self,button):
        perspective,orthographic = self.view_controller.cameras
        perspective : gfx.PerspectiveCamera
        orthographic : gfx.OrthographicCamera        
        extent = perspective.height
        factor = 0.5 / m.tan(0.5 * m.radians(perspective.fov))
        distance = extent * factor
        origin = perspective.local.position
        direction = perspective.local.forward
        target = origin + direction * distance
        perspective.local.position = target + np.array([0,0,-1]) * distance
        orthographic.local.position = target + np.array([0,0,-1]) * distance
        perspective.local.euler = np.array([m.pi,0,perspective.local.euler_z])
        orthographic.local.euler = np.array([m.pi,0,orthographic.local.euler_z])

    @Gtk.Template.Callback()
    def on_left_clicked(self,button):
        perspective,orthographic = self.view_controller.cameras
        perspective : gfx.PerspectiveCamera
        orthographic : gfx.OrthographicCamera        
        extent = perspective.height
        factor = 0.5 / m.tan(0.5 * m.radians(perspective.fov))
        distance = extent * factor
        origin = perspective.local.position
        direction = perspective.local.forward
        target = origin + direction * distance
        perspective.local.position = target + np.array([-1,0,0]) * distance
        orthographic.local.position = target + np.array([-1,0,0]) * distance
        perspective.look_at(target)
        orthographic.look_at(target)

    @Gtk.Template.Callback()
    def on_right_clicked(self,button):
        perspective,orthographic = self.view_controller.cameras
        perspective : gfx.PerspectiveCamera
        orthographic : gfx.OrthographicCamera        
        extent = perspective.height
        factor = 0.5 / m.tan(0.5 * m.radians(perspective.fov))
        distance = extent * factor
        origin = perspective.local.position
        direction = perspective.local.forward
        target = origin + direction * distance
        perspective.local.position = target + np.array([1,0,0]) * distance
        orthographic.local.position = target + np.array([1,0,0]) * distance
        perspective.look_at(target)
        orthographic.look_at(target)

    @Gtk.Template.Callback()
    def on_front_clicked(self,button):
        perspective,orthographic = self.view_controller.cameras
        perspective : gfx.PerspectiveCamera
        orthographic : gfx.OrthographicCamera        
        extent = perspective.height
        factor = 0.5 / m.tan(0.5 * m.radians(perspective.fov))
        distance = extent * factor
        origin = perspective.local.position
        direction = perspective.local.forward
        target = origin + direction * distance
        perspective.local.position = target + np.array([0,-1,0]) * distance
        orthographic.local.position = target + np.array([0,-1,0]) * distance
        perspective.look_at(target)
        orthographic.look_at(target)
        
    @Gtk.Template.Callback()
    def on_back_clicked(self,button):
        perspective,orthographic = self.view_controller.cameras
        perspective : gfx.PerspectiveCamera
        orthographic : gfx.OrthographicCamera        
        extent = perspective.height
        factor = 0.5 / m.tan(0.5 * m.radians(perspective.fov))
        distance = extent * factor
        origin = perspective.local.position
        direction = perspective.local.forward
        target = origin + direction * distance
        perspective.local.position = target + np.array([0,1,0]) * distance
        orthographic.local.position = target + np.array([0,1,0]) * distance 
        perspective.look_at(target)
        orthographic.look_at(target)

    @Gtk.Template.Callback()
    def on_persp_clicked(self,button):
        if '透视' == button.get_label():
            button.set_label('正交')
        else:
            button.set_label('透视')

@Gtk.Template(filename='ui/hotbar.ui')
class Hotbar (Gtk.ScrolledWindow):
    __gtype_name__ = "Hotbar"
    stored_items = Gtk.Template.Child('stored_items')

    def __init__(self):
        provider = Gtk.CssProvider.new()
        provider.load_from_path('ui/hotbar.css')
        Gtk.StyleContext.add_provider_for_display(self.get_display(),provider,Gtk.STYLE_PROVIDER_PRIORITY_USER)

    def set_viewbar(self,area : Gtk.DrawingArea,viewbar : Viewbar,geom_panel : Panel,editor : Editor):
        self.viewbar = viewbar
        self.area = area
        self.editor = editor
        self.geoms = geom_panel

    @Gtk.Template.Callback()
    def on_point_toggled(self,sender,*args):
        def add_point(sender,n_press,x,y):
            camera = self.viewbar.get_view_camera()
            screen_width,screen_height = self.area.get_width(),self.area.get_height()

            pixel_x,pixel_y = x,y
            origin = camera.local.position
            rot = camera.local.euler
            pixel_center_x = screen_width / 2
            pixel_center_y = screen_height / 2 
            centered_x = pixel_x - pixel_center_x
            centered_y = pixel_center_y - pixel_y
            ndc_x = centered_x / pixel_center_x
            ndc_y = centered_y / pixel_center_y

            if type(camera) == gfx.OrthographicCamera:
                ray_direction = la.vec_unproject([ndc_x,ndc_y], camera.projection_matrix)
                ray_direction[2] = origin[2]
                ray_direction = la.vec_transform_quat(ray_direction,la.quat_from_euler(rot))
                position = (ray_direction).astype(np.float32)
            else:
                ray_direction = la.vec_unproject([ndc_x,ndc_y], camera.projection_matrix) * 100
                ray_direction = la.vec_transform_quat(ray_direction,la.quat_from_euler(rot))
                position = (origin + ray_direction).astype(np.float32)
            geometry = gfx.Geometry(positions=[position])
            material = gfx.PointsMaterial(color="black", size=2)
            point = gfx.Points(geometry, material)
            self.editor.add(point)
            
            self.geoms.add('点-'+str(point.id))

        if sender.get_active():
            self.controller = Gtk.GestureClick.new()
            self.controller.connect("released", add_point)
            self.area.add_controller(self.controller)
            self.sig_clicked = sender.connect("clicked", lambda sender: sender.set_active(False))
        else:
            self.area.remove_controller(self.controller)
            sender.disconnect(self.sig_clicked)

    @Gtk.Template.Callback()
    def on_line_toggled(self,sender):
        n = 0

        def add_line(sender,n_press,x,y):
            nonlocal n
            n += 1
            camera = self.viewbar.get_view_camera()
            screen_width,screen_height = self.area.get_width(),self.area.get_height()

            pixel_x,pixel_y = x,y
            origin = camera.local.position
            rot = camera.local.euler
            pixel_center_x = screen_width / 2
            pixel_center_y = screen_height / 2 
            centered_x = pixel_x - pixel_center_x
            centered_y = pixel_center_y - pixel_y
            ndc_x = centered_x / pixel_center_x
            ndc_y = centered_y / pixel_center_y

            if type(camera) == gfx.OrthographicCamera:
                ray_direction = la.vec_unproject([ndc_x,ndc_y], camera.projection_matrix)
                ray_direction[2] = origin[2]
                ray_direction = la.vec_transform_quat(ray_direction,la.quat_from_euler(rot))
                position = (ray_direction).astype(np.float32)
            else:
                ray_direction = la.vec_unproject([ndc_x,ndc_y], camera.projection_matrix) * 100
                ray_direction = la.vec_transform_quat(ray_direction,la.quat_from_euler(rot))

            position = (origin + ray_direction).astype(np.float32)

            if n % 2 == 1:                
                geometry = gfx.Geometry(positions=[position])
                material = gfx.PointsMaterial(color="orange", size=2)
                self.begin = gfx.Points(geometry, material)
                self.editor.add(self.begin)
            else:
                begin = self.begin.geometry.positions.data[0]
                self.editor.remove(self.begin)

                end = position
                geometry = gfx.Geometry(positions=[begin,end])
                material = gfx.LineMaterial(thickness=1.0, color='black')
                line = gfx.Line(geometry, material)
                self.editor.add(line)
                self.geoms.add('线-'+str(line.id))

        if sender.get_active():
            self.controller = Gtk.GestureClick.new()
            self.controller.connect("released", add_line)
            self.area.add_controller(self.controller)
            self.sig_clicked = sender.connect("clicked", lambda sender: sender.set_active(False))
        else:
            self.area.remove_controller(self.controller)
            sender.disconnect(self.sig_clicked)
            self.editor.remove(self.begin)

    @Gtk.Template.Callback()
    def on_surface_toggled(self,sender):
        n = 0
        def add_line(sender,n_press,x,y):
            nonlocal n
            n += 1
            camera = self.viewbar.get_view_camera()
            screen_width,screen_height = self.area.get_width(),self.area.get_height()

            pixel_x,pixel_y = x,y
            origin = camera.local.position
            rot = camera.local.euler
            pixel_center_x = screen_width / 2
            pixel_center_y = screen_height / 2 
            centered_x = pixel_x - pixel_center_x
            centered_y = pixel_center_y - pixel_y
            ndc_x = centered_x / pixel_center_x
            ndc_y = centered_y / pixel_center_y

            if type(camera) == gfx.OrthographicCamera:
                ray_direction = la.vec_unproject([ndc_x,ndc_y], camera.projection_matrix)
                ray_direction[2] = origin[2]
                ray_direction = la.vec_transform_quat(ray_direction,la.quat_from_euler(rot))
                position = (ray_direction).astype(np.float32)
            else:
                ray_direction = la.vec_unproject([ndc_x,ndc_y], camera.projection_matrix) * 100
                ray_direction = la.vec_transform_quat(ray_direction,la.quat_from_euler(rot))
 
            position = (origin + ray_direction).astype(np.float32)
            if n == 1:
                self.begin = np.array([x,y])
                geometry = gfx.Geometry(positions=[position])
                material = gfx.PointsMaterial(color="orange", size=2)
                self.points = gfx.Points(geometry, material)
                self.editor.add(self.points)

                geometry = gfx.Geometry(positions=self.points.geometry.positions)
                material = gfx.LineMaterial(thickness=1.0, color='black')
                self.line = gfx.Line(geometry, material)
                self.editor.add(self.line)

            elif n > 1:                
                begin = self.begin
                end = np.array([x,y])
                distance = (np.linalg.norm(begin - end))
                if distance <= 5:
                    n = 0
                    position = self.points.geometry.positions.data[0]

                old_positions = self.points.geometry.positions.data
                new_positions = np.vstack([old_positions, position])
                self.points.geometry.positions = gfx.Buffer(new_positions)
                self.points.geometry.positions.update_range()
                self.line.geometry.positions = gfx.Buffer(new_positions)
                self.line.geometry.positions.update_range()

                if n == 0:                    
                    positions = self.points.geometry.positions.data
                    tri = Delaunay(positions[:, :2])
                    indices = tri.simplices
                    geometry = gfx.Geometry(positions=positions, indices=indices)
                    material = gfx.MeshPhongMaterial(color=(0.5, 0.5, 1.0, 0.8))
                    mesh = gfx.Mesh(geometry, material)
                    self.editor.add(mesh)
                    self.geoms.add('面-'+str(mesh.id))

                    self.editor.remove(self.points)
                    self.editor.remove(self.line)

        if sender.get_active():
            self.controller = Gtk.GestureClick.new()
            self.controller.connect("released", add_line)
            self.area.add_controller(self.controller)
            self.sig_clicked = sender.connect("clicked", lambda sender: sender.set_active(False))
        else:
            self.area.remove_controller(self.controller)
            sender.disconnect(self.sig_clicked)
