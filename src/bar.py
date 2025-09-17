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
    tools = Gtk.Template.Child('tools')

    def __init__(self):
        provider = Gtk.CssProvider.new()
        provider.load_from_path('ui/hotbar.css')
        Gtk.StyleContext.add_provider_for_display(self.get_display(),provider,Gtk.STYLE_PROVIDER_PRIORITY_USER)

    @GObject.Signal(return_type=bool, arg_types=(object,))
    def item_added(self,*args): 
        pass
        
    def set_items(self,items):
        widget = self.tools.get_first_child()
        while widget:
            next_widget = widget.get_next_sibling()
            self.tools.remove(widget)
            widget = next_widget

        for text,action,icon in items:
            def callback(sender,f):
                self.emit('item_added', f())

            button = Gtk.Button()
            button.connect('clicked',callback, action)
            button.set_label(text)
            button.set_size_request(50,50)
            if icon: button.set_icon_name(icon)
            self.tools.append(button)