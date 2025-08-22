import gi
gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk, Gio

import subprocess as sp
import numpy as np
import os
import pygfx as gfx
import threading
import shutil
import zipfile
import trimesh
import io
from simtoy import *

class ImportDialog (Gtk.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.progress = Gtk.ProgressBar()
        self.progress.set_fraction(0)  # 设置50%进度
        
        self.set_child(self.progress)
        self.set_resizable(False)
        self.progress.set_valign(Gtk.Align.CENTER)
        self.progress.set_margin_start(20)
        self.progress.set_margin_end(20)
        self.progress.set_margin_top(20)
        self.progress.set_margin_bottom(20)

        bar = Gtk.HeaderBar()
        title_label = Gtk.Label()
        title_label.set_text('导入')
        bar.set_title_widget(title_label)
        
        bar.set_show_title_buttons(False)
        self.set_titlebar(bar)
                
    def input(self, file_path, editor, panel):
        self.stdout_thread = threading.Thread(target=self.doing,args=[file_path, editor, panel])
        self.stdout_thread.start()

    def doing(self, file_path, editor, panel):
        with zipfile.ZipFile(file_path, 'r') as zf:
            count = len(zf.namelist())
            for i, archive_path in enumerate(zf.namelist()):
                dir_name = os.path.dirname(archive_path)
                file_name = os.path.basename(archive_path)
                                    
                if archive_path.endswith('.las'):
                    item = panel.get(file_name)
                    if item: continue

                    # 读取点数据
                    points_data = zf.read(archive_path)
                    points = np.frombuffer(points_data, dtype=np.float64).reshape(-1, 3)

                    # 读取颜色数据
                    colors_data = zf.read(archive_path + '.colors')
                    colors = np.frombuffer(colors_data, dtype=np.float32).reshape(-1, 3)

                    geometry = gfx.Geometry(positions=points.astype(np.float32), colors=colors)
                    material = gfx.PointsMaterial(color_mode="vertex", size=1,pick_write=True)
                    obj = PointCloud(geometry, material)
                    obj.name = file_name
                    editor.add(obj)
                    item = panel.add(obj)
                elif archive_path.endswith('.npy'):
                    item = panel.get(dir_name)

                    if not item: continue
                    # 读取点数据
                    points_data = zf.read(archive_path)
                    points = np.frombuffer(points_data, dtype=np.float64).reshape(-1, 3)

                    # 读取颜色数据
                    colors_data = zf.read(archive_path + '.colors')
                    colors = np.frombuffer(colors_data, dtype=np.float32).reshape(-1, 3)

                    geometry = gfx.Geometry(positions=points.astype(np.float32), colors=colors)
                    material = gfx.PointsMaterial(color_mode="vertex", size=1,pick_write=True)
                    sub_obj = PointCloud(geometry, material)
                    sub_obj.name = file_name
    
                    item.obj.add(sub_obj)
                    panel.add_sub(item,[sub_obj])
                elif archive_path.endswith('.obj'):
                    item = panel.get(dir_name)
                    if not item: continue

                    bbo = io.BytesIO(zf.read(archive_path))
                    from pygfx.utils.load import meshes_from_trimesh
                    mesh = meshes_from_trimesh(trimesh.load(bbo,file_type='obj'), apply_transforms=True)[0]
                    pc = mesh.geometry.positions.data
                    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
                    offset = [(x.max()-x.min())/2,(y.max()-y.min())/2,0]
                    origin = np.array([np.min(x),np.min(y),0])
                    x = x - np.min(x)
                    y = y - np.min(y)
                    z = z
                    pc = np.column_stack([x,y,z]) - [(x.max()-x.min())/2,(y.max()-y.min())/2,0]
                    mesh.geometry.positions.data[:] = pc.astype(np.float32)

                    mesh.material.side = gfx.VisibleSide.both
                    mesh.material.color = (0.8, 0.8, 0.8)  # 设置材质颜色为白色
                    mesh.material.shininess = 0  # 降低高光强度
                    mesh.material.specular = (0.0, 0.0, 0.0, 1.0)  # 降低高光色
                    mesh.material.emissive = (0.8, 0.8, 0.8)  # 设置微弱自发光
                    mesh.material.flat_shading = True  # 启用平面着色
                    mesh.material.pick_write = True

                    building = Building()
                    building.geometry = mesh.geometry
                    building.material = mesh.material
                    building.local.position = origin + offset
                    building.name = file_name

                    if archive_path + '.roof' in zf.namelist():         
                        building.roof_mesh_content = zf.read(archive_path + '.roof')

                    item.obj.add(building)
                    panel.add_sub(item,[building])
                else:
                    pass

                self.progress.set_fraction((i+1)/count)
                
        self.progress.set_fraction(1)
        GLib.idle_add(self.close)