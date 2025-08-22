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

class ExportDialog (Gtk.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.progress = Gtk.ProgressBar()
        self.progress.set_fraction(0) 
        
        self.set_child(self.progress)
        self.set_resizable(False)
        self.progress.set_valign(Gtk.Align.CENTER)
        self.progress.set_margin_start(20)
        self.progress.set_margin_end(20)
        self.progress.set_margin_top(20)
        self.progress.set_margin_bottom(20)

        bar = Gtk.HeaderBar()
        title_label = Gtk.Label()
        title_label.set_text('导出')
        bar.set_title_widget(title_label)
        
        bar.set_show_title_buttons(False)
        self.set_titlebar(bar)
                
    def input(self, file_path, editor, panel):
        self.stdout_thread = threading.Thread(target=self.doing,args=[file_path, editor, panel])
        self.stdout_thread.start()

    def doing(self, file_path, editor, panel):
        count = panel.model.get_n_items()
        for i, item in enumerate(panel.model):
            for j, sub_item in enumerate(item.model):
                count += 1

        with zipfile.ZipFile(file_path,'w') as zf:
            n = 0
            for i, item in enumerate(panel.model):
                print(item.obj.name)
                points = item.obj.geometry.positions.data + item.obj.local.position
                colors = item.obj.geometry.colors.data
                zf.writestr(item.obj.name, points.tobytes())
                zf.writestr(item.obj.name + '.colors', colors.tobytes())
        
                for j, sub_item in enumerate(item.model):
                    print(sub_item.obj.name)
                    if type(sub_item.obj) == PointCloud:
                        points = sub_item.obj.geometry.positions.data + sub_item.obj.local.position
                        colors = sub_item.obj.geometry.colors.data
                        zf.writestr(os.path.join(item.obj.name, sub_item.obj.name), points.tobytes())
                        zf.writestr(os.path.join(item.obj.name, sub_item.obj.name + '.colors'), colors.tobytes())
                    elif type(sub_item.obj) == Building:
                        positions = sub_item.obj.geometry.positions.data + sub_item.obj.local.position
                        faces = sub_item.obj.geometry.indices.data if sub_item.obj.geometry.indices is not None else None
                        tm = trimesh.Trimesh(vertices=positions, faces=faces)
                        bbo = io.BytesIO()
                        tm.export(bbo,file_type='obj')
                        zf.writestr(os.path.join(item.obj.name, sub_item.obj.name), bbo.getvalue())

                        if sub_item.obj.roof_mesh_content:
                            zf.writestr(os.path.join(item.obj.name, sub_item.obj.name + '.roof'), sub_item.obj.roof_mesh_content)
                    else:
                        pass

                self.progress.set_fraction((n+1)/count)

        self.progress.set_fraction(1)
        GLib.idle_add(self.close)