import gi

from simtoy.tools import engravtor
gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk, GObject, Gio, Gdk

import pygfx as gfx
from simtoy import *

@Gtk.Template(filename='ui/panel.ui')
class Panel (Gtk.Paned):
    __gtype_name__ = "Panel"
    provider = Gtk.CssProvider.new()

    listview = Gtk.Template.Child('geoms')
    expander_device = Gtk.Template.Child('expander_device')
    expander_gcode = Gtk.Template.Child('expander_gcode')
    textview_gcode = Gtk.Template.Child('textview_gcode')

    expander_text = Gtk.Template.Child('expander_text')
    expander_bitmap = Gtk.Template.Child('expander_bitmap')
    btn_connect = Gtk.Template.Child('btn_connect')
    dp_com_port = Gtk.Template.Child('dp_com_port')

    menu_add = Gtk.Template.Child('popover_menu_add')
    menu = Gtk.Template.Child('popover_menu')

    def __init__(self):
        Gtk.StyleContext.add_provider_for_display(self.get_display(),self.provider,Gtk.STYLE_PROVIDER_PRIORITY_USER)

        self.model = Gio.ListStore(item_type=GObject.Object)
        self.tree_model = Gtk.TreeListModel.new(self.model,passthrough=False,autoexpand=False,create_func=lambda item: item.model)

        self.selection_model = Gtk.SingleSelection.new(self.tree_model)
        self.selection_model.set_autoselect(True)
        self.selection_model.set_can_unselect(True)
        self.selection_model.connect('selection-changed', self.listview_selection_changed)
        self.cur_item_index = Gtk.INVALID_LIST_POSITION
        
        factory = Gtk.SignalListItemFactory()
        factory.connect("setup", self.setup_listitem)
        factory.connect("bind", self.bind_listitem)
                
        self.listview.set_model(self.selection_model)
        self.listview.set_factory(factory)
        
        # # 创建右键点击手势
        # left_click_gesture = Gtk.GestureClick()
        # left_click_gesture.set_button(1)  # 3 代表鼠标右键
        # left_click_gesture.connect("pressed", self.listview_left_clicked)
        # self.listview.add_controller(left_click_gesture)

        # 创建右键点击手势
        right_click_gesture = Gtk.GestureClick()
        right_click_gesture.set_button(3)  # 3 代表鼠标右键
        right_click_gesture.connect("pressed", self.listview_right_clicked)
        self.listview.add_controller(right_click_gesture)

    def listview_selection_changed(self, model, *args):
        i = model.get_selected()
        listviewitem = model.get_item(i)
        item = listviewitem.get_item()

        self.expander_device.set_visible(item.obj.__class__.__name__ == 'Engravtor')
        self.expander_gcode.set_visible(item.obj.__class__.__name__ == 'Engravtor')
        self.expander_text.set_visible(item.obj.__class__.__name__ == 'Text')
        self.expander_bitmap.set_visible(item.obj.__class__.__name__ == 'Bitmap')

    def listview_right_clicked(self, gesture, n_press, x, y):
        popover = Gtk.PopoverMenu()
        popover.set_parent(gesture.get_widget())

        if self.cur_item_index == Gtk.INVALID_LIST_POSITION:
            popover.set_menu_model(self.menu_add)
            self.selection_model.unselect_all()
        else:
            popover.set_menu_model(self.menu)

        rect = Gdk.Rectangle()
        rect.x = x
        rect.y = y
        popover.set_pointing_to(rect)

        self.selection_model.set_can_unselect(False)
        i = self.cur_item_index

        popover.popup()

        self.selection_model.set_selected(i)
        self.selection_model.set_can_unselect(True)

    def set_viewbar(self,viewbar):
        self.viewbar = viewbar

    def setup_listitem(self, factory, listitem):
        # 创建一个水平排列的容器
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        
        name_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        expander = Gtk.TreeExpander()
        name_box.append(expander)

        # 创建图标（使用默认的文件夹图标）
        icon = Gtk.Image.new_from_icon_name("printer")
        # icon.set_active(True)
        # icon.set_has_frame(False)
        css = """
            .borderless-toggle-button {
                background: none;
            }
            """
        # self.provider.load_from_data(css)
        # icon.get_style_context().add_class("borderless-toggle-button")
        # icon.connect("toggled", self.item_visible_toggled, listitem)
        name_box.append(icon)

        label = Gtk.Label()
        name_box.append(label)

        # 将图标和标签添加到容器中
        box.append(name_box)

        # 设置列表项的显示内容
        listitem.set_child(box)

    def bind_listitem(self, factory, list_item):
        tree_row = list_item.get_item()
        box = list_item.get_child()
        name_box = box.get_first_child()
        expander = name_box.get_first_child()
        icon = expander.get_next_sibling()
        icon.set_visible(0 == tree_row.get_depth())
        label = icon.get_next_sibling()

        item = tree_row.get_item()
        item.row = tree_row
        item.widget = box

        expander.set_list_row(tree_row)
        label.set_label(item.obj.name)

        if item.model.get_n_items():
            expander.set_hide_expander(False)
        else:
            expander.set_hide_expander(True)

    def add(self, obj : WorldObject, parent : WorldObject = None):
        parent_item = None
        model = self.model

        for item in self.model:
            if item.obj == parent:
                parent_item = item
                model = parent_item.model
                break

        start = 0
        if parent_item:
            start = parent_item.model.get_n_items()

        item = GObject.Object()
        item.obj = obj
        item.parent = parent_item
        item.model = Gio.ListStore(item_type=GObject.Object)
        model.append(item)

        if parent_item:
            expanded = parent_item.row.get_expanded()
            b,i = self.model.find(parent_item)
            self.model.items_changed(i,1,1)
            parent_item.model.items_changed(start,0,parent_item.model.get_n_items() - start)
            parent_item.row.set_expanded(expanded)
        
    def remove(self, obj):
        for i,item in enumerate(self.model):
            if item.obj == obj:
                self.model.remove(i)
                break

    def get(self, name):
        for item in self.model:
            if item.obj.name == name:
                return item
            
            for sub_item in item.model:
                if sub_item.obj.name == name:
                    return sub_item

        return None

    def item_visible_toggled(self,sender,list_item):
        tree_row = list_item.get_item()
        item = tree_row.get_item()
        
        if sender.get_active():
            sender.set_icon_name("display-brightness-symbolic")
            item.obj.material.opacity = 1
        else:
            sender.set_icon_name("")
            item.obj.material.opacity = 0
    
    def focus_clicked(self,sender,list_item):
        tree_item = list_item.get_item()
        item = tree_item.get_item()
        camera = self.viewbar.get_view_camera()
        camera.show_object(item.obj)

    @Gtk.Template.Callback()
    def point_size_value_changed(self,spin_button):
        value = spin_button.get_value()
        i = self.selection_model.get_selected()
        item = self.selection_model.get_item(i).get_item()
        item.obj.material.size = value

    @Gtk.Template.Callback()
    def btn_connect_toggled(self,sender,*args):
        model = self.listview.get_model()
        i = model.get_selected()
        listviewitem = model.get_item(i)
        item = listviewitem.get_item()
        engravtor = item.obj

        if sender.get_active():
            engravtor.controller.connect(self.dp_com_port.get_selected_item().get_string())
            sender.set_label('关闭')
        else:
            engravtor.controller.disconnect()
            sender.set_label('连接')

    @Gtk.Template.Callback()
    def btn_gcode_clicked(self,sender,*args):
        model = self.listview.get_model()
        i = model.get_selected()
        listviewitem = model.get_item(i)
        item = listviewitem.get_item()
        engravtor = item.obj

        import tempfile
        temp_file = tempfile.NamedTemporaryFile(delete=False); temp_file.close()
        svg_filepath = temp_file.name + '.svg'
        gc_filepath = temp_file.name + '.gc'
        
        width,heigh = engravtor.export_svg(svg_filepath)
        if self.export_gcode_from_svg(svg_filepath,gc_filepath,width,heigh):
            return

        with open(gc_filepath,'r') as f:
            gcode = f.read()
            self.textview_gcode.get_buffer().set_text(gcode)


    @Gtk.Template.Callback()
    def btn_preview_clicked(self,sender,*args):
        model = self.listview.get_model()
        i = model.get_selected()
        listviewitem = model.get_item(i)
        item = listviewitem.get_item()
        engravtor = item.obj
        buffer = self.textview_gcode.get_buffer()
        start,end = buffer.get_bounds()
        gcode = buffer.get_text(start,end,True)
        engravtor.preview(gcode)

    @Gtk.Template.Callback()
    def btn_run_clicked(self,sender,*args):
        model = self.listview.get_model()
        i = model.get_selected()
        listviewitem = model.get_item(i)
        item = listviewitem.get_item()
        engravtor = item.obj

        import tempfile
        temp_file = tempfile.NamedTemporaryFile(delete=False); temp_file.close()
        svg_filepath = temp_file.name + '.svg'
        gc_filepath = temp_file.name + '.gc'
        
        width,heigh = engravtor.export_svg(svg_filepath)
        if self.export_gcode_from_svg(svg_filepath,gc_filepath,width,heigh):
            return

        with open(gc_filepath,'r') as f:
            gcode = f.read()
            self.textview_gcode.get_buffer().set_text(gcode)

        buffer = self.textview_gcode.get_buffer()
        start,end = buffer.get_bounds()
        gcode = buffer.get_text(start,end,True)
        engravtor.run(gcode)

    def export_gcode_from_svg(self,svg_filepath,gc_filepath,width,height):
        from svg2gcode.__main__ import svg2gcode
        from svg2gcode.svg_to_gcode import css_color
        import argparse
        import re

        # defaults
        cfg = {
            "pixelsize_default": 0.1,
            "imagespeed_default": 800,
            "cuttingspeed_default": 1000,
            "imagepower_default": 300,
            "poweroffset_default": 0,
            "cuttingpower_default": 850,
            "xmaxtravel_default": 300,
            "ymaxtravel_default": 400,
            "rapidmove_default": 10,
            "noise_default": 0,
            "overscan_default": 0,
            "pass_depth_default": 0,
            "passes_default": 1,
            "rotate_default": 0,
            "colorcoded_default": "",
            "constantburn_default": True,
        }

        # Define command line argument interface
        parser = argparse.ArgumentParser(description='Convert svg to gcode for GRBL v1.1 compatible diode laser engravers.')
        parser.add_argument('svg', type=str, help='svg file to be converted to gcode')
        parser.add_argument('gcode', type=str, help='gcode output file')
        parser.add_argument('--showimage', action='store_true', default=False, help='show b&w converted image' )
        parser.add_argument('--selfcenter', action='store_true', default=False, help='self center the gcode (--origin cannot be used at the same time)' )
        parser.add_argument('--pixelsize', default=cfg["pixelsize_default"], metavar="<default:" + str(cfg["pixelsize_default"])+">",type=float, help="pixel size in mm (XY-axis): each image pixel is drawn this size")
        parser.add_argument('--imagespeed', default=cfg["imagespeed_default"], metavar="<default:" + str(cfg["imagespeed_default"])+">",type=int, help='image draw speed in mm/min')
        parser.add_argument('--cuttingspeed', default=cfg["cuttingspeed_default"], metavar="<default:" + str(cfg["cuttingspeed_default"])+">",type=int, help='cutting speed in mm/min')
        parser.add_argument('--imagepower', default=cfg["imagepower_default"], metavar="<default:" +str(cfg["imagepower_default"])+ ">",type=int, help="maximum laser power while drawing an image (as a rule of thumb set to 1/3 of the machine maximum for a 5W laser)")
        parser.add_argument('--poweroffset', default=cfg["poweroffset_default"], metavar="<default:" +str(cfg["poweroffset_default"])+ ">",type=int, help="pixel intensity to laser power: shift power range [0-imagepower]")
        parser.add_argument('--cuttingpower', default=cfg["cuttingpower_default"], metavar="<default:" +str(cfg["cuttingpower_default"])+ ">",type=int, help="sets laser power of line (path) cutting")
        parser.add_argument('--passes', default=cfg["passes_default"], metavar="<default:" +str(cfg["passes_default"])+ ">",type=int, help="Number of passes (iterations) for line drawings, only active when pass_depth is set")
        parser.add_argument('--pass_depth', default=cfg["pass_depth_default"], metavar="<default:" + str(cfg["pass_depth_default"])+">",type=float, help="cutting depth in mm for one pass, only active for passes > 1")
        parser.add_argument('--rapidmove', default=cfg["rapidmove_default"], metavar="<default:" + str(cfg["rapidmove_default"])+ ">",type=int, help='generate G0 moves between shapes, for images: G0 moves when skipping more than 10mm (default), 0 is no G0 moves' )
        parser.add_argument('--noise', default=cfg["noise_default"], metavar="<default:" +str(cfg["noise_default"])+ ">",type=int, help='reduces image noise by not emitting pixels with power lower or equal than this setting')
        parser.add_argument('--overscan', default=cfg["overscan_default"], metavar="<default:" +str(cfg["overscan_default"])+ ">",type=int, help="overscan image lines to avoid incorrect power levels for pixels at left and right borders, number in pixels, default off")
        parser.add_argument('--showoverscan', action='store_true', default=False, help='show overscan pixels (note that this is visible and part of the gcode emitted!)' )
        parser.add_argument('--constantburn', action=argparse.BooleanOptionalAction, default=cfg["constantburn_default"], help='default constant burn mode (M3)')
        parser.add_argument('--origin', default=None, nargs=2, metavar=('delta-x', 'delta-y'),type=float, help="translate origin by vector (delta-x,delta-y) in mm (default not set, option --selfcenter cannot be used at the same time)")
        parser.add_argument('--scale', default=None, nargs=2, metavar=('factor-x', 'factor-y'),type=float, help="scale svg with (factor-x,factor-y) (default not set)")
        parser.add_argument('--rotate', default=cfg["rotate_default"], metavar="<default:" +str(cfg["rotate_default"])+ ">",type=int, help="number of degrees to rotate")
        parser.add_argument('--splitfile', action='store_true', default=False, help='split gcode output of SVG path and image objects' )
        parser.add_argument('--pathcut', action='store_true', default=False, help='alway cut SVG path objects! (use laser power set with option --cuttingpower)' )
        parser.add_argument('--nofill', action='store_true', default=False, help='ignore SVG fill attribute' )
        parser.add_argument('--xmaxtravel', default=cfg["xmaxtravel_default"], metavar="<default:" +str(cfg["xmaxtravel_default"])+ ">",type=int, help="machine x-axis lengh in mm")
        parser.add_argument('--ymaxtravel', default=cfg["ymaxtravel_default"], metavar="<default:" +str(cfg["ymaxtravel_default"])+ ">",type=int, help="machine y-axis lengh in mm")
        parser.add_argument( '--color_coded', action = 'store', default=cfg["colorcoded_default"], metavar="<default:\"" + str(cfg["colorcoded_default"])+ "\">",type = str, help = 'set action for path with specific stroke color "[color = [cut|engrave|ignore] *]*"'', example: --color_coded "black = ignore purple = cut blue = engrave"' )
        parser.add_argument('--fan', action='store_true', default=False, help='set machine fan on' )
        parser.add_argument('-V', '--version', action='version', version='%(prog)s ' + '3.3.6', help="show version number and exit")

        # 使用临时文件作为输出路径
        args = parser.parse_args([svg_filepath, gc_filepath,'--imagepower','10','--imagespeed','100', '--pixelsize','1','--origin',str(-width/2),str(-height/2)])
        
        
        if args.color_coded != "":
            if args.pathcut:
                print("options --color_coded and --pathcut cannot be used at the same time, program abort")
                return 1
            # check argument validity (1)

            # category names
            category = ["cut", "engrave", "ignore"]
            # get css color names
            colors = str([*css_color.css_color_keywords])
            colors = re.sub(r"(,|\[|\]|\'| )", '', colors.replace(",", "|"))

            # make a color list
            colors = colors.split("|")

            # get all names from color_code
            names_regex = "[a-zA-Z]+"
            match = re.findall(names_regex, args.color_coded)
            names = [i for i in match]

            for name in names:
                if not (name in colors or name in category):
                    print(f"argument error: '--color_coded {args.color_coded}' has a name '{name}' that does not correspond to a css color or category (cut|engrave|ignore).")
                    return 1

        if args.origin is not None and args.selfcenter:
            print("options --selfcenter and --origin cannot be used at the same time, program abort")
            return 1

        return svg2gcode(args)
