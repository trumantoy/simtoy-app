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
        self.selection_model.set_can_unselect(False)
        self.cur_item_index = Gtk.INVALID_LIST_POSITION
        self.selected_item = None

        factory = Gtk.SignalListItemFactory()
        factory.connect("setup", self.setup_listitem)
        factory.connect("bind", self.bind_listitem)
                
        self.listview.set_model(self.selection_model)
        self.listview.set_factory(factory)
        
        # 创建右键点击手势
        left_click_gesture = Gtk.GestureClick()
        left_click_gesture.set_button(1)  # 3 代表鼠标右键
        left_click_gesture.connect("pressed", self.listview_left_clicked)
        self.listview.add_controller(left_click_gesture)

        # 创建右键点击手势
        right_click_gesture = Gtk.GestureClick()
        right_click_gesture.set_button(3)  # 3 代表鼠标右键
        right_click_gesture.connect("pressed", self.listview_right_clicked)
        self.listview.add_controller(right_click_gesture)

    def listview_left_clicked(self, gesture, n_press, x, y):
        model = self.listview.get_model()
        i = model.get_selected()
        listviewitem = model.get_item(i)
        item = listviewitem.get_item()

        self.expander_device.set_visible(False)
        self.expander_text.set_visible(False)
        self.expander_bitmap.set_visible(False)

        if type(item.obj).__name__ == 'Engravtor':
            self.expander_device.set_visible(True)
        elif type(item.obj).__name__ == 'Text':
            self.expander_text.set_visible(True)
        elif type(item.obj).__name__ == 'Bitmap':
            self.expander_bitmap.set_visible(True)
        else:
            item = None

        self.selected_item = item

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
    def roofcolor_activated(self,sender,*args):
        print('11')

    @Gtk.Template.Callback()
    def roomcolor_activated(self,sender,*args):
        i = self.selection_model.get_selected()
        item = self.selection_model.get_item(i).get_item()
        color = self.roomcolor.get_color()
        item.obj.material.color = (color.red, color.green, color.blue)

    @Gtk.Template.Callback()
    def btn_connect_toggled(self,sender,*args):
        if sender.get_active():
            sender.set_label('关闭')
        else:
            sender.set_label('连接')

    @Gtk.Template.Callback()
    def btn_preview_clicked(self,sender,*args):
        model = self.listview.get_model()
        i = model.get_selected()
        listviewitem = model.get_item(i)
        item = listviewitem.get_item()
        engravtor = item.obj
        engravtor.preview()


    @Gtk.Template.Callback()
    def btn_run_clicked(self,sender,*args):
        model = self.listview.get_model()
        i = model.get_selected()
        listviewitem = model.get_item(i)
        item = listviewitem.get_item()
        engravtor = item.obj
        engravtor.run()