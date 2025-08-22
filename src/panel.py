import gi
gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk, GObject, Gio, Gdk

import pygfx as gfx
from simtoy import *

@Gtk.Template(filename='ui/panel.ui')
class Panel (Gtk.Paned):
    __gtype_name__ = "Panel"
    provider = Gtk.CssProvider.new()

    listview = Gtk.Template.Child('geoms')
    expander_position = Gtk.Template.Child('position')
    expander_pointcloud = Gtk.Template.Child('pointcloud')
    expander_mesh = Gtk.Template.Child('mesh')

    spin_x = Gtk.Template.Child('x')
    spin_y = Gtk.Template.Child('y')
    spin_z = Gtk.Template.Child('z')
        
    menu_add = Gtk.Template.Child('popover_menu_add')
    menu = Gtk.Template.Child('popover_menu')

    def __init__(self):
        Gtk.StyleContext.add_provider_for_display(self.get_display(),self.provider,Gtk.STYLE_PROVIDER_PRIORITY_USER)

        self.model = Gio.ListStore(item_type=GObject.Object)
        self.tree_model = Gtk.TreeListModel.new(self.model,passthrough=False,autoexpand=False,create_func=lambda item: item.model)

        self.selection_model = Gtk.SingleSelection.new(self.tree_model)
        self.selection_model.set_autoselect(False)
        self.selection_model.set_can_unselect(True)
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
        if self.selected_item:
            self.selected_item.obj.set_bounding_box_visible(False)

        if self.cur_item_index == Gtk.INVALID_LIST_POSITION:
            self.selection_model.unselect_all()
            self.selected_item = None
            return
        
        item = self.selection_model.get_item(self.cur_item_index).get_item()
        self.spin_x.set_value(item.obj.local.x)
        self.spin_y.set_value(item.obj.local.y)
        self.spin_z.set_value(item.obj.local.z)

        if type(item.obj) == PointCloud:
            self.expander_position.set_visible(True)
            self.expander_pointcloud.set_visible(True)
            self.expander_mesh.set_visible(False)
            item.obj.set_bounding_box_visible(True)
        elif type(item.obj) == Building:
            self.expander_position.set_visible(True)
            self.expander_pointcloud.set_visible(False)
            self.expander_mesh.set_visible(True)
            item.obj.set_bounding_box_visible(True)
        else:
            self.expander_position.set_visible(False)
            self.expander_pointcloud.set_visible(False)
            self.expander_mesh.set_visible(False)
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

        label = Gtk.Label()
        name_box.append(label)

        # 创建图标（使用默认的文件夹图标）
        focus = Gtk.Button()
        focus.set_icon_name("find-location-symbolic")
        # focus.set_has_frame(False)
        focus.connect("clicked", self.focus_clicked, listitem)
        name_box.append(focus)

        # 将图标和标签添加到容器中
        box.append(name_box)

        # 创建图标（使用默认的文件夹图标）
        icon = Gtk.ToggleButton()
        icon.set_icon_name("display-brightness-symbolic")
        icon.set_active(True)
        icon.set_has_frame(False)
        css = """
            .borderless-toggle-button {
                background: none;
            }
            """
        self.provider.load_from_data(css)
        icon.get_style_context().add_class("borderless-toggle-button")
        icon.connect("toggled", self.item_visible_toggled, listitem)
        box.append(icon)

        # 设置列表项的显示内容
        listitem.set_child(box)

        # 添加鼠标进入/离开控制器
        motion_controller = Gtk.EventControllerMotion()
        motion_controller.connect("enter", self.listitem_enter,listitem)  # 鼠标进入事件
        motion_controller.connect("leave", self.listitem_leave)  # 鼠标离开事件
        box.add_controller(motion_controller)


    def listitem_enter(self, controller, x, y,listitem):
        self.cur_item_index = listitem.get_position()

    def listitem_leave(self, controller):
        self.cur_item_index = Gtk.INVALID_LIST_POSITION

    def bind_listitem(self, factory, list_item):
        tree_row = list_item.get_item()
        box = list_item.get_child()
        name_box = box.get_first_child()
        expander = name_box.get_first_child()
        label = expander.get_next_sibling()

        item = tree_row.get_item()
        item.row = tree_row
        item.widget = box

        expander.set_list_row(tree_row)
        label.set_label(item.obj.name)

        if item.model.get_n_items():
            expander.set_hide_expander(False)
        else:
            expander.set_hide_expander(True)

    def add(self, obj : WorldObject):
        item = GObject.Object()
        item.obj = obj
        item.parent = None
        item.model = Gio.ListStore(item_type=GObject.Object)
        self.model.append(item)
        obj.material.pick_write = True

        return item
    
    def add_sub(self,item,objs):
        start = item.model.get_n_items()
        for obj in objs:
            obj.material.pick_write = True
            sub_item = GObject.Object()
            sub_item.obj = obj
            sub_item.parent = item
            sub_item.model = Gio.ListStore(item_type=GObject.Object)
            item.model.append(sub_item)
        
        b,i = self.model.find(item)
        self.model.items_changed(i,1,1)
        item.model.items_changed(start,0,item.model.get_n_items() - start)
        
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
    def assessment_value_changed(self, spin_button):
        assessment = spin_button.get_value()

        for item in self.model:
            if type(item.obj) != PointCloud:
                continue
            
            for sub_item in item.model:
                if type(sub_item.obj) != Building:
                    continue

                if sub_item.obj.assessment < assessment:
                    sub_item.obj.material.opacity = 1
                else:
                    sub_item.obj.material.opacity = 0
            
    # @Gtk.Template.Callback()
    def x_value_changed(self, spin_button):
        value = spin_button.get_value()
        i = self.selection_model.get_selected()
        item = self.selection_model.get_item(i).get_item()
        item.obj.local.x = value

    # @Gtk.Template.Callback()
    def y_value_changed(self, spin_button):
        value = spin_button.get_value()
        i = self.selection_model.get_selected()
        item = self.selection_model.get_item(i).get_item()
        item.obj.local.y = value

    # @Gtk.Template.Callback()
    def z_value_changed(self, spin_button):
        value = spin_button.get_value()
        i = self.selection_model.get_selected()
        item = self.selection_model.get_item(i).get_item()
        item.obj.local.z = value

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