import sys
sys.path.append('.')

import numpy as np
import pygfx as gfx

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk, Gio

from app_window import *

if __name__ == '__main__':
    GLib.set_application_name('Simtoy')

    settings = Gtk.Settings.get_default()
    settings.set_property('gtk-application-prefer-dark-theme', True)

    def do_activate(app):
        builder = Gtk.Builder.new_from_file('ui/app.ui')
        app_window = builder.get_object('app_window')
        app.add_window(app_window) 

    app = Gtk.Application(application_id="xyz.simtoy.app")
    app.connect('activate',do_activate)

    exit_status = app.run(sys.argv)
    sys.exit(exit_status)