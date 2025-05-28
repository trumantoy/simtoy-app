from socket import *
from time import time
import traceback as tb
import functools
import cv2
import numpy as np

def log(func):
    @functools.wraps(func)
    def newfunc(*args): return func(*args)
    return newfunc

class Projector:
    steps = list()

    def __init__(self):
        self.model_path = ''
        self.picture = Projector.make_bg(600,800)
        pass
        
    def __del__(self):
        self.steps.clear()
        pass

    def step(self,dt):
        if not self.steps: return
        fun,args = self.steps[0]
        fun(*args)
        self.steps.pop(0)
        pass
   
    def power(self,on):
        if on:
            sock = socket(AF_INET, SOCK_STREAM)
            sock.setblocking(False)
            sock.settimeout(0.0)
            sock.bind(('localhost', 8898))
            sock.listen()
            f = self.accepting,[sock]
            self.steps.append(f)
            self.sock = sock
        else: 
            if 'conn' in vars(self): self.conn.close()
            if 'sock' in vars(self): self.sock.close()
            del self.conn
            del self.sock
        pass

    def accepting(self,sock):
        try:
            conn,addr = sock.accept()
        except BlockingIOError:
            f = self.accepting,[sock]
            self.steps.append(f)
        except:
            tb.print_exc()
        else:
            self.conn = conn
            buff = bytes()
            f = self.receiving,[conn,addr,buff]
            self.steps.append(f)
        pass

    def receiving(self,conn,addr,buff):
        try:
            buff = conn.recv(1024)
            end = buff.index(b'\n')
            line = buff[:end]
            buff = buff[end+1:]
            self.command(line.decode())
        except BlockingIOError:
            pass
        except:
            tb.print_exc()
    
        f = self.receiving,[conn,addr,buff]
        self.steps.append(f)
        pass
    
    def command(self,line : str):
        try:
            eval(f'self.{line}')
        except:
            tb.print_exc()
        pass

    def make_screen(self,left,top,right,bottom):
        img = Projector.make_bg(bottom,right)
        screen = (left, top, img)
        self.screens.append(screen)

    def make_control_points(self):
        self.points = Projector.make_control_point(self.picture)

    @staticmethod
    def make_bg(height, width):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        colors = [
            (255, 204, 153), # light orange
            (153, 255, 153), # light green
            (153, 204, 255), # light blue
        ]

        grid_size = width // 10
        for x in range(0, width, grid_size):
            # color = colors[x // grid_size % len(colors)]
            color = colors[int(time()) % len(colors)]
            cv2.rectangle(img, (x, 0), (x + grid_size, height), color, -1)
            cv2.line(img, (x, 0), (x, height), (0, 0, 0), 1)
        for y in range(0, height, grid_size):
            cv2.line(img, (0, y), (width, y), (0, 0, 0), 1)
        return img
    
    @staticmethod
    def make_control_point(img):
        h,w,channel = img.shape
        lt = [0,0]
        rt = [w,0]
        lb = [0,h]
        rb = [w,h]
        control_points = [lt,rt,lb,rb]
        r = w // 10 // 5
        for p in control_points:
            cv2.circle(img, (p[0], p[1]), r, [255, 0, 0],-1)
        return img,[lb,rb,lt,rt]