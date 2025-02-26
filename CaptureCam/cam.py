import cv2
import sys
import numpy as np
import os
import time
import mttkinter as tkinter
import tkinter as tk #GUI
import tkinter.font as tkFont
from tkinter import ttk, filedialog # Import filedialog module for folder selection
from tkinter import *
from threading import Thread
from datetime import datetime
from io import BytesIO
import win32clipboard #Clipboard
from PIL import Image, ImageTk #pillow
import mss #Screen Capture
import win32gui
import win32ui
import win32con
import win32api
from win32gui import FindWindow, GetWindowRect #Get window size and location
import ctypes #for Find window
from ctypes import windll, wintypes
import FreeSimpleGUI as sg #Direct capture windwo
from pygrabber.dshow_graph import FilterGraph #Get camera list
import keyboard
import shutil
import pygetwindow

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

def setClickthrough(hwnd):
    print("setting window properties")
    try:
        styles = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        styles = win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
        time.sleep(0.5)
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, styles)
        win32gui.SetLayeredWindowAttributes(hwnd, 0, 255, win32con.LWA_ALPHA)
    except Exception as e:
        print(e)


def capture_win_target(handle, window_capture_area_name: str, capture_full_window, window_margin):
    # Adapted from https://stackoverflow.com/questions/19695214/screenshot-of-inactive-window-printwindow-win32gui
    
    #Find target windwow and capture window
    
    window_title = win32gui.GetWindowText(handle)

    hwnd = win32gui.FindWindow(None, window_title)
    
    
    if capture_full_window == False:

        hwnd_a = win32gui.FindWindow(None, window_capture_area_name)
    
        bar_thickness = get_title_bar_thickness(hwnd_a)
    
    margin = window_margin
    
    #Get target window position and area
    try:
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
    except:
        return np.zeros((512, 512, 3), dtype=np.uint8)
        print("Wrong window handle!")

    x, y, w, h = win32gui.GetWindowRect(hwnd)
    
    left_w, top_w, right_w, bottom_w = win32gui.GetWindowRect(hwnd)
    
    
    if capture_full_window == False:
    
        left_a, top_a, right_a, bottom_a = win32gui.GetWindowRect(hwnd_a)
        
        top_a = top_a + bar_thickness - margin
        
        left_a = left_a + margin
        
        right_a = right_a - margin
        
        bottom_a = bottom_a - margin
    
    #Set crot area
    if capture_full_window == False:
        if left_a>left_w:
            x_crop = left_a-left_w
        else:
            x_crop = 0
        
        if top_a>top_w:
            y_crop = top_a-top_w
        else:
            y_crop = 0
        
        if right_a<right_w:
            x2_crop = right_a-right_w
        else:
            x2_crop = right_w
        
        if bottom_a<bottom_w:
            y2_crop = bottom_a-bottom_w
        else:
            y2_crop = bottom_w

    w = right - left
    h = bottom - top
    
    try:
        #Create bitmap
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mfc_dc, w, h)
        save_dc.SelectObject(bitmap)

        # If Special K is running, this number is 3. If not, 1
        result = windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 3)

        bmpinfo = bitmap.GetInfo()
        bmpstr = bitmap.GetBitmapBits(True)
        
    except:
        pass

    try:
        img = np.frombuffer(bmpstr, dtype=np.uint8).reshape((bmpinfo["bmHeight"], bmpinfo["bmWidth"], 4))
    except:
        return np.zeros((512, 512, 3), dtype=np.uint8)
    img = np.ascontiguousarray(img)[..., :-1]  # make image C_CONTIGUOUS and drop alpha channel
    
    img = np.array(img, dtype=np.uint8)

    #Delete hwnd
    try:
        win32gui.DeleteObject(bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)
    except:
        pass
    
    #Image crop
    if capture_full_window == False:
        crop_img = img[y_crop:y2_crop, x_crop:x2_crop]
        
        crop_h,crop_w,crop_c = crop_img.shape

    if not result:  # result should be 1
        win32gui.DeleteObject(bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)
        print(f"Unable to acquire capture! Result: {result}")
        return np.zeros((512, 512, 3), dtype=np.uint8)
    
    if capture_full_window == False:
        if((crop_h>0) and (crop_w>0)):
            return crop_img
        else:
            return np.zeros((512, 512, 3), dtype=np.uint8)
       
    if capture_full_window == True:
        return img

#Find windwow by name

def get_active_window_title():
    
    active_window = win32gui.GetForegroundWindow()
    window_title = win32gui.GetWindowText(active_window)
    
    return (window_title)


def find_window(name):
    
    try:
        hwnd = ctypes.windll.user32.FindWindowW(0, name)
        
        if hwnd:
            return(True)
        else:
            return(False)
    except:
        return(False)
        
#Find video devices
def get_available_cameras() :

    devices = FilterGraph().get_input_devices()

    available_cameras = []

    for device_name in devices:
        available_cameras.append(device_name)

    return available_cameras
    
#Find visible Windows
def get_available_windows() :
    
    def enum_windows_callback(hwnd, titles):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                titles.append(title)

    window_titles = []
    win32gui.EnumWindows(enum_windows_callback, window_titles)
    
    return window_titles


#send image to clipboard
def send_to_clipboard(clip_type, data):
        
    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardData(clip_type, data)
    win32clipboard.CloseClipboard()
  
#face detection  
def face_detection(frame, mask_folder, mask_format, m_scale):
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=6,
        minSize=(10, 10),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
        
    fh, fw, fc = frame.shape
    

    if (m_scale != ''):
        mask_scale = int(m_scale)
    else:
        mask_scale = 1

    
    black_canvas = np.zeros((fh, fw, 3), np.uint8)
    
    
    for (x, y, w, h) in faces:
        cv2.rectangle(black_canvas, (x-mask_scale, y-mask_scale), (x+w+mask_scale, y+h+mask_scale), (255, 255, 255), -1)
        cv2.rectangle(frame, (x-mask_scale, y-mask_scale), (x+w+mask_scale, y+h+mask_scale), (0, 255, 0), 1)
            
    mask_path = os.path.join(mask_folder, f"face_mask.{mask_format}")
        
    cv2.imwrite(mask_path, black_canvas)
    
    return(frame)

#Get title bar thickness
def get_title_bar_thickness(hwnd):
    rect = ctypes.wintypes.RECT()
    ctypes.windll.user32.GetWindowRect(hwnd, ctypes.pointer(rect))
    client_rect = ctypes.wintypes.RECT()
    ctypes.windll.user32.GetClientRect(hwnd, ctypes.pointer(client_rect))
    title_bar_thickness = (rect.bottom - rect.top) - (client_rect.bottom - client_rect.top)
    
    return title_bar_thickness
    

#Capture window area
def Capture_window(name:str, margin)-> tuple:
    
    with mss.mss() as mss_instance:
    
        hwnd = ctypes.windll.user32.FindWindowW(0, name)
        
        if hwnd:
            title_bar_thickness = get_title_bar_thickness(hwnd)
            
            rect = ctypes.wintypes.RECT()
            ctypes.windll.user32.GetWindowRect(hwnd, ctypes.pointer(rect))
            
            monitor_number = 1
            
            mon = mss_instance.monitors[monitor_number]
            
            monitor = {
                "top": rect.top + title_bar_thickness - margin,
                "left": rect.left + margin,
                "width": rect.right - (rect.left + margin) - margin,
                "height": rect.bottom - rect.top - title_bar_thickness,
                "mon": monitor_number,
            }

            screenshot = mss_instance.grab(monitor)

            
            try:
                img = np.array(screenshot, dtype=np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            except:
                pass
            
            return(img)
         
        else:
            img = np.zeros((512, 512, 3), np.uint8)

            return(img)
            
   
def Diredt_window_keep_aspect_ratio(target_width, target_height, window_name, margin):
    if not((target_width == 0) or (target_height == 0)):
        try:
            hwnd = win32gui.FindWindow(None, window_name)
            
            bar_thickness = get_title_bar_thickness(hwnd)
            
            add_w = target_width - margin*2
            
            add_h = target_height - bar_thickness
            
            if hwnd:
                x0, y0, x1, y1 = win32gui.GetWindowRect(hwnd)
                
                w = x1 - x0 
                h = y1 - y0
                
                w_mult = (w-(margin*2)) / target_width
                
                new_height = int((target_height * w_mult)+bar_thickness)
                
                if not h == new_height:
                    win32gui.MoveWindow(hwnd, x0, y0, w, new_height, True)
        except:
            pass
     

class CaptureWindow(tk.Toplevel):
        
    def __init__(self, name, target, target_handle, crop, layer, index_n, init_width, init_height, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #Window var
        self.title(name)
        self.geometry(f"{init_width}x{init_height}")
        self.configure(background='green')
        self.attributes('-transparentcolor', 'green', '-topmost', 1)
        self.keep_aspect_ratio = False
        self.aspect_ratio = init_width/init_height
        self.bind("<Configure>", self.on_resize)
        self.attributes("-alpha", 1.0)
        self.update_width = init_width
        self.update_height = init_height
        
        #Other var
        self.capture_window_name = name
        self.target_window_name = target
        self.target_window_handle = target_handle
        self.target_crop_enable = crop
        self.layer_list = layer
        self.current_layer = 0
        self.ai_image_enable = False
        self.overlay_alpha = 0.9
        self.render_path =""

    
    def show_ai_image(self, render_path):
        
        if os.path.exists(render_path):
            if self.ai_image_enable :
            
                try:
                    self.render_path = render_path
                    self.attributes("-alpha", self.overlay_alpha)
                    self.image = Image.open(self.render_path)
                    self.background_image = ImageTk.PhotoImage(self.image)
                    self.background = tk.Label(self, image=self.background_image)
                    self.background.pack(fill=tk.BOTH, expand=tk.YES)
                    time.sleep(0.1)
                    setClickthrough(self.background.winfo_id())
                except:
                    pass
                
            else:
                self.attributes("-alpha", 1.0)
                try:
                    time.sleep(0.5)
                    self.background.pack_forget()
                except:
                    pass
        else :
            print(f"Failed to find : {render_path}")

    def update_ai_image(self):
        
        if self.ai_image_enable :
            try:

                if os.path.exists(self.render_path):
                    image = Image.open(self.render_path)
                           
                    image = image.resize((self.update_width, self.update_height))
                
                    self.background_image = ImageTk.PhotoImage(image)
                
                    self.background.configure(image=self.background_image)
                    
                    self.after(100, self.update_ai_image)
                else :
                    print(f"Failed to find : {self.render_path}")
            except:
                self.after(100, self.update_ai_image)
    
    def on_resize(self, event):
    
        self.update_width = event.width
        self.update_height = event.height
    
        if  self.keep_aspect_ratio:
                
            new_width = event.width
            new_height = int(new_width / self.aspect_ratio)
            
            if event.width != new_width or event.height != new_height:
                self.geometry(f"{new_width}x{new_height}")       
    
    def set_aspect_ratio(self, width, height, enable):
        
        self.keep_aspect_ratio = enable
        
        if enable :
            self.aspect_ratio = width/height
        
        self.aspect_ratio = width / height
  

class WebcamApp:
    def __init__(self, output_folder, render_folder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
        #Dpi scaling awarness setting
        ctypes.windll.shcore.SetProcessDpiAwareness(2)

        #Init var
        self.output_folder = output_folder
        self.render_folder = render_folder
        self.cam_index = 0  # Default webcam index
        self.width = 512  # Default width
        self.height = 0  # Default height
        self.cap_win_a = 1.0 # Defautl direct capture windw alpha
        self.cap = cv2.VideoCapture(self.cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cam_list = get_available_cameras()
        self.window_list = get_available_windows()
        self.window_list.insert(0, "Disable")
        self.cam_list.append("Region Capture")
        self.cam_list.append("Window Capture")
        self.Direct_capture_enable = False
        self.capture_image_name = "capture" #Default capture image name
        self.target_window_name ="" #Find window name for capture mode selection
        self.target_window_capture_mode = False
        self.current_target_window_name = ""
        self.capture_list = [] #Saved Capture_window list
        #self.active_capture = CaptureWindow("capture", "Disable", 0, False, [], 0) #Active capture window & target & mode
        self.active_capture = None
        self.keypress = False
        self.first_run = True
        self.target_window_full_name = "" #target window name for window capture
        self.window_capture_enable = False #Enable Full window capture
        self.mask_paint_mode = False
        self.mask_drawing = False
        self.mask_brush_size = 30 # Mask brush init size
        self.mask_left_pressed = False
        self.mask_image = np.zeros((512, 512, 3), dtype=np.uint8) #Create init mask
        self.current_active_index = 0 #Starting active capture window index
        self.created_window_name = "" #Created window name init
        self.margin = 11 #Window border size
        self.pause_capture = False #Pause capture
        self.layer_save_mode = False
        self.current_layer_index = 0
        self.top_most_window = False
        self.region_window_list = [] 
       
        # Set UI scale
        self.ui_scale = 1.0
        
        # Set Font size
        self.font_size = 10

        self.root = tk.Tk()
        self.root.title("Webcam App")

        # Set GUI size
        self.root.geometry(str(int(645*self.ui_scale))+"x"+str(int(500*self.ui_scale))+"+100+100")
        
        self.root.resizable(width=1, height=1)
        
        self.sct = mss.mss()
        
        self.uifont = tkFont.Font(family="Arial", size= int(self.font_size*self.ui_scale))
        
        self.root.option_add('*Font', self.uifont)
        
        ####################################### GUI ##############################################

        # Webcam selection dropdown
        self.webcam_label = tk.Label(self.root, text="Webcam:", font=self.uifont)
        self.webcam_label.grid(row=0, column=0)

        self.webcam_combobox = ttk.Combobox(self.root, values=self.cam_list)  # Select video device
        self.webcam_combobox.set(self.cam_list[-2])
        self.webcam_combobox.grid(row=0, column=1)
        
        # Create Direct capture window       
        self.direct_button = tk.Button(self.root, text="Add Region window", command=self.direct_capture)
        self.direct_button.grid(row=0, column=2, sticky='nesw', columnspan=15)
        
        #Set Capture image name
        self.capture_name_entry = tk.Entry(self.root)
        self.capture_name_entry.insert(0, self.capture_image_name)  # Default capture image name
        self.capture_name_entry.grid(row=1, column=2)
        
        #Select Target Window list
        self.window_combobox = ttk.Combobox(self.root, values=self.window_list)  # Select Target Window
        self.window_combobox.set(self.window_list[0])
        self.window_combobox.grid(row=2, column=2)

        # Resolution input fields
        self.width_label = tk.Label(self.root, text="Width")
        self.width_label.grid(row=1, column=0)

        self.width_entry = tk.Entry(self.root)
        self.width_entry.insert(0, "512")  # Default width
        self.width_entry.grid(row=1, column=1)

        self.height_label = tk.Label(self.root, text="Height")
        self.height_label.grid(row=2, column=0)

        self.height_entry = tk.Entry(self.root)
        self.height_entry.insert(0, "0")  # Default height
        self.height_entry.grid(row=2, column=1)
        
        # FPS input field
        self.fps_label = tk.Label(self.root, text="FPS:")
        self.fps_label.grid(row=3, column=0)

        self.fps_entry = tk.Entry(self.root)
        self.fps_entry.insert(0, "20")  # Default fps (adjust as needed)
        self.fps_entry.grid(row=3, column=1)

        # Checkbox for show preview
        self.show_preview_var = tk.IntVar()
        self.show_preview_var.set(0)  # Default checked
        self.show_preview_checkbox = tk.Checkbutton(self.root, text="Webcam", variable=self.show_preview_var)
        self.show_preview_checkbox.grid(row=4, column=0)
        
        # Checkbox for show render
        self.show_render_var = tk.IntVar()
        self.show_render_var.set(0)  # Default checked
        self.show_render_checkbox = tk.Checkbutton(self.root, text="AI Render", variable=self.show_render_var)
        self.show_render_checkbox.grid(row=4, column=1)
        
        # Checkbox for preivew always on top
        self.show_top_var = tk.IntVar()
        self.show_top_var.set(0)  # Default checked
        self.show_top_checkbox = tk.Checkbutton(self.root, text="Always on top", variable=self.show_top_var)
        self.show_top_checkbox.grid(row=4, column=2)
        
        # Checkbox for keep aspect ratio
        self.use_aspect_var = tk.IntVar()
        self.use_aspect_var.set(0)  # Default checked
        self.use_aspect_checkbox = tk.Checkbutton(self.root, text="Keep aspect ratio", variable=self.use_aspect_var, command=self.toggle_keep_aspect)
        self.use_aspect_checkbox.grid(row=5, column=2)
        
        # Checkbox for face detection
        self.face_detect_var = tk.IntVar()
        self.face_detect_var.set(0)  # Default checked
        self.face_detect_checkbox = tk.Checkbutton(self.root, text="Face detect", variable=self.face_detect_var)
        self.face_detect_checkbox.grid(row=5, column=0)
        
        # Face Mask scale
        self.mask_size_entry = tk.Entry(self.root)
        self.mask_size_entry.insert(0, "20")  # Default mask size (adjust as needed)
        self.mask_size_entry.grid(row=5, column=1)

        # Output folder selection
        self.output_folder_label = tk.Label(self.root, text="Capture Path")
        self.output_folder_label.grid(row=7, column=0)

        self.output_folder_entry = tk.Entry(self.root)
        self.output_folder_entry.grid(row=7, column=1)

        self.output_folder_button = tk.Button(self.root, text="Browse", command=self.browse_output_folder)
        self.output_folder_button.grid(row=7, column=2, sticky='nesw')
        
        # Render file selection
        self.render_folder_label = tk.Label(self.root, text="Render iamge")
        self.render_folder_label.grid(row=8, column=0)

        self.render_folder_entry = tk.Entry(self.root)
        self.render_folder_entry.grid(row=8, column=1)

        self.render_folder_button = tk.Button(self.root, text="Browse", command=self.browse_render_file)
        self.render_folder_button.grid(row=8, column=2, sticky='nesw')
        
        # Image format selection
        self.format_label = tk.Label(self.root, text="Save format")
        self.format_label.grid(row=9, column=0)

        self.format_combobox = ttk.Combobox(self.root, values=["png", "jpg", "bmp"])
        self.format_combobox.set("jpg")
        self.format_combobox.grid(row=9, column=1)
        
        #Padding type selection
        self.padding_type_combobox = ttk.Combobox(self.root, values=["Replicate", "Reflect", "Wrap", "Constant"])
        self.padding_type_combobox.set("Constant")
        self.padding_type_combobox.grid(row=9, column=2)

        # Start & Stop
        
        self.start_button = tk.Button(self.root, text="Start", command=self.start_capture)
        self.start_button.grid(row=10, column=0, sticky='nesw', columnspan=15)

        self.stop_button = tk.Button(self.root, text="Stop", command=self.stop_capture, state=tk.DISABLED)
        self.stop_button.grid(row=11, column=0, sticky='nesw', columnspan=15)
        
        # Export video
        
        self.export_button = tk.Button(self.root, text="Export Video", command=self.export_video)
        self.export_button.grid(row=12, column=0)

        self.exportfps_entry = tk.Entry(self.root)
        self.exportfps_entry.insert(0, "12")  # Default exportfps (adjust as needed)
        self.exportfps_entry.grid(row=12, column=1)

        # Checkbox for image delete after export
        self.delete_images_var = tk.IntVar()
        self.delete_images_var.set(0)  # Default checked
        self.delete_images_checkbox = tk.Checkbutton(self.root, text="Clear after export", variable=self.delete_images_var)
        self.delete_images_checkbox.grid(row=12, column=2)
        
        # Refresh list button
        
        self.reload_button = tk.Button(self.root, text="Reload list", command=self.reload_list)
        self.reload_button.grid(row=3, column=2, sticky='nesw', columnspan=15)
        
        # Toggle top most 
        
        self.topmost_button = tk.Button(self.root, text="Pin app", command=self.top_most_main)
        self.topmost_button.grid(row=13, column=2)
        
        # AI overlay alpha
        self.overlay_label = tk.Label(self.root, text="Overlay alhpa:", font=self.uifont)
        self.overlay_label.grid(row=13, column=0)
        self.overlay_alpha_entry = tk.Entry(self.root)
        self.overlay_alpha_entry.insert(0, "0.9")  # Default alpha value (adjust as needed)
        self.overlay_alpha_entry.grid(row=13, column=1)

        self.is_capturing = False

        self.thread = None
        
        # Mapping keyboard
        
        keyboard.on_press(self.on_press_reaction)
    
    class CaptureRegion(tk.Toplevel):

        def __init__(self, title, init_width, init_height, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.title(title)
            self.geometry(f"{init_width}x{init_height}")
            self.configure(background="green")
            self.attributes('-transparentcolor', 'green', '-topmost', 1)
            self.keep_aspect_ratio = True
            self.aspect_ratio = init_width/init_height
            self.bind("<Configure>", self.on_resize)
        
        def show_image(self, render_image):
        
            self.attributes("-alpha", 0.6)
            self.image = Image.open(render_image)
            self.img_copy = self.image.copy()
            self.background_image = ImageTk.PhotoImage(self.image)
            self.background = tk.Label(self, image=self.background_image)
            self.background.pack(fill=tk.BOTH, expand=tk.YES)
            self.background.bind('<Configure>', self._resize_image)
            setClickthrough(self.background.winfo_id())

        def on_resize(self, event):
        
            if self.use_aspect_var.get():
                    
                new_width = event.width
                new_height = int(new_width / self.aspect_ratio)
                
                if event.width != new_width or event.height != new_height:
                    self.geometry(f"{new_width}x{new_height}")       
        
        def set_aspect_ratio(self):
        
            self.aspect_ratio = self.winfo_width() / self.winfo_height()
    
    
    def on_press_reaction(self, event):
        if not self.keypress:
            if event.name == "c":
                active_title = get_active_window_title()
                
                for saved_list in self.capture_list:
                    if saved_list.capture_window_name == active_title:
                        
                        self.active_capture = saved_list
                        
                        if not(self.active_capture.target_window_name == "Disable"):
                            
                            self.target_window_capture_mode = True
                            print(f"Capture : {active_title}")
                            self.keypress = False
                            break
                            
                        else:
                            self.target_window_capture_mode = False
                            print(f"Capture : {active_title}")
                            self.keypress = False
                            break

            if event.name == "f":
                if not self.keypress:
                    active_title = get_active_window_title()
                    
                    if self.active_capture:
                    
                        if (active_title == self.active_capture.capture_window_name):
                            
                            if self.active_capture.target_crop_enable:
                                self.active_capture.target_crop_enable = False
                            else:
                                self.active_capture.target_crop_enable = True
                                
                            print(f"{self.active_capture.capture_window_name} / Capture Full window : {self.active_capture.target_crop_enable}")
                            self.keypress = False

            if event.name == "p":
                if not self.keypress and not(self.is_capturing) :
                    active_title = get_active_window_title()
             
                    for saved_list in self.capture_list:
                        if saved_list.capture_window_name == active_title:
                            if not saved_list.target_window_name == "Disable" :
                                if saved_list.ai_image_enable:
                                    print("AI render overlay disabled")
                                    saved_list.ai_image_enable = False
                                    saved_list.overlay_alpha = self.overlay_alpha_entry.get()
                                    render_path = os.path.join(self.render_folder)
                                    time.sleep(0.5)
                                    saved_list.show_ai_image(render_path)
                                else:
                                    print("AI render overlay enabled")
                                    saved_list.ai_image_enable = True
                                    render_path = os.path.join(self.render_folder)
                                    saved_list.show_ai_image(render_path)
                                    time.sleep(0.5)
                                    saved_list.update_ai_image()
                            else:
                                print("Cannot be used without a target window.")
                    self.keypress = False
                else:
                    print("It cannot be changed during capture")
        else:
            self.keypress = True
    
    def draw_circle(self, event, x, y, flags, param):
        
        if self.mask_paint_mode:
        
            if event == cv2.EVENT_LBUTTONDOWN:
                self.mask_drawing = True
                ix, iy = x,y
                self.mask_left_pressed = True
                
            if event == cv2.EVENT_RBUTTONDOWN:    
                self.mask_drawing = True
                ix, iy = x,y
                self.mask_left_pressed = False
                
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.mask_drawing == True:
                    if self.mask_left_pressed == True:
                        cv2.circle(self.mask_image, (x,y), self.mask_brush_size, (255, 255, 255), -1)
                    else :
                        cv2.circle(self.mask_image, (x,y), self.mask_brush_size, (0, 0, 0), -1)
                    
            elif event == cv2.EVENT_LBUTTONUP:
                self.mask_drawing = False
                self.mask_left_pressed = False
             
            elif event == cv2.EVENT_RBUTTONUP:
                 self.mask_drawing = False
            
            elif event == cv2.EVENT_MOUSEWHEEL: 
                if flags >0:
                    self.mask_brush_size +=1
                elif self.mask_brush_size >1:
                    self.mask_brush_size -=1
    
    
    def update_w_alpha(self, value):
        
        self.cap_win_a = float(value)
    
    def direct_capture_window(self, name):
    
        self.remove_closed_window_list()
        
        target_window = self.window_combobox.get()
        
        try:
            init_width = int(self.width_entry.get())

            init_height = int(self.height_entry.get())
            
            if init_height == 0:
                init_height = init_width
            if init_width == 0:
                init_width == init_height
        except:
            init_width = 512

            init_height = 512  
            
        window_name = ""
        
        if ((target_window == "Disable")==True):
            window_name = name
            layer_name = name.split(',')
            
            region_window = CaptureWindow(window_name, "Disable", 0, False, layer_name, 0, init_width, init_height)
            
            region_window.keep_aspect_ratio =  self.use_aspect_var.get()
 
            self.capture_list.append(region_window)
            
            print(f"Layer list : {layer_name}")

        else:
            window_name = name + "_" + target_window
            layer_name = name.split(',')
            
            region_window = CaptureWindow(window_name, target_window, win32gui.FindWindow(None, target_window), False, layer_name, 0, init_width, init_height)
            
            region_window.keep_aspect_ratio =  self.use_aspect_var.get()
     
            self.capture_list.append(region_window)
            
            print(f"Layer list : {layer_name}")

        self.created_window_name = window_name
        print(f"[{window_name}] is created")
    
    def remove_closed_window_list(self):
        #Remove closed window
        for check_list in self.capture_list:
            #print(f"checking: {check_list.capture_window_name}")
            if not(find_window(check_list.capture_window_name)):
                #print(f"remove {check_list.capture_window_name}")
                self.capture_list.remove(check_list)

    def get_window_handle(self, name):
        if (name == "Disable"):
            return(0)
        else:
            return(win32gui.FindWindow(None, name))
    
    def browse_output_folder(self):
        self.output_folder = filedialog.askdirectory()
        self.output_folder_entry.delete(0, tk.END)
        self.output_folder_entry.insert(0, self.output_folder)
        
    def browse_render_file(self):
        self.render_folder = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.bmp;")])
        self.render_folder_entry.delete(0, tk.END)
        self.render_folder_entry.insert(0, self.render_folder)
        
        
    #Return new image size
    def resize_capture(self, width, height, entered_width, entered_height):
        try:
            if entered_width > 0 and entered_height == 0:
                # Calculate height based on the aspect ratio
                aspect_ratio = width / height
                new_width = entered_width
                new_height = int(new_width / aspect_ratio)
            elif entered_height > 0 and entered_width == 0:
                # Calculate width based on the aspect ratio
                aspect_ratio = width / height
                new_height = entered_height
                new_width = int(new_height * aspect_ratio)
            else:
                # Use the entered width and height
                new_width = entered_width
                new_height = entered_height
                
            return(new_width, new_height)
        except ZeroDivisionError:
            return()
    
    
    def toggle_keep_aspect(self):
    
        if self.capture_list:
            entered_width = int(self.width_entry.get())
            entered_height = int(self.height_entry.get())
            
            for saved_list in self.capture_list:
                try:
                    saved_list.set_aspect_ratio(entered_width, entered_height, self.use_aspect_var.get())
                except:
                    pass
        print(f"Keep aspect ratio : {self.use_aspect_var.get()}")

    #Start Capture

    def start_capture(self):
        
        self.remove_closed_window_list()
        
        self.mask_paint_mode = False
        
        self.current_active_index = 0
        
        self.target_window_full_name = win32gui.FindWindow(None, self.window_combobox.get())
        
        #Fist run 
  
        if self.first_run:
            if (self.capture_list):
            
                try:
                    self.active_capture =  self.capture_list[-1]

                except:
                
                    pass
            
        self.target_window_name = self.window_combobox.get()
        
        if (self.target_window_name == "Disable"):
            self.target_window_capture_mode = False
            self.window_capture_enable = False
        else:
            self.target_window_capture_mode = True
        
        self.keypress = False

        self.capture_image_name = self.capture_name_entry.get()
        
        
        
        # Parse width and height from the entry fields
        
        try:
            entered_width = int(self.width_entry.get())
        except:
            entered_width = 0
        try:    
            entered_height = int(self.height_entry.get())
        except:
            entered_height = 0

        #Check direct capture enable
        if (self.webcam_combobox.get() == ("Region Capture")) or self.webcam_combobox.get() == ("Window Capture"):
            self.Direct_capture_enable = True
            
            if self.webcam_combobox.get() == ("Window Capture"):
                self.window_capture_enable = True
            else :
                self.window_capture_enable = False
        else:
            self.Direct_capture_enable = False
            self.window_capture_enable = False
            
        #Capture image from webcam or direct window
        if (self.Direct_capture_enable) == False:
             
            self.cam_index = self.cam_list.index(self.webcam_combobox.get())
            self.cap = cv2.VideoCapture(self.cam_index)

        # If no output folder is provided, create a 'capture' folder in the current directory
        if not self.output_folder:
            self.output_folder = os.path.join(os.getcwd(), "capture")
            
        os.makedirs(self.output_folder, exist_ok=True)
        
        if not self.render_folder:
            self.output_folder = os.path.join(os.getcwd(), "render")
        
        # Set the selected image format
        self.format = self.format_combobox.get()

        self.is_capturing = True
        self.start_button.config(state=tk.DISABLED)
        self.reload_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.webcam_combobox.config(state=tk.DISABLED)

        # Change UI color to yellow during capture
        self.root.configure(background='yellow')


        #Thread start - Capture frames
        
        self.thread = Thread(target=self.capture_frames, daemon=True)
        self.thread.start()
        
    #Stop cpature

    def stop_capture(self):
        self.is_capturing = False
        self.start_button.config(state=tk.NORMAL)
        self.reload_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.webcam_combobox.config(state=tk.NORMAL)
        self.first_run = False
        self.remove_closed_window_list()

        # Return UI color to the original color after stopping capture
        self.root.configure(background='SystemButtonFace')
    
    #Refresh cam & window list
    
    def reload_list(self):
        self.cam_list = get_available_cameras()
        self.cam_list.append("Region Capture")
        self.cam_list.append("Window Capture")
        self.webcam_combobox['values'] = self.cam_list
        self.window_list = get_available_windows()
        self.window_list.insert(0, "Disable")
        self.window_combobox['values'] = self.window_list

    def top_most_main(self):
        
        if not self.top_most_window :
            self.top_most_window = True
            self.root.attributes('-topmost', 1)
            self.topmost_button.configure(bg="red")
        else :
            self.top_most_window = False
            self.root.attributes('-topmost', 0)
            #self.root.title("Webcam App")
            self.topmost_button.configure(bg="SystemButtonFace")

    def direct_capture(self):
        
        self.direct_capture_window(self.capture_name_entry.get())
        
        
        self.cam_index = self.cam_list.index(self.webcam_combobox.get())
        
    #Combine & export ai images        
    def export_video(self):
    
        image_folder = os.path.dirname(self.render_folder)
        
        images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
        
        # Output video file name
        output_video = image_folder+'/export_'+str(datetime.now().strftime("%Y%m%d_%H%M%S"))+'.mp4'
        
        # Sort the images based on their filenames (ensure proper order)
        images.sort()
        
        #Delete first (last frame)
        images.remove(images[0])

        # Get the image dimensions from the first image
        img = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = img.shape

        # Define the video codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        export_fps = self.exportfps_entry.get()
        video = cv2.VideoWriter(output_video, fourcc, int(export_fps), (width, height))

        # Iterate through each image and add it to the video
        for image in images:
            img = cv2.imread(os.path.join(image_folder, image))
            video.write(img)

        # Release the VideoWriter object
        video.release()
        print("Exported!")
        
        # Remove the image files after video export
        if self.delete_images_var.get():
            for image in images:
                os.remove(os.path.join(image_folder, image))
            print("Render images deleted.")


    #Capture image
    def capture_frames(self):
        
        #Capture Webcam or Direct window
        while self.is_capturing:
            
            wrong_size = False
            
            aspect_ratio = 1.0
            
            frame = np.zeros((512, 512, 3), dtype=np.uint8)
            
            if (self.Direct_capture_enable) == False:
                ret, frame = self.cap.read()
                
                if not ret:

                    break
                    
            elif ((self.Direct_capture_enable == True) and (self.window_capture_enable == False)):

                if self.capture_list and self.active_capture:
                #Test image size valid
                    h,w,c = frame.shape
                    if h>0 and w>0:
                        pass  
                    else:
                        wrong_size = True
                    if ((find_window(self.active_capture.capture_window_name)) == False):
                        wrong_size = True

                #Capture direct window        
                    if(wrong_size==False):
                        if not(self.active_capture.target_window_name == "Disable"):
                            frame = capture_win_target(self.active_capture.target_window_handle, self.active_capture.capture_window_name, self.active_capture.target_crop_enable, self.margin)
                        else:
                            frame = Capture_window(self.active_capture.capture_window_name, self.margin)
                #If image is wrong        
                else:
                    wrong_size = True
                    frame = np.zeros((512, 512, 3), dtype=np.uint8)
            
            elif ((self.Direct_capture_enable == True) and (self.window_capture_enable == True)):
                
                if not(self.target_window_full_name == "Disable"):
                    frame = capture_win_target(self.target_window_full_name, "", True, self.margin)
                    
            
            #Get image size    
            fh, fw, fc = frame.shape
            
            try:
                entered_width = int(self.width_entry.get())
            except:
                entered_width = 0
            try:    
                entered_height = int(self.height_entry.get())
            except:
                entered_height = 0
           
            #Get new image size
            if not(entered_width == 0 and entered_height == 0):
                try:
                    self.width, self.height = self.resize_capture(fw, fh, entered_width, entered_height)
                except:
                    wrong_size = True
                    print("Wrong size!!")
        
                # Resize the frame based on the entered width and height
                try:
                    #Add padding to image
                    if self.use_aspect_var.get():
                        # Read the input image
                        original_image = frame
                        
                        target_width = entered_width
                        
                        target_height = entered_height
                        
                        if target_height == 0:
                            target_height = target_width
                        if target_width == 0:
                            target_width = target_height

                        # Get the original image dimensions
                        original_height, original_width = original_image.shape[:2]

                        # Calculate the scaling factors for width and height
                        width_scale = target_width / original_width
                        height_scale = target_height / original_height

                        # Choose the minimum scaling factor to maintain the aspect ratio
                        min_scale = min(width_scale, height_scale)

                        # Calculate the new dimensions
                        new_width = int(original_width * min_scale)
                        new_height = int(original_height * min_scale)

                        # Resize the image while maintaining the aspect ratio
                        resized_image = cv2.resize(original_image, (new_width, new_height))

                        # Calculate the borders to fill the space
                        top_border = (target_height - new_height) // 2
                        bottom_border = target_height - new_height - top_border
                        left_border = (target_width - new_width) // 2
                        right_border = target_width - new_width - left_border

                        # Use copyMakeBorder to add borders and achieve the desired size
                        
                        padding_type = self.padding_type_combobox.get()
                        
                        if padding_type == "Replicate" :
                            frame = cv2.copyMakeBorder(resized_image, top_border, bottom_border, left_border, right_border, cv2.BORDER_REPLICATE)
                        elif padding_type == "Reflect" :
                            frame = cv2.copyMakeBorder(resized_image, top_border, bottom_border, left_border, right_border, cv2.BORDER_REFLECT)
                        elif padding_type == "Wrap" :
                            frame = cv2.copyMakeBorder(resized_image, top_border, bottom_border, left_border, right_border, cv2.BORDER_WRAP)
                        elif padding_type == "Constant" :
                            frame = cv2.copyMakeBorder(resized_image, top_border, bottom_border, left_border, right_border, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    
                    else:
                        frame = cv2.resize(frame, (self.width, self.height))
                except:
                    frame = np.zeros((512, 512, 3), dtype=np.uint8)
                    print("Wrong size!!!")

                    wrong_size = True
                    fh, fw = 512, 512
            
            #Save Capture image
            if (wrong_size==False):
            
                if self.window_capture_enable == False and self.Direct_capture_enable == True:    
                    if (self.active_capture.target_window_name == "Disable"):
                        self.capture_image_name = self.active_capture.layer_list[self.active_capture.current_layer]
                        
                    else:

                        self.capture_image_name = self.active_capture.layer_list[self.active_capture.current_layer]

                frame_path = os.path.join(self.output_folder, f"{self.capture_image_name}.{self.format}")
                
                if(self.pause_capture==False):
                    cv2.imwrite(frame_path, frame)
                

                
                #Face detect mask generate
                if self.face_detect_var.get():
                    frame = face_detection(frame, self.output_folder, self.format, self.mask_size_entry.get())



            #Load ai render image
            render_path = os.path.join(self.render_folder)
            
            if os.path.exists(render_path):
                renderimage = cv2.imread(render_path)
            else :
                renderimage =  np.zeros((fh, fw), dtype=np.uint8)
                
            if renderimage is None:
                renderimage =  np.zeros((fh, fw), dtype=np.uint8)
             
            if not wrong_size:
                #Get aspect ratio
                aspect_ratio = frame.shape[1] / frame.shape[0]
            
            
            
            # try:
                # if self.active_capture.ai_image_enable:
                    # self.active_capture.update_ai_image(render_path)
            # except:
                # pass

            # Show or hide the preview based on the checkbox
            if self.show_preview_var.get():
                if find_window("Webcam_Capture") == False:
                    cv2.namedWindow("Webcam_Capture", cv2.WINDOW_NORMAL)
                
                if (self.show_top_var.get() == 1) and (cv2.getWindowProperty("Webcam_Capture", cv2.WND_PROP_TOPMOST) == 0):
                    cv2.setWindowProperty("Webcam_Capture", cv2.WND_PROP_TOPMOST, 1)
                    
                if (self.Direct_capture_enable == True) and (self.window_capture_enable == True):
                    #text_overlay = self.active_capture.capture_window_name
                    text_overlay = self.capture_image_name
                else :
                    text_overlay = self.capture_image_name
                 
                if self.pause_capture:
                    save_status = "-Pause"
                else:
                    save_status = ""
                    
                 
                #Mask mode 
                if self.mask_paint_mode:    
                    
                    h,w,c = frame.shape
                    
                    self.mask_image = cv2.resize(self.mask_image, (w, h))

                    mask_overlay_image = cv2.addWeighted(frame, 0.7, self.mask_image, 0.3, 0)

                    cv2.imshow("Webcam_Capture", cv2.putText(mask_overlay_image, text_overlay+"_mask"+save_status, (10,35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2))
                    
                    cv2.imwrite(os.path.join(self.output_folder, f"{self.capture_image_name}_mask.{self.format}"), self.mask_image)
                
                else:                
                    cv2.imshow("Webcam_Capture", cv2.putText(frame, text_overlay+save_status, (10,35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2))
                
                if self.mask_paint_mode == True:
                    try:
                        cv2.setMouseCallback('Webcam_Capture', self.draw_circle)
                    except:
                        pass
                
                
                cv2.resizeWindow("Webcam_Capture", cv2.getWindowImageRect("Webcam_Capture")[2], int(cv2.getWindowImageRect("Webcam_Capture")[2]/aspect_ratio))

            else:
                if (cv2.getWindowProperty("Webcam_Capture", cv2.WND_PROP_VISIBLE) > 0):
                    cv2.destroyWindow("Webcam_Capture")
            

            # Show or hide the Render preview based on the checkbox
            if self.show_render_var.get():
            
                if (self.Direct_capture_enable == True) and (self.window_capture_enable == True):
                    #text_overlay = self.active_capture.capture_window_name
                    text_overlay = self.capture_image_name
                else :
                    text_overlay = self.capture_image_name
                
                if self.pause_capture:
                    save_status = "-Pause"
                else:
                    save_status = ""
                
                if find_window("AI_Render_Preview") == False:
                    cv2.namedWindow("AI_Render_Preview", cv2.WINDOW_NORMAL)  
                
                if (self.show_top_var.get() == 1) and (cv2.getWindowProperty("AI_Render_Preview", cv2.WND_PROP_TOPMOST) == 0):
                    cv2.setWindowProperty("AI_Render_Preview", cv2.WND_PROP_TOPMOST, 1)

                cv2.imshow("AI_Render_Preview", cv2.putText(renderimage, text_overlay+save_status, (10,35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2))
                
                cv2.resizeWindow("AI_Render_Preview", cv2.getWindowImageRect("AI_Render_Preview")[2], int(cv2.getWindowImageRect("AI_Render_Preview")[2]/aspect_ratio))
            else:
                if (cv2.getWindowProperty("AI_Render_Preview", cv2.WND_PROP_VISIBLE) > 0):
                    cv2.destroyWindow("AI_Render_Preview")

                        
            #Copy preview image to clipboard
            
            key_pressed = cv2.waitKeyEx(1) & 0xFF
            
            if key_pressed == ord('q'):
                
                clip_path = os.path.join(self.render_folder)
                
                if os.path.exists(clip_path):
                    image = Image.open(clip_path)

                    output = BytesIO()
                    image.convert("RGB").save(output, "BMP")
                    data = output.getvalue()[14:]
                    output.close()

                    send_to_clipboard(win32clipboard.CF_DIB, data)
                    
                    print("Copied to clipboard!")
                    
            elif key_pressed == ord('m'):
                if self.mask_paint_mode == False:
                    self.mask_paint_mode = True
                    print("Mask paint mode enable")
                    
                else:
                    self.mask_paint_mode = False
                    print("Mask paint mode disable")
                    
            elif key_pressed == ord('n'):
                self.mask_image = np. zeros((512, 512, 3), np.uint8)
                
            elif key_pressed == ord('c'):
                self.remove_closed_window_list()
                
                if self.Direct_capture_enable:
                    try:
                    
                        if (self.active_capture.capture_window_name == self.capture_list[self.current_active_index].capture_window_name):
                            self.current_active_index = (self.current_active_index+1) % (len(self.capture_list))
                    
                        self.active_capture = self.capture_list[self.current_active_index]
                        
                        print(f"Capture : {self.active_capture.capture_window_name}")
                        
                    except:
                        pass
                    
            

            
            elif key_pressed == ord('s'): 
                
                render_path = os.path.join(self.render_folder)
                image_format = os.path.splitext(render_path)[1]
            
                if os.path.exists(render_path):
                    save_path=(os.path.dirname(render_path)+"/saved_frame")
                    os.makedirs(save_path, exist_ok=True)
                    current_time = str(datetime.now().strftime("%Y%m%d_%H%M%S%f"))
                    shutil.copyfile(render_path, save_path+f"/saved_frame_{current_time}{image_format}")
                    print("Frame saved! / "+ save_path+f"/saved_frame_{current_time}{image_format}")
            
            
            elif key_pressed == ord('a'):
                try:
                    win = pygetwindow.getWindowsWithTitle(self.active_capture.capture_window_name)[0]
                    win.minimize()
                    win.restore()
                    win.activate()
                except :
                    pass
                    
            elif key_pressed == ord('z'): #Pause capture

                if self.pause_capture == True:
                
                    self.pause_capture = False
                    print("Save start")
                    
                else :
                
                    self.pause_capture = True
                    print("Save pause")
                  
  
            elif key_pressed == ord('x'): #Change save layer
            
                if self.active_capture:
                    try:
                        self.active_capture.current_layer = (self.active_capture.current_layer+1) % (len(self.active_capture.layer_list))
                        print(f"Layer : {self.active_capture.layer_list[self.active_capture.current_layer]}")
                    except:
                        self.active_capture.current_layer = 0
             
            # Parse fps from the entry field
            try:
                fps = float(self.fps_entry.get())
            except:
                fps = 20
            
            if (fps > 0) :
                
                try:
                    self.delay_seconds = 1.0 / fps  # Convert fps to seconds
                except:
                    self.delay_seconds - 0.1
            

                #Wait
                time.sleep(self.delay_seconds)
            
            
        self.cap.release()
        cv2.destroyAllWindows()
        
        
        
    def callback():
        global after_id
        after_id = root.after(500, callback)
    
    #Close app
    def quit():
        """Cancel all scheduled callbacks and quit."""
        self.stop_capture
        self.is_capturing = False
        cv2.destroyAllWindows()
        root.after_cancel(after_id)
        root.destroy()    
        

    def run(self):
        self.callback

        self.root.protocol('WM_DELETE_WINDOW', quit)
        
        self.root.mainloop()
    
    
    def close_app(self):
        self.is_capturing = False
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    
    output_folder = "captured_frames"
    render_folder = "rendered_frames/render.jpg"
    app = WebcamApp(output_folder, render_folder)
    app.run()
   
