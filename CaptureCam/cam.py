import cv2
import sys
import numpy as np
import os
import time
import tkinter as tk
from tkinter import ttk, filedialog # Import filedialog module for folder selection
from threading import Thread
from datetime import datetime
from io import BytesIO
import win32clipboard
from PIL import Image

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
    
def send_to_clipboard(clip_type, data):
        
    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardData(clip_type, data)
    win32clipboard.CloseClipboard()
    
def face_detection(frame, mask_folder, mask_format, m_scale):
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=6,
        minSize=(10, 10),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
        
    fh, fw, fc = frame.shape
    
    mask_scale = int(m_scale)
    
    frame = np.zeros((fh, fw, 3), np.uint8)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x-mask_scale, y-mask_scale), (x+w+mask_scale, y+h+mask_scale), (255, 255, 255), -1)
            
    mask_path = os.path.join(mask_folder, f"face_mask.{mask_format}")
        
    cv2.imwrite(mask_path, frame)

    

class WebcamApp:
    def __init__(self, output_folder, render_folder):
        self.output_folder = output_folder
        self.render_folder = render_folder
        self.cam_index = 0  # Default webcam index
        self.width = 512  # Default width
        self.height = 0  # Default height
        self.cap = cv2.VideoCapture(self.cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.root = tk.Tk()
        self.root.title("Webcam App")

        # Add margins left and right
        self.root.geometry("250x980+100+100")

        # Webcam selection dropdown
        self.webcam_label = tk.Label(self.root, text="Select Webcam:")
        self.webcam_label.pack(pady=5, padx=10)

        self.webcam_combobox = ttk.Combobox(self.root, values=[f"Webcam {i}" for i in range(10)])  # Adjust the range as needed
        self.webcam_combobox.set("Webcam 0")
        self.webcam_combobox.pack(pady=10, padx=10)

        # Resolution input fields
        self.width_label = tk.Label(self.root, text="Enter Width:")
        self.width_label.pack(pady=5, padx=10)

        self.width_entry = tk.Entry(self.root)
        self.width_entry.insert(0, "512")  # Default width
        self.width_entry.pack(pady=5, padx=10)

        self.height_label = tk.Label(self.root, text="Enter Height:")
        self.height_label.pack(pady=5, padx=10)

        self.height_entry = tk.Entry(self.root)
        self.height_entry.insert(0, "0")  # Default height
        self.height_entry.pack(pady=10, padx=10)
        
        # FPS input field
        self.fps_label = tk.Label(self.root, text="Enter FPS:")
        self.fps_label.pack(pady=5, padx=10)

        self.fps_entry = tk.Entry(self.root)
        self.fps_entry.insert(0, "12")  # Default fps (adjust as needed)
        self.fps_entry.pack(pady=5, padx=10)

        # Checkbox for show preview
        self.show_preview_var = tk.IntVar()
        self.show_preview_var.set(0)  # Default checked
        self.show_preview_checkbox = tk.Checkbutton(self.root, text="Show Webcan Preview", variable=self.show_preview_var)
        self.show_preview_checkbox.pack(pady=5, padx=10)
        
        # Checkbox for show render
        self.show_render_var = tk.IntVar()
        self.show_render_var.set(0)  # Default checked
        self.show_render_checkbox = tk.Checkbutton(self.root, text="Show AI Render", variable=self.show_render_var)
        self.show_render_checkbox.pack(pady=5, padx=10)
        
        # Checkbox for preivew always on top
        self.show_top_var = tk.IntVar()
        self.show_top_var.set(0)  # Default checked
        self.show_top_checkbox = tk.Checkbutton(self.root, text="Preview always on top", variable=self.show_top_var)
        self.show_top_checkbox.pack(pady=5, padx=10)
        
        # Checkbox for face detection
        self.face_detect_var = tk.IntVar()
        self.face_detect_var.set(0)  # Default checked
        self.face_detect_checkbox = tk.Checkbutton(self.root, text="Face detect mask", variable=self.face_detect_var)
        self.face_detect_checkbox.pack(pady=5, padx=10)
        
        # Mask scale
        
        self.mask_size_label = tk.Label(self.root, text="Mask padding:")
        self.mask_size_label.pack(pady=5, padx=10)
        self.mask_size_entry = tk.Entry(self.root)
        self.mask_size_entry.insert(0, "20")  # Default exportfps (adjust as needed)
        self.mask_size_entry.pack(pady=5, padx=10)

        # Output folder selection
        self.output_folder_label = tk.Label(self.root, text="Select Output Folder:")
        self.output_folder_label.pack(pady=5, padx=10)

        self.output_folder_entry = tk.Entry(self.root)
        self.output_folder_entry.pack(pady=5, padx=10)

        self.output_folder_button = tk.Button(self.root, text="Browse", command=self.browse_output_folder)
        self.output_folder_button.pack(pady=5, padx=10)
        
        # Render file selection
        self.render_folder_label = tk.Label(self.root, text="Select rendered image:")
        self.render_folder_label.pack(pady=5, padx=10)

        self.render_folder_entry = tk.Entry(self.root)
        self.render_folder_entry.pack(pady=5, padx=10)

        self.render_folder_button = tk.Button(self.root, text="Browse", command=self.browse_render_file)
        self.render_folder_button.pack(pady=5, padx=10)
        
        # Image format selection
        self.format_label = tk.Label(self.root, text="Select Image Format:")
        self.format_label.pack(pady=5, padx=10)

        self.format_combobox = ttk.Combobox(self.root, values=["png", "jpg"])
        self.format_combobox.set("jpg")
        self.format_combobox.pack(pady=10, padx=10)

        # Start & Stop
        self.start_button_label = tk.Label(self.root, text="Start Stop capture:")
        self.start_button_label.pack(pady=5, padx=10)
        
        self.start_button = tk.Button(self.root, text="Start", command=self.start_capture)
        self.start_button.pack(pady=5, padx=10)

        self.stop_button = tk.Button(self.root, text="Stop", command=self.stop_capture, state=tk.DISABLED)
        self.stop_button.pack(pady=5, padx=10)
        
        # Export video
        
        self.export_button_label = tk.Label(self.root, text="Export Video to rendered Folder:")
        self.export_button_label.pack(pady=5, padx=10)
        
        self.export_button = tk.Button(self.root, text="Export", command=self.export_video)
        self.export_button.pack(pady=5, padx=10)
        
        # Checkbox for image delete after export
        self.delete_images_var = tk.IntVar()
        self.delete_images_var.set(0)  # Default checked
        self.delete_images_checkbox = tk.Checkbutton(self.root, text="Remove images after export", variable=self.delete_images_var)
        self.delete_images_checkbox.pack(pady=5, padx=10)
        
        # FPS input field

        self.exportfps_entry = tk.Entry(self.root)
        self.exportfps_entry.insert(0, "12")  # Default exportfps (adjust as needed)
        self.exportfps_entry.pack(pady=5, padx=10)

        self.is_capturing = False

        self.thread = None
        
    def browse_output_folder(self):
        self.output_folder = filedialog.askdirectory()
        self.output_folder_entry.delete(0, tk.END)
        self.output_folder_entry.insert(0, self.output_folder)
        
    def browse_render_file(self):
        self.render_folder = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg")])
        self.render_folder_entry.delete(0, tk.END)
        self.render_folder_entry.insert(0, self.render_folder)

    def start_capture(self):
        self.cam_index = int(self.webcam_combobox.get().split()[-1])
        self.cap = cv2.VideoCapture(self.cam_index)

        # Parse width and height from the entry fields
        entered_width = int(self.width_entry.get())
        entered_height = int(self.height_entry.get())

        # Parse fps from the entry field
        fps = float(self.fps_entry.get())
        self.delay_seconds = 1.0 / fps  # Convert fps to seconds

        # Calculate missing width or height based on the entered value and aspect ratio
        if entered_width > 0 and entered_height == 0:
            # Calculate height based on the aspect ratio
            aspect_ratio = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) / self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.width = entered_width
            self.height = int(self.width / aspect_ratio)
        elif entered_height > 0 and entered_width == 0:
            # Calculate width based on the aspect ratio
            aspect_ratio = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) / self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.height = entered_height
            self.width = int(self.height * aspect_ratio)
        else:
            # Use the entered width and height
            self.width = entered_width
            self.height = entered_height

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
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
        self.stop_button.config(state=tk.NORMAL)

        # Change UI color to yellow during capture
        self.root.configure(background='yellow')

        self.thread = Thread(target=self.capture_frames)
        self.thread.start()
        
        # Parse fps from the entry field
        fps = float(self.fps_entry.get())
        self.delay_seconds = 1.0 / fps  # Convert fps to seconds

    def stop_capture(self):
        self.is_capturing = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

        # Return UI color to the original color after stopping capture
        self.root.configure(background='SystemButtonFace')
        

    def export_video(self):
    
        image_folder = os.path.dirname(self.render_folder)
        
        images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
        
        # Output video file name
        output_video = image_folder+'/export_'+str(datetime.now().strftime("%Y%m%d_%H%M%S"))+'.mp4'
        
        # Sort the images based on their filenames (ensure proper order)
        images.sort()

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
            print("Image files deleted.")

    def capture_frames(self):
        while self.is_capturing:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
                
            # Resize the frame based on the entered width and height
            frame = cv2.resize(frame, (self.width, self.height))

            frame_path = os.path.join(self.output_folder, f"capture.{self.format}")
            cv2.imwrite(frame_path, frame)
            
            #Face detect mask generate
            if self.face_detect_var.get():
            
                face_detection(frame, self.output_folder, self.format, self.mask_size_entry.get())
              
            #Save render preview
            render_path = os.path.join(self.render_folder)
            
            if os.path.exists(render_path):
                renderimage = cv2.imread(render_path)
            else :
                h, w, c = frame.shape
                renderimage =  np.zeros((h, w), dtype=np.uint8)
                
            if renderimage is None:
                h, w, c = frame.shape
                renderimage =  np.zeros((h, w), dtype=np.uint8)
                
            aspect_ratio = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) / self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            # Show or hide the preview based on the checkbox
            if self.show_preview_var.get():
                cv2.namedWindow("Webcam_Capture", cv2.WINDOW_NORMAL)
                
                if self.show_top_var.get(): 
                    cv2.setWindowProperty("Webcam_Capture", cv2.WND_PROP_TOPMOST, 1)                
                cv2.imshow("Webcam_Capture", frame)

                cv2.resizeWindow("Webcam_Capture", cv2.getWindowImageRect("Webcam_Capture")[2], int(cv2.getWindowImageRect("Webcam_Capture")[2]/aspect_ratio))

            else:
                if (cv2.getWindowProperty("Webcam_Capture", cv2.WND_PROP_VISIBLE) > 0):
                    cv2.destroyWindow("Webcam_Capture")
                
            if self.show_render_var.get():
                cv2.namedWindow("Render_Preview", cv2.WINDOW_NORMAL)  
                
                if self.show_top_var.get(): 
                    cv2.setWindowProperty("Render_Preview", cv2.WND_PROP_TOPMOST, 1)
                    
                cv2.imshow("Render_Preview", renderimage)
                
                cv2.resizeWindow("Render_Preview", cv2.getWindowImageRect("Render_Preview")[2], int(cv2.getWindowImageRect("Render_Preview")[2]/aspect_ratio))
            else:
                if (cv2.getWindowProperty("Render_Preview", cv2.WND_PROP_VISIBLE) > 0):
                    cv2.destroyWindow("Render_Preview")
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                
                clip_path = os.path.join(self.render_folder)
                
                if os.path.exists(clip_path):
                    image = Image.open(clip_path)

                    output = BytesIO()
                    image.convert("RGB").save(output, "BMP")
                    data = output.getvalue()[14:]
                    output.close()

                    send_to_clipboard(win32clipboard.CF_DIB, data)

            time.sleep(self.delay_seconds)

        self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    output_folder = "captured_frames"
    render_folder = "rendered_frames/render.jpg"
    app = WebcamApp(output_folder, render_folder)
    app.run()
