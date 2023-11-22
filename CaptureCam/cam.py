import cv2
import os
import time
import tkinter as tk
from tkinter import ttk, filedialog # Import filedialog module for folder selection
from threading import Thread

class WebcamApp:
    def __init__(self, output_folder):
        self.output_folder = output_folder
        self.cam_index = 0  # Default webcam index
        self.width = 512  # Default width
        self.height = 0  # Default height
        self.cap = cv2.VideoCapture(self.cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.root = tk.Tk()
        self.root.title("Webcam App")

        # Add margins left and right
        self.root.geometry("250x650+100+100")

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
        self.show_preview_checkbox = tk.Checkbutton(self.root, text="Show Preview", variable=self.show_preview_var)
        self.show_preview_checkbox.pack(pady=5, padx=10)

        # Output folder selection
        self.output_folder_label = tk.Label(self.root, text="Select Output Folder:")
        self.output_folder_label.pack(pady=5, padx=10)

        self.output_folder_entry = tk.Entry(self.root)
        self.output_folder_entry.pack(pady=5, padx=10)

        self.output_folder_button = tk.Button(self.root, text="Browse", command=self.browse_output_folder)
        self.output_folder_button.pack(pady=5, padx=10)
        
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

        self.is_capturing = False

        self.thread = None
        
    def browse_output_folder(self):
        self.output_folder = filedialog.askdirectory()
        self.output_folder_entry.delete(0, tk.END)
        self.output_folder_entry.insert(0, self.output_folder)

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

            # Show or hide the preview based on the checkbox
            if self.show_preview_var.get():
                cv2.imshow("Webcam Capture", frame)
            else:
                cv2.destroyAllWindows()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(self.delay_seconds)

        self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    output_folder = "captured_frames"
    app = WebcamApp(output_folder)
    app.run()