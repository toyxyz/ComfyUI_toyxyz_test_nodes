from PIL import Image, ImageFile
import comfy.utils
import numpy as np
import torch
import os
import time
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CaptureWebcam:

    @classmethod
    def INPUT_TYPES(cls):
        return {
                "required": {
                    "select_webcam": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "display": "number"
                })
                },
            }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = "load_image"
    
    CATEGORY = "ToyxyzTestNodes"

    def load_image(self, select_webcam):
    
        # Open the webcam (default camera)
        cap = cv2.VideoCapture(select_webcam)

        # Check if the webcam is opened successfully
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            
            return
        else:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Check if the frame is captured successfully
            if not ret:
                print("Error: Could not read frame.")
                
                return
            
            i = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

        image = i

        image = image.convert('RGB')
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
           
        return (image, )

    @classmethod
    def IS_CHANGED(cls):

        return
        
class LoadWebcamImage:

    @classmethod
    def INPUT_TYPES(cls):
        return {
                "required": {
                    "image_path": ("STRING", {"default": './ComfyUI/custom_nodes/toyxyz_test/CaptureCam/captured_frames/capture.jpg', "multiline": False}), 
                }
            }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_image"
    
    CATEGORY = "ToyxyzTestNodes"

    def load_image(self, image_path):

        try:
            i = Image.open(image_path)
            i.verify()
            i = Image.open(image_path)
            
        except OSError as e:
            print("Load fail")

            try:
                time.sleep(0.05)
                i = Image.open(image_path)
                print("Try again!")
                    
            except OSError as e:
                try:
                    time.sleep(0.05)
                    i = Image.open(image_path)
                    print("Try again!")
                
                except OSError as e:
                    print("Image doesn't exist!")
                    i = Image.new(mode='RGB', size=(512, 512), color=(0, 0, 0))
   
        if not i:
            return
            
        image = i  

        image = image.convert('RGB')
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
     
        return (image, )

    @classmethod
    def IS_CHANGED(cls, image_path):
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

NODE_CLASS_MAPPINGS = {
    "CaptureWebcam": CaptureWebcam,
    "LoadWebcamImage": LoadWebcamImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CaptureWebcam": "Capture Webcam",
    "LoadWebcamImage": "Load Webcam Image",
}