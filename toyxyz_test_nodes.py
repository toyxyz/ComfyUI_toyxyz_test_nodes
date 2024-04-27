from PIL import Image, ImageFile
import comfy.utils
import numpy as np
import torch
import torchvision.transforms.functional as tf
import torchvision.transforms.v2 as T
import os
import time
import cv2
from pathlib import Path
from nodes import MAX_RESOLUTION, SaveImage, common_ksampler

ImageFile.LOAD_TRUNCATED_IMAGES = True

def p(image):
    return image.permute([0,3,1,2])
def pb(image):
    return image.permute([0,2,3,1])

def save_image(img: torch.Tensor, path, image_format, jpg_quality, png_compress_level):
    path = str(path)

    if len(img.shape) != 3:
        raise ValueError(f"can't take image batch as input, got {img.shape[0]} images")

    img = img.permute(2, 0, 1)
    if img.shape[0] != 3:
        raise ValueError(f"image must have 3 channels, but got {img.shape[0]} channels")

    img = img.clamp(0, 1)
    img = tf.to_pil_image(img)
    
    ext = image_format

    if ext == ".jpg":
        # Save as JPEG with specified quality
        img.save(path, format="JPEG", quality=jpg_quality)
    elif ext == ".png":
        # Save as PNG with specified compression level
        img.save(path, format="PNG", compress_level=png_compress_level)
    elif ext == ".bmp":
        img.save(path, format="bmp")
    else:
        # Raise an error for unsupported file formats
        raise ValueError(f"Unsupported file format: {ext}")

    subfolder, filename = os.path.split(path)

    return {"filename": filename, "subfolder": subfolder, "type": "output"}

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

    def __init__(self):
        self.webcam_index = 0

    def load_image(self, select_webcam):
        capture = cv2.VideoCapture(select_webcam, cv2.CAP_DSHOW)

        try:
            # should be instantly opened
            if not capture.isOpened():
                print("Error: Could not open webcam.")

                return
            else:
                # Capture frame-by-frame
                # fake read first because the first frame is warmup and sometimes contains artifacts
                ret, frame = capture.read()
                ret, frame = capture.read()

                image = Image.fromarray(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if (image is None):
                print("Error: Could not read frame.")
                return

            image = image.convert('RGB')
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]

            return (image,)
        finally:
            capture.release()

    @classmethod
    def IS_CHANGED(cls):

        return
        
class LoadWebcamImage:

    @classmethod
    def INPUT_TYPES(cls):
        return {
                "required": {
                    "image_path": ("STRING", {"default": './ComfyUI/custom_nodes/ComfyUI_toyxyz_test_nodes/CaptureCam/captured_frames/capture.jpg', "multiline": False}), 
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

class SaveImagetoPath:

    INPUT_TYPES = lambda: {
        "required": {
            "path": ("STRING", {"default": "./ComfyUI/custom_nodes/ComfyUI_toyxyz_test_nodes/CaptureCam/rendered_frames/render.jpg"}),
            "image": ("IMAGE",),
            "save_sequence": (("false", "true"), {"default": "false"}),
            "image_format": ((".jpg", ".png", ".bmp"), {"default": ".jpg"}),
            "jpg_quality": ("INT", {
                    "default": 70,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
            "png_compression": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 9,
                    "step": 1,
                    "display": "number"
                }),
        },
    }
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "execute"
    
    CATEGORY = "ToyxyzTestNodes"

    def execute(
        self,
        path: str,
        image_format: str,
        image: torch.Tensor,
        save_sequence: str,
        jpg_quality,
        png_compression,
    ):
        assert isinstance(path, str)
        assert isinstance(image_format, str)
        assert isinstance(image, torch.Tensor)
        assert isinstance(save_sequence, str)

        save_sequence: bool = save_sequence == "true"

        path: Path = Path(path)

        path2 = path
        
        if save_sequence :
            count = 0
            base_filename, file_extension = path2.stem, path2.suffix
            while path2.exists():
                filename = f"{base_filename}_{format(count, '06')}{file_extension}"
                path2 = Path(path2.parent, filename)
                count += 1

        path.parent.mkdir(exist_ok=True)
        
        if image.shape[0] == 1:
            # batch has 1 image only
            save_image(
                image[0],
                path,
                image_format,
                jpg_quality,
                png_compression,
            )
            if save_sequence :
                save_image(
                    image[0],
                    path2,
                    image_format,
                    jpg_quality,
                    png_compression,
                )
            
        else:
            # batch has multiple images
            for i, img in enumerate(image):
                subpath = path.with_stem(f"{path.stem}-{i}")
                save_image(
                    img,
                    subpath,
                    image_format,
                    jpg_quality,
                    png_compression,
                )
                for i, img in enumerate(image):
                    subpath = path.with_stem(f"{path2.stem}-{i}")
                    save_image(
                        img,
                        subpath,
                        image_format,
                        jpg_quality,
                        png_compression,
                    )

        return ()
        

class LatentDelay:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "latent": ("LATENT",),
                              "delaytime": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "LatentDelay"
    
    CATEGORY = "ToyxyzTestNodes"

    def LatentDelay(self, latent, delaytime):
        time.sleep(delaytime)
        return (latent,)

class ImageResize_Padding:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "interpolation": (["nearest", "bilinear", "bicubic", "area", "nearest-exact", "lanczos"],),
                "padding": ("BOOLEAN", { "default": True }),
                "Red":("FLOAT", { "default": 0, "min": 0, "max": 1, "step": 0.1, }),
                "Green":("FLOAT", { "default": 0, "min": 0, "max": 1, "step": 0.1, }),
                "Blue":("FLOAT", { "default": 0, "min": 0, "max": 1, "step": 0.1, }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "width", "height",)
    FUNCTION = "execute"
    CATEGORY = "ToyxyzTestNodes"

    def execute(self, image, width, height, padding, Red, Green, Blue, interpolation="nearest"):
        _, oh, ow, _ = image.shape
        
        if padding is True:
            oAspectRatio = ow/oh
            tAspectRatio = width/height
            
            if oAspectRatio > tAspectRatio:
                pady = int(((ow/tAspectRatio)-oh)/2)
                padx = 0
            if oAspectRatio < tAspectRatio:
                padx = int(((oh*tAspectRatio)-ow)/2)
                pady = 0
            if oAspectRatio == tAspectRatio:
                padx = 0
                pady = 0
                
            pad = (padx, pady, padx, pady)
            image = pb(T.functional.pad(p(image), pad, fill=(Red, Green, Blue)))
            

        outputs = p(image)
        
        outputs = comfy.utils.lanczos(outputs, width, height)

        outputs = pb(outputs)

        return(outputs, outputs.shape[2], outputs.shape[1],)
       

NODE_CLASS_MAPPINGS = {
    "CaptureWebcam": CaptureWebcam,
    "LoadWebcamImage": LoadWebcamImage,
    "SaveImagetoPath": SaveImagetoPath,
    "LatentDelay": LatentDelay,
    "ImageResize_Padding": ImageResize_Padding,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CaptureWebcam": "Capture Webcam",
    "LoadWebcamImage": "Load Webcam Image",
    "SaveImagetoPath": "Save Image to Path",
    "LatentDelay": "LatentDelay",
    "ImageResize_Padding": "ImageResize_Padding",
}

