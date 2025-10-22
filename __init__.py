from .ComfyCouple.comfy_couple import ComfyCoupleMask, ComfyCoupleRegion
from .ComfyCouple.Extractor import ComfyCoupleRegionExtractor
from .ComfyCouple.Visualizer import ComfyCoupleRegionVisualizer
from .nodes.toyxyz_test_nodes import CaptureWebcam, LoadWebcamImage, SaveImagetoPath, LatentDelay, ImageResize_Padding, Direct_screenCap, Depth_to_normal, Remove_noise, Export_glb, Load_Random_Text_From_File
from .nodes.visual_area_mask import VisualAreaMask
from .openposeeditor.openpose_editor_nodes import OpenposeEditorNode, PoseToMaskNode
from .openposeeditor.poseinter import Pose_Inter, PoseKeypointToCoordStr, JoinPose


NODE_CLASS_MAPPINGS = {
    "ComfyCoupleMask": ComfyCoupleMask,
    "ComfyCoupleRegion": ComfyCoupleRegion,
    "VisualAreaMask": VisualAreaMask,
    "CaptureWebcam": CaptureWebcam,
    "LoadWebcamImage": LoadWebcamImage,
    "SaveImagetoPath": SaveImagetoPath,
    "LatentDelay": LatentDelay,
    "ImageResize_Padding": ImageResize_Padding,
    "Direct Screen Capture": Direct_screenCap,
    "Depth to normal": Depth_to_normal,
    "Remove noise": Remove_noise,
    "Export glb": Export_glb,
    "Load Random Text From File": Load_Random_Text_From_File,
    "OpenposeEditorNode": OpenposeEditorNode,
    "PoseToMaskNode": PoseToMaskNode,
    "Pose_Inter": Pose_Inter,
    "PoseKeypointToCoordStr": PoseKeypointToCoordStr,
    "JoinPose": JoinPose,
    "ComfyCoupleRegionExtractor": ComfyCoupleRegionExtractor,
    "ComfyCoupleRegionVisualizer": ComfyCoupleRegionVisualizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyCoupleMask": "ComfyCouple Mask",
    "ComfyCoupleRegion": "ComfyCouple Region",
    "VisualAreaMask": "Visual Area Mask",
    "CaptureWebcam": "Capture Webcam",
    "LoadWebcamImage": "Load Webcam Image",
    "SaveImagetoPath": "Save Image to Path",
    "LatentDelay": "LatentDelay",
    "ImageResize_Padding": "ImageResize_Padding",
    "Direct_screenCap": "Direct_screenCap",
    "Depth_to_normal": "Depth_to_normal",
    "Remove_noise": "Remove_noise",
    "Export_glb": "Export_glb",
    "Load_Random_Text_From_File": "Load_Random_Text_From_File",
    "OpenposeEditorNode": "Openpose Editor Node",
    "PoseToMaskNode": "Pose Keypoints to Mask",
    "Pose_Inter": "Pose Interpolation",
    "PoseKeypointToCoordStr": "POSE_KEYPOINT to Coord_str",
    "JoinPose": "join_pose",
    "ComfyCoupleRegionExtractor": "Comfy Couple Region Extractor",
    "ComfyCoupleRegionVisualizer": "Region Visualizer (Comfy Couple)",
}

WEB_DIRECTORY = "./web"
