# ComfyUI_toyxyz_test_nodes

This is a custom node that collects the tools I use frequently.

https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes/assets/8006000/8536e96a-514a-48b2-b1aa-8eccbd3fa853

(This video is at 4x speed)

Update 

2025/10/30 - Add lora hook support to ComfyCouple Region node

2025/10/21 - Add Openpose Editor Node, Pose Interpolation, ComfyCouple Region, ComfyCouple Mask, Comfy Couple Region Extractor

2025/03/10 - Add Visual area mask node

2024/11/14 - Add Load Random Text From File node

2024/11/04 - Add Export glb node.

2024/11/02 - Add remove noise node for normal map. Added sobel ratio for more accurate Noraml.

2024/10/25 - Add depth to normal node.

2024/08/11 - Add Direct_screenCap node. 

2023/11/24 - AddSave image to path node. Add Render preview, Add export video, Add face detection (After the update, you will need to run CaptrueCam/setup.bat one more time.)

2023/11/29 - Add Region Capture. Made the Webcam app UI smaller. 

2023/12/01 - Add Ai render overlay 
         

## Installation

1. Git clone this repo to the ComfyUI/custom_nodes path.

   `git clone https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes`

2. Run setup.bat in `ComfyUI/custom_nodes/ComfyUI_toyxyz_test_nodes/CaptureCam`


## Usage

Default workflow
![workflow (36)](https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes/assets/8006000/7a8644d2-59f9-4ed5-a32a-82c75cdb0997)
(Workflow embedded)

Render preview workflow
![workflow (38)](https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes/assets/8006000/ef16937e-30a0-4af8-a1f9-0f0eb080b103)
(Workflow embedded)

Direct Webcam capture workflow (without webcam app)
![workflow (40)](https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes/assets/8006000/cac4b89b-c2a1-4007-8906-fb8de9e26213)
(Workflow embedded)


## Visual area mask

  Creates masks for the specified regions. Useful for regional prompting.

  Image_width: Specify the width of the mask
  
  Image_height: Specify the height of the mask
  
  area_number: Specify the number of areas to create. Maximum 12. 

  area_id : Area number to adjust. Starts from 0.

  x : X position of the area selected in area_id.

  y : Y position of the area selected in area_id.

  width : Width of the area selected in area_id.

  height: Height of the selected area at area_id.

  strength: Strength of the selected area at area_id.

  mask_overlap_method: default, subtract - Subtracts the masks from other regions from a single mask.

  Update outputs: Update nodes according to the number in area_number.

<img width="1576" height="1591" alt="image" src="https://github.com/user-attachments/assets/dcc54f06-7d5c-4a2c-844c-11f8ec8088ae" />

## Openpose Editor Node

  Modify each body part of OpenPose

<img width="2140" height="1800" alt="image" src="https://github.com/user-attachments/assets/c54a6c62-a8aa-4418-a8b7-6bd65d5cce82" />

## Pose Interpolation

  Generate interpolated poses between two OpenPose poses.

<img width="2408" height="1558" alt="image" src="https://github.com/user-attachments/assets/c76ae523-d09b-4d2a-b21f-447c76fdf36e" />

## ComfyCouple Region / ComfyCouple Mask 

  Regional Prompting Node. Supported models are SD 1.5, SDXL, and Flux. To disable Auto_inject_flux, you must free the model cache. To use Lora_hook, set skip_positive_conditioning to false.

<img width="2376" height="1818" alt="image" src="https://github.com/user-attachments/assets/d78d5d1d-67e7-4a0c-bc1f-26569cc638b6" />

## Comfy Couple Region Extractor

  Cut out the masked region from the couple region. It can be used in the face detailing workflow.

<img width="2505" height="1667" alt="image" src="https://github.com/user-attachments/assets/5b8871f0-24db-46e8-b69a-4c0e8aa844cf" />



## Load Random Text From File

  Retrieves the entire text or random lines from a txt file at the entered path.  

  file_paht : The path to the text file or the path where the files are located 

  seed : seed for random line

  edit_text : Edit tag_(tag) to tag \(tag\)

  get_random_line : Get random line from txt. False for get entire text

  get_random_txt_from_path : Randomly use one of all text files located in the entered path instead of one text file. 

  strength : Adjust the strength of the prompt. 

  ban_tag : Prompts to exclude.

  text : Multi-line text as an alternative to text files 

  use_index : Gets text from the line corresponding to index instead of a random line. 

  index : The index of the line. 

  ![image](https://github.com/user-attachments/assets/77bac635-4b6d-4972-a502-02978414adbb)




   
## Export glb

  Export a flat .glb file with a color image, normal map, and alpha mask. 

  You can specify the roughness, metallic, and save path. 

  ![image](https://github.com/user-attachments/assets/202744b2-3b91-4bc0-a4bd-ab8b1a1a8874)


## Remove noise

   guided_first : Apply guided filter first.

   Remove noise from an image. Can be used to clean a normal map. 

   bilateral_loop: The number of times to apply the bilateralFilter. If 0, it is not used. 

   d/sigma_color/sigma_space : bilateralFilter parameters 

   guided_loop: The number of iterations of the guidedFilter. If 0, it is not used. 

   radius/eps: guidedFilter parameters. 

   ![image](https://github.com/user-attachments/assets/814aa202-8150-4c67-bdab-bdc7d012fab5)

   ![image](https://github.com/user-attachments/assets/734448f1-9bf6-43e1-9d82-7e66ac032f94)



## Depth to normal

   Converts a depth image to a normal map. It works very well with 2D images and DepthAnything v2. 

   depth_min : Depths lower than this value are replaced with 0. 

   blue_depth : Adjusts the intensity of the blue channel of the normal map to emphasize depth. The lower this number, the stronger the depth. 

   sobel_ratio : Makes the Normal map more stereoscopically accurate. Values between 0.1 and 0.3 are recommended. 

   ![image](https://github.com/user-attachments/assets/0cf453a7-4d3d-4484-b914-6b45ca66046d)

   ![image](https://github.com/user-attachments/assets/b3c55fab-1d81-4d9f-aafa-c1add3ccf7b4)

   ![image](https://github.com/user-attachments/assets/6ef06f69-52b8-4c31-945d-ad0e11cf474b)



  

### Direct_screenCap

 Captures an image from a specified window or screen.

 capture_mode
  Default : Capture a defined area of the monitor
  window : Captures the area of the window entered in target_window 
  window_crop : Same as window, but cuts off and captures the area relative to that window.

 target_window : Name of the window to capture. You can find its name in the list of windows in the Webcam app. 

![image](https://github.com/user-attachments/assets/3d8ffb75-70f2-45df-879d-188dfc7b7a81)


### Load Webcam Image

 Load an image from a path. 

 To use this node with webcam, you must first run run.bat in `ComfyUI/custom_nodes/ComfyUI_toyxyz_test_nodes/CaptureCam`.

 And in the Webcam app, you'll need to select your webcam and run capture with start.

![image](https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes/assets/8006000/89e829c2-54eb-4965-8a8f-db9f4b73bfd8)



### Capture Webcam

Captures an image directly from the webcam selected with 'select_webcam'. (Usually 0)

This is unstable compared to the Load Webcam Image node. 

If you're using obs, I recommend using the Load Webcam Image node.

![image](https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes/assets/8006000/108baad7-842b-44af-9ed2-f8f6c63ad899)


### Save image to path

This node saves the generated images to a defined path. 

If save_sequence is true, it saves the images in order without overwriting them.

![image](https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes/assets/8006000/71c3cb06-1a7a-42ee-8ebf-59ea59b82562)

### LatentDelay 

![image](https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes/assets/8006000/a4f4f9c2-e8dc-4461-a526-ab08b936a32f)

Set the delay between image generation. 

### ImageResize_Padding

Resizes the image while maintaining its proportions and painting the margins with the color you specify. 

![image](https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes/assets/8006000/ad8d24b1-df8b-4923-8e38-9d92fa16b0c1)


### Webcam app

This script captures the selected webcam and saves it as an image file in real-time. 

You can specify the resolution, format, and path of the image to be saved. 

If you don't enter a path, it will be saved to the default path. 

You can combine a sequence of saved images into a video using the Export button. Make sure to set save_sequence in Save Image to Path to true. 

To use AI Render, you need a Save Image to Path node.

If you entered a location other than the default path in Save Image to Path, you must select a newly created render image in Select Rendered image.

Run_hide_cmd.vbs : Hide the cmd and run the app.

![image](https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes/assets/8006000/b187a874-8fb7-4757-b459-5cc66c793877)


Webcam : List of camera devices connected to your computer

Width: Sets the width of the image. 

Height: Sets the height of the image. 

If either the width or height is zero, it will be automatically adjusted to fit the other values entered. 

FPS : Set how often the capture occurs. If you enter 0, it is unlimited.

Webcam(Checkbox) : Preview the captured image.

Al Render: Preview the generated image in ComfyUi.

Always on top : Webcam, AI Render is always visible on top. To disable it, you need to close the preview window. 

Face detect : Automatically recognize faces and generate masks. It is stored as face_mask.jpg. Use with inpainting.

Keep aspect ratio : Correct the aspect ratio of the image and the capture window. 

Capture Path: The path where the captured image will be saved.

Render image: Path to the image generated by ComfyUI. Required to use AI Render. 

If you don't enter a path, the default path is used. 

Save format: Set the image format to be saved

Overlay alhpa : The alpha value of the overlay image displayed above the region capture window. 

![image](https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes/assets/8006000/406d5656-52d0-41fd-8d1f-1a157d1bf7e1)


Padding: Set how to fill the margins of the image when using Keep aspect ratio.

Export video: Combines the image sequences located in the render image folder into a single video. Enter the desired FPS value.

Clear after export: Deletes the image sequence after the video is exported.

![image](https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes/assets/8006000/fd6fb4bd-659c-40d2-830f-b13883fee865)

Add region window: Creates a window to specify the region to capture. 

The name of the window is the same as entered in Save name. If you enter comma-separated text (e.g., A,B,C), you can use one Region window to capture three images, A, B, and C, alternating between them. 

Save name: Set a name for the captured image. 

Window list: Select the windows to capture. Other windows will not be captured. If set to Disable, it will be captured as it is displayed on the screen. Window capture is often unstable depending on the program. Be careful when using it.

Reload list: Refresh the list of webcams and windows.

The name of the currently captured and saved image file and the selected Region window are displayed in the Webcam/Ai Render preview window.

![image](https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes/assets/8006000/0adbe3f0-8578-4c61-8b2b-225194ea024b)


If you select 'Region Capture' from the Webcam list, it will capture the region of the window added with 'Add Region window'. If you select 'Window Capture', it will capture the entire window selected in the window list.


https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes/assets/8006000/8723e014-caa5-4e16-8c8c-5c5edac6f141
![image](https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes/assets/8006000/9d2a41df-2cd7-426b-94a4-e5f01a32370a)
![image](https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes/assets/8006000/7d4196ba-3577-4d58-b551-c80aaae66e30)
![image](https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes/assets/8006000/aab16eba-fd6e-4045-86f6-920218ba980d)


### Hotkeys :

S: Saves the image displayed in the current AI render to the render folder. Activate (click) the AI render window and then use it.

F : Toggles the Region widow (target window list selected) to full window mode. Select the target Region window and then click Use.

A : Select the Region window currently being captured. Enable the AI render or Webcam preview window, then use it.

C: Change the region window. Select the desired Region window and press the key to switch to that window. When used in the Ai Render or Webcam window, the Region windows are switched in the order they were created.

X: If the Region windows were created separated by commas, pressing the key while Ai Render or Webcam is active will switch the name of the captured image. 

Q: Copy the image displayed in AI render to the clipboard. Activate AI render and then use.

M: Mask paint toggle. Allows you to paint the mask directly in the Webcam preview window. Paint with the left mouse and erase with the right. You can change the brush size with the mouse wheel. The mask is saved as the captured image name + '_mask'.

N: Erase all painted masks.

Z: Pause image capture. Use for Webcam or AI Render.

P: Displays the AI Render image as an overlay on the currently selected Region window. This is unstable and should be used with caution. It can only be enabled/disabled when capture is stopped and requires a target window to be set. Does not run even if there is no render image to load. 

### Render preview

Load the image file saved with the Save image to path node. Pressing 'Q' while the window is active will copy the preview image to the clipboard. 

![image](https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes/assets/8006000/ff2b4d68-cf61-4665-b75d-eff1b65b7606)


### Face detection

Detect faces and create masks.  Use it for inpainting with the Load Webcam Image node.

![image](https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes/assets/8006000/633e46fe-5f04-4f95-9cb1-3be335a63bb3)


### Note

The ControlNet preprocessor slows down the process, so I recommend using other tools to prepare the ControlNet image.

If you want ComfyUI to run continuously, use Auto Queue.

![image](https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes/assets/8006000/eafebe66-5a7e-4bfb-a0e9-cfa06e679813)

For maximum speed, set the VAE to taesd.

![image](https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes/assets/8006000/c14ad91e-f2c5-4bbb-a26b-fdc6583a8c88)


