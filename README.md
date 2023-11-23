# ComfyUI_toyxyz_test_nodes

This node was created to send a webcam to ComfyUI in real time. 

This node is recommended for use with LCM. 

https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes/assets/8006000/8536e96a-514a-48b2-b1aa-8eccbd3fa853

(This video is at 4x speed)

Update 

2023/11/24 - AddSave image to path node. Add Render preview, Add export video

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

### 1. Load Webcam Image

 Load an image from a path. 

 To use this node with webcam, you must first run run.bat in `ComfyUI/custom_nodes/ComfyUI_toyxyz_test_nodes/CaptureCam`.

 And in the Webcam app, you'll need to select your webcam and run capture with start.

![image](https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes/assets/8006000/89e829c2-54eb-4965-8a8f-db9f4b73bfd8)



### 2. Capture Webcam

Captures an image directly from the webcam selected with 'select_webcam'.

This is very slow compared to the Load Webcam Image node. 

If you're using LCM, I recommend using the Load Webcam Image node.

![image](https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes/assets/8006000/108baad7-842b-44af-9ed2-f8f6c63ad899)


### 3. Save image to path

This node saves the generated images to a defined path. 

If save_sequence is true, it saves the images in order without overwriting them.

![image](https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes/assets/8006000/71c3cb06-1a7a-42ee-8ebf-59ea59b82562)


### 4. Webcam app

This script captures the selected webcam and saves it as an image file in real-time. 

You can specify the resolution, format, and path of the image to be saved. 

If you don't enter a path, it will be saved to the default path. 

If either the width or height is zero, it will be automatically adjusted to fit the other values entered. 

You can combine a sequence of saved images into a video using the Export button. Make sure to set save_sequence in Save Image to Path to true. 

To use AI Render, you need a Save Image to Path node.

If you entered a location other than the default path in Save Image to Path, you must select a newly created render image in Select Rendered image.

![image](https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes/assets/8006000/de13b447-e5c1-435c-8e0b-86b79d218c46)


https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes/assets/8006000/8723e014-caa5-4e16-8c8c-5c5edac6f141


### 5. AI Render preview

Load the image file saved with the Save image to path node. Pressing 'Q' while the window is active will copy the preview image to the clipboard. 

![image](https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes/assets/8006000/ff2b4d68-cf61-4665-b75d-eff1b65b7606)


### Note

The ControlNet preprocessor slows down the process, so I recommend using other tools to prepare the ControlNet image.

If you want ComfyUI to run continuously, use Auto Queue.

![image](https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes/assets/8006000/eafebe66-5a7e-4bfb-a0e9-cfa06e679813)

