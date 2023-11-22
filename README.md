# ComfyUI_toyxyz_test_nodes

This node was created to send a webcam to ComfyUI in real time. 

This node is recommended for use with LCM. 

https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes/assets/8006000/8536e96a-514a-48b2-b1aa-8eccbd3fa853


## Installation

1. Git clone this repo to the ComfyUI/custom_nodes path.

   `git clone https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes`

2. Run setup.bat in `ComfyUI/custom_nodes/ComfyUI_toyxyz_test_nodes/CaptureCam`


## Usage
![workflow (36)](https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes/assets/8006000/7a8644d2-59f9-4ed5-a32a-82c75cdb0997)
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


### 3. Webcam app

This script captures the selected webcam and saves it as an image file in real-time. 

You can specify the resolution, format, and path of the image to be saved. 

If you don't enter a path, it will be saved to the default path. 

If either the width or height is zero, it will be automatically adjusted to fit the other values entered. 

![image](https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes/assets/8006000/75f536e9-c3b1-4640-aca3-be433c96612e)

### Note

The ControlNet preprocessor slows down the process, so I recommend using other tools to prepare the ControlNet image.


