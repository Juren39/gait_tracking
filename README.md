# Project Running Guide.  
[CN](./README_CN.md)

## Project Overview
This project primarily utilizes the YOLOv8 model for object detection and tracking, supporting the integration of various trackers such as DeepSORT, OCSORT, ByteTrack, and more.

The configuration file `config.yaml` is used to manage project parameters, allowing users to conveniently adjust model paths, runtime settings, and tracking parameters.

## How to Run the Project

### 1. Prepare the Environment
First, clone the repository from GitHub:
```bash
git clone https://github.com/Juren39/yolo_track.git
cd yolo_track
```
Ensure the required dependencies are installed:
```bash
conda create --name yolov8 python=3.10
conda activate yolov8
pip install boxmot
pip install git+https://github.com/mikel-brostrom/ultralytics.git
```

### 2. Edit the Configuration File
Modify the `config.yaml` file based on your needs.[Go to config intro](./configs/configs_intro.md)

Example: Change the YOLO model path and input video path:
```yaml
yolo_model: "./weights/yolov8s.pt"
source: "./data/demo.mp4"
```
To change the video being tracked, update the `source` path to the desired input video path.

### 3. Run the Project
Run the main program directly from the root folder:
```bash
python main.py
```
The project will automatically read the parameters from `config.yaml` and start processing.

### 4. Output Results
1. **Processing Results**:
   - The processed video will be saved in the `output/exp/videos` folder.
   - The processed labels will be saved in the `output/exp/labels` folder.
2. **Log Information**:
   - The console will display detailed information for each processed frame.

### 5. Annotation Assistance Tool
   - ```bash
      cd dataset
      python -m http.server 8000
      ```
      you can use 8000 or other port
   - Open your browser and visit `http://127.0.0.1:8000/index.html`

This tool provides a simple web interface that allows:
1. Uploading video files (mp4/avi) and corresponding TXT files (in JSON format internally)
2. Starting video playback and visualization by clicking "Play"
3. Modifying the Track ID of a specific object
4. Automatically saving the modified content every 10 minutes
5. Manually saving (supports custom file names)
6. Resetting to the initial state by clicking "Reupload"

## References
This project is inspired by and references content from BoxMOT. [BoxMOT](https://github.com/username/boxmot).

Special thanks to the author of BoxMOT for their contributions to the community.
