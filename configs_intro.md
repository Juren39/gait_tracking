# Configuration File Overview
The project uses the `config.yaml` file, located in the `/configs` folder. The main fields and their purposes are as follows:

```yaml
yolo_model: "./weights/yolov8n.pt" # Path to the YOLO model (Path type)

reid_model: "./weights/osnet_x0_25_msmt17.pt" # Path to the ReID model (Path type)

tracking_method: "deepocsort" # Tracking method (supports 'deepocsort', 'botsort', 'ocsort', 'bytetrack', 'imprassoc')

source: "./data/video.mp4" # Input source (file path/directory/URL/webcam)

imgsz: [640, 640] # Inference image size (list of integers for width and height)

conf: 0.5 # Confidence threshold (float)

iou: 0.7 # IoU threshold (float)

device: "cuda:0" # Device to use ('cuda:0', 'cpu', etc.)

show: false # Whether to display tracking video results

classes: [0] # Target classes to filter (list of integers, using YOLO classes, 0 represents humans)

name: "exp" # Name of the experiment for saving results

exist_ok: true # Whether to allow overwriting existing projects

half: false # Whether to use FP16 half-precision inference

vid_stride: 1 # Video frame stride (integer)

show_labels: true # Whether to display labels

show_conf: true # Whether to display confidence scores

show_trajectories: true # Whether to display trajectories

save_id_crops: false # Whether to save each cropped result to the corresponding ID folder

line_width: 2 # Border width (integer or None)

per_class: false # Whether to separate tracking by class

agnostic_nms: false # Whether to use class-agnostic NMS

verbose: false # Whether to output detailed information: false for showing the progress bar, true for detailed information output. Default is false.

videofps: 10 # Video frames, how many frames per second, Default is 10

```

[Go to README](../README.md)