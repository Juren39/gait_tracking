import argparse
import cv2
import numpy as np
from functools import partial
from pathlib import Path
import yaml
import json
import torch
import subprocess

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS
from boxmot.utils.checks import RequirementsChecker
from tracking.detectors import (get_yolo_inferer, default_imgsz,
                                is_ultralytics_model, is_yolox_model)

checker = RequirementsChecker()
checker.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.data.utils import VID_FORMATS
from ultralytics.utils.plotting import save_one_box

from dataset.data_store import init_database

def on_predict_start(predictor, persist=False):
    assert predictor.custom_args['tracking_method'] in TRACKERS, \
        f"'{predictor.custom_args['tracking_method']}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = TRACKER_CONFIGS / (predictor.custom_args['tracking_method'] + '.yaml')
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.custom_args['tracking_method'],
            tracking_config,
            predictor.custom_args['reid_model'],
            predictor.device,
            predictor.custom_args['half'],
            predictor.custom_args['per_class']
        )
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers

@torch.no_grad()
def run(args):
    yolo_model = args['yolo_model']
    reid_model = args['reid_model']
    tracking_method = args['tracking_method']
    source = args['source']
    imgsz = args['imgsz'] or default_imgsz(yolo_model)
    conf = args['conf']
    iou = args['iou']
    device = args['device']
    show = args['show']
    classes = args['classes']
    name = args['name']
    exist_ok = args['exist_ok']
    half = args['half']
    vid_stride = args['vid_stride']
    show_labels = args['show_labels']
    show_conf = args['show_conf']
    show_trajectories = args['show_trajectories']
    save_id_crops = args['save_id_crops']
    line_width = args['line_width']
    per_class = args['per_class']
    verbose = args['verbose']
    agnostic_nms = args['agnostic_nms']

    if imgsz is None:
        imgsz = default_imgsz(yolo_model)

    yolo = YOLO(
        yolo_model if is_ultralytics_model(yolo_model)
        else 'yolov8n.pt',
    )

    source_file_name = Path(source).stem
    # origin video file to mp4
    origin_to_mp4_path = Path('./output') / 'videos' / f'{source_file_name}.mp4'
    convert_to_mp4(source, origin_to_mp4_path)

    results = yolo.track(
        source=origin_to_mp4_path,
        conf=conf,
        iou=iou,
        agnostic_nms=agnostic_nms,
        show=False,
        stream=True,
        device=device,
        show_conf=show_conf,
        show_labels=show_labels,
        verbose=verbose,
        exist_ok=exist_ok,
        name=name,
        classes=classes,
        imgsz=imgsz,       
        vid_stride=vid_stride,
        line_width=line_width
    )
    # 获取视频属性
    vid = cv2.VideoCapture(origin_to_mp4_path)
    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    vid.release()

    video_output_path = Path('./output') / name / 'videos' / f'{source_file_name}_tracked.mp4'
    label_output_path = Path('./output') / name / 'labels' / f'{source_file_name}_labels.txt'
    video_output_path.parent.mkdir(parents=True, exist_ok=True)
    label_output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_output_path), fourcc, fps, (frame_width, frame_height))
    yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))

    if not is_ultralytics_model(yolo_model):
        # replace yolov8 model
        m = get_yolo_inferer(yolo_model)
        yolo_model = m(model=yolo_model, device=yolo.predictor.device,
                       args=yolo.predictor.args)
        yolo.predictor.model = yolo_model

        # If current model is YOLOX, change the preprocess and postprocess
        if is_yolox_model(yolo_model):
            # add callback to save image paths for further processing
            yolo.add_callback("on_predict_batch_start",
                              lambda p: yolo_model.update_im_paths(p))
            yolo.predictor.preprocess = (
                lambda imgs: yolo_model.preprocess(im=imgs))
            yolo.predictor.postprocess = (
                lambda preds, im, im0s:
                yolo_model.postprocess(preds=preds, im=im, im0s=im0s))

    yolo.predictor.custom_args = args
    labels = {
        "video_name": source_file_name,
        "fps": fps,
        "width": frame_width,
        "height": frame_height,
        "frames": []
    }
    frame_idx = 0  # 记录帧编号 
    for r in results:
        tracks_info = [] = []
        img = yolo.predictor.trackers[0].plot_results(r.orig_img, show_trajectories)
        # detections = r.boxes
        # dets = []
        # for box in detections:
        #     x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 获取检测框坐标
        #     conf = box.conf[0].cpu().item()            # 置信度
        #     cls = box.cls[0].cpu().item()              # 类别
        #     dets.append([x1, y1, x2, y2, conf, cls])
        # dets = np.array(dets)
        # if len(dets) == 0:
        #     tracks = []
        # else:
        #     tracks = yolo.predictor.trackers[0].update(dets, r.orig_img)
        # draw_tracking_results(r.orig_img, tracks)
        # img = r.orig_img  # Pass the original image without annotations
        # # 将跟踪后的目标信息写入 frame_detections
        # for track in tracks:
        #     x1, y1, x2, y2, track_id = track[:5]
        #     # 如需添加类别、置信度等，需要结合 dets 或额外返回，示例仅记录 track_id 和 bbox
        #     frame_detections.append({
        #         "track_id": int(track_id),
        #         "bbox": [float(x1), float(y1), float(x2), float(y2)]
        #     })
        # 该绘制方法存在问题，会导致只能追踪到一个单位，先暂时注释，准备用来给strongsort进行绘制用
        for box in r.boxes:
            # 坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf_val = float(box.conf[0].cpu().item())
            cls_val = int(box.cls[0].cpu().item())
            # track_id 可能为空，需要做判断
            if box.id is not None:
                track_id = int(box.id[0].item())
            else:
                track_id = -1    
            tracks_info.append({
                "track_id": track_id,
                "class_id": cls_val,
                "confidence": conf_val,
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            })
        # 将该帧的信息存入 labels
        labels["frames"].append({
            "frame_id": frame_idx,
            "detections": tracks_info
        })
        frame_idx += 1
        out.write(img)
    out.release()
    with open(label_output_path, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=4)

    print(f"video file saved to: {video_output_path.resolve()}")
    print(f"label file saved to: {label_output_path.resolve()}")

def convert_to_mp4(input_path, output_path, fps=None):

    command = [
        "ffmpeg",
        "-i", input_path,           # 输入视频
        "-c:v", "libx264",          # 视频编码器
        "-preset", "fast",          # 压缩速度/质量平衡
        "-crf", "23",               # 视频质量（数值越小越清晰）
        "-c:a", "aac",              # 音频编码器
        "-b:a", "128k",             # 音频比特率
        "-movflags", "+faststart",  # 优化文件头
        output_path
    ]
    subprocess.run(command, check=True)


def draw_tracking_results(image, tracks):
    for track in tracks:
        x1, y1, x2, y2, track_id = track[:5]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f"ID: {int(track_id)}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
def parse_opt():
    """
    从 YAML 文件读取配置参数。
    """
    with open('./configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config['yolo_model'] = Path(config['yolo_model'])  
    config['reid_model'] = Path(config['reid_model'])  
    config['tracking_method'] = str(config['tracking_method'])  
    config['source'] = str(config['source'])  
    config['imgsz'] = list(map(int, config['imgsz'])) if isinstance(config['imgsz'], list) else config['imgsz']  
    config['conf'] = float(config['conf']) 
    config['iou'] = float(config['iou']) 
    config['device'] = str(config['device'])  
    config['show'] = bool(config['show'])  
    config['classes'] = list(map(int, config['classes'])) if config['classes'] is not None else None 
    config['name'] = str(config['name']) 
    config['exist_ok'] = bool(config['exist_ok'])  
    config['half'] = bool(config['half'])  
    config['vid_stride'] = int(config['vid_stride']) 
    config['show_labels'] = bool(config['show_labels'])  
    config['show_conf'] = bool(config['show_conf'])  
    config['show_trajectories'] = bool(config['show_trajectories'])  
    config['save_id_crops'] = bool(config['save_id_crops'])  
    config['line_width'] = int(config['line_width']) if config['line_width'] is not None else None  
    config['per_class'] = bool(config['per_class'])  
    config['verbose'] = bool(config['verbose'])  
    config['agnostic_nms'] = bool(config['agnostic_nms'])  

    return config

if __name__ == "__main__":
    config = parse_opt()
    conn = init_database("./dataset/data/tracking.db")
    run(config)