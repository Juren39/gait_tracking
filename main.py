import argparse
import cv2
import numpy as np
from functools import partial
from pathlib import Path
import yaml
import json
import torch
import subprocess
from tqdm import tqdm

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

from dataset.data_store import save_id_name_mapping

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
            predictor.custom_args['db_config'],
            predictor.custom_args['mode'],
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
    agnostic_nms = args['agnostic_nms']
    verbose = args['verbose']
    videofps = args['videofps']
    db_config = args['db_config']
    mode = args['mode']

    if mode == "registration":
        source_path = Path(source) / 'videos'
    else:
        source_path = Path(source)
    if source_path.is_dir():
        video_files = []
        for ext in ("*.mp4", "*.avi"):
            video_files.extend(source_path.glob(ext))
    else:
        video_files = [source_path]

    registered_ids = set()  
    id_info_list = []  # 详细信息

    for video_path in video_files:

        source_file_name = video_path.stem
        video_output_path = Path('./output') / name / 'visualization' / f'{source_file_name}_tracked.mp4'
        label_output_path = Path('./output') / name / 'labels' / f'{source_file_name}_labels.txt'
        origin_to_mp4_path = Path('./output') / name / 'videos' / f'{source_file_name}.mp4'
        id_name_out_path = Path('./dataset/id_name_mapping.txt')
        id_name_out_path.parent.mkdir(parents=True, exist_ok=True)
        video_output_path.parent.mkdir(parents=True, exist_ok=True)
        label_output_path.parent.mkdir(parents=True, exist_ok=True)
        origin_to_mp4_path.parent.mkdir(parents=True, exist_ok=True)

        convert_to_mp4(video_path, origin_to_mp4_path, videofps)

        vid = cv2.VideoCapture(origin_to_mp4_path)
        frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        vid.release()
        
        if mode == "registration":
            tracking_config = TRACKER_CONFIGS / (args['tracking_method'] + '.yaml')
            tracker = create_tracker(
                tracker_type=args['tracking_method'],
                tracker_config=tracking_config,  
                reid_weights=args['reid_model'],
                device=device,
                half=args['half'],
                db_config=db_config,
                mode=mode,
                per_class=args['per_class'],
            )
            cap = cv2.VideoCapture(str(origin_to_mp4_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = 0

            progress_bar = (
                tqdm(total=total_frames, desc="Processing frames", unit="frame", dynamic_ncols=True, leave=True)
                if ~verbose else None
            )

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                dets, box_id = load_json_bboxes(Path(source) / 'labels' / f'{source_file_name}_labels.txt', frame_idx)  
                # 假设返回 (N,6): [x1,y1,x2,y2,score,cls]

                # 调用 tracker.update(dets, frame)
                tracks = tracker.update(dets, frame, box_id)  # DeepOcSort
                # 统计出现的 ID
                for track in tracks:
                    track_id = track[4]  # 获取当前 track_id
                    if track_id not in registered_ids:
                        registered_ids.add(track_id)
                        id_info_list.append({
                            "track_id": track_id,
                            "frame_idx": frame_idx
                        })
                progress_bar.update(1)
                frame_idx += 1
            
            cap.release()
            progress_bar.close()

        else:
            if imgsz is None:
                imgsz = default_imgsz(yolo_model)

            yolo = YOLO(
                yolo_model if is_ultralytics_model(yolo_model)
                else 'yolov8n.pt',
            )

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
                exist_ok=exist_ok,
                name=name,
                classes=classes,
                imgsz=imgsz,       
                vid_stride=vid_stride,
                line_width=line_width,
                verbose=verbose
            )
            frame_idx = 0  # 记录帧编号 

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
            progress_bar = (
                tqdm(total=total_frames, desc="Processing frames", unit="frame", dynamic_ncols=True, leave=True)
                if ~verbose else None
            )
            for r in results:
                tracks_info = []
                img = yolo.predictor.trackers[0].plot_results(r.orig_img, show_trajectories)
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
                if ~verbose:
                    num_people = len(tracks_info)
                    progress_bar.set_postfix({"Frame": frame_idx, "People Detected": num_people})
                    progress_bar.update(1)
                    
                frame_idx += 1
                out.write(img)
            progress_bar.close()
            out.release()
            
            with open(label_output_path, 'w', encoding='utf-8') as f:
                json.dump(labels, f, ensure_ascii=False, indent=4)
            print(f"video file saved to: {video_output_path.resolve()}")
            print(f"label file saved to: {label_output_path.resolve()}")
    
    if mode == 'registration':
        save_id_name_mapping(db_config, id_name_out_path)
        print(f"📌 Registration 模式完成，共注册 {len(registered_ids)} 个 ID")
        print(f"🏷️ 相关映射关系已经打印至: {id_name_out_path}")
        print("📋 详细信息:")
        for info in id_info_list:
            print(f"🔹 First Appear Frame :{info['frame_idx']}, ID :{info['track_id']}")
    else:
        print(f"📌 Recognition 模式完成，共检测 {len(video_files)} 个视频")
    
def convert_to_mp4(input_path, output_path, videofps):

    command = [
        "ffmpeg",
        '-n',
        "-i", input_path,           # 输入视频
        "-c:v", "libx264",          # 视频编码器
        "-preset", "fast",          # 压缩速度/质量平衡
        "-crf", "23",               # 视频质量（数值越小越清晰）
        "-c:a", "aac",              # 音频编码器
        "-b:a", "128k",             # 音频比特率
        "-movflags", "+faststart",  # 优化文件头
        "-r", videofps,
        output_path
    ]
    dataset_path = Path("./dataset/ffmpeg_log.txt")
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with open("./dataset/ffmpeg_log.txt", "w") as log_file:
        subprocess.run(command, stdout=log_file, stderr=log_file)

def load_json_bboxes(json_path: str, frame_idx: int) -> np.ndarray:
    """
    从上述 JSON 文件中读取指定 frame_idx 的检测框信息，返回形如 (N, 6) 的 array:
      [x1, y1, x2, y2, score, cls]
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    frames = data['frames']
    cur_frame_data = None
    for fdata in frames:
        if fdata['frame_id'] == frame_idx:
            cur_frame_data = fdata
            break
    if cur_frame_data is None:
        return np.empty((0, 6))
    
    # 提取 detections
    detections = cur_frame_data.get('detections', [])
    bboxes = []
    box_id = []
    for det in detections:
        # 解析
        confidence = float(det['confidence'])
        cls = float(det['class_id'])  # 如果 class_id 全是 int，可以转成 float32
        x1, y1, x2, y2 = det['bbox']  # bbox是list: [x1, y1, x2, y2]
        bboxes.append([x1, y1, x2, y2, confidence, cls])
        box_id.append(det['track_id'])

    if len(bboxes) == 0:
        return np.empty((0, 6), dtype=np.float32), box_id
    
    return np.array(bboxes, dtype=np.float32), box_id

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
    with open('./config.yaml', 'r') as f:
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
    config['agnostic_nms'] = bool(config['agnostic_nms'])  
    config['verbose'] = bool(config['verbose'])
    config['videofps'] = str(config['videofps'])
    config['db_config'] = dict(config['db_config'])
    config['mode'] = str(config['mode'])

    return config

if __name__ == "__main__":
    config = parse_opt()
    run(config)