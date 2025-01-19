# 项目运行指南

## 项目简介
该项目主要使用 YOLOv8 模型进行目标检测和跟踪，并支持多种跟踪器（如 DeepSORT、OCSORT、ByteTrack 等）的集成。

通过配置文件 `config.yaml` 来管理项目参数，用户可以方便地调整模型路径、运行设置以及跟踪参数。

---
## 主文件

### 配置文件介绍
项目中使用的配置文件为 `config.yaml`，该文件位于 `/configs` 文件夹中，其主要字段和用途如下：

```yaml
yolo_model: "./weights/yolov8n.pt" # YOLO 模型路径 (Path 类型)

reid_model: "./weights/osnet_x0_25_msmt17.pt" # ReID 模型路径 (Path 类型)

tracking_method: "deepocsort" # 跟踪方法 (支持 'deepocsort', 'botsort', 'ocsort', 'bytetrack', 'imprassoc')

source: "./data/video.mp4" # 输入源 (文件路径/目录/URL/摄像头)

imgsz: [640, 640] # 推理图像大小 (整数列表，表示宽和高)

conf: 0.5 # 置信度阈值 (浮点数)

iou: 0.7 # IoU 阈值 (浮点数)

device: "cuda:0" # 使用的设备 ('cuda:0', 'cpu' 等)

show: false # 是否显示追踪视频结果

classes: [0] # 筛选目标类别 (整数列表，采用yolo类别，0代表人类)

name: "exp" # 保存结果的实验名称

exist_ok: true # 是否允许覆盖已有项目

half: false # 是否使用 FP16 半精度推理

vid_stride: 1 # 视频帧步长 (整数)

show_labels: true # 是否显示标签

show_conf: true # 是否显示置信度

show_trajectories: true # 是否显示轨迹

save_id_crops: false # 是否保存每个裁剪结果到对应 ID 的文件夹

line_width: 2 # 边框宽度 (整数或 None)

per_class: false # 是否按类别区分跟踪

verbose: true # 是否显示每帧详细信息

agnostic_nms: false # 是否使用类别无关的 NMS
```

---

## 项目运行方式

### 1. 准备环境
确保已安装必要的依赖项。

```bash
conda create --name yolov8 python=3.10
conda activate yolov8
pip install poetry
poetry install --with yolo —no-root
pip uninstall torch torchvision torchaudio -y
pip install torch==2.2.1+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 配置文件编辑
根据需求修改 `config.yaml` 文件。
示例：修改 YOLO 模型路径和输入视频路径。
```yaml
yolo_model: "./weights/yolov8s.pt"
source: "./data/demo.mp4"
```
要更改追踪的视频请将source路径改为输入视频路径。
### 3. 运行项目
直接在主文件夹中运行主程序：
```bash
python main.py
```
项目会自动读取 `config.yaml` 中的参数并开始处理。
---
### 注意事项
1. 确保所有路径（如 `yolo_model`, `reid_model`, `source` 等）在配置文件中正确。
2. 如果使用 GPU，请确保安装了正确版本的 CUDA 和 cuDNN，并且 PyTorch 与之兼容。
3. 如果出现视频无法解码的问题，可以使用 `ffmpeg` 转换为兼容格式（如 H.264）：
   ```bash
   ffmpeg -i input.mp4 -c:v libx264 -preset fast -crf 23 output.mp4
   ```
4. 支持的跟踪方法包括：`deepocsort`, `botsort`, `strongsort`, `ocsort`, `bytetrack`, `imprassoc`。

---

### 输出结果
1. **处理结果**：
   - 处理后的视频会保存在 `output/exp/videos` 文件夹中。
   - 处理后的视频会保存在 `output/exp/labels` 文件夹中。
2. **日志信息**：
   - 控制台会输出每帧的处理详情。

## 标注辅助工具

本工具提供一个简易的网页界面，能够：

1. **上传视频文件（mp4/avi）** 与 **上传 TXT 文件（内部为 JSON 格式）**  
2. 点击「播放」后，**自动启动视频播放与可视化**（在 Canvas 上绘制检测/跟踪框和 ID）  
3. **支持修改某个目标的 Track ID**（一处修改，全局生效）  
4. **每隔 10 分钟自动保存**当前修改后的内容  
5. **可手动保存**（支持输入自定义文件名）  
6. 点击「重新上传」回到初始状态

> **注意**：由于浏览器安全限制，**自动保存**和**手动保存**均以“下载文件”的方式出现。文件名中若带子目录（如 `auto_saved/`）可能不会真正创建本地文件夹，而是取决于浏览器的下载行为。

---

### 功能概述

- **拖拽/点击上传**  
  - 视频拖拽区：只允许 `.mp4`, `.avi`  
  - TXT 文件拖拽区：只允许 `.txt`（**文件内部实际为 JSON 格式**）  
- **三种状态**  
  1. `initial`：初始状态，尚未满足播放条件  
  2. `playable`：已上传视频和 TXT，允许点击「播放」  
  3. `playing`：点击「播放」后进入，禁用「播放」按钮，同时开始自动保存  
- **可视化**  
  - 在 Canvas 上以红色矩形和 ID 文字显示当前帧的检测结果  
  - 可修改 ID，点击「更新」后全局替换该 ID  
- **自动保存**  
  - 默认每 10 分钟一次，将修改后的 JSON 下载保存  
  - 文件名示例：`auto_saved/原文件名_1.json`, `auto_saved/原文件名_2.json` 等  
- **手动保存**  
  - 点击「保存JSON」，可在对话框中输入想要的保存文件名（默认基于 TXT 原文件名）  
  - 同样以下载形式保存  
- **重新上传**  
  - 回到初始状态并停止自动保存

---

### 使用方法

1. **本地启动服务器**（推荐）  
   - 在命令行进入该文件（`index.html`）所在目录  
   - 运行 `python -m http.server 8000`（或其他任意 HTTP 服务）  
   - 在浏览器中访问 `http://127.0.0.1:8000/index.html`  

2. **上传文件**  
   - 打开页面后，你会看到「上传视频文件 (mp4/avi)」和「上传 TXT 文件」两个拖拽/点击区域  
   - 分别拖拽（或点击选择）一个 `.mp4/.avi` 和 `.txt` 文件（请确认 `.txt` 文件中是合法的 JSON 格式）

3. **点击「播放」**  
   - 当两个文件都上传后，「播放」按钮会变得可点（`playable` 状态）  
   - 点击后进入 `playing` 状态，禁用「播放」按钮并启动自动保存  
   - 视频将在 `<video>` 中播放，并在上方 Canvas 同步绘制 ID 框

4. **修改 ID**  
   - 页面下方的「修正区」会显示当前帧所有目标的 Track ID  
   - 可在输入框中更改 ID 并点击「更新」按钮  
   - 更新后会全局替换该 ID，画面下次刷新时即可看到新 ID

5. **自动保存**  
   - 每 10 分钟浏览器会自动触发一次下载，文件名如 `auto_saved/xxx_1.json`  
   - 不同浏览器可能不会真正创建 `auto_saved` 文件夹，而是把该名字作为文件名一部分

6. **手动保存**  
   - 点击「保存JSON」可弹出提示，让你输入保存文件名  
   - 默认名形如 `xxx_updated.json`；你可修改为任意名称后下载

7. **重新上传**  
   - 点击「重新上传」按钮后，会终止自动保存并回到初始状态（`initial`），清空视频/Canvas/ID 修正信息  
   - 可再次上传新的文件

---

### 注意事项

1. **TXT 文件必须是合法 JSON**  
   - 如果上传的 `.txt` 文件内容不是 JSON 格式，脚本会在解析时报错  
2. **自动/手动保存**  
   - 均以浏览器下载方式进行  
   - 不能真正写入本地特定目录，如需此功能需配合后端或使用浏览器的高级 API  
3. **大文件性能**  
   - 如果视频分辨率很大，浏览器实时绘制会较吃性能，可考虑修改脚本做降分辨率或跳帧  

## 参考
本项目受 BoxMOT 的启发并参考了其内容。 [BoxMOT](https://github.com/username/boxmot).

特别感谢 BoxMOT 的作者对社区的贡献。