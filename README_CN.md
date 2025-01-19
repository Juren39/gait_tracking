# 项目运行指南

## 项目简介
该项目主要使用 YOLOv8 模型进行目标检测和跟踪，并支持多种跟踪器（如 DeepSORT、OCSORT、ByteTrack 等）的集成。

通过配置文件 `config.yaml` 来管理项目参数，用户可以方便地调整模型路径、运行设置以及跟踪参数。

## 项目运行方式

### 1. 准备环境
先从git上克隆整个仓库
```bash
git clone https://github.com/Juren39/yolo_track.git
cd yolo_track
```
确保已安装必要的依赖项。

```bash
conda create --name yolov8 python=3.10
conda activate yolov8
pip install boxmot
pip install git+https://github.com/mikel-brostrom/ultralytics.git
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

### 4. 输出结果
1. **处理结果**：
   - 处理后的视频会保存在 `output/exp/videos` 文件夹中。
   - 处理后的标签会保存在 `output/exp/labels` 文件夹中。
   - 原视频会被处理成固定格式的mp4文件后保存在 `output/videos` 文件夹中。
2. **日志信息**：
   - 控制台会输出每帧的处理详情。

### 5. 标注辅助工具
   - 在命令行进入该文件（`index.html`）所在目录(`cd dataset`)
   - 运行 `python -m http.server 8000`（或其他任意 HTTP 服务）  
   - 在浏览器中访问 `http://127.0.0.1:8000/index.html` 

本工具提供一个简易的网页界面，能够：
1. 上传视频文件（mp4/avi）与上传 TXT 文件（内部为 JSON 格式  
2. 点击「播放」后，自动启动视频播放与可视化  
3. 支持修改某个目标的 Track ID）  
4. 每隔 10 分钟自动保存当前修改后的内容  
5. 可手动保存（支持输入自定义文件名）  
6. 点击「重新上传」回到初始状态

## 参考
本项目受 BoxMOT 的启发并参考了其内容。 [BoxMOT](https://github.com/username/boxmot).

特别感谢 BoxMOT 的作者对社区的贡献。