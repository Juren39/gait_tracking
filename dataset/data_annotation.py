from flask import Flask, request, jsonify
import torch
from sam2.sam2.build_sam import build_sam2_video_predictor

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

app = Flask(__name__)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)