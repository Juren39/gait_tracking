import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image

class GaitRecognitionBackend:
    """
    步态识别模型封装类
    Args:
        weights (str): 模型权重文件路径
        device (torch.device): 模型运行设备，如 'cpu' 或 'cuda'
        half (bool): 是否使用半精度推理
    """
    def __init__(self, weights: str, device: torch.device, half: bool = False):
        self.device = device
        self.half = half
        # 构建步态识别模型
        self.model = GaitRecognitionModel()
        self.model.to(device)
        if half:
            self.model.half()
        # 如果提供权重，则加载
        if weights is not None:
            self.load_weights(weights)
        self.model.eval()
        
        # 定义图像预处理流程，这里假设步态模型要求的输入尺寸为 (128, 64)
        self.preprocess_transform = transforms.Compose([
            transforms.Resize((128, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def load_weights(self, weights_path: str):
        """
        加载预训练模型权重
        """
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
    
    def get_features(self, bboxes: np.ndarray, img: np.ndarray) -> np.ndarray:
        """
        根据检测框从原图中提取步态特征
        Args:
            bboxes: numpy数组，形状为 (N, 4)，每行格式为 [x1, y1, x2, y2]
            img: numpy数组，形状为 (H, W, 3)，原图（RGB格式）
        Returns:
            features: numpy数组，形状为 (N, num_features)
        """
        crops = []
        # 如果图像不是uint8格式，则转换
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        for bbox in bboxes:
            # 将检测框的坐标转换为整数
            x1, y1, x2, y2 = bbox.astype(int)
            # 裁剪目标区域
            crop = img[y1:y2, x1:x2, :]
            if crop.size == 0:
                continue
            # 将裁剪区域转为 PIL Image，以便使用 torchvision 的预处理
            crop_img = Image.fromarray(crop)
            crop_tensor = self.preprocess_transform(crop_img)
            crops.append(crop_tensor)
        
        if len(crops) == 0:
            # 如果没有有效的目标，返回空数组
            return np.empty((0, 512))
        
        # 拼接成批处理数据，并送入模型
        crops_tensor = torch.stack(crops, dim=0).to(self.device)
        if self.half:
            crops_tensor = crops_tensor.half()
        with torch.no_grad():
            features = self.model(crops_tensor)
        # 将特征转回 CPU 并转换为 numpy 数组
        features = features.cpu().numpy()
        return features