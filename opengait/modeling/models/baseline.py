import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
from pathlib import Path
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks
from opengait.utils import get_valid_args, is_list, is_dict, np2var, ts2np, list2var, get_attr_from, align_human_vertical
from .. import backbones
from einops import rearrange

class Baseline(nn.Module):
    def __init__(self, model_cfg: dict):
        super().__init__()
        self.build_network(model_cfg)
        
    def build_network(self, model_cfg):
        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])
        self.pretrained_gaitbase = model_cfg["pretrained_gaitbase"]
        self.init_GaitBase()

    def init_GaitBase(self):
        load_dict = torch.load(self.pretrained_gaitbase, map_location=torch.device("cpu"))['model']
        msg = self.load_state_dict(load_dict, strict=True)

    def get_backbone(self, backbone_cfg):
        """Get the backbone of the model."""
        if is_dict(backbone_cfg):
            Backbone = get_attr_from([backbones], backbone_cfg['type'])
            valid_args = get_valid_args(Backbone, backbone_cfg, ['type'])
            return Backbone(**valid_args)
        if is_list(backbone_cfg):
            Backbone = nn.ModuleList([self.get_backbone(cfg)
                                      for cfg in backbone_cfg])
            return Backbone
        raise ValueError(
            "Error type for -Backbone-Cfg-, supported: (A list of) dict.")
    
    def input_pretreament(self, seq_tensors, device):
        with torch.no_grad():
            model = YOLO("yolov8n-seg.pt")
            results = model(seq_tensors, verbose=False)
        all_silhouettes = []
        for i, result in enumerate(results):
            if not hasattr(result, 'masks') or result.masks is None:
                # 如果这一帧没有任何分割结果
                silhouette = torch.zeros((128, 128), dtype=torch.float32).to(device)
                all_silhouettes.append(silhouette)
                continue

            boxes = result.boxes  # 各目标的检测框信息 (包括类别 cls, 置信度 conf 等)
            masks = result.masks  # 分割信息: result.masks.data shape [num_objs, H, W]
            mask_data = masks.data  # (num_objs, img_size, img_size)

            # 根据类别筛选
            person_mask_list = []
            for box_idx, box in enumerate(boxes):
                cls_id = int(box.cls.item())  # 当前目标的类别ID
                if cls_id == 0:
                    # 取出对应的mask
                    m = mask_data[box_idx]  # shape [img_size, img_size]
                    person_mask_list.append(m)

            # 合并多个mask (如果检测到多个同类目标，做"或"操作)
            if len(person_mask_list) > 0:
                combined_mask = torch.sum(torch.stack(person_mask_list), dim=0) > 0
                silhouette = combined_mask.float().to(device)
            else:
                silhouette = torch.zeros((128, 128), dtype=torch.float32).to(device)

        all_silhouettes.append(silhouette)
        silhouettes_tensor = torch.stack(all_silhouettes, dim=0)
        return silhouettes_tensor

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        sils = ipts[0]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
        else:
            sils = rearrange(sils, 'n s c h w -> n c s h w')

        del ipts
        outs = self.Backbone(sils)  # [n, c, s, h, w]

        # Temporal Pooling, TP
        outs = self.TP(outs, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]
        embed = embed_1

        batch_size = embed.size(0)
        embed_flat = embed.view(batch_size, -1)  # [n, embedding_dim]
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs},
            },
            'visual_summary': {
                'image/sils': rearrange(sils, 'n c s h w -> (n s) c h w')
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval
    
        # for i in range(sils[0].shape[0]):
        #     frame = sils[0][i]  # shape: [128, 88]
        #     # 如果在 GPU 上，先移回 CPU
        #     frame = frame.cpu().numpy()  
        #     plt.figure(figsize=(18, 6))
        #     plt.subplot(1, 3, 1)
        #     save_path = Path('/home/jsj/gait_tracking/dataset/images')
        #     plt.imshow(frame, cmap='gray')
        #     plt.axis('off')
        #     plt.savefig(save_path, bbox_inches='tight')
        #     print(f"Figure saved to {save_path}")