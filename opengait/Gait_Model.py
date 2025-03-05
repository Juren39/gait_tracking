import os
import torch
import torch.nn as nn
from typing import Union, Dict, Any
from torchvision import transforms
from collections import deque
import cv2
import numpy as np

from opengait.modeling import models
from opengait.utils import config_loader



class GaitModel():
    def __init__(
        self,
        cfgs_path: str
    ) -> None:
        """
        Initializes the engine with arguments (instead of using argparse).

        Args:
            local_rank (int): Local rank for distributed training.
            cfgs_path (str): Path to the YAML config file.
            phase (str): 'train' or 'test'.
            log_to_file (bool): Whether to log output to file.
            iteration (int): Checkpoint iteration to restore; 0 means none.
        """
        self.cfgs: Dict[str, Any] = config_loader(cfgs_path)

    def loadModel(self):
        model_cfg = self.cfgs['model_cfg']
        Model = getattr(models, model_cfg['model'])
        model = Model(model_cfg)
        return model
    
    def imgs_resize_input(self, img_deque, labs=None, ty=None, vi=None, seqL=None):
        """
        将 deque 中的图像统一 resize 到 (128,128)，转换为指定格式，并与 
        labs, ty, vi, seqL 一起打包成 (ipts, labs, ty, vi, seqL)。

        参数：
        ----------
        img_deque:  存放图像的 deque，长度应当是 n*s
        n:          batch 大小（或 sequence 个数）
        s:          每个 batch/sequence 包含的帧数

        返回：
        ----------
        inputs = (ipts, labs, ty, vi, seqL)

        其中 ipts = (sils, ratios)
        - sils   : shape: [n, s, c, 128, 128]
        - ratios : shape: [n, s]
          每个元素是原始图像的宽高比 (w/h)
        """
        n = 1
        s = len(img_deque)
        images_list = list(img_deque)
        sils_list = []
        ratios_list = []
        seq_tensors = []
        seq_ratios = []
        for idx in range(s):
            img = images_list[idx]
            w, h = img.shape[:2]
            img = cv2.resize(img, (128, 128))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ratio = w / float(h)
            img_tensor = img.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_tensor).permute(2,0,1)
            seq_tensors.append(img_tensor)
            seq_ratios.append(ratio)
        seq_tensors = torch.stack(seq_tensors, dim=0) # [s, C, 128, 128]
        sils_list.append(seq_tensors) # [1, s, C, 128, 128]
        ratios_list.append(seq_ratios)
        sils = torch.stack(sils_list, dim=0)
        ratios = torch.tensor(ratios_list, dtype=torch.float32)
        ipts = (sils, ratios)
        return ipts
    
        ####### 输入 sils 为某个轨迹一系列的检测框图像
    def extract_gait_feature(self, sils):
        gaitmodel = self.loadModel()
        gaitmodel.requires_grad_(False)
        gaitmodel.eval()
        ipts = self.imgs_resize_input(sils)
        ipts = gaitmodel.inputs_pretreament(ipts)
        embs = gaitmodel.forward(ipts)
        embs = embs.detach().cpu().numpy()
        return embs    
    
    def update_gait_feature(self, sils, old_embs):
        gaitmodel = self.loadModel()
        gaitmodel.requires_grad_(False)
        gaitmodel.eval()
        ipts = self.imgs_resize_input([sils])
        ipts = gaitmodel.inputs_pretreament(ipts)
        embs = gaitmodel.forward(ipts)
        new_embs = gaitmodel.merge_features_with_maxpool(embs, old_embs)
        return new_embs  
