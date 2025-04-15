import pickle
import os
import torch
import torch.nn as nn
from typing import Union, Dict, Any
from torchvision import transforms
import torch.utils.data as tordata
from collections import deque
import cv2
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from opengait.modeling import models
from opengait.utils import config_loader, get_attr_from, get_valid_args
from opengait.data.collate_fn import CollateFn
from opengait.data.dataset import DataSet
import opengait.data.sampler as Samplers
from opengait.data.transform import get_transform

pkl_path = '/data4/Gait3D-sils-64-44-pkl/0000/camid0_videoid2/seq0/seq0.pkl'
cfgs_path = './opengait/configs/biggait/BigGait_CCPG.yaml'

cfgs = config_loader(cfgs_path)
data_cfg = cfgs['data_cfg']
sampler_cfg = cfgs['evaluator_cfg']['sampler']
dataset = DataSet(data_cfg, False)

Sampler = get_attr_from([Samplers], sampler_cfg['type'])
vaild_args = get_valid_args(Sampler, sampler_cfg, free_keys=[
    'sample_type', 'type'])
sampler = Sampler(dataset, **vaild_args)

loader = tordata.DataLoader(
    dataset=dataset,
    batch_sampler=sampler,
    collate_fn=CollateFn(dataset.label_set, sampler_cfg),
    num_workers=data_cfg['num_workers'])

# 拿到 DataLoader 的迭代器
loader_iter = iter(loader)
# 从迭代器里取下一个 batch（也就是第一个 batch）
first_batch = next(loader_iter)

print(type(first_batch[0]))
    