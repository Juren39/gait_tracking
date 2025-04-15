from pathlib import Path
import torch
from opengait.Gait_Model import GaitModel
import time
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
gait_model = GaitModel(
            device,
            cfgs_path=Path('./opengait/configs/biggait/BigGait_CCPG.yaml')
        )
gait_model.eval_model()