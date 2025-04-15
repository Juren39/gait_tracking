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
import pickle
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from opengait.utils.common import NoOp, Odict

from opengait.modeling import models
from opengait.utils import config_loader, get_attr_from, get_valid_args, is_list, is_dict, np2var, ts2np, list2var, get_msg_mgr
from opengait.evaluation import evaluator as eval_functions
from opengait.data.collate_fn import CollateFn
from opengait.data.dataset import DataSet
import opengait.data.sampler as Samplers
from opengait.data.transform import get_transform


class GaitModel():
    def __init__(
        self,
        device,
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
        self.msg_mgr = get_msg_mgr()
        self.cfgs: Dict[str, Any] = config_loader(cfgs_path)
        self.model = self.loadModel()
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.test_loader = self.get_loader(
                self.cfgs['data_cfg'], train=False)
        self.evaluator_trfs = get_transform(
            self.cfgs['evaluator_cfg']['transform'])
        self.sils_transform = int(self.cfgs['evaluator_cfg']['sils_transform'])
        engine_cfg = self.cfgs['evaluator_cfg']
        output_path = os.path.join('output/', self.cfgs['data_cfg']['dataset_name'],
                            self.cfgs['model_cfg']['model'], engine_cfg['save_name'])
        self.msg_mgr.init_logger(output_path, False)

    def loadModel(self):
        model_cfg = self.cfgs['model_cfg']
        Model = getattr(models, model_cfg['model'])
        model = Model(model_cfg)
        return model
    
    def trajectory_pretreament(self, img_deque, labs=None, ty=None, vi=None, seqL=None):
        s = len(img_deque)
        images_list = list(img_deque)
        trf = self.evaluator_trfs[self.sils_transform]
        sils_list = []
        ratios_list = []
        seq_tensors = []
        seq_ratios = []
        for idx in range(s):
            img = images_list[idx]
            h, w = img.shape[:2] # (H, W, C)
            img = cv2.resize(img, (128, 128))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ratio = w / float(h)
            img_tensor = img.astype(np.float32)
            img_tensor = torch.from_numpy(img_tensor).permute(2,0,1)
            seq_tensors.append(img_tensor)
            seq_ratios.append(ratio)
        seq_tensors = torch.stack(seq_tensors, dim=0) # [s, C, 128, 128]
        seq_tensors = self.model.input_pretreament(seq_tensors, self.device)
        if trf is not None: seq_tensors = trf(seq_tensors)  
        sils_list.append(seq_tensors) # [1, s, C, 128, 128]
        ratios_list.append(seq_ratios)
        sils = torch.stack(sils_list, dim=0).to(self.device) # [1, s, C, 128, 128]
        ratios = torch.tensor(ratios_list, dtype=torch.float32).to(self.device)
        ipts = (sils, ratios)
        return ipts, labs, ty, vi, seqL     
    
        ####### 输入 sils 为某个轨迹一系列的检测框图像
    def extract_gait_feature(self, sils):
        inputs = self.trajectory_pretreament(sils)
        embs = self.model.forward(inputs)['inference_feat']['embeddings']
        ####### 改变输入形式
        embs = embs.detach().cpu().numpy()
        return embs
    
    def get_loader(self, data_cfg, train=False):
        sampler_cfg = self.cfgs['evaluator_cfg']['sampler']
        dataset = DataSet(data_cfg, train)
        Sampler = get_attr_from([Samplers], sampler_cfg['type'])
        vaild_args = get_valid_args(Sampler, sampler_cfg, free_keys=[
            'sample_type', 'type'])
        sampler = Sampler(dataset, **vaild_args)

        loader = tordata.DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            collate_fn=CollateFn(dataset.label_set, sampler_cfg),
            num_workers=data_cfg['num_workers'])
        return loader
    
    def dataset_pretreament(self, inputs):
        """Conduct transforms on input data.
        Args:
            inputs: the input data.
        Returns:
            tuple: training data including inputs, labels, and some meta data.
        """
        seqs_batch, labs_batch, typs_batch, vies_batch, seqL_batch = inputs
        seq_trfs =  self.evaluator_trfs
        if len(seqs_batch) != len(seq_trfs):
            raise ValueError(
                "The number of types of input data and transform should be same. But got {} and {}".format(len(seqs_batch), len(seq_trfs)))
        seqs = [np2var(np.asarray([trf(fra) for fra in seq]), requires_grad=False).float().to(self.device)
                for trf, seq in zip(seq_trfs, seqs_batch)]
        typs = typs_batch
        vies = vies_batch
        labs = list2var(labs_batch).long()
        if seqL_batch is not None:
            seqL_batch = np2var(seqL_batch).int()
        seqL = seqL_batch

        if seqL is not None:
            seqL_sum = int(seqL.sum().data.cpu().numpy())
            ipts = [_[:, :seqL_sum] for _ in seqs]
        else:
            ipts = seqs
        del seqs
        return ipts, labs, typs, vies, seqL

    def inference(self, train=False):
        if train:
            data_loader = None
        else:
            data_loader = self.test_loader
            total_size = len(self.test_loader)
        pbar = tqdm(total=total_size, desc='Transforming')
        batch_size = data_loader.batch_sampler.batch_size
        rest_size = total_size
        info_dict = Odict()
        for inputs in data_loader:
            ipts = self.dataset_pretreament(inputs) # inputs <class 'list'>
            retval = self.model.forward(ipts)
            inference_feat = retval['inference_feat']
            del retval  
            for k, v in inference_feat.items():
                inference_feat[k] = ts2np(v)
            info_dict.append(inference_feat)
            rest_size -= batch_size
            if rest_size >= 0:
                update_size = batch_size
            else:
                update_size = total_size % batch_size
            pbar.update(update_size)
        pbar.close()
        for k, v in info_dict.items():
            v = np.concatenate(v)[:total_size]
            info_dict[k] = v
        return info_dict
    
    def eval_model(self):
        evaluator_cfg = self.cfgs['evaluator_cfg']
        with torch.no_grad():
            info_dict = self.inference()
        loader = self.test_loader
        label_list = loader.dataset.label_list
        types_list = loader.dataset.types_list
        views_list = loader.dataset.views_list
        info_dict.update({
            'labels': label_list,
            'types': types_list,
            'views': views_list
        })
        if 'eval_func' in evaluator_cfg:
            eval_func = evaluator_cfg["eval_func"]
        else:
            eval_func = 'identification'  
        eval_func = getattr(eval_functions, eval_func)
        valid_args = get_valid_args(eval_func, evaluator_cfg, ['metric'])

        try:
            dataset_name = self.cfgs['data_cfg']['test_dataset_name']
        except KeyError:
            dataset_name = self.cfgs['data_cfg']['dataset_name']
        return eval_func(info_dict, dataset_name, **valid_args)

