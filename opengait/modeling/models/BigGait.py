# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py


import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange
from ..base_model import BaseModel
from torch.nn import functional as F
from kornia import morphology as morph
import random
import cv2
import numpy as np

# import GaitBase & DINOv2_small
from .BigGait_utils.BigGait_GaitBase import Baseline
from .BigGait_utils.DINOv2 import vit_small
from .BigGait_utils.save_img import save_image, pca_image

# ######################################## BigGait ###########################################

class infoDistillation(nn.Module):
    def __init__(self, source_dim, target_dim, p, softmax, Relu, Up=True):
        super(infoDistillation, self).__init__()
        self.dropout = nn.Dropout(p=p)
        self.bn_s = nn.BatchNorm1d(source_dim, affine=False)
        self.bn_t = nn.BatchNorm1d(target_dim, affine=False)
        if Relu:
            self.down_sampling = nn.Sequential(
                nn.Linear(source_dim, source_dim//2),
                nn.BatchNorm1d(source_dim//2, affine=False),
                nn.GELU(),
                nn.Linear(source_dim//2, target_dim),
                )
            if Up:
                self.up_sampling = nn.Sequential(
                    nn.Linear(target_dim, source_dim//2),
                    nn.BatchNorm1d(source_dim//2, affine=False),
                    nn.GELU(),
                    nn.Linear(source_dim//2, source_dim),
                    )
        else:
            self.down_sampling = nn.Linear(source_dim, target_dim)
            if Up:
                self.up_sampling = nn.Linear(target_dim, source_dim)
        self.softmax = softmax
        self.mse = nn.MSELoss()
        self.Up = Up

    def forward(self, x):
        # [n, c]
        d_x = self.down_sampling(self.bn_s(self.dropout(x)))
        if self.softmax:
            d_x = F.softmax(d_x, dim=1)
            if self.Up:
                u_x = self.up_sampling(d_x)
                return d_x, torch.mean(self.mse(u_x, x))
            else:
                return d_x, None
        else:
            if self.Up:
                u_x = self.up_sampling(d_x)
                return torch.sigmoid(self.bn_t(d_x)), torch.mean(self.mse(u_x, x))
            else:
                return torch.sigmoid(self.bn_t(d_x)), None


def padding_resize(x, ratios, target_h, target_w):
    n,h,w = x.size(0),target_h, target_w
    ratios = ratios.view(-1)
    need_w = (h * ratios).int()
    need_padding_mask = need_w < w
    pad_left = torch.where(need_padding_mask, (w - need_w) // 2, torch.tensor(0).to(x.device))
    pad_right = torch.where(need_padding_mask, w - need_w - pad_left, torch.tensor(0).to(x.device)).tolist()
    need_w = need_w.tolist()
    pad_left = pad_left.tolist()
    x = torch.concat([F.pad(F.interpolate(x[i:i+1,...], (h, need_w[i]), mode="bilinear", align_corners=False), (pad_left[i], pad_right[i]))  if need_padding_mask[i] else F.interpolate(x[i:i+1,...], (h, need_w[i]), mode="bilinear", align_corners=False)[...,pad_left[i]:pad_left[i]+w]  for i in range(n)], dim=0)
    return x

class BigGait__Dinov2_Gaitbase(nn.Module):
    def __init__(self, model_cfg: dict, msg_mgr=None, device=None):
        """
        Args:
            model_cfg (dict): 配置参数, 内含 pretrained_dinov2, pretrained_mask_branch 等子字段.
            msg_mgr (optional): 日志管理器(可为None或你的logger).
            device (optional): 当前使用的设备, 可为 torch.device 或字符串等.
        """
        super().__init__()
        self.msg_mgr = msg_mgr
        self.device = device

        # 在构造函数里调用你原先的 build_network
        self.build_network(model_cfg)

        # 在构造函数里执行模型权重初始化, 包括 init_DINOv2(), init_Mask_Branch() 等
        self.init_parameters()
        
    def build_network(self, model_cfg):
        # get pretained models
        self.pretrained_dinov2 = model_cfg["pretrained_dinov2"]
        self.pretrained_mask_branch = model_cfg["pretrained_mask_branch"]
        self.pretrained_biggait = model_cfg["pretrained_biggait"]

        # set input size
        self.image_size = model_cfg["image_size"]
        self.sils_size = model_cfg["sils_size"]

        # set feature dim
        self.f4_dim = model_cfg["Mask_Branch"]['source_dim']
        self.fc_dim = self.f4_dim*4
        self.mask_dim = model_cfg["Mask_Branch"]['target_dim']
        self.app_dim = model_cfg["Appearance_Branch"]['target_dim']
        self.denoising_dim = model_cfg["Denoising_Branch"]['target_dim']

        # init submodules
        # self.Denoising_Branch = infoDistillation(**model_cfg["Denoising_Branch"])
        # self.Appearance_Branch = infoDistillation(**model_cfg["Appearance_Branch"])
        # self.Mask_Branch = infoDistillation(**model_cfg["Mask_Branch"])
        self.part_info = infoDistillation(**model_cfg["Denoising_Branch"])
        self.rgb_info = infoDistillation(**model_cfg["Appearance_Branch"])
        self.body_info = infoDistillation(**model_cfg["Mask_Branch"])
        self.gait_net = Baseline(model_cfg)
        self.backbone = vit_small()

    # def init_DINOv2(self):
    #     self.backbone = vit_small()
    #     # self.msg_mgr.log_info(f'load model from: {self.pretrained_dinov2}')
    #     pretrain_dict = torch.load(self.pretrained_dinov2)
    #     msg = self.backbone.load_state_dict(pretrain_dict, strict=True)
    #     # n_parameters = sum(p.numel() for p in self.backbone.parameters())
    #     # self.msg_mgr.log_info('Missing keys: {}'.format(msg.missing_keys))
    #     # self.msg_mgr.log_info('Unexpected keys: {}'.format(msg.unexpected_keys))
    #     # self.msg_mgr.log_info(f"=> loaded successfully '{self.pretrained_dinov2}'")
    #     # self.msg_mgr.log_info('DINOv2 Count: {:.5f}M'.format(n_parameters / 1e6))

    def init_BigGait(self):
        # self.msg_mgr.log_info(f'load model from: {self.pretrained_biggait}')
        load_dict = torch.load(self.pretrained_biggait, map_location=torch.device("cpu"))['model']
        msg = self.load_state_dict(load_dict, strict=True)
        # n_parameters = sum(p.numel() for p in self.Mask_Branch.parameters())
        # self.msg_mgr.log_info('Missing keys: {}'.format(msg.missing_keys))
        # self.msg_mgr.log_info('Unexpected keys: {}'.format(msg.unexpected_keys))
        # self.msg_mgr.log_info(f"=> loaded successfully '{self.pretrained_mask_branch}'")
        # self.msg_mgr.log_info('SegmentationBranch Count: {:.5f}M'.format(n_parameters / 1e6))
    
    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

        n_parameters = sum(p.numel() for p in self.parameters())
        # self.msg_mgr.log_info('Expect backbone Count: {:.5f}M'.format(n_parameters / 1e6))

        self.init_BigGait()
        self.eval()
        self.requires_grad_(False)
        
        # n_parameters = sum(p.numel() for p in self.parameters())
        # self.msg_mgr.log_info('All Backbone Count: {:.5f}M'.format(n_parameters / 1e6))
        # self.msg_mgr.log_info("=> init successfully")

    # resize image
    def preprocess(self, sils, image_size, mode='bilinear'):
        # shape: [nxs,c,h,w] / [nxs,c,224,112]
        return F.interpolate(sils, (image_size*2, image_size), mode=mode, align_corners=False)

    def min_max_norm(self, x):
        return (x - x.min())/(x.max() - x.min())

    # cal foreground
    def get_body(self, mask):
        # value: [0,1]  shape: [nxs, h, w, c]
        def judge_edge(image, edge=1):
            # [nxs,h,w]
            edge_pixel_count = image[:, :edge, :].sum(dim=(1,2)) + image[:, -edge:, :].sum(dim=(1,2))
            return edge_pixel_count > (image.size(2)) * edge
        condition_mask = torch.round(mask[...,0]) - mask[...,0].detach() + mask[...,0]
        condition_mask = judge_edge(condition_mask, 5)
        mask[condition_mask, :, :, 0] = mask[condition_mask, :, :, 1]
        return mask[...,0]
    
    def connect_loss(self, images, n, s, c):
        images = images.view(n*s,c,self.sils_size*2,self.sils_size)
        gradient_x = F.conv2d(images, torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])[None,None,...].repeat(1,c,1,1).to(images.dtype).to(images.device), padding=1)
        gradient_y = F.conv2d(images, torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])[None,None,...].repeat(1,c,1,1).to(images.dtype).to(images.device), padding=1)
        loss_connectivity = (torch.sum(torch.abs(gradient_x)) + torch.sum(torch.abs(gradient_y))) / (n*s*c*self.sils_size*2*self.sils_size)
        return loss_connectivity
    
    # Binarization and Closing operations to enhance foreground
    def get_edge(self, sils, threshold=1):
        mask_sils = torch.round(sils * threshold)
        kernel = torch.ones((3,3))
        dilated_mask = morph.dilation(mask_sils, kernel.to(sils.device)).detach()  # Dilation
        kernel = torch.ones((5,5))
        eroded_mask = morph.erosion(dilated_mask, kernel.to(sils.device)).detach()  # Erosion
        edge_mask = (dilated_mask > 0.5) ^ (eroded_mask > 0.5)
        sils = edge_mask * sils + (eroded_mask > 0.5) * torch.ones_like(sils, dtype=sils.dtype, device=sils.device)
        return sils

    def diversity_loss(self, images, max_p):
        # [ns, hw, c]
        p = torch.sum(images, dim=1) / (torch.sum(images, dim=(1,2)) + 1e-6).view(-1,1).repeat(1,max_p)
        entropies = -torch.sum(p * torch.log2(p + 1e-6), dim=1)
        max_p = torch.Tensor([1/max_p]).repeat(max_p).to(images.dtype).to(images.device)
        max_entropies = -torch.sum(max_p * torch.log2(max_p), dim=0)
        return torch.mean(max_entropies - entropies)
    
    def inputs_pretreament(self, ipts):
        mean_array = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std_array = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        sils = ipts[0]
        ratios = ipts[1]
        sils = (sils - mean_array) / std_array
        ipts = (sils, ratios)
        return ipts
    
    def merge_features_with_maxpool(old_features: torch.Tensor,
                                new_feature: torch.Tensor) -> torch.Tensor:
        if new_feature.dim() == 1:
            new_feature = new_feature.unsqueeze(0)  # [1, D]
        combined = torch.cat([old_features, new_feature], dim=0)  # [N+1, D]
        merged_feature, _ = torch.max(combined, dim=0)
        return merged_feature

    def forward(self, ipts):
        sils = ipts[0]                      # input_images;         shape: [n,s,c,h,w];
        ratios = ipts[1]                   # real_image_ratios     shape: [n,s,ratio];     ratio: w/h,  e.g. 112/224=0.5;
        seqL = None
        del ipts

        with torch.no_grad():
            n,s,c,h,w = sils.size()
            sils = rearrange(sils, 'n s c h w -> (n s) c h w').contiguous()
            outs = self.preprocess(padding_resize(sils, ratios, 256, 128), self.image_size)         # [ns,c,448,224]    if have not used pad_resize for input images
            outs = self.backbone(outs, is_training=True) # [ns,h*w,c]
            outs_last1 = outs["x_norm_patchtokens"].contiguous()
            outs_last4 = outs["x_norm_patchtokens_mid4"].contiguous()

            outs_last1 = rearrange(outs_last1.view(n, s, self.image_size//7, self.image_size//14, -1), 'n s h w c -> (n s) c h w').contiguous()
            outs_last4 = rearrange(outs_last4.view(n, s, self.image_size//7, self.image_size//14, -1), 'n s h w c -> (n s) c h w').contiguous()
            outs_last1 = self.preprocess(outs_last1, self.sils_size) # [ns,c,64,32]
            outs_last4 = self.preprocess(outs_last4, self.sils_size) # [ns,c,64,32]
            outs_last1 = rearrange(outs_last1.view(n, s, -1, self.sils_size*2, self.sils_size), 'n s c h w -> (n s) (h w) c').contiguous()
            outs_last4 = rearrange(outs_last4.view(n, s, -1, self.sils_size*2, self.sils_size), 'n s c h w -> (n s) (h w) c').contiguous()

        # get foreground
        mask = torch.ones_like(outs_last1[...,0], device=outs_last1.device, dtype=outs_last1.dtype).view(n*s,1,self.sils_size*2,self.sils_size)
        mask = padding_resize(mask, ratios, self.sils_size*2, self.sils_size)
        foreground = outs_last1.view(-1, self.f4_dim)[mask.view(-1) != 0]
        fore_feat, loss_mse1 = self.body_info(foreground)
        foreground = torch.zeros_like(mask, dtype=fore_feat.dtype, device=fore_feat.device).view(-1,1).repeat(1,self.mask_dim)
        foreground[mask.view(-1) != 0] = fore_feat
        loss_connectivity_shape = self.connect_loss(foreground, n, s, self.mask_dim)
        foreground = foreground.detach().clone()
        foreground = self.get_body(foreground.view(n*s,self.sils_size*2,self.sils_size,self.mask_dim)).view(n*s,-1) # [n*s,h*w]
        foreground = self.get_edge(foreground.view(n*s,1,self.sils_size*2,self.sils_size)).view(n*s,-1) # [n*s,h*w]
        del fore_feat, mask

        # get denosing
        denosing = outs_last4.view(-1, self.fc_dim)[foreground.view(-1) != 0]
        den_feat, _ = self.part_info(denosing)
        denosing = torch.zeros_like(foreground, dtype=den_feat.dtype, device=den_feat.device).view(-1,1).repeat(1,self.denoising_dim)
        denosing[foreground.view(-1) != 0] = den_feat
        loss_connectivity_part = self.connect_loss(denosing.view(n*s,-1,self.denoising_dim)[...,:-1].permute(0,2,1), n, s, (self.denoising_dim-1))
        loss_diversity_part = self.diversity_loss(denosing.view(n*s,-1,self.denoising_dim), self.denoising_dim)
        del den_feat

        # get appearance
        appearance = outs_last4.view(-1, self.fc_dim)[foreground.view(-1) != 0]
        app_feat, _ = self.rgb_info(appearance)
        appearance = torch.zeros_like(foreground, dtype=app_feat.dtype, device=app_feat.device).view(-1,1).repeat(1,self.app_dim)
        appearance[foreground.view(-1) != 0] = app_feat
        appearance = appearance.view(n*s,-1,self.app_dim)
        del app_feat

        # get embeding
        embed_1, logits = self.gait_net(
                                        denosing.view(n,s,self.sils_size*2,self.sils_size,self.denoising_dim).permute(0, 4, 1, 2, 3).contiguous(),
                                        appearance.view(n,s,self.sils_size*2,self.sils_size,self.app_dim).permute(0, 4, 1, 2, 3).contiguous(),
                                        seqL,
                                        )
        return embed_1
