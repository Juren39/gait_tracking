import torch
import torch.nn.functional as F

# 0 Background; 1 Hand; 2 Head; 3 Leg; 4 Body; 5 Feet; 6 Shoulder;
def find_channel_centroid(self, parsing_map, channel_idx):
    """计算指定通道的质心坐标"""
    # 提取目标通道并二值化 [N, H, W]
    mask = (parsing_map[:, channel_idx] > 0.5).float()
    # 生成坐标网格
    y_coords, x_coords = torch.meshgrid(
        torch.arange(parsing_map.size(2), device=parsing_map.device),
        torch.arange(parsing_map.size(3), device=parsing_map.device),
        indexing='ij'
    )
    # 批量计算加权坐标 [N, 2]
    total = mask.sum(dim=(1,2)) + 1e-6  # 防零除
    centroid_x = (mask * x_coords).sum(dim=(1,2)) / total
    centroid_y = (mask * y_coords).sum(dim=(1,2)) / total
    return torch.stack([centroid_x, centroid_y], dim=1)  # [N, 2]

def calculate_rotation_angle(self, p1, p2):
    """计算两点间旋转角度（弧度）"""
    delta = p2 - p1
    angle = torch.atan2(delta[:,0], delta[:,1])  # atan2(dx, dy)
    return angle  # [N]

def rotate_feature_maps(self, feature_maps, angle):
    """执行旋转操作"""
    N, _, H, W = feature_maps.shape
    # angle_deg = torch.rad2deg(angle)
    # 构建旋转矩阵
    rotation_matrix = torch.zeros(N, 2, 3, device=feature_maps.device)
    rotation_matrix[:,0,0] = rotation_matrix[:,1,1] = torch.cos(angle)
    rotation_matrix[:,0,1] = -torch.sin(angle)
    rotation_matrix[:,1,0] = torch.sin(angle)
    # 生成仿射网格
    grid = F.affine_grid(rotation_matrix, [N, 1, H, W], align_corners=False)
    # 执行双线性插值
    rotated = F.grid_sample(
        feature_maps.float(), grid, 
        mode='bilinear', 
        padding_mode='border',
        align_corners=False
    ) 
    return rotated

# 0 Background; 1 Hand; 2 Head; 3 Leg; 4 Body; 5 Feet; 6 Shoulder;
# align_human_vertical(parsing_maps=parsing_feat, feature_maps=parsing_feat, mask=mask, align_fn=self.Gait_Align, ratios=ratios)
def align_human_vertical(self, parsing_maps, feature_maps, mask, align_fn, ratios):
    """人体垂直对齐主函数"""
    # 输入参数检查
    assert parsing_maps.dim() == 4, "Input must be NCHW tensor"
    # Step 1: 获取关键点
    # upper_centers = (find_channel_centroid(parsing_maps, 2) + find_channel_centroid(parsing_maps, 4)) / 2  # Head + Body
    upper_centers = find_channel_centroid(parsing_maps, 4)  # Body
    lower_centers = find_channel_centroid(parsing_maps, 3)  # Leg
    # Step 2: 有效性验证
    valid_mask = ~torch.isnan(upper_centers).any(1) & ~torch.isnan(lower_centers).any(1)
    # Step 3: 计算旋转角度
    rotation_angles = torch.zeros(parsing_maps.size(0), device=parsing_maps.device)
    rotation_angles[valid_mask] = -calculate_rotation_angle(
        upper_centers[valid_mask], 
        lower_centers[valid_mask]
    )
    feature_maps = F.pad(feature_maps, (10, 10, 20, 20), mode='constant', value=0)
    mask = F.pad(mask, (10, 10, 20, 20), mode='constant', value=0)
    feature_maps = F.interpolate(feature_maps, scale_factor=2, mode='bilinear')
    mask = F.interpolate(mask, scale_factor=2, mode='bilinear')
    feature_maps = self.rotate_feature_maps(feature_maps, rotation_angles)
    mask = self.rotate_feature_maps(mask, rotation_angles)
    feature_maps = F.interpolate(feature_maps, scale_factor=0.5, mode='bilinear')
    mask = F.interpolate(mask, scale_factor=0.5, mode='bilinear')
    feature_maps = align_fn(feature_maps, mask, ratios) # [n, c, h, w]
    mask = align_fn(mask, mask, ratios) # [n, c, h, w]
    return feature_maps, mask