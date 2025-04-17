#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import pickle
from mmdet.models import LOSSES


@LOSSES.register_module()
class PlanningLoss(nn.Module):
    """轨迹规划损失函数，用于计算预测轨迹和真实轨迹之间的误差"""
    def __init__(self, loss_type='L2'):
        """
        初始化函数
        Args:
            loss_type: 损失类型，默认为'L2'距离
        """
        super(PlanningLoss, self).__init__()
        self.loss_type = loss_type
    
    def forward(self, sdc_traj, gt_sdc_fut_traj, mask):
        """
        计算轨迹规划损失
        Args:
            sdc_traj: 预测的自动驾驶车辆轨迹 (B, T, 2/3)
            gt_sdc_fut_traj: 真实的未来轨迹 (B, T, 2/3)
            mask: 有效帧的掩码 (B, T)
        Returns:
            err: 计算得到的损失值
        """
        # 计算位置误差（x,y坐标）
        err = sdc_traj[..., :2] - gt_sdc_fut_traj[..., :2]
        err = torch.pow(err, exponent=2)
        err = torch.sum(err, dim=-1)
        err = torch.pow(err, exponent=0.5)
        err = torch.sum(err * mask)/(torch.sum(mask) + 1e-5)
        # 如果轨迹包含朝向信息，额外计算朝向误差
        if sdc_traj.shape[2] == 3:
            yaw_err = sdc_traj[..., 2] - gt_sdc_fut_traj[..., 2]
            yaw_err = torch.pow(yaw_err, exponent=2)
            yaw_err = torch.pow(yaw_err, exponent=0.5)
            err += torch.sum(yaw_err * mask)/(torch.sum(mask) + 1e-5)
        return err


@LOSSES.register_module()
class CollisionLoss(nn.Module):
    def __init__(self, delta=0.5, weight=1.0):
        """
        初始化函数
        Args:
            delta: 车辆边界框的膨胀量，用于安全裕度
            weight: 损失权重
        """        
        super(CollisionLoss, self).__init__()
        self.w = 1.85 + delta  # 车辆宽度（加上安全裕度）
        self.h = 4.084 + delta  # 车辆高度（加上安全裕度）
        self.weight = weight
    
    def forward(self, sdc_traj_all, sdc_planning_gt, sdc_planning_gt_mask, future_gt_bbox):
        """
        计算碰撞损失
        Args:
            sdc_traj_all: 预测的自动驾驶车辆轨迹 (1, 6, 2/3)
            sdc_planning_gt: 真实的规划轨迹 (1, 6, 3)
            sdc_planning_gt_mask: 有效帧的掩码 (1, 6)
            future_gt_bbox: 未来帧中其他物体的边界框列表
        Returns:
            inter_sum: 加权后的碰撞损失
        """
        # sdc_traj_all (1, 6, 2)
        # sdc_planning_gt (1,6,3)
        # sdc_planning_gt_mask (1, 6)
        # future_gt_bbox 6x[lidarboxinstance]
        n_futures = len(future_gt_bbox)
        inter_sum = sdc_traj_all.new_zeros(1, )
        dump_sdc = []
        for i in range(n_futures):
            if not sdc_planning_gt_mask[0][i]:  # scene last_frame
                continue
            elif future_gt_bbox[i] is None:     # scene last_frame
                continue
            elif len(future_gt_bbox[i]) == 0:      # no bbox
                continue
            elif len(future_gt_bbox[i].tensor) > 0: # 下一帧有box
                future_gt_bbox_corners = future_gt_bbox[i].corners[:, [0,3,4,7], :2] # (N, 8, 3) -> (N, 4, 2) only bev  # 第i帧lidar坐标系下
                # sdc_yaw = -sdc_planning_gt[0, i, 2].to(sdc_traj_all.dtype) - 1.5708
                if sdc_traj_all.shape[2] == 3:  # pred_yaw
                    sdc_yaw = sdc_traj_all[0, i, 2]
                else:   # gt yaw
                    sdc_yaw = sdc_planning_gt[0, i, 2].to(sdc_traj_all.dtype)
                sdc_bev_box = self.to_corners([sdc_traj_all[0, i, 0], sdc_traj_all[0, i, 1], self.w, self.h, sdc_yaw])  # 第i帧lidar坐标系下
                dump_sdc.append(sdc_bev_box.cpu().detach().numpy())
                for j in range(future_gt_bbox_corners.shape[0]):    # N个bbox
                    inter_sum += self.inter_bbox(sdc_bev_box, future_gt_bbox_corners[j].to(sdc_traj_all.device))
        return inter_sum * self.weight
        
    def inter_bbox(self, corners_a, corners_b):
        xa1, ya1 = torch.max(corners_a[:, 0]), torch.max(corners_a[:, 1])
        xa2, ya2 = torch.min(corners_a[:, 0]), torch.min(corners_a[:, 1])
        xb1, yb1 = torch.max(corners_b[:, 0]), torch.max(corners_b[:, 1])
        xb2, yb2 = torch.min(corners_b[:, 0]), torch.min(corners_b[:, 1])
        
        xi1, yi1 = min(xa1, xb1), min(ya1, yb1)
        xi2, yi2 = max(xa2, xb2), max(ya2, yb2)
        intersect = max((xi1 - xi2), xi1.new_zeros(1, ).to(xi1.device)) * max((yi1 - yi2), xi1.new_zeros(1,).to(xi1.device))
        return intersect

    def to_corners(self, bbox):
        x, y, w, l, theta = bbox
        corners = torch.tensor([
            [w/2, -l/2], [w/2, l/2], [-w/2, l/2], [-w/2,-l/2]  
        ]).to(x.device) # 4,2
        rot_mat = torch.tensor(
            [[torch.cos(theta), torch.sin(theta)],
             [-torch.sin(theta), torch.cos(theta)]]
        ).to(x.device)
        new_corners = rot_mat @ corners.T + torch.tensor(bbox[:2])[:, None].to(x.device)
        return new_corners.T