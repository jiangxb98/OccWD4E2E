import numpy as np
import torch
import torch.nn as nn
from mmdet.models.builder import BACKBONES

@BACKBONES.register_module()
class RewardConvNet(nn.Module):
    def __init__(self, 
                 input_channels: int = 256, 
                 hidden_dim: int = 256,
                 fut_traj_num: int = 3,
                 bev_h: int = 200,
                 bev_w: int = 200,):
        """
        Initialize RewardConvNet.

        Args:
            input_channels (int): Number of channels in the input feature map. Default is 512.
            conv1_out_channels (int): Number of output channels for the first convolution. Default is 256.
            conv2_out_channels (int): Number of output channels for the second convolution. Default is 256.
        """
        super(RewardConvNet, self).__init__()
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fut_traj_num = fut_traj_num

        # 合并所有卷积层到一个Sequential中
        self.conv_reward_net = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            
            # 第二个卷积块
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            
            # Adaptive average pooling to reduce spatial dimensions to 1x1
            nn.AdaptiveAvgPool2d(1),
        )

        # Encode for trajectory
        # self.trajectory_encoder = nn.Sequential(
        #     nn.Linear(fut_traj_num * 2, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        # )

        self.trajectory_single_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Encode for trajectory and fut_bev feature
        self.cat_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # MLP head for scoring
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )


    def forward_multi(self, fut_bev_feature, traj) -> torch.Tensor:
        """
        Forward propagation.

        Args:
            fut_bev_feature (torch.Tensor): Future BEV feature, shape [L_out, inter_num, bs, bev_h*bev_w, dims]
            traj (torch.Tensor): Trajectory, shape [batch_size*num_traj, fut_traj_num, 2]
        Returns:
            torch.Tensor: Scoring features, shape [batch_size*num_traj, 128, 1, 1]
        """
        # 取inter_num的最后一个时间步的BEV特征
        fut_traj_num, inter_num, bs, bev_h_w, dims = fut_bev_feature.shape
        fut_bev_feature = fut_bev_feature[:, -1, ...].reshape(fut_traj_num, bs, self.bev_h, self.bev_w, dims).reshape(fut_traj_num*bs, self.bev_h, self.bev_w, dims)
        fut_bev_feature = fut_bev_feature.mean(0).unsqueeze(0).permute(0, 3, 1, 2)
        # 展平traj
        sample_traj_nums = traj.shape[0]
        traj_feats = traj.reshape(-1, self.fut_traj_num * 2)

        reward_feats = self.conv_reward_net(fut_bev_feature)  # [bs, 256, 1, 1]
        traj_feats = self.trajectory_encoder(traj_feats)  # [n_traj*bs, 256]

        reward_feats = reward_feats.repeat(sample_traj_nums, 1, 1, 1).squeeze(-1).squeeze(-1)
        
        x = self.cat_encoder(torch.cat([reward_feats, traj_feats], dim=1))
        x = self.reward_head(x)
        return reward_feats, x



    def forward_single(self, fut_bev_feature, traj) -> torch.Tensor:
        """
        Forward propagation.

        Args:
            fut_bev_feature (torch.Tensor): Future BEV feature, shape [bs, bev_h*bev_w, dims]
            traj (torch.Tensor): Trajectory, shape [batch_size*num_traj, fut_traj_num, 2]
        Returns:
            torch.Tensor: Scoring features, shape [batch_size*num_traj, 128, 1, 1]
        """
        # 取inter_num的最后一个时间步的BEV特征
        bs, bev_h_w, dims = fut_bev_feature.shape
        fut_bev_feature = fut_bev_feature.reshape(bs, self.bev_h, self.bev_w, dims)
        fut_bev_feature = fut_bev_feature.permute(0, 3, 1, 2)
        # 展平traj
        bs, num_traj = traj.shape[0], traj.shape[1]
        traj_feats = traj.reshape(-1, 2)  # [bs*num_traj, 2]

        reward_feats = self.conv_reward_net(fut_bev_feature)  # [bs, 256, 1, 1]
        traj_feats = self.trajectory_single_encoder(traj_feats)  # [bs*num_traj, 256]

        reward_feats = reward_feats.repeat(bs*num_traj, 1, 1, 1).squeeze(-1).squeeze(-1)
        
        x = self.cat_encoder(torch.cat([reward_feats, traj_feats], dim=1))
        x = self.reward_head(x)

        # select the max reward
        multi_traj_scores = x.reshape(bs, num_traj)
        multi_traj_scores = multi_traj_scores.softmax(dim=1)
        max_reward_idx = multi_traj_scores.argmax(dim=1)
        max_reward = multi_traj_scores[:, max_reward_idx]
        best_traj = traj[:, max_reward_idx, :]  # [bs, 1, 2]

        # 返回multi_traj_scores, best_traj_idx, best_traj
        # 计算损失函数的时候监督multi_traj_scores，参考WoTE的损失函数的设计
        return multi_traj_scores, max_reward_idx, best_traj

if __name__ == "__main__":
    # reward_conv_net = RewardConvNet()
    # x = torch.randn(3, 3, 1, 40000, 256)
    # traj = torch.randn(10, 3, 2)
    # print(x.shape)
    # y = reward_conv_net.forward_multi(x, traj)
    # print(y.shape)


    reward_conv_net = RewardConvNet()
    fut_bev_feature = torch.randn(1, 40000, 256)
    traj = torch.randn(1, 20, 2)
    print(fut_bev_feature.shape)
    print(traj.shape)
    y = reward_conv_net.forward_single(fut_bev_feature, traj)
    print(y.shape)

