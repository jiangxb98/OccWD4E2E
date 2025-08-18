import numpy as np
import torch
import torch.nn as nn
from mmdet.models.builder import BACKBONES
from ..dense_heads import BevFeatureSlicer
import copy
from einops import rearrange

@BACKBONES.register_module()
class RewardConvNet(nn.Module):
    def __init__(self, 
                 input_channels: int = 256, 
                 hidden_dim: int = 256,
                 fut_traj_num: int = 3,
                 bev_h: int = 200,
                 bev_w: int = 200,
                 sim_reward_nums: int = 0,
                 use_sim_reward: bool = False,
                 use_im_reward: bool = False,
                 extra_bev_adapter: bool = False):
        super(RewardConvNet, self).__init__()
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fut_traj_num = fut_traj_num
        self.sim_reward_nums = sim_reward_nums
        self.use_sim_reward = use_sim_reward
        self.use_im_reward = use_im_reward
        self.extra_bev_adapter = extra_bev_adapter

        if self.extra_bev_adapter:
            bevformer_bev_conf = {
                'xbound': [-51.2, 51.2, 0.512],
                'ybound': [-51.2, 51.2, 0.512],
                'zbound': [-10.0, 10.0, 20.0],
            }
            plan_grid_conf = {
                'xbound': [-50.0, 50.0, 0.5],
                'ybound': [-50.0, 50.0, 0.5],
                'zbound': [-10.0, 10.0, 20.0],
            }
            self.bev_sampler =  BevFeatureSlicer(bevformer_bev_conf, plan_grid_conf)

            self.embed_dims = input_channels
            bev_adapter_block = nn.Sequential(
                nn.Conv2d(self.embed_dims, self.embed_dims // 2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims // 2, self.embed_dims, kernel_size=1),
            )
            N_Blocks = 3
            bev_adapter = [copy.deepcopy(bev_adapter_block) for _ in range(N_Blocks)]
            self.bev_adapter = nn.Sequential(*bev_adapter)     


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
        self.reward_head = None
        if self.use_im_reward:
            self.reward_head = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

        # for sim reward
        self.sim_reward_heads = None
        if self.sim_reward_nums > 0 and self.use_sim_reward:
            self.sim_reward_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1),
                ) for _ in range(self.sim_reward_nums)
            ])

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

    def forward_reward_feat(self, fut_bev_feature, traj):
        pass

    def forward_single(self, fut_bev_feature, traj) -> torch.Tensor:
        """
        Forward propagation.

        Args:
            fut_bev_feature (torch.Tensor): Future BEV feature, shape [bs, bev_h*bev_w, dims]
            traj (torch.Tensor): Trajectory, shape [batch_size*num_traj, planning_steps, 2]
        Returns:
            torch.Tensor: Scoring features, shape [batch_size*num_traj, 128, 1, 1]
        """
        # 取inter_num的最后一个时间步的BEV特征
        bs, bev_h_w, dims = fut_bev_feature.shape
        fut_bev_feature = fut_bev_feature.reshape(bs, self.bev_h, self.bev_w, dims)
        fut_bev_feature = fut_bev_feature.permute(0, 3, 1, 2)
        # 展平traj
        num_traj, planning_steps, _ = traj.shape
        traj_feats = traj.reshape(-1, 2)  # [bs*num_traj, 2]

        reward_feats = self.conv_reward_net(fut_bev_feature)  # [bs, 256, 1, 1]
        traj_feats = self.trajectory_single_encoder(traj_feats)  # [bs*num_traj, 256]

        reward_feats = reward_feats.repeat(bs*num_traj, 1, 1, 1).squeeze(-1).squeeze(-1)
        
        x = self.cat_encoder(torch.cat([reward_feats, traj_feats], dim=1))
        x = self.reward_head(x)

        # select the max reward
        multi_traj_scores = x.reshape(bs, num_traj)
        multi_traj_scores = multi_traj_scores.softmax(dim=1)
        # max_reward_idx = multi_traj_scores.argmax(dim=1)
        # max_reward = multi_traj_scores[:, max_reward_idx]
        # traj = traj.reshape(bs, num_traj, planning_steps, 2)
        # best_traj = traj[:, max_reward_idx, :].squeeze(1)  # [bs, 1, 2]

        # 返回multi_traj_scores, best_traj_idx, best_traj
        # 计算损失函数的时候监督multi_traj_scores，参考WoTE的损失函数的设计
        return multi_traj_scores  #, max_reward_idx, best_traj

    def forward_single_im_sim(self, fut_bev_feature, traj) -> torch.Tensor:
        """
        Forward propagation.

        Args:
            fut_bev_feature (torch.Tensor): Future BEV feature, shape [bs, bev_h*bev_w, dims]
            traj (torch.Tensor): Trajectory, shape [batch_size*num_traj, planning_steps, 2]
        Returns:
            torch.Tensor: Scoring features, shape [batch_size*num_traj, 128, 1, 1]
        """
        if self.extra_bev_adapter:
            fut_bev_feature = rearrange(fut_bev_feature, 'b (w h) c -> b c h w', h=self.bev_h, w=self.bev_w)
            fut_bev_feature = self.bev_sampler(fut_bev_feature)
            fut_bev_feature = fut_bev_feature + self.bev_adapter(fut_bev_feature)
            fut_bev_feature = rearrange(fut_bev_feature, 'b c h w -> b (w h) c', h=self.bev_h, w=self.bev_w)

        bs, bev_h_w, dims = fut_bev_feature.shape
        fut_bev_feature = fut_bev_feature.reshape(bs, self.bev_h, self.bev_w, dims)
        fut_bev_feature = fut_bev_feature.permute(0, 3, 1, 2)
        # 展平traj
        num_traj, planning_steps, _ = traj.shape
        traj_feats = traj.reshape(-1, 2)  # [bs*num_traj, 2]

        # 建议这里额外加一个transformer,对fut_bev_feature进行处理，和traj_feats进行交互


        reward_feats = self.conv_reward_net(fut_bev_feature)  # [bs, 256, 1, 1]
        traj_feats = self.trajectory_single_encoder(traj_feats)  # [bs*num_traj, 256]
        # wote 额外加了个transformer
        # self.cluster_encoder = nn.TransformerEncoder(encoder_layer, num_layers=TRANSFORMER_NUM_LAYERS)
        # traj_feats = self.cluster_encoder(traj_feats)

        reward_feats = reward_feats.repeat(bs*num_traj, 1, 1, 1).squeeze(-1).squeeze(-1)
        
        x_cat = self.cat_encoder(torch.cat([reward_feats, traj_feats], dim=1))
        
        # for imitation reward
        if self.use_im_reward:
            x = self.reward_head(x_cat)
            im_traj_scores = x.reshape(bs, num_traj)
            im_traj_scores = im_traj_scores.softmax(dim=1)
        else:
            im_traj_scores = None

        # for sim reward
        if self.use_sim_reward:
            sim_reward_scores = []
            for i in range(self.sim_reward_nums):
                x_sim = self.sim_reward_heads[i](x_cat)
                x_sim = x_sim.reshape(bs, num_traj)
                sim_reward_scores.append(x_sim)
            # mean for sim reward
            sim_traj_scores = torch.cat(sim_reward_scores, dim=0).reshape(bs*self.sim_reward_nums, num_traj)
        else:
            sim_traj_scores = None

        return im_traj_scores, sim_traj_scores

class CrossAttentionTransformer(nn.Module):
    """2层transformer用于reward_feats和traj_feats的cross attention交互"""
    
    def __init__(self, hidden_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 第一层cross attention
        self.cross_attention_1 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 第一层feed forward
        self.feed_forward_1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # 第一层layer norm
        self.norm_1 = nn.LayerNorm(hidden_dim)
        self.norm_2 = nn.LayerNorm(hidden_dim)
        
        # 第二层cross attention
        self.cross_attention_2 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 第二层feed forward
        self.feed_forward_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # 第二层layer norm
        self.norm_3 = nn.LayerNorm(hidden_dim)
        self.norm_4 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, traj_feats, reward_feats):
        """
        Args:
            traj_feats: [bs*num_traj, hidden_dim] - 轨迹特征作为query
            reward_feats: [bs*num_traj, hidden_dim] - BEV特征作为key和value
        Returns:
            enhanced_feats: [bs*num_traj, hidden_dim] - 交互后的特征
        """
        # 第一层cross attention
        # query: traj_feats, key&value: reward_feats
        attn_output_1, _ = self.cross_attention_1(
            query=traj_feats,
            key=reward_feats,
            value=reward_feats
        )
        
        # 残差连接和layer norm
        traj_feats_1 = self.norm_1(traj_feats + self.dropout(attn_output_1))
        
        # 第一层feed forward
        ff_output_1 = self.feed_forward_1(traj_feats_1)
        traj_feats_1 = self.norm_2(traj_feats_1 + self.dropout(ff_output_1))
        
        # 第二层cross attention
        attn_output_2, _ = self.cross_attention_2(
            query=traj_feats_1,
            key=reward_feats,
            value=reward_feats
        )
        
        # 残差连接和layer norm
        traj_feats_2 = self.norm_3(traj_feats_1 + self.dropout(attn_output_2))
        
        # 第二层feed forward
        ff_output_2 = self.feed_forward_2(traj_feats_2)
        enhanced_feats = self.norm_4(traj_feats_2 + self.dropout(ff_output_2))
        
        return enhanced_feats


@BACKBONES.register_module()
class RewardConvNet_v2(nn.Module):
    def __init__(self, 
                 input_channels: int = 256, 
                 hidden_dim: int = 256,
                 fut_traj_num: int = 3,
                 bev_h: int = 200,
                 bev_w: int = 200,
                 sim_reward_nums: int = 0,
                 use_sim_reward: bool = False,
                 use_im_reward: bool = False,
                 extra_bev_adapter: bool = False,
                 use_cross_attention: bool = False):
        super(RewardConvNet_v2, self).__init__()
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fut_traj_num = fut_traj_num
        self.sim_reward_nums = sim_reward_nums
        self.use_sim_reward = use_sim_reward
        self.use_im_reward = use_im_reward
        self.extra_bev_adapter = extra_bev_adapter

        if self.extra_bev_adapter:
            bevformer_bev_conf = {
                'xbound': [-51.2, 51.2, 0.512],
                'ybound': [-51.2, 51.2, 0.512],
                'zbound': [-10.0, 10.0, 20.0],
            }
            plan_grid_conf = {
                'xbound': [-50.0, 50.0, 0.5],
                'ybound': [-50.0, 50.0, 0.5],
                'zbound': [-10.0, 10.0, 20.0],
            }
            self.bev_sampler =  BevFeatureSlicer(bevformer_bev_conf, plan_grid_conf)

            self.embed_dims = input_channels
            bev_adapter_block = nn.Sequential(
                nn.Conv2d(self.embed_dims, self.embed_dims // 2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims // 2, self.embed_dims, kernel_size=1),
            )
            N_Blocks = 3
            bev_adapter = [copy.deepcopy(bev_adapter_block) for _ in range(N_Blocks)]
            self.bev_adapter = nn.Sequential(*bev_adapter)     


        # Transformer Encoder for trajectory_feat
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=8, 
            dim_feedforward=512, 
            dropout=0.1,
            batch_first=True
        )
        self.cluster_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # BEV feature 和 trajectory_feat 进行cross attention
        if use_cross_attention:
            self.cross_attention_transformer = CrossAttentionTransformer(
                hidden_dim=hidden_dim,
                num_heads=8,
                dropout=0.1
            )
        else:
            self.cross_attention_transformer = None
        self.use_cross_attention = use_cross_attention        


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
        self.reward_head = None
        if self.use_im_reward:
            self.reward_head = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

        # for sim reward
        self.sim_reward_heads = None
        if self.sim_reward_nums > 0 and self.use_sim_reward:
            self.sim_reward_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1),
                ) for _ in range(self.sim_reward_nums)
            ])

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

    def forward_reward_feat(self, fut_bev_feature, traj):
        pass

    def forward_single(self, fut_bev_feature, traj) -> torch.Tensor:
        """
        Forward propagation.

        Args:
            fut_bev_feature (torch.Tensor): Future BEV feature, shape [bs, bev_h*bev_w, dims]
            traj (torch.Tensor): Trajectory, shape [batch_size*num_traj, planning_steps, 2]
        Returns:
            torch.Tensor: Scoring features, shape [batch_size*num_traj, 128, 1, 1]
        """
        # 取inter_num的最后一个时间步的BEV特征
        bs, bev_h_w, dims = fut_bev_feature.shape
        fut_bev_feature = fut_bev_feature.reshape(bs, self.bev_h, self.bev_w, dims)
        fut_bev_feature = fut_bev_feature.permute(0, 3, 1, 2)
        # 展平traj
        num_traj, planning_steps, _ = traj.shape
        traj_feats = traj.reshape(-1, 2)  # [bs*num_traj, 2]

        reward_feats = self.conv_reward_net(fut_bev_feature)  # [bs, 256, 1, 1]
        traj_feats = self.trajectory_single_encoder(traj_feats)  # [bs*num_traj, 256]

        reward_feats = reward_feats.repeat(bs*num_traj, 1, 1, 1).squeeze(-1).squeeze(-1)
        
        x = self.cat_encoder(torch.cat([reward_feats, traj_feats], dim=1))
        x = self.reward_head(x)

        # select the max reward
        multi_traj_scores = x.reshape(bs, num_traj)
        multi_traj_scores = multi_traj_scores.softmax(dim=1)
        # max_reward_idx = multi_traj_scores.argmax(dim=1)
        # max_reward = multi_traj_scores[:, max_reward_idx]
        # traj = traj.reshape(bs, num_traj, planning_steps, 2)
        # best_traj = traj[:, max_reward_idx, :].squeeze(1)  # [bs, 1, 2]

        # 返回multi_traj_scores, best_traj_idx, best_traj
        # 计算损失函数的时候监督multi_traj_scores，参考WoTE的损失函数的设计
        return multi_traj_scores  #, max_reward_idx, best_traj

    def forward_single_im_sim(self, fut_bev_feature, traj) -> torch.Tensor:
        """
        Forward propagation.

        Args:
            fut_bev_feature (torch.Tensor): Future BEV feature, shape [bs, bev_h*bev_w, dims]
            traj (torch.Tensor): Trajectory, shape [batch_size*num_traj, planning_steps, 2]
        Returns:
            torch.Tensor: Scoring features, shape [batch_size*num_traj, 128, 1, 1]
        """
        if self.extra_bev_adapter:
            fut_bev_feature = rearrange(fut_bev_feature, 'b (w h) c -> b c h w', h=self.bev_h, w=self.bev_w)
            fut_bev_feature = self.bev_sampler(fut_bev_feature)
            fut_bev_feature = fut_bev_feature + self.bev_adapter(fut_bev_feature)
            fut_bev_feature = rearrange(fut_bev_feature, 'b c h w -> b (w h) c', h=self.bev_h, w=self.bev_w)

        bs, bev_h_w, dims = fut_bev_feature.shape
        fut_bev_feature = fut_bev_feature.reshape(bs, self.bev_h, self.bev_w, dims)
        fut_bev_feature = fut_bev_feature.permute(0, 3, 1, 2)
        # 展平traj
        num_traj, planning_steps, _ = traj.shape
        traj_feats = traj.reshape(-1, 2)  # [bs*num_traj, 2]

        # 建议这里额外加一个transformer,对fut_bev_feature进行处理，和traj_feats进行交互


        reward_feats = self.conv_reward_net(fut_bev_feature)  # [bs, 256, 1, 1]
        traj_feats = self.trajectory_single_encoder(traj_feats)  # [bs*num_traj, 256]
        # WOTE 额外加了个transformer
        traj_feats = self.cluster_encoder(traj_feats)

        
        if self.use_cross_attention:
            # reward_feats 和 traj_feats 进行attention交互
            # query 是traj_feats，key和value是reward_feats
            enhanced_feats = self.cross_attention_transformer(traj_feats, reward_feats)
            x_cat = self.cat_encoder(torch.cat([reward_feats, enhanced_feats], dim=1))
        else:
            reward_feats = reward_feats.repeat(bs*num_traj, 1, 1, 1).squeeze(-1).squeeze(-1)
            x_cat = self.cat_encoder(torch.cat([reward_feats, traj_feats], dim=1))
        
        # for imitation reward
        if self.use_im_reward:
            x = self.reward_head(x_cat)
            im_traj_scores = x.reshape(bs, num_traj)
            im_traj_scores = im_traj_scores.softmax(dim=1)
        else:
            im_traj_scores = None

        # for sim reward
        if self.use_sim_reward:
            sim_reward_scores = []
            for i in range(self.sim_reward_nums):
                x_sim = self.sim_reward_heads[i](x_cat)
                x_sim = x_sim.reshape(bs, num_traj)
                sim_reward_scores.append(x_sim)
            # mean for sim reward
            sim_traj_scores = torch.cat(sim_reward_scores, dim=0).reshape(bs*self.sim_reward_nums, num_traj)
        else:
            sim_traj_scores = None

        return im_traj_scores, sim_traj_scores


if __name__ == "__main__":
    # reward_conv_net = RewardConvNet()
    # x = torch.randn(3, 3, 1, 40000, 256)
    # traj = torch.randn(10, 3, 2)
    # print(x.shape)
    # y = reward_conv_net.forward_multi(x, traj)
    # print(y.shape)


    # reward_conv_net = RewardConvNet()
    # fut_bev_feature = torch.randn(1, 40000, 256)
    # traj = torch.randn(20, 1, 2)
    # print(fut_bev_feature.shape)
    # print(traj.shape)
    # y = reward_conv_net.forward_single(fut_bev_feature, traj)
    # print(y.shape)


    reward_conv_net = RewardConvNet_v2(use_im_reward=True, 
                                       use_sim_reward=True, 
                                       sim_reward_nums=1, 
                                       use_cross_attention=True, 
                                       extra_bev_adapter=True)
    fut_bev_feature = torch.randn(1, 40000, 256)
    traj = torch.randn(20, 1, 2)
    y = reward_conv_net.forward_single_im_sim(fut_bev_feature, traj)
    print(y.shape)


