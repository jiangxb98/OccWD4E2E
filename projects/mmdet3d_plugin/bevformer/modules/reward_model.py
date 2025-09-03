import numpy as np
import torch
import torch.nn as nn
from mmdet.models.builder import BACKBONES
import copy
from einops import rearrange
import torch.nn.functional as F

def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    """
    Parameters
    ----------
        x_bounds: Forward direction in the ego-car.
        y_bounds: Sides
        z_bounds: Height

    Returns
    -------
        bev_resolution: Bird's-eye view bev_resolution
        bev_start_position Bird's-eye view first element
        bev_dimension Bird's-eye view tensor spatial dimension
    """
    bev_resolution = torch.tensor(
        [row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = torch.tensor(
        [row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = torch.tensor([(row[1] - row[0]) / row[2]
                                 for row in [x_bounds, y_bounds, z_bounds]], dtype=torch.long)

    return bev_resolution, bev_start_position, bev_dimension

# Grid sampler
# Sample a smaller receptive-field bev from larger one
class BevFeatureSlicer(nn.Module):
    def __init__(self, grid_conf, map_grid_conf):
        super().__init__()
        if grid_conf == map_grid_conf:
            self.identity_mapping = True
        else:
            self.identity_mapping = False

            bev_resolution, bev_start_position, bev_dimension= calculate_birds_eye_view_parameters(
                grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound']
            )

            map_bev_resolution, map_bev_start_position, map_bev_dimension = calculate_birds_eye_view_parameters(
                map_grid_conf['xbound'], map_grid_conf['ybound'], map_grid_conf['zbound']
            )

            self.map_x = torch.arange(
                map_bev_start_position[0], map_grid_conf['xbound'][1], map_bev_resolution[0])

            self.map_y = torch.arange(
                map_bev_start_position[1], map_grid_conf['ybound'][1], map_bev_resolution[1])

            # convert to normalized coords
            self.norm_map_x = self.map_x / (- bev_start_position[0])
            self.norm_map_y = self.map_y / (- bev_start_position[1])

            tmp_m, tmp_n = torch.meshgrid(
                self.norm_map_x, self.norm_map_y)  # indexing 'ij'
            tmp_m, tmp_n = tmp_m.T, tmp_n.T  # change it to the 'xy' mode results

            self.map_grid = torch.stack([tmp_m, tmp_n], dim=2)

    def forward(self, x):
        # x: bev feature map tensor of shape (b, c, h, w)
        if self.identity_mapping:
            return x
        else:
            grid = self.map_grid.unsqueeze(0).type_as(
                x).repeat(x.shape[0], 1, 1, 1)

            return F.grid_sample(x, grid=grid, mode='bilinear', align_corners=True)

@BACKBONES.register_module()
class RewardConvNet(nn.Module):
    def __init__(self, 
                 input_channels: int = 256, 
                 hidden_dim: int = 256,
                 fut_traj_num: int = 3,
                 bev_h: int = 200,
                 bev_w: int = 200,
                 sim_reward_nums: int = 5,
                 use_sim_reward: bool = False,
                 use_im_reward: bool = False,
                 extra_bev_adapter: bool = False,
                 ):
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
            traj (torch.Tensor): Trajectory, shape [batch_size, num_traj, planning_steps, 2]
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
        bs, num_traj, planning_steps, _ = traj.shape
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

    def reward_distillation_alignment(self, model_a_trajectories, model_b_trajectory, fut_bev_feature, return_distance_loss=False):
        """
        使用reward蒸馏实现跨模型对齐
        
        Args:
            model_a_trajectories: torch.Tensor [bs, times, 20, 2] - 模型A的20个轨迹点
            model_b_trajectory: torch.Tensor [bs, times, 2] - 模型B的轨迹点, 小模型
            reward_model: 训练好的reward模型
            fut_bev_feature: torch.Tensor [bs, times, bev_h*bev_w, dims] - 未来times帧BEV特征
        Returns:
            alignment_loss: torch.Tensor - 对齐损失
        """
        bs, times, bev_h_w, dims = fut_bev_feature.shape
        num_traj = model_a_trajectories.shape[2]
        assert bs == 1
        # bs = bs x times
        # bs = times
        # 编码未来bev特征
        fut_bev_feature = fut_bev_feature.reshape(bs*times, bev_h_w, dims)
        fut_bev_feature = fut_bev_feature.reshape(bs*times, self.bev_h, self.bev_w, dims)
        fut_bev_feature = fut_bev_feature.permute(0, 3, 1, 2)        
        reward_feats = self.conv_reward_net(fut_bev_feature)  # [bsxtimes, 256, 1, 1]
        reward_feats = reward_feats.reshape(bs, times, 256, 1, 1)  # [bs, times, 256, 1, 1]
        # 编码模型a的轨迹
        model_a_trajectories_embed = model_a_trajectories.reshape(-1, 2)  # [bs*times*20, 2]
        model_a_trajectories_embed = self.trajectory_single_encoder(model_a_trajectories_embed)  # [bs*times*20, 256]
        model_a_trajectories_embed = model_a_trajectories_embed.reshape(bs, times, num_traj, 256)
        # 编码模型b的轨迹
        model_b_trajectory_embed = model_b_trajectory.reshape(-1, 2)  # [bs*times, 2]
        model_b_trajectory_embed = self.trajectory_single_encoder(model_b_trajectory_embed)  # [bs*times, 256]
        model_b_trajectory_embed = model_b_trajectory_embed.reshape(bs, times, 256)

        # 使用reward model评估模型a的轨迹
        im_traj_scores_a_list = []
        for i in range(times):
            reward_feats_i = reward_feats[:, i, ...].repeat(bs*num_traj, 1, 1, 1).squeeze(-1).squeeze(-1)
            x_cat_i = self.cat_encoder(torch.cat([reward_feats_i, model_a_trajectories_embed[:, i, ...].reshape(bs*num_traj, -1)], dim=1))
            x_i = self.reward_head(x_cat_i)
            im_traj_scores_a_i = x_i.reshape(bs, num_traj)
            im_traj_scores_a_list.append(im_traj_scores_a_i)
        im_traj_scores_a = torch.stack(im_traj_scores_a_list, dim=1)  # [bs, times, num_traj]
        # 选择最大的reward轨迹的索引
        best_traj_idx = torch.argmax(im_traj_scores_a, dim=2)  # [bs, times]
        best_traj_a = model_a_trajectories[:, torch.arange(times), best_traj_idx.squeeze(0)]  # [bs, times, 2]

        if return_distance_loss:
            # 计算模型B轨迹与最佳轨迹A的距离
            distance_loss = torch.norm(model_b_trajectory - best_traj_a, dim=1).mean()
        else:
            distance_loss = 0
        
        
        # 4. Reward对齐损失：让模型B的轨迹获得与最佳轨迹A相同的reward
        # 将模型B的轨迹送入reward model，得到reward值
        im_traj_scores_b_list = []
        for i in range(times):
            reward_feats_i = reward_feats[:, i, ...].squeeze(-1).squeeze(-1)
            x_cat_i = self.cat_encoder(torch.cat([reward_feats_i, model_b_trajectory_embed[:, i, ...]], dim=1))
            x_i = self.reward_head(x_cat_i)
            im_traj_scores_b_i = x_i.reshape(bs, 1)  # bs, 1
            im_traj_scores_b_list.append(im_traj_scores_b_i)
        im_traj_scores_b = torch.stack(im_traj_scores_b_list, dim=1).squeeze(-1)  # [bs, times, 1]-->[bs, times]
        # 得到模型A最佳轨迹的reward
        best_traj_a_reward = im_traj_scores_a[:, torch.arange(times), best_traj_idx.squeeze(0)]  # [bs, times]
        
        # Reward对齐损失
        reward_alignment_loss = torch.nn.functional.mse_loss(im_traj_scores_b, best_traj_a_reward)  # [bs, times]
        
        # 总损失
        total_loss = distance_loss + reward_alignment_loss
        
        return total_loss



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
                 extra_bev_adapter: bool = True,
                 use_cross_attention: bool = True):
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
        traj_feats = traj.reshape(bs, num_traj, 2)  # [bs*num_traj, 2]

        reward_feats = self.conv_reward_net(fut_bev_feature)  # [bs, 256, 1, 1]
        reward_feats = reward_feats.squeeze(2).squeeze(2).unsqueeze(1)
        traj_feats = self.trajectory_single_encoder(traj_feats)  # [bs, num_traj, 256]
        # WOTE 额外加了个transformer
        traj_feats = self.cluster_encoder(traj_feats)

        
        if self.use_cross_attention:
            # reward_feats 和 traj_feats 进行attention交互
            # query 是traj_feats，key和value是reward_feats
            enhanced_feats = self.cross_attention_transformer(traj_feats, reward_feats)
            x_cat = self.cat_encoder(torch.cat([traj_feats, enhanced_feats], dim=2))
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


