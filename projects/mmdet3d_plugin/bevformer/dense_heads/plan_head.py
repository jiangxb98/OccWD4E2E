import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmcv.cnn import xavier_init, constant_init
from mmdet.models import HEADS, build_head, build_loss
from mmdet.models.utils import build_transformer
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner.base_module import BaseModule
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import xavier_init
from torch.nn.init import normal_
from einops import rearrange
import copy
from projects.mmdet3d_plugin.bevformer.modules.collision_optimization import CollisionNonlinearOptimizer
from projects.mmdet3d_plugin.bevformer.utils.cost import Cost_Function

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


@HEADS.register_module()
class PoseEncoder(BaseModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers=2,
        num_modes=3,
        num_fut_ts=1,
        init_cfg=None
    ):
        super().__init__(init_cfg)
        self.num_modes = num_modes
        self.num_fut_ts = num_fut_ts
        assert num_fut_ts == 1
        
        pose_encoder = []

        for _ in range(num_layers - 1):
            pose_encoder.extend([
                nn.Linear(in_channels, out_channels),
                nn.ReLU(True)])
            in_channels = out_channels
        pose_encoder.append(nn.Linear(out_channels, out_channels))
        self.pose_enc = nn.Sequential(*pose_encoder)
    
    def forward(self,x):
        # x: N*2,
        pose_feat = self.pose_enc(x)
        return pose_feat


@HEADS.register_module()
class PoseDecoder(BaseModule):

    def __init__(
            self, 
            in_channels,
            num_layers=2,
            num_modes=3,
            num_fut_ts=1,
            init_cfg = None):
        super().__init__(init_cfg)

        self.num_modes = num_modes
        self.num_fut_ts = num_fut_ts
        assert num_fut_ts == 1

        pose_decoder = []
        for _ in range(num_layers - 1):
            pose_decoder.extend([
                nn.Linear(in_channels, in_channels),
                nn.ReLU(True)])
        pose_decoder.append(nn.Linear(in_channels, num_modes*num_fut_ts*2))
        self.pose_dec = nn.Sequential(*pose_decoder)

    def forward(self, x):
        # x: ..., D
        rel_pose = self.pose_dec(x).reshape(*x.shape[:-1], self.num_modes, 2)
        rel_pose = rel_pose.squeeze(1)
        return rel_pose

@HEADS.register_module()
class PlanHead_v1(BaseModule):
    """Head of Ego-Trajectory Planning.
    """

    def __init__(self,
                 # Architecture.
                 with_adapter=True,
                 transformer=None,
                 plan_grid_conf=None,

                 # class
                 instance_cls = [2,3,4,5,6,7,9,10],
                 drivable_area_cls = [11],

                 # positional encoding
                 bev_h=200,
                 bev_w=200,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),

                 # loss
                 loss_planning=None,
                 loss_collision=None,

                 output_multi_traj=False,
                 sample_traj_nums=1,
                 use_sim_reward=False,
                 use_im_reward=False,
                 plan_query_nums=1,
                 plan_query_mode='first',  # 'mean', 'max', 'min', 'first'
                 plan_traj_for_sim_reward_epoch=999999,
                 use_gt_occ_for_sim_reward=False,
                 random_select=False,
                 sim_reward_nums=1,
                 *args,
                 **kwargs):

        # BEV configuration of reference frame.
        super().__init__(**kwargs)
        self.cost_function = Cost_Function(plan_grid_conf)
        self.output_multi_traj = output_multi_traj
        self.sample_traj_nums = sample_traj_nums
        self.use_sim_reward = use_sim_reward
        self.use_im_reward = use_im_reward
        self.plan_query_mode = plan_query_mode
        self.plan_query_nums = plan_query_nums
        self.plan_traj_for_sim_reward_epoch = plan_traj_for_sim_reward_epoch
        self.random_select = random_select
        self.use_gt_occ_for_sim_reward = use_gt_occ_for_sim_reward
        self.sim_reward_nums = sim_reward_nums
        # cls
        self.instance_cls = torch.tensor(instance_cls, requires_grad=False)  # 'bicycle', 'bus', 'car', 'construction', 'motorcycle', 'pedestrian', 'trailer', 'truck'
        self.drivable_area_cls = torch.tensor(drivable_area_cls, requires_grad=False)  # 'drivable_area'

        # sample trajs
        self.sample_num = 1800
        assert self.sample_num % 3 == 0
        self.num = int(self.sample_num / 3)

        bevformer_bev_conf = {
            'xbound': [-51.2, 51.2, 0.512],
            'ybound': [-51.2, 51.2, 0.512],
            'zbound': [-10.0, 10.0, 20.0],
        }
        self.bev_sampler =  BevFeatureSlicer(bevformer_bev_conf, plan_grid_conf)

        # TODO: reimplement it with down-scaled feature_map
        self.embed_dims = transformer.embed_dims
        self.with_adapter = with_adapter
        if with_adapter:
            bev_adapter_block = nn.Sequential(
                nn.Conv2d(self.embed_dims, self.embed_dims // 2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims // 2, self.embed_dims, kernel_size=1),
            )
            N_Blocks = 3
            bev_adapter = [copy.deepcopy(bev_adapter_block) for _ in range(N_Blocks)]
            self.bev_adapter = nn.Sequential(*bev_adapter)
        
        self.costvolume_head = nn.Sequential(
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.embed_dims, 1, kernel_size=1, padding=0),
        )

        # build encoder
        self.pose_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.ReLU(True),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        # build transformer architecture.
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)

        # build decoder
        self.planning_steps = 1
        self.reg_branch = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.planning_steps * 2),
        )

        # loss
        self.loss_planning = build_loss(loss_planning)
        self.loss_collision = []
        for cfg in loss_collision:
            self.loss_collision.append(build_loss(cfg))
        self.loss_collision = nn.ModuleList(self.loss_collision)

        self._init_layers()

    def _init_layers(self):
        """Initialize BEV prediction head."""
        # plan query for the next frame.
        self.plan_embedding = nn.Embedding(self.plan_query_nums, self.embed_dims)
        # navi embed.
        self.navi_embedding = nn.Embedding(3, self.embed_dims)
        # mlp_fuser
        fuser_dim = 2
        self.mlp_fuser = nn.Sequential(
                nn.Linear(self.embed_dims*fuser_dim, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
            )

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        try:
            self.transformer.init_weights()
            # Initialization of embeddings.
            normal_(self.plan_embedding)
            normal_(self.navi_embedding)
            xavier_init(self.mlp_fuser, distribution='uniform', bias=0.)
        except:
            pass
    
    def loss(self, outs_planning, sdc_planning, sdc_planning_mask, future_gt_bbox=None):
        """
            outs_planning:      B,Lout,mode=1,2
            sdc_planning:       B,Lout,3
            sdc_planning_mask:  B,Lout,2   valid_frmae=1
            future_gt_bbox:     Lout*[N_box个bbox_3d]
        """
        loss_dict = dict()
        for i in range(len(self.loss_collision)):
            loss_collision = self.loss_collision[i](outs_planning, sdc_planning[..., :3], torch.any(sdc_planning_mask, dim=-1), future_gt_bbox)
            loss_dict[f'loss_collision_{i}'] = loss_collision          
        loss_ade = self.loss_planning(outs_planning, sdc_planning, torch.any(sdc_planning_mask, dim=-1))
        loss_dict.update(dict(loss_ade=loss_ade))
        return loss_dict

    def loss_cost(self, trajs, gt_trajs, cost_volume, instance_occupancy, drivable_area):
        '''
        trajs: torch.Tensor (B, N, 3)
        gt_trajs: torch.Tensor (B, 3)
        cost_volume: torch.Tensor (B, 200, 200)
        instance_occupancy: torch.Tensor(B, 200, 200)
        drivable_area: torch.Tensor(B, 200, 200)
        '''
        if gt_trajs.ndim == 2:
            gt_trajs = gt_trajs[:, None]

        gt_cost_fo = self.cost_function(cost_volume, gt_trajs[:,:,:2], instance_occupancy, drivable_area)

        sm_cost_fo = self.cost_function(cost_volume, trajs[:,:,:2], instance_occupancy, drivable_area)

        L = F.relu(gt_cost_fo - sm_cost_fo)  # 为什么这里用relu?，如果sm的更好，那么就为0，如果gt的更好，那么就为relu(gt-sm)

        return torch.mean(L)

    def cal_sim_reward(self, trajs, gt_trajs, cost_volume, instance_occupancy, drivable_area):
        '''
        这个是为了计算sim reward的
        trajs: torch.Tensor (B, N, 3)
        gt_trajs: torch.Tensor (B, 3)
        cost_volume: torch.Tensor (B, 200, 200)
        instance_occupancy: torch.Tensor(B, 200, 200)
        drivable_area: torch.Tensor(B, 200, 200)
        '''
        if gt_trajs.ndim == 2:
            gt_trajs = gt_trajs[:, None]

        # if self.use_gt_occ_for_sim_reward and self.training:
        #     # 训练阶段可以使用GT的occupancy来计算sim reward
        #     cost = self.cost_function.forward_sim(gt_trajs[:,:,:2], instance_occupancy, drivable_area)
        # else:
        cost = self.cost_function.forward_sim(trajs[:,:,:2], instance_occupancy, drivable_area)

        # cost=0表示没有碰撞
        pos_mask = cost <= 0
        neg_mask = cost > 0
        
        cost[pos_mask] = 1
        cost[neg_mask] = 0

        return cost
    
    def select(self, trajs, cost_volume, instance_occupancy, drivable_area, k=1):
        '''
        trajs: torch.Tensor (B, N, 3)
        cost_volume: torch.Tensor (B, 200, 200)
        instance_occupancy: torch.Tensor(B, 200, 200)
        drivable_area: torch.Tensor(B, 200, 200)
        '''
        sm_cost_fo = self.cost_function(cost_volume, trajs[:,:,:2], instance_occupancy, drivable_area)

        CS = sm_cost_fo
        CC, KK = torch.topk(CS, k, dim=-1, largest=False)   # B,N_sample

        ii = torch.arange(len(trajs))
        select_traj = trajs[ii[:,None], KK].squeeze(1) # (B, 3)

        return select_traj

    def select_multi_traj(self, trajs, cost_volume, instance_occupancy, drivable_area, k=1, random_select=False):
        '''
        trajs: torch.Tensor (B, N, 3)
        cost_volume: torch.Tensor (B, 200, 200)
        instance_occupancy: torch.Tensor(B, 200, 200)
        drivable_area: torch.Tensor(B, 200, 200)
        '''
        sm_cost_fo = self.cost_function(cost_volume, trajs[:,:,:2], instance_occupancy, drivable_area)
        CS = sm_cost_fo

        if random_select:
            sample_best_k = 5
            _, KK = torch.topk(CS, sample_best_k, dim=-1, largest=False)   # B,N_sample
            ii = torch.arange(len(trajs))
            select_traj = trajs[ii[:,None], KK].squeeze(1) # (B, N_sample, 3)
            reset_k = k - sample_best_k
            select_traj_ = trajs[:, torch.randperm(trajs.shape[1])[:reset_k]]
            select_traj = torch.cat([select_traj, select_traj_], dim=1)
        else:
            CC, KK = torch.topk(CS, k, dim=-1, largest=False)   # B,N_sample
            ii = torch.arange(len(trajs))
            select_traj = trajs[ii[:,None], KK].squeeze(1) # (B, N_sample, 3)

        return select_traj

    @auto_fp16(apply_to=('bev_feats'))
    def forward(self, bev_feats, trajs, sem_occupancy, command, gt_trajs=None, multi_traj=False, training_epoch=0):
        """ Forward function for each frame.

        Args:
            bev_feats: bev feats of current frame, with shape of (bs, bev_h * bev_w, embed_dim)
            trajs:    bs, sample_num, 3     current -> next frmae, under ref_lidar
            gt_trajs: bs, 2                 current -> next frame, under ref_lidar
            sem_occ:  bs, H,W,D             semantic occupancy
            command: bs                    0:Right  1:Left  2:Forward
            gt_trajs: bs, 3                 current -> next frame, under ref_liar
        """
        cur_trajs = []
        # 根据导航命令选择相应的轨迹子集, 这个command就是gt信息吧，导航信息算是gt吗？
        for i in range(len(command)):
            command_i = command[i]
            traj = trajs[i]
            # 根据导航命令选择相应的轨迹子集
            if command_i == 1:    # Left
                cur_trajs.append(traj[:self.num].repeat(3, 1))
            elif command_i == 2:  # Forward
                cur_trajs.append(traj[self.num:self.num * 2].repeat(3, 1))
            elif command_i == 0:  # Right
                cur_trajs.append(traj[self.num * 2:].repeat(3, 1))
            else:
                cur_trajs.append(traj)
        cur_trajs = torch.stack(cur_trajs)  # B,N_sample,3

        # bev_feat
        # grid sample
        bev_feats = rearrange(bev_feats, 'b (w h) c -> b c h w', h=self.bev_h, w=self.bev_w)
        bev_feats = self.bev_sampler(bev_feats)
        # plugin adapter
        if self.with_adapter:
            bev_feats = bev_feats + self.bev_adapter(bev_feats)  # residual connection

        # cost_volume
        # 通过costvolume_head计算成本体积，表示每个位置的成本
        costvolume = self.costvolume_head(bev_feats).squeeze(1) # b,h,w
        # instance_occupancy
        instance_occupancy = torch.isin(sem_occupancy, self.instance_cls.to(sem_occupancy)).float()
        instance_occupancy = instance_occupancy.max(-1)[0].detach()  # b,h,w
        # drivable_area
        drivable_area = torch.isin(sem_occupancy, self.drivable_area_cls.to(sem_occupancy)).float()
        drivable_area = drivable_area.max(-1)[0].detach()   # b,h,w

        if self.training:
            loss = self.loss_cost(cur_trajs, gt_trajs, costvolume, instance_occupancy, drivable_area)
        else:
            loss = None

        if self.output_multi_traj and multi_traj:
            # 去掉多余的重复轨迹用来计算sim_reward
            cur_trajs = cur_trajs[:, :self.num, :]
            # 1. select_traj
            select_traj_ = self.select_multi_traj(cur_trajs, costvolume, instance_occupancy, drivable_area, self.sample_traj_nums, self.random_select)  # B,num_traj,3
            # 2. random select traj_nums from cur_trajs
            # select_traj_ = cur_trajs[torch.randperm(cur_trajs.shape[1])[:, :self.sample_traj_nums]]

            # select_traj -> encoder
            select_traj = self.pose_encoder(select_traj_.float())   # B,1,C
            select_traj = select_traj.permute(1, 0, 2)

            # bev refine
            bs = bev_feats.shape[0]
            dtype = bev_feats.dtype
            bev_feats = rearrange(bev_feats, 'b c h w -> b (w h) c')

            # # 1. plan_query
            plan_query = self.plan_embedding.weight.to(dtype)
            if plan_query.shape[0] == self.sample_traj_nums:
                plan_query = plan_query[:, None]
            else:
                plan_query = plan_query[None].repeat(self.sample_traj_nums, 1, 1)
            # navi_embed
            navi_embed = self.navi_embedding.weight[command]
            navi_embed = navi_embed[None].repeat(self.sample_traj_nums, 1, 1)
            # mlp_fuser
            plan_query = torch.cat([plan_query, navi_embed], dim=-1)
            plan_query = self.mlp_fuser(plan_query)

            # 3. bev_feats
            bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                                device=plan_query.device).to(dtype)
            bev_pos = self.positional_encoding(bev_mask).to(dtype)  # bs, bev_dims, bev_h, bev_w

            # 5. do transformer layers to get pose features.
            plan_query = self.transformer(
                plan_query,
                bev_feats.expand(self.sample_traj_nums, *bev_feats.shape[1:]),
                prev_pose=select_traj,
                bev_pos=bev_pos.expand(self.sample_traj_nums, *bev_pos.shape[1:]),
            )
            plan_query = plan_query.permute(1, 0, 2)
            # for list
            # plan_query_list = []
            # for j in range(self.sample_traj_nums):
            #     plan_query_ = self.transformer(
            #         plan_query,
            #         bev_feats,
            #         prev_pose=select_traj[:,:,j],
            #         bev_pos=bev_pos.repeat(self.sample_traj_nums, 1, 1, 1),
            #     )
            #     plan_query_list.append(plan_query_)
            # plan_query = torch.cat(plan_query_list, dim=1)  # bs, traj_nums, C

            # 6. plan regression
            bs, sample_traj_nums, C = plan_query.shape
            next_pose = self.reg_branch(plan_query).view((-1, self.planning_steps, 2))   # B*traj_nums,mode=1,2

            # 计算sim_rewards
            sim_rewards = None
            if self.use_sim_reward and self.training:
                if training_epoch < self.plan_traj_for_sim_reward_epoch:
                    # 使用初始化的多模轨迹来计算sim_reward
                    sim_rewards = self.cal_sim_reward(select_traj_, gt_trajs, None, instance_occupancy, drivable_area)
                else:
                    # 使用预测的多模轨迹来计算sim_reward（加这个的原因是，尝试用预测的结果来计算sim_reward）
                    sim_rewards = self.cal_sim_reward(next_pose.detach().clone().transpose(1, 0), gt_trajs, None, instance_occupancy, drivable_area)

            return next_pose, loss, select_traj_.to(torch.float32), sim_rewards
        else:
            # select_traj
            select_traj = self.select(cur_trajs, costvolume, instance_occupancy, drivable_area)  # B,3
            # select_traj -> encoder
            select_traj = self.pose_encoder(select_traj.float()).unsqueeze(1)   # B,1,C

            # bev refine
            bs = bev_feats.shape[0]
            dtype = bev_feats.dtype
            bev_feats = rearrange(bev_feats, 'b c h w -> b (w h) c')

            # 1. plan_query
            plan_query = self.plan_embedding.weight.to(dtype)
            plan_query = plan_query[None]
            # 当使用多个plan_query时，取平均
            if plan_query.shape[1] > 1:
                if self.plan_query_mode == 'first':
                    plan_query = plan_query[:, 0, :][:, None]
                elif self.plan_query_mode == 'mean':
                    plan_query = plan_query.mean(1)[:, None]
                elif self.plan_query_mode == 'max':
                    plan_query = plan_query.max(1)[:, None]
                elif self.plan_query_mode == 'min':
                    plan_query = plan_query.min(1)[:, None]
            
            # navi_embed
            navi_embed = self.navi_embedding.weight[command]
            navi_embed = navi_embed[None]
            # mlp_fuser
            plan_query = torch.cat([plan_query, navi_embed], dim=-1)
            plan_query = self.mlp_fuser(plan_query)

            # 3. bev_feats
            bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                                device=plan_query.device).to(dtype)
            bev_pos = self.positional_encoding(bev_mask).to(dtype)  # bs, bev_dims, bev_h, bev_w

            # 5. do transformer layers to get pose features.
            plan_query = self.transformer(
                plan_query,
                bev_feats,
                prev_pose=select_traj,
                bev_pos=bev_pos,
            )
            
            # 6. plan regression
            next_pose = self.reg_branch(plan_query).view((-1, self.planning_steps, 2))   # B,mode=1,2
            return next_pose, loss

@HEADS.register_module()
class PlanHead_v2(BaseModule):
    """Head of Ego-Trajectory Planning.
    """

    def __init__(self,
                 # Architecture.
                 with_adapter=True,
                 transformer=None,
                 plan_grid_conf=None,

                 # positional encoding
                 bev_h=200,
                 bev_w=200,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),

                 # loss
                 loss_planning=None,
                 loss_collision=None,

                 *args,
                 **kwargs):

        # BEV configuration of reference frame.
        super().__init__(**kwargs)
        bevformer_bev_conf = {
            'xbound': [-51.2, 51.2, 0.512],
            'ybound': [-51.2, 51.2, 0.512],
            'zbound': [-10.0, 10.0, 20.0],
        }
        self.bev_sampler =  BevFeatureSlicer(bevformer_bev_conf, plan_grid_conf)

        # TODO: reimplement it with down-scaled feature_map
        self.embed_dims = transformer.embed_dims
        self.with_adapter = with_adapter
        if with_adapter:
            bev_adapter_block = nn.Sequential(
                nn.Conv2d(self.embed_dims, self.embed_dims // 2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims // 2, self.embed_dims, kernel_size=1),
            )
            N_Blocks = 3
            bev_adapter = [copy.deepcopy(bev_adapter_block) for _ in range(N_Blocks)]
            self.bev_adapter = nn.Sequential(*bev_adapter)


        # build transformer architecture.
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)

        # build decoder
        self.planning_steps = 1
        self.reg_branch = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.planning_steps * 2),
        )

        # loss
        self.loss_planning = build_loss(loss_planning)
        self.loss_collision = []
        for cfg in loss_collision:
            self.loss_collision.append(build_loss(cfg))
        self.loss_collision = nn.ModuleList(self.loss_collision)

        self._init_layers()

    def _init_layers(self):
        """Initialize BEV prediction head."""
        # plan query for the next frame.
        self.plan_embedding = nn.Embedding(1, self.embed_dims)
        # navi embed.
        self.navi_embedding = nn.Embedding(3, self.embed_dims)
        # mlp_fuser
        fuser_dim = 2
        self.mlp_fuser = nn.Sequential(
                nn.Linear(self.embed_dims*fuser_dim, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
            )

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        try:
            self.transformer.init_weights()
            # Initialization of embeddings.
            normal_(self.plan_embedding)
            normal_(self.navi_embedding)
            xavier_init(self.mlp_fuser, distribution='uniform', bias=0.)
        except:
            pass
    
    def loss(self, outs_planning, sdc_planning, sdc_planning_mask, future_gt_bbox=None):
        """
            outs_planning:      B,Lout,mode=1,2
            sdc_planning:       B,Lout,3
            sdc_planning_mask:  B,Lout,2
            future_gt_bbox:     Lout*[N_box个bbox_3d]
        """
        loss_dict = dict()
        for i in range(len(self.loss_collision)):
            loss_collision = self.loss_collision[i](outs_planning, sdc_planning[..., :3], torch.any(sdc_planning_mask, dim=-1), future_gt_bbox)
            loss_dict[f'loss_collision_{i}'] = loss_collision          
        loss_ade = self.loss_planning(outs_planning, sdc_planning, torch.any(sdc_planning_mask, dim=-1))
        loss_dict.update(dict(loss_ade=loss_ade))
        return loss_dict

    @auto_fp16(apply_to=('bev_feats'))
    def forward(self, bev_feats, command):
        """ Forward function for each frame.

        Args:
            bev_feats: bev feats of current frame, with shape of (bs, bev_h * bev_w, embed_dim)
            command: bs                    0:Right  1:Left  2:Forward
        """
        # bev_feat
        # grid sample
        bev_feats = rearrange(bev_feats, 'b (w h) c -> b c h w', h=self.bev_h, w=self.bev_w)
        bev_feats = self.bev_sampler(bev_feats)
        # plugin adapter
        if self.with_adapter:
            bev_feats = bev_feats + self.bev_adapter(bev_feats)  # residual connection

        # bev refine
        bs = bev_feats.shape[0]
        dtype = bev_feats.dtype
        bev_feats = rearrange(bev_feats, 'b c h w -> b (w h) c')

        # # 1. plan_query
        # plan_query = select_traj
        plan_query = self.plan_embedding.weight.to(dtype)
        plan_query = plan_query[None]
        # navi_embed
        navi_embed = self.navi_embedding.weight[command]
        navi_embed = navi_embed[None]
        # mlp_fuser
        plan_query = torch.cat([plan_query, navi_embed], dim=-1)
        plan_query = self.mlp_fuser(plan_query)

        # 3. bev_feats
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=plan_query.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)  # bs, bev_dims, bev_h, bev_w

        # 5. do transformer layers to get pose features.
        plan_query = self.transformer(
            plan_query,
            bev_feats,
            bev_pos=bev_pos,
        )
        
        # 6. plan regression
        next_pose = self.reg_branch(plan_query).view((-1, self.planning_steps, 2))   # B,mode=1,2
        return next_pose