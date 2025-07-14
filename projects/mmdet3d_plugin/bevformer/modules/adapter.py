import torch.nn as nn
import torch
import torch.nn.functional as F

class FutureBEVAdapter(nn.Module):
    def __init__(self, 
                 in_channels, 
                 n_blocks=2, 
                 reduction=8,
                 bev_h=200,
                 bev_w=200):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            block = nn.Sequential(
                # 普通3x3卷积
                nn.Conv2d(in_channels, in_channels//2, 3, padding=1),
                nn.ReLU(inplace=True),
                # 1x1卷积整合特征
                nn.Conv2d(in_channels//2, in_channels, 1),
                # SE注意力做通道调整
                SELayer(in_channels, reduction=reduction)
            )
            self.blocks.append(block)
            
    def forward(self, x):
        inter_num, bs, hw, c = x.shape
        x = x.reshape(inter_num*bs, self.bev_h, self.bev_w, c).permute(0, 3, 1, 2).contiguous()
        for block in self.blocks:
            x = block(x) + x
        x = x.permute(0, 2, 3, 1).reshape(inter_num, bs, hw, c).contiguous()
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        # 较小的reduction（如4或8）：更强的特征重标定能力，但参数更多
        # 较大的reduction（如16或32）：更轻量，但特征重标定能力较弱
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 降维
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),  # 升维
            nn.Sigmoid()  # 归一化到0-1
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: 空间维度压缩
        y = self.avg_pool(x).view(b, c).contiguous()  # [B,C,H,W] -> [B,C]
        # Excitation: 通过FC产生通道权重
        y = self.fc(y).view(b, c, 1, 1).contiguous()  # [B,C] -> [B,C,1,1]
        # Scale: 重新加权
        return x * y.expand_as(x)  # 广播乘法，调整每个通道的权重
    




class TemporalFusionAdapter(nn.Module):
    def __init__(self, 
                 in_channels,
                 n_future=6,
                 reduction=8,
                 bev_h=200,
                 bev_w=200
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.n_future = n_future
        self.bev_h = bev_h
        self.bev_w = bev_w
        
        # 时间编码
        self.temporal_encoding = nn.Parameter(torch.zeros(n_future, in_channels))
        nn.init.normal_(self.temporal_encoding, mean=0, std=1.0)
        
        # 时序特征处理
        self.temporal_fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 时序注意力
        self.temporal_attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )
        
        # 添加最终融合层
        self.final_fusion = nn.Sequential(
            # 先融合
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            # nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # 再调整
            nn.Conv2d(in_channels, in_channels, 1),
            # nn.BatchNorm2d(in_channels)
        )
        
    def forward(self, future_feats):
        """
        Args:
            future_feats: 未来帧BEV特征(bs, 6, HxW, C)
        """
        B, T, HW, C = future_feats.shape
        assert B == 1, "B must be 1"
        future_feats = future_feats.squeeze(0).contiguous()
        future_feats = future_feats.reshape(T, self.bev_h, self.bev_w, C).permute(0, 3, 1, 2).contiguous()
        
        # 1. 处理每个时序特征
        processed_feats = []
        for t, feat in enumerate(future_feats):
            feat = feat.unsqueeze(0)
            # 添加时间编码
            time_code = self.temporal_encoding[t].view(1, -1, 1, 1).expand(B, -1, self.bev_h, self.bev_w)
            feat = feat + time_code
            
            # 处理特征
            feat = self.temporal_fusion(feat)
            processed_feats.append(feat)
            
        # 2. 计算时序注意力权重
        stacked_feats = torch.stack(processed_feats, dim=1)  # (B, T, C, H, W)
        global_feats = torch.mean(stacked_feats, dim=[3, 4])  # (B, T, C)
        
        # 3. 计算时序注意力权重
        temporal_weights = self.temporal_attention(global_feats)  # (B, T, C)
        temporal_weights = temporal_weights.unsqueeze(-1).unsqueeze(-1)  # (B, T, C, 1, 1)
        
        # 4. 加权特征
        weighted_feats = stacked_feats * temporal_weights
        
        # 5. 先sum再通过融合层
        sum_feat = torch.sum(weighted_feats, dim=1)  # (B, C, H, W)
        fused_feat = self.final_fusion(sum_feat)
        
        # 6. 残差连接
        fused_feat = fused_feat + sum_feat

        # reshape for output
        # bs, C, H, W
        fused_feat = fused_feat.permute(0, 2, 3, 1).reshape(B, HW, C).contiguous()
        
        return fused_feat