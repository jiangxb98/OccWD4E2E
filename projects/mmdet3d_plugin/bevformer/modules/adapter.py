import torch.nn as nn

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
        x = x.reshape(inter_num*bs, self.bev_h, self.bev_w, c).permute(0, 3, 1, 2)
        for block in self.blocks:
            x = block(x) + x
        x = x.permute(0, 2, 3, 1).reshape(inter_num, bs, hw, c)
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
        y = self.avg_pool(x).view(b, c)  # [B,C,H,W] -> [B,C]
        # Excitation: 通过FC产生通道权重
        y = self.fc(y).view(b, c, 1, 1)  # [B,C] -> [B,C,1,1]
        # Scale: 重新加权
        return x * y.expand_as(x)  # 广播乘法，调整每个通道的权重