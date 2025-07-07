import torch.nn as nn

class FutureBEVAdapter(nn.Module):
    def __init__(self, in_channels, n_blocks=2, reduction=8):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            block = nn.Sequential(
                # 普通3x3卷积
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                # 1x1卷积整合特征
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                # SE注意力做通道调整
                SELayer(in_channels, reduction=reduction)
            )
            self.blocks.append(block)
            
    def forward(self, x):
        identity = x
        for block in self.blocks:
            x = block(x) + x
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