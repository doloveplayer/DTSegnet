import math
import torch.nn as nn
from utils import trunc_normal_


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels=[32, 64, 160, 256], drop_rate=0.1):
        """
        初始化特征融合模块。
        :param in_channels_p2: P2 的输入通道数
        :param in_channels_p3: P3 的输入通道数
        :param in_channels_p4: P4 的输入通道数
        :param drop_rate: dropout 的比率，默认为 0.0（不使用）
        """
        super(FeatureFusionModule, self).__init__()

        # Conv to downsample and align P2 to P3's channels
        self.conv_p2_to_p3 = nn.Sequential(
            nn.Conv2d(in_channels[1], in_channels[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels[2]),
            nn.ReLU()
        )

        # Conv to downsample and align (P2 + P3) to P4's channels
        self.conv_p3_to_p4 = nn.Sequential(
            nn.Conv2d(in_channels[2], in_channels[3], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels[3]),
            nn.ReLU()
        )

        # Optional dropout for regularization
        self.dropout = nn.Dropout2d(p=drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, P2, P3, P4):
        """
        前向传播逻辑。
        :param P2: 特征图 P2 (高分辨率，低通道数)
        :param P3: 特征图 P3 (中分辨率，中通道数)
        :param P4: 特征图 P4 (低分辨率，高通道数)
        :return: 融合后的特征图
        """
        # Step 1: Downsample P2 and align to P3's channel size
        P2_downsampled = self.conv_p2_to_p3(P2)
        fused_p2_p3 = P2_downsampled + P3

        # Optional dropout
        fused_p2_p3 = self.dropout(fused_p2_p3)

        # Step 2: Downsample (P2 + P3) and align to P4's channel size
        fused_p2_p3_downsampled = self.conv_p3_to_p4(fused_p2_p3)
        fused_result = fused_p2_p3_downsampled + P4

        # Optional dropout
        fused_result = self.dropout(fused_result)

        return fused_result
