import torch
import math
import torch.nn as nn
from utils import trunc_normal_


class UpsampleDecoder(nn.Module):
    def __init__(self, embed_dims=[32, 64, 160, 256]):
        """
        上采样解码器模块，带有跳跃连接。
        :param embed_dims: 每一层的通道数列表，从 H/4 到 H/32 的通道数顺序 [H/4通道, H/8通道, H/16通道, H/32通道]
        """
        super(UpsampleDecoder, self).__init__()

        # 构建上采样模块
        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 两倍上采样
                nn.Conv2d(embed_dims[len(embed_dims) - i - 1] * 2, embed_dims[len(embed_dims) - i - 2], kernel_size=3,
                          stride=1, padding=1),  # 融合跳跃连接
                nn.BatchNorm2d(embed_dims[len(embed_dims) - i - 2]),  # 批归一化
                nn.ReLU(inplace=True)  # 激活函数
            )
            for i in range(len(embed_dims) - 1)  # 每次两倍上采样直到 H/4
        ])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, skip_features):
        """
        前向传播
        :param skip_features: 跳跃连接特征图列表 [H/4特征图, H/8特征图, H/16特征图, H/32特征图]
        :return: 恢复的特征图列表
        """
        outputs = []
        for i, block in enumerate(self.upsample_blocks):
            # 跳跃连接特征图从高分辨率到低分辨率依次使用
            skip = skip_features[len(skip_features) - 1 - i]  # 对应 H/16 -> H/8 -> H/4
            x = torch.cat([x, skip], dim=1)  # 拼接跳跃连接
            x = block(x)  # 上采样并卷积
            outputs.append(x)  # 保存结果
        return outputs


# 测试代码
if __name__ == "__main__":
    # 模拟输入特征图，顺序为 [H/4, H/8, H/16, H/32]
    skip_h4 = torch.rand(2, 32, 64, 64)  # [b=2, H/4通道=32, H/4=64, W/4=64]
    skip_h8 = torch.rand(2, 64, 32, 32)  # [b=2, H/8通道=64, H/8=32, W/8=32]
    skip_h16 = torch.rand(2, 160, 16, 16)  # [b=2, H/16通道=160, H/16=16, W/16=16]
    skip_h32 = torch.rand(2, 256, 8, 8)  # [b=2, H/32通道=256, H/32=8, W/32=8]

    # 初始化解码器
    decoder = UpsampleDecoder(embed_dims=[32, 64, 160, 256])  # embed_dims 顺序为 [H/4, H/8, H/16, H/32]

    # 前向传播
    outputs = decoder(skip_h32, [skip_h4, skip_h8, skip_h16, skip_h32])

    # 打印输出特征图形状
    for i, feature in enumerate(outputs):
        print(f"输出特征图 {i + 1} 形状:", feature.shape)
