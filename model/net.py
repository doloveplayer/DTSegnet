from .Decoder import UpsampleDecoder
from .Encoder import DiffTransformerEncoder
from .Feature_Fusion import FeatureFusionModule
from .Bi_Transformer import BiDirectionalAttentionModule

import torch
from torchinfo import summary
import torch.nn as nn
import math
from utils import trunc_normal_


# Define Configurations
class ModelConfig:
    def __init__(self, in_chans, embed_dims, num_heads_dt, depths_dt, drop_rate, num_heads_ba, mlp_dim, depth_ba):
        self.in_chans = in_chans
        self.embed_dims = embed_dims
        self.drop_rate = drop_rate
        # DiffTransformerEncoder
        self.num_heads_dt = num_heads_dt
        self.depths_dt = depths_dt
        # BiAttention
        self.num_heads_ba = num_heads_ba
        self.mlp_dim = mlp_dim
        self.depth_ba = depth_ba


configs = {
    "v0": ModelConfig(
        in_chans=3,
        embed_dims=[32, 64, 160, 256],
        num_heads_dt=[1, 2, 4, 8],
        depths_dt=[3, 4, 6, 3],
        drop_rate=0.1,
        num_heads_ba=4,
        mlp_dim=1024,
        depth_ba=6,
    )
}


class net(nn.Module):
    def __init__(self, config_name: str,
                 num_classes=21,
                 input_size=(512, 512)):
        """
                :param Encoder: 编码器
                :param Decoder: 解码器
                :param FusionModule: 特征融合模块
                :param BiAttention: 双向注意力模块
                :param num_classes: 最终分割类别数，默认为 3
                :param input_size: 输入图像的大小 (H, W)，用于初始化位置嵌入
                """
        super(net, self).__init__()
        self.config = configs[config_name]
        self.Encoder = DiffTransformerEncoder(in_chans=self.config.in_chans, embed_dims=self.config.embed_dims,
                                              num_heads=self.config.num_heads_dt, depths=self.config.depths_dt,
                                              drop_rate=self.config.drop_rate)
        self.Decoder = UpsampleDecoder(embed_dims=self.config.embed_dims)
        self.FusionModule = FeatureFusionModule(in_channels=self.config.embed_dims, drop_rate=self.config.drop_rate)
        self.BiAttention = BiDirectionalAttentionModule(c1=self.config.embed_dims[0], c4=self.config.embed_dims[3],
                                                        num_heads=self.config.num_heads_ba, depth=self.config.depth_ba,
                                                        mlp_dim=self.config.mlp_dim)

        # 输入位置嵌入，形状 [1, C1, H, W]
        self.input_pos_embed = nn.Parameter(torch.randn(1, 3, *input_size))

        # BiAttention 的位置嵌入，形状 [1, C1, H/4, W/4]
        self.bi_attention_pos_embed = nn.Parameter(torch.randn(1, 32, input_size[0] // 4, input_size[1] // 4))

        # 通道对齐模块
        self.attention_to_decoder = nn.Conv2d(256, 32, kernel_size=1)  # 将 p1_attention 的 C4 映射到 C1
        self.output_to_classes = nn.Conv2d(32, num_classes, kernel_size=1)  # 将最终输出映射到类别数

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
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

    def forward(self, x):
        """
        :param x: 输入图像 [b, 3, H, W]
        :return: 分割掩码 [b, num_classes, H, W]
        """
        b, c, h, w = x.size()

        # 输入加上位置嵌入
        x = x + self.input_pos_embed

        # 编码器提取多尺度特征图
        # feature_maps:
        # feature_maps[0]: [b, 3, H, W]
        # feature_maps[1]: [b, 32, H/4, W/4] (高分辨率空间特征)
        # feature_maps[2]: [b, 64, H/8, W/8]
        # feature_maps[3]: [b, 160, H/16, W/16]
        # feature_maps[4]: [b, 256, H/32, W/32] (低分辨率语义特征)
        # print("Input shape:", x.shape)
        feature_maps = self.Encoder(x)
        # for i, feature_map in enumerate(feature_maps):
        #     print(f"Feature map at stage {i} shape: {feature_map.shape}")

        # 双向注意力模块，输入 [b, C1, H/4, W/4] 和 [b, C4, H/32, W/32]
        # 使用 bi_attention_pos_embed 作为位置嵌入
        # p1_attention: [b, 256, H/4, W/4] (与 Decoder 输出通道不一致)
        # P4_feature_map: [b, 256, H/32, W/32]
        p1_attention, P4_feature_map = self.BiAttention(
            feature_maps[1], feature_maps[4], self.bi_attention_pos_embed
        )

        # 特征融合模块，将 [b, 64, H/8, W/8] 和 [b, 160, H/16, W/16] 与 P4_feature_map 融合
        # fusion_map: [b, 256, H/32, W/32]
        fusion_map = self.FusionModule(feature_maps[2], feature_maps[3], P4_feature_map)

        # 解码器逐步上采样并恢复到更高分辨率
        # output: [b, 32, H/4, W/4] (解码后的高分辨率特征图)
        outputs = self.Decoder(fusion_map, [feature_maps[1], feature_maps[2], feature_maps[3], feature_maps[4]])
        output = outputs[-1]  # 取解码器的最后一个输出

        # 通道对齐：将 p1_attention 映射到 C1 以便与 output 相加
        aligned_attention = self.attention_to_decoder(p1_attention)  # [b, 32, H/4, W/4]

        # 融合解码器输出和对齐的注意力特征
        c_mask = output + aligned_attention  # [b, 32, H/4, W/4]

        # 将融合结果映射到类别数并恢复到原始分辨率
        c_mask = self.output_to_classes(c_mask)  # [b, 3, H/4, W/4]
        c_mask = nn.functional.interpolate(c_mask, size=(h, w), mode='bilinear', align_corners=False)  # [b, 3, H, W]

        return c_mask

#
# 测试代码
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 输入模拟
    batch_size = 1
    height, width = 512, 512
    x = torch.randn(batch_size, 3, height, width)
    x = x.to(device)
    model = net("v0").to(device)
    model.eval()
    # Use torchinfo to summarize the model
    print("Model Summary:")
    summary(
        model,
        input_size=(1, 3, 512, 512),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        # depth=3  # Control the depth of details in the output
    )
#
#     # 测试前向传播
#     with torch.no_grad():
#         c_mask = model(x)
#     print("输出分割掩码形状:", c_mask.shape)  # 应为 [b, 3, H, W]
