from .Decoder import UpsampleDecoder
from .Encoder import DiffTransformerEncoder, TransformerEncoder
from .Feature_Fusion import FeatureFusionModule
from .Bi_Transformer import BiDirectionalAttentionModule

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        num_heads_dt=[1, 2, 5, 8],
        depths_dt=[2, 2, 2, 2],
        drop_rate=0.5,
        num_heads_ba=2,
        mlp_dim=512,
        depth_ba=3,
    )
}


class net(nn.Module):
    def __init__(self, config_name: str,
                 num_classes=21,
                 input_size=(512, 512),
                 is_pretraining=False):
        """
                :param Encoder: 编码器
                :param Decoder: 解码器
                :param FusionModule: 特征融合模块
                :param BiAttention: 双向注意力模块
                :param num_classes: 最终分割类别数，默认为 3
                :param input_size: 输入图像的大小 (H, W)，用于初始化位置嵌入
                """
        super(net, self).__init__()
        self.is_pretraining = is_pretraining
        self.config = configs[config_name]
        self.Encoder = TransformerEncoder(in_chans=self.config.in_chans, embed_dims=self.config.embed_dims,
                                           num_heads=self.config.num_heads_dt, depths=self.config.depths_dt,
                                           drop_rate=self.config.drop_rate)
        # self.Encoder = DiffTransformerEncoder(in_chans=self.config.in_chans, embed_dims=self.config.embed_dims,
        #                                       num_heads=self.config.num_heads_dt, depths=self.config.depths_dt,
        #                                       drop_rate=self.config.drop_rate)

        self.FusionModule = FeatureFusionModule(in_channels=self.config.embed_dims, drop_rate=self.config.drop_rate)

        # 输入位置嵌入，形状 [1, C1, H, W]
        self.input_pos_embed = nn.Parameter(torch.randn(1, 3, *input_size))

        # 修改输出层：如果是分类任务，则替换为一个线性分类头
        if is_pretraining:
            self.output_to_classes = nn.Linear(
                self.config.embed_dims[3] * (input_size[0] // 32) * (input_size[0] // 32), num_classes)  # 分类头
        else:
            # BiAttention 的位置嵌入，形状 [1, C1, H/4, W/4]
            self.bi_attention_pos_embed = nn.Parameter(
                torch.randn(1, self.config.embed_dims[0], input_size[0] // 4, input_size[1] // 4))
            self.Decoder = UpsampleDecoder(embed_dims=self.config.embed_dims)
            self.BiAttention = BiDirectionalAttentionModule(c1=self.config.embed_dims[0], c4=self.config.embed_dims[3],
                                                            num_heads=self.config.num_heads_ba,
                                                            depth=self.config.depth_ba,
                                                            embed_dim=self.config.mlp_dim)
            self.output_to_classes = nn.Conv2d(self.config.embed_dims[0], num_classes, kernel_size=1)  # 将最终输出映射到类别数

    def freeze_layers(self, freeze_encoder=False, freeze_decoder=False, freeze_fusion=False, freeze_biattention=False):
        """
        冻结指定的层
        :param freeze_encoder: 是否冻结编码器层
        :param freeze_decoder: 是否冻结解码器层
        :param freeze_fusion: 是否冻结特征融合模块层
        """
        if freeze_encoder:
            for param in self.Encoder.parameters():
                param.requires_grad = False

        if freeze_decoder:
            for param in self.Decoder.parameters():
                param.requires_grad = False

        if freeze_fusion:
            for param in self.FusionModule.parameters():
                param.requires_grad = False

        if freeze_biattention:
            for param in self.BiAttention.parameters():
                param.requires_grad = False

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
        feature_maps = self.Encoder(x)

        if self.is_pretraining:
            fusion_map = self.FusionModule(feature_maps[2], feature_maps[3], feature_maps[4])
            out = fusion_map.view(b, -1)
            c_mask = self.output_to_classes(out)
        else:
            # 双向注意力模块，输入 [b, C1, H/4, W/4] 和 [b, C4, H/32, W/32]
            # 使用 bi_attention_pos_embed 作为位置嵌入
            # p1_feature_map: [b, C1, H/4, W/4] (与 Decoder 输出通道不一致)
            # P4_feature_map: [b, C4, H/32, W/32]
            p1_feature_map, P4_feature_map = self.BiAttention(
                feature_maps[1], feature_maps[4], self.bi_attention_pos_embed
            )
            # 特征融合模块，将 [b, C2, H/8, W/8] 和 [b, C3, H/16, W/16] 与 P4_feature_map 融合
            # fusion_map: [b, C4, H/32, W/32]
            fusion_map = self.FusionModule(feature_maps[2], feature_maps[3], feature_maps[4]) + P4_feature_map
            # 解码器逐步上采样并恢复到更高分辨率
            # output: [b, 32, H/4, W/4] (解码后的高分辨率特征图)
            outputs = self.Decoder(fusion_map, [feature_maps[1], feature_maps[2], feature_maps[3], feature_maps[4]])
            # outputs = self.Decoder(fusion_map)
            output = outputs[-1]  # 取解码器的最后一个输出

            # 融合解码器输出和对齐的注意力特征
            c_mask = output + p1_feature_map  # [b, 32, H/4, W/4]

            # 恢复到原始分辨率
            c_mask = nn.functional.interpolate(c_mask, size=(h, w), mode='bilinear',
                                               align_corners=False)  # [b, 32, H, W]
            # 将融合结果映射到类别数并恢复到原始分辨率
            c_mask = self.output_to_classes(c_mask)  # [b, classes, H/4, W/4]
            # 应用 softmax 激活函数，将 logits 转换为概率分布
            c_mask = F.softmax(c_mask, dim=1)  # 在类别维度上应用 softmax

        return c_mask
