VIVIT_BASE_UCF101 = {
    # 输入配置
    "image_size":    224,   # 单帧分辨率，ViT 标准
    "num_frames":     16,   # 输入视频帧数，ViViT 论文标准
    "num_channels":    3,   # RGB 视频通道数
    "num_classes":   101,   # UCF101 固定 101 类

    # Tubelet 嵌入配置
    "patch_size":     16,   # 空间 patch 大小，ViT-B/16 标准
    "tube_size":       2,   # 时间 tube 大小，ViViT 论文最优

    # Transformer 主干配置（对齐 ViT-Base）
    "hidden_size":   768,   # 特征嵌入维度
    "spatial_num_layers":  12,   # 空间 Transformer 层数
    "spatial_num_heads":   12,   # 空间注意力头数
    "temporal_num_layers":  4,   # 时间 Transformer 层数
    "temporal_num_heads":  12,   # 时间注意力头数
    "qkv_bias":      True,       # QKV 线性层偏置

    # 正则化配置
    "drop_out_prob":      0.1,   # Dropout 概率
    "attn_dropout_prob":  0.0,   # Attention Dropout，默认关闭
    "drop_path_rate":     0.2,   # DropPath 最大概率
}
