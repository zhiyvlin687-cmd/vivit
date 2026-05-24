import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


class DropPath(nn.Module):
    def __init__(self, droppath_prob: float = 0.0):
        super().__init__()
        self.droppath_prob = droppath_prob

    def forward(self, x):
        if self.droppath_prob == 0 or not self.training:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        keep_prob = 1 - self.droppath_prob
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        random_tensor.div_(keep_prob)
        return x * random_tensor


class MultiHeadAttention(nn.Module):
    def __init__(self, config, num_attention_head):
        super().__init__()
        self.hidden_size  = config["hidden_size"]
        self.num_head     = num_attention_head
        self.head_size    = self.hidden_size // self.num_head
        self.qkv          = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=config["qkv_bias"])
        self.proj         = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_dropout_p = config["attn_dropout_prob"]
        self.proj_dropout = nn.Dropout(config["drop_out_prob"])
        self.q_norm       = nn.LayerNorm(self.head_size)
        self.k_norm       = nn.LayerNorm(self.head_size)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_head, self.head_size).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout_p if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
        attn_proj   = self.proj(attn_output)
        attn_proj   = self.proj_dropout(attn_proj)
        return attn_proj


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense1  = nn.Linear(config["hidden_size"], config["hidden_size"] * 4)
        self.dense2  = nn.Linear(config["hidden_size"] * 4, config["hidden_size"])
        self.dropout = nn.Dropout(config["drop_out_prob"])
        self.act     = nn.GELU()

    def forward(self, x):
        x = self.dense1(x)
        x = self.act(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config, droppath_prob, num_attention_head):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(config, num_attention_head)
        self.layernorm1 = nn.LayerNorm(config["hidden_size"])
        self.mlp        = MLP(config)
        self.layernorm2 = nn.LayerNorm(config["hidden_size"])
        self.DropPath1  = DropPath(droppath_prob)
        self.DropPath2  = DropPath(droppath_prob)

    def forward(self, x):
        x = x + self.DropPath1(self.multi_head_attention(self.layernorm1(x)))
        x = x + self.DropPath2(self.mlp(self.layernorm2(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, config, num_hidden_layers, droppath_prob, num_attention_head):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(config, droppath_prob[i], num_attention_head)
            for i in range(num_hidden_layers)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class CNNEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size  = config["hidden_size"]
        self.num_channels = config["num_channels"]
        self.patch_size   = config["patch_size"]
        self.image_size   = config["image_size"]
        self.tube_size    = config["tube_size"]
        self.kernel_size  = (3, 3, 3)
        self.num_frames   = config["num_frames"]
        self.stem_ch1     = self.hidden_size // 8   # 96
        self.stem_ch2     = self.hidden_size // 4   # 192

        # shortcut：从原始输入直接 patchify，保证残差信息充分
        self.shortcut = nn.Conv3d(
            self.num_channels, self.hidden_size,
            kernel_size=(self.tube_size, self.patch_size, self.patch_size),
            stride=(self.tube_size, self.patch_size, self.patch_size),
        )

        # 主路径：三层逐步提取时空特征，通道数平滑递进
        self.cnn = nn.Sequential(
            nn.Conv3d(self.num_channels, self.stem_ch1,
                      kernel_size=self.kernel_size, stride=1, padding=1),
            nn.BatchNorm3d(self.stem_ch1),
            nn.GELU(),
            nn.Conv3d(self.stem_ch1, self.stem_ch2,
                      kernel_size=self.kernel_size, stride=1, padding=1),
            nn.BatchNorm3d(self.stem_ch2),
            nn.GELU(),
            nn.Conv3d(self.stem_ch2, self.hidden_size,
                      kernel_size=(self.tube_size, self.patch_size, self.patch_size),
                      stride=(self.tube_size, self.patch_size, self.patch_size)),
        )

        # 动态推算 num_patches，不依赖手动计算公式
        with torch.no_grad():
            sample = torch.zeros(
                1, self.num_channels, self.num_frames,
                self.image_size, self.image_size
            )
            self.num_patches = self.cnn(sample).flatten(2).shape[2]

    def forward(self, x):
        res = self.shortcut(x)
        x   = self.cnn(x)
        return (x + res).flatten(2).transpose(1, 2)


class ViViT_Factorised_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_channels       = config["num_channels"]
        self.image_size         = config["image_size"]
        self.num_frames         = config["num_frames"]
        self.num_classes        = config["num_classes"]
        self.patch_size         = config["patch_size"]
        self.tube_size          = config["tube_size"]
        self.hidden_size        = config["hidden_size"]
        self.spatial_num_layers = config["spatial_num_layers"]
        self.spatial_num_heads  = config["spatial_num_heads"]
        self.temporal_num_layers = config["temporal_num_layers"]
        self.temporal_num_heads  = config["temporal_num_heads"]

        self.embedded            = CNNEmbedding(config)
        self.num_temporal_tokens = self.num_frames // self.tube_size
        self.num_spatial_tokens  = self.embedded.num_patches // self.num_temporal_tokens

        self.spatial_pos_embed  = nn.Parameter(torch.zeros(1, self.num_spatial_tokens,  self.hidden_size))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, self.num_temporal_tokens, self.hidden_size))
        self.pos_drop           = nn.Dropout(config["drop_out_prob"])

        self.spatial_dpr  = [x.item() for x in torch.linspace(0, config["drop_path_rate"], self.spatial_num_layers)]
        self.temporal_dpr = [x.item() for x in torch.linspace(0, config["drop_path_rate"], self.temporal_num_layers)]

        self.spatial_encoder  = Encoder(config, self.spatial_num_layers,  self.spatial_dpr,  self.spatial_num_heads)
        self.temporal_encoder = Encoder(config, self.temporal_num_layers, self.temporal_dpr, self.temporal_num_heads)
        self.head             = nn.Linear(self.hidden_size, self.num_classes)
        self.spatial_norm     = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.temporal_norm    = nn.LayerNorm(self.hidden_size, eps=1e-6)

        self.apply(self._init_weights)
        # nn.Parameter 不会被 apply 遍历，需要单独初始化
        trunc_normal_(self.spatial_pos_embed,  std=0.02)
        trunc_normal_(self.temporal_pos_embed, std=0.02)

    def forward(self, x):
        x = self.embedded(x)
        x = x.reshape(-1, self.num_temporal_tokens, self.num_spatial_tokens, self.hidden_size)
        x = x + self.spatial_pos_embed
        x = self.pos_drop(x)

        B, T, S, C = x.shape
        x = x.reshape(B * T, S, C)
        x = self.spatial_encoder(x)
        x = self.spatial_norm(x)

        x = x.reshape(B, T, S, C)
        temporal_tokens = x.mean(dim=2)

        temporal_tokens = temporal_tokens + self.temporal_pos_embed
        temporal_tokens = self.pos_drop(temporal_tokens)
        temporal_tokens = self.temporal_encoder(temporal_tokens)
        temporal_tokens = self.temporal_norm(temporal_tokens)

        final_feature = temporal_tokens.mean(dim=1)
        logits        = self.head(final_feature)
        return logits

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv3d, nn.Linear)):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm3d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
