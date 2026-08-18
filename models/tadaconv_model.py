"""
TAdaConvModel - 完整模型组装 (TAdaConvNeXtV2 + STSA + CPAM + 双分类头)。

严格遵循:
  1. 主干 = 官方 TAdaConv-main 的 TAdaConvNeXtV2 (third_party/tadaconv),
     直接复用官方 TAdaConvNeXtV2Block + LayerNorm, 不修改官方代码。
  2. 主干组装方式照抄官方 ConvNeXt (backbone.py): stem + 3 downsample + 4 stages。
  3. STSA 插在 Stage3/4 之后 (主路径, 即插即用, 零新增主干参数)。
  4. CPAM 取 4 个 STSA 之前的原始 stage 特征做辅助监督 (论文 Figure 1)。
  5. 双分类头: 主头 (官方 BaseHead 同构) + CPAM 辅助头。

可开关: use_cpam / use_stsa 控制模块启用。
"""
import os
import sys
import torch
import torch.nn as nn
from types import SimpleNamespace
from collections import OrderedDict

# 确保 third_party 可导入 (官方 tadaconv 包)
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TP = os.path.join(_PKG_ROOT, "third_party")
if _TP not in sys.path:
    sys.path.insert(0, _TP)

# 官方 TAdaConvNeXtV2 组件 (third_party/tadaconv)
from tadaconv.models.module_zoo.branches.tadaconvnextv2 import TAdaConvNeXtV2Block
from tadaconv.models.module_zoo.ops.misc import LayerNorm

from models.modules.cpam import CPAM
from models.modules.stsa import STSA


def _official_block_cfg():
    """构造喂给官方 TAdaConvNeXtV2Block 的最小 cfg (官方 Tiny 默认值)."""
    return SimpleNamespace(
        VIDEO=SimpleNamespace(
            BACKBONE=SimpleNamespace(
                BRANCH=SimpleNamespace(
                    ROUTE_FUNC_R=2,        # 官方 tiny 默认
                    ROUTE_FUNC_K=[3, 3],   # 官方 tiny 默认
                    HEAD_DIM=48,           # 官方 tiny 默认
                )
            )
        )
    )


def load_pretrained(model, weight_path, device="cpu"):
    """加载官方 .pyth 预训练权重 (K400/SSv2), 兼容 prefix: backbone./module.backbone./无前缀.

    只加载主干部分 (downsample_layers/stages/norm), 跳过 head/CPAM/STSA (各自独立初始化)。
    """
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"pretrained weights not found: {weight_path}")
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    pretrained = ckpt["model_state"] if "model_state" in ckpt else ckpt
    if isinstance(pretrained, dict) and "state_dict" in pretrained:
        pretrained = pretrained["state_dict"]

    model_state = model.state_dict()
    new_state = OrderedDict()
    matched = 0
    for k, v in pretrained.items():
        name = k
        for pref in ("backbone.", "module.backbone.", ""):
            if k.startswith(pref):
                name = k[len(pref):]
                break
        if name in model_state and model_state[name].shape == v.shape:
            new_state[name] = v
            matched += 1
    model.load_state_dict(new_state, strict=False)
    print(f"[TAdaConv] loaded {matched}/{len(model_state)} params from {weight_path}")
    return model


class TAdaConvModel(nn.Module):
    """TAdaConvNeXtV2 主干 + STSA(Stage3/4) + CPAM(辅助) + 双分类头.

    Args:
        num_classes (int): 分类数.
        depths (list[int]): 各 stage block 数, 默认官方 Tiny [3,3,9,3].
        dims (list[int]): 各 stage 通道, 默认 [96,192,384,768].
        t_stride (int): stem 时间步长, 1 -> T 全程不变 (32帧进32帧出), 2 -> 16帧.
        t_kernel_size (int): stem 时间核.
        drop_path (float): stochastic depth.
        use_cpam (bool): 启用 CPAM 辅助分支.
        use_stsa (bool): 启用 STSA (Stage3/4 之后).
        in_chans (int): 输入通道 (RGB=3).
    """

    def __init__(self, num_classes=5, depths=(3, 3, 9, 3), dims=(96, 192, 384, 768),
                 t_stride=1, t_kernel_size=3, drop_path=0.2,
                 use_cpam=True, use_stsa=True, in_chans=3):
        super().__init__()
        self.num_classes = num_classes
        self.dims = list(dims)
        self.depths = list(depths)
        self.use_cpam = use_cpam
        self.use_stsa = use_stsa
        n_stages = len(self.dims)
        assert n_stages == 4, "TAdaConvNeXtV2 固定 4 个 stage"

        blk_cfg = _official_block_cfg()
        ln_cfg = {"eps": 1e-6, "data_format": "channels_first"}

        # ---- 下采样层: stem + 3 个中间下采样 (照抄官方 ConvNeXt) ----
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0],
                      kernel_size=(t_kernel_size, 4, 4), stride=(t_stride, 4, 4),
                      padding=((t_kernel_size - 1) // 2, 0, 0)),
            LayerNorm(dims[0], **ln_cfg),
        )
        self.downsample_layers.append(stem)
        for i in range(n_stages - 1):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i], **ln_cfg),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            ))

        # ---- 4 个 stage (官方 TAdaConvNeXtV2Block) ----
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path, sum(self.depths))]
        cur = 0
        for i in range(n_stages):
            stage = nn.Sequential(*[
                TAdaConvNeXtV2Block(blk_cfg, dim=dims[i], drop_path=dp_rates[cur + j],
                                    layer_scale_init_value=1e-6)
                for j in range(self.depths[i])
            ])
            self.stages.append(stage)
            cur += self.depths[i]

        # ---- 最终 norm (官方 ConvNeXt 同款) ----
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)

        # ---- STSA (Stage3/4 之后, 即插即用) ----
        self.stsa3 = STSA(dims[2]) if use_stsa else None
        self.stsa4 = STSA(dims[3]) if use_stsa else None

        # ---- CPAM 辅助分支 (取 4 个原始 stage 特征) ----
        self.cpam = CPAM(stage_channels=dims, num_classes=num_classes,
                         out_ch=dims[-1]) if use_cpam else None

        # ---- 主分类头 (官方 BaseHead 同构: 输入已 GAP 为 (N,C) -> Linear) ----
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(dims[-1], num_classes),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # ---- 与旧接口兼容的辅助方法 (scripts/train.py, diag_*, 03/04 脚本复用) ----
    def set_head(self, num_classes, class_counts=None):
        """替换分类头, 并用稀有类偏置初始化 (DeepEthogram rare_class_bias)."""
        self.num_classes = num_classes
        lin = self.head[1]  # head = Sequential(Dropout, Linear)
        dev = lin.weight.device
        new_lin = nn.Linear(lin.in_features, num_classes).to(dev)
        nn.init.trunc_normal_(new_lin.weight, std=0.02)
        if class_counts is not None:
            cc = torch.as_tensor(class_counts, dtype=torch.float32, device=dev).clamp(min=1.0)
            pi = cc / cc.sum()
            b = -torch.log((1.0 - pi) / (pi + 1e-8))
            new_lin.bias.data.copy_(b.clamp(-5.0, 5.0))
        else:
            nn.init.constant_(new_lin.bias, 0)
        self.head[1] = new_lin

    def freeze_backbone(self):
        """LP 阶段: 冻结主干(含STSA/norm), 只训 head + cpam (新增模块保持可训)."""
        for p in self.downsample_layers.parameters():
            p.requires_grad = False
        for p in self.stages.parameters():
            p.requires_grad = False
        for p in self.norm.parameters():
            p.requires_grad = False
        if self.stsa3 is not None:
            for p in self.stsa3.parameters():
                p.requires_grad = True
        if self.stsa4 is not None:
            for p in self.stsa4.parameters():
                p.requires_grad = True
        if self.cpam is not None:
            for p in self.cpam.parameters():
                p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def freeze_first_stages(self, n=3):
        """LP-FT 阶段: 冻结前 n 个 stage (+ 对应 downsample), 训 Stage4+STSA+CPAM+head."""
        for i in range(min(n, 4)):
            for p in self.downsample_layers[i].parameters():
                p.requires_grad = False
            for p in self.stages[i].parameters():
                p.requires_grad = False

    def forward_features(self, x):
        """返回 (x_normed, stage_features). stage_features 在 STSA 之前收集."""
        stages_feat = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            stages_feat.append(x)               # 收集原始 stage 输出 (STSA 之前)
            if i == 2 and self.stsa3 is not None:
                x = self.stsa3(x)               # Stage3 后插 STSA
            elif i == 3 and self.stsa4 is not None:
                x = self.stsa4(x)               # Stage4 后插 STSA
        # 官方 ConvNeXt: norm(x.mean([-3,-2,-1])) = GAP 后接 LayerNorm
        pooled = x.mean([-3, -2, -1])           # (N, C)
        return self.norm(pooled), stages_feat

    def forward(self, x, return_aux=True):
        """x: (N, C, T, H, W). 返回 (logits, aux_logits).

        logits: (N, num_classes) 主头输出.
        aux_logits: (N, num_classes) CPAM 辅助头输出, 未启用 CPAM 时为 None.
        """
        if x.dim() == 4:
            x = x.unsqueeze(0)
        x = x.permute(0, 2, 1, 3, 4) if x.dim() == 5 and x.shape[1] != 3 else x
        # 确保格式 (N, C, T, H, W)
        if x.shape[1] not in (3, 4):
            # 若输入是 (N,T,C,H,W) 转成 (N,C,T,H,W)
            x = x.permute(0, 2, 1, 3, 4)

        feats, stage_feats = self.forward_features(x)

        # 主头: 全局池化 (照官方 BaseHead: GAP -> Linear)
        logits = self.head(feats)

        aux_logits = None
        if self.cpam is not None:
            _, aux_logits = self.cpam(stage_feats)

        if return_aux:
            return logits, aux_logits
        return logits

    def num_parameters(self, only_trainable=False):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad or not only_trainable)