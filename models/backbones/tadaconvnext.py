"""
TAdaConvNeXtV2 封装（纯官方，无自定义附加模块）。

结构忠实迁移自官方 alibaba-mmai-research/TAdaConv（Apache-2.0）的 TAdaConvNeXtV2：
  - 4 个 stage 全部用官方 TAdaConvNeXtV2Block（third_party/DeepBehavior/TAdaConv/tadaconv）
  - 分类头 = 官方 BaseHead（AdaptiveAvgPool3d(1) -> Linear），见官方 base_blocks.py

导入说明：官方代码内部以 `tadaconv...` 包路径互相引用，
因此本项目把官方源码放在 `third_party/DeepBehavior/TAdaConv/tadaconv`，
并把 `third_party` 加入 sys.path，保持官方源码零改动。

保留（成熟的训练技巧）：
  - set_head 稀有类偏置初始化（DeepEthogram rare_class_bias）
  - freeze_backbone / unfreeze_all（LP-FT 两段式微调）
"""
import os
import sys
from types import SimpleNamespace
from collections import OrderedDict

_THIRD_PARTY = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "third_party")
if _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)

import torch
import torch.nn as nn

from tadaconv.models.module_zoo.ops.misc import LayerNorm
from tadaconv.models.module_zoo.branches.tadaconvnextv2 import TAdaConvNeXtV2Block
from tadaconv.models.utils.init_helper import trunc_normal_
from models.cpam import CPAM


def build_backbone_cfg(num_classes, drop_path=0.1):
    """构造官方 block 所需的 cfg.VIDEO.BACKBONE（参数与官方 tiny 一致）。"""
    return SimpleNamespace(
        DEPTH=[3, 3, 9, 3],
        NUM_FILTERS=[96, 192, 384, 768],
        STEM=SimpleNamespace(T_KERNEL_SIZE=3, T_STRIDE=1, T_PADDING=1),
        DROP_PATH=drop_path,
        LARGE_SCALE_INIT_VALUE=1e-6,
        NUM_INPUT_CHANNELS=3,
        NUM_OUT_FEATURES=768,
        BRANCH=SimpleNamespace(ROUTE_FUNC_R=2, ROUTE_FUNC_K=(3, 3), HEAD_DIM=48,
                               WITH_BIAS_CAL=True, ZERO_INIT_CAL=True),
        HEAD=SimpleNamespace(NUM_CLASSES=num_classes, LOSS_FUNC="cross_entropy"),
    )


class TAdaConvNeXtV2(nn.Module):
    """纯官方 TAdaConvNeXtV2-Tiny + 官方 BaseHead（无自定义附加模块）。"""

    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_classes = model_cfg.num_classes
        b_cfg = build_backbone_cfg(self.num_classes,
                                   drop_path=getattr(model_cfg, "drop_path", 0.1))
        # 包装成 block 需要的全局 cfg（cfg.VIDEO.BACKBONE）
        self.cfg = SimpleNamespace(VIDEO=SimpleNamespace(BACKBONE=b_cfg))

        # ---- 下采样层（stem + 3）----
        # 输入通道固定 = 3 (RGB)。3D 骨干在内部建模运动, 不需要帧差/光流通道。
        in_ch = b_cfg.NUM_INPUT_CHANNELS
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv3d(in_ch, b_cfg.NUM_FILTERS[0],
                      kernel_size=(b_cfg.STEM.T_KERNEL_SIZE, 4, 4),
                      stride=(b_cfg.STEM.T_STRIDE, 4, 4),
                      padding=((b_cfg.STEM.T_KERNEL_SIZE - 1) // 2, 0, 0), bias=False),
            LayerNorm(b_cfg.NUM_FILTERS[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            ds = nn.Sequential(
                LayerNorm(b_cfg.NUM_FILTERS[i], eps=1e-6, data_format="channels_first"),
                nn.Conv3d(b_cfg.NUM_FILTERS[i], b_cfg.NUM_FILTERS[i + 1],
                          kernel_size=(1, 2, 2), stride=(1, 2, 2), bias=False),
            )
            self.downsample_layers.append(ds)

        # ---- Stages（官方 TAdaConvNeXtV2Block）----
        self.stages = nn.ModuleList()
        dp_rates = torch.linspace(0, b_cfg.DROP_PATH, sum(b_cfg.DEPTH))
        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[
                TAdaConvNeXtV2Block(cfg=self.cfg, dim=b_cfg.NUM_FILTERS[i],
                                    drop_path=dp_rates[cur + j].item(),
                                    layer_scale_init_value=b_cfg.LARGE_SCALE_INIT_VALUE)
                for j in range(b_cfg.DEPTH[i])
            ])
            self.stages.append(stage)
            cur += b_cfg.DEPTH[i]

        # ---- CPAM (Swin-Panda: 多stage融合 + 辅助损失, 局部细粒度) ----
        self.cpam = None
        if getattr(model_cfg, "cpam", True):
            out_ch = b_cfg.NUM_FILTERS[-1]
            self.cpam = CPAM(b_cfg.NUM_FILTERS, out_ch, self.num_classes,
                             interpolate_size=(16, 7, 7))
            print("[TAdaConv] CPAM auxiliary module enabled")

        # ---- 分类头 = 官方 BaseHead（AdaptiveAvgPool3d(1) -> Linear）----
        self.norm = LayerNorm(b_cfg.NUM_FILTERS[-1], eps=1e-6, data_format="channels_first")
        drop_rate = float(getattr(model_cfg, "dropout", 0.0) or 0.0)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(1),
            nn.Dropout(p=drop_rate),
            nn.Linear(b_cfg.NUM_FILTERS[-1], self.num_classes),
        )
        self.aux_head = None
        self._frozen_bn = False
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        stages = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            stages.append(x)
        return self.norm(x), stages

    def forward(self, x, return_aux=False, pose=None, kp_map=None):
        if isinstance(x, dict):
            x = x["video"]
        if x.dim() == 4:
            x = x.unsqueeze(0)
        feats, stages = self.forward_features(x)
        logits = self.head(feats)
        aux_logits = None
        if self.cpam is not None:
            _, aux_logits = self.cpam(stages)
        if return_aux or (aux_logits is not None and not self.training and False):
            return logits, aux_logits
        if aux_logits is not None:
            return logits, aux_logits
        return logits

    def set_aux_head(self, num_aux_classes):
        in_dim = self.head[3].in_features
        dev = self.head[3].weight.device
        self.aux_head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(1),
            nn.Dropout(p=float(getattr(self.model_cfg, "dropout", 0.0) or 0.0)),
            nn.Linear(in_dim, num_aux_classes),
        ).to(dev)
        nn.init.trunc_normal_(self.aux_head[3].weight, std=.02)
        if self.aux_head[3].bias is not None:
            nn.init.constant_(self.aux_head[3].bias, 0)

    def freeze_backbone(self):
        for p in self.downsample_layers.parameters():
            p.requires_grad = False
        for p in self.stages.parameters():
            p.requires_grad = False
        for p in self.norm.parameters():
            p.requires_grad = False
        for p in self.cpam.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = True
        # 冻结 BN 统计量：避免小批量更新 running stats → 特征漂移
        self._frozen_bn = True
        self._apply_bn_eval()

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
        self._frozen_bn = False

    def _apply_bn_eval(self):
        for m in (self.downsample_layers, self.stages, self.norm, self.cpam):
            for mod in m.modules():
                if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm3d)):
                    mod.eval()

    def train(self, mode=True):
        super().train(mode)
        if mode and self._frozen_bn:
            self._apply_bn_eval()
        return self

    def set_head(self, num_classes, class_counts=None):
        """替换分类头，并用稀有类偏置初始化（DeepEthogram rare_class_bias）。"""
        self.num_classes = num_classes
        dev = self.head[3].weight.device
        self.head[3] = nn.Linear(self.head[3].in_features, num_classes).to(dev)
        nn.init.trunc_normal_(self.head[3].weight, std=.02)
        if class_counts is not None and self.head[3].bias is not None:
            cc = torch.as_tensor(class_counts, dtype=torch.float32, device=dev).clamp(min=1.0)
            pi = cc / cc.sum()
            b = -torch.log((1.0 - pi) / (pi + 1e-8))
            self.head[3].bias.data.copy_(b.clamp(-5.0, 5.0))
        elif self.head[3].bias is not None:
            nn.init.constant_(self.head[3].bias, 0)


def load_pretrained(model, weight_path, device="cpu"):
    """加载官方 .pyth 预训练权重（K400/SSv2），兼容 prefix：backbone./module.backbone./无前缀。"""
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"pretrained weights not found: {weight_path}")
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    pretrained = ckpt["model_state"] if "model_state" in ckpt else ckpt
    if isinstance(pretrained, dict) and "state_dict" in pretrained:
        pretrained = pretrained["state_dict"]

    model_state = model.state_dict()
    new_state = OrderedDict()
    matched = 0
    for prefixes in (["backbone.", "module.backbone.", ""],):
        for k, v in pretrained.items():
            for pref in prefixes:
                name = k[len(pref):] if k.startswith(pref) else k
                if name in model_state and model_state[name].shape == v.shape:
                    new_state[name] = v
                    matched += 1
                    break
    if matched == 0:
        for k, v in pretrained.items():
            if k in model_state and model_state[k].shape == v.shape:
                new_state[k] = v
                matched += 1
    model.load_state_dict(new_state, strict=False)
    print(f"[TAdaConv] loaded {matched}/{len(model_state)} params from {weight_path}")
    return model


def build_model(cfg):
    """cfg: 完整 config.yaml dict（自动取 MODEL 段）或 MODEL 段 dict。"""
    if isinstance(cfg, dict):
        if "MODEL" in cfg:
            cfg = cfg["MODEL"]
        cfg = SimpleNamespace(**cfg)
    model = TAdaConvNeXtV2(cfg)
    w = getattr(cfg, "init_weights", None)
    if w and os.path.exists(w):
        load_pretrained(model, w)
    return model
