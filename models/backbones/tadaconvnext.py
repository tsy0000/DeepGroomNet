import os
import sys
from types import SimpleNamespace
from collections import OrderedDict
_THIRD_PARTY = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'third_party')
if _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)
import torch
import torch.nn as nn
from DeepBehavior.TAdaConv.tadaconv.models.module_zoo.ops.misc import LayerNorm
from DeepBehavior.TAdaConv.tadaconv.models.module_zoo.branches.tadaconvnextv2 import TAdaConvNeXtV2Block
from DeepBehavior.TAdaConv.tadaconv.models.utils.init_helper import trunc_normal_
from models.attention import build_attention, MotionAttention, MotionStatAttention

def build_backbone_cfg(num_classes, drop_path=0.1):
    return SimpleNamespace(DEPTH=[3, 3, 9, 3], NUM_FILTERS=[96, 192, 384, 768], STEM=SimpleNamespace(T_KERNEL_SIZE=3, T_STRIDE=1, T_PADDING=1), DROP_PATH=drop_path, LARGE_SCALE_INIT_VALUE=1e-06, NUM_INPUT_CHANNELS=3, NUM_OUT_FEATURES=768, BRANCH=SimpleNamespace(ROUTE_FUNC_R=2, ROUTE_FUNC_K=(3, 3), HEAD_DIM=48, WITH_BIAS_CAL=True, ZERO_INIT_CAL=True), HEAD=SimpleNamespace(NUM_CLASSES=num_classes, LOSS_FUNC='cross_entropy'))

class TAdaConvNeXtV2WithAttention(nn.Module):

    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_classes = model_cfg.num_classes
        b_cfg = build_backbone_cfg(self.num_classes, drop_path=getattr(model_cfg, 'drop_path', 0.1))
        self.cfg = SimpleNamespace(VIDEO=SimpleNamespace(BACKBONE=b_cfg))
        in_ch = b_cfg.NUM_INPUT_CHANNELS
        in_ch += 1 if getattr(model_cfg, 'motion_channel', False) else 0
        in_ch += 2 if getattr(model_cfg, 'flow_channel', False) else 0
        self.motion_channel = in_ch > b_cfg.NUM_INPUT_CHANNELS
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(nn.Conv3d(in_ch, b_cfg.NUM_FILTERS[0], kernel_size=(b_cfg.STEM.T_KERNEL_SIZE, 4, 4), stride=(b_cfg.STEM.T_STRIDE, 4, 4), padding=((b_cfg.STEM.T_KERNEL_SIZE - 1) // 2, 0, 0), bias=False), LayerNorm(b_cfg.NUM_FILTERS[0], eps=1e-06, data_format='channels_first'))
        self.downsample_layers.append(stem)
        for i in range(3):
            ds = nn.Sequential(LayerNorm(b_cfg.NUM_FILTERS[i], eps=1e-06, data_format='channels_first'), nn.Conv3d(b_cfg.NUM_FILTERS[i], b_cfg.NUM_FILTERS[i + 1], kernel_size=(1, 2, 2), stride=(1, 2, 2), bias=False))
            self.downsample_layers.append(ds)
        self.stages = nn.ModuleList()
        dp_rates = torch.linspace(0, b_cfg.DROP_PATH, sum(b_cfg.DEPTH))
        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[TAdaConvNeXtV2Block(cfg=self.cfg, dim=b_cfg.NUM_FILTERS[i], drop_path=dp_rates[cur + j].item(), layer_scale_init_value=b_cfg.LARGE_SCALE_INIT_VALUE) for j in range(b_cfg.DEPTH[i])])
            self.stages.append(stage)
            cur += b_cfg.DEPTH[i]
        attn_name = getattr(model_cfg, 'attention', 'none')
        positions = getattr(model_cfg, 'attention_positions', [0, 1, 2, 3])
        red_map = dict(getattr(model_cfg, 'attention_reductions', {}) or {})
        center_prior = float(getattr(model_cfg, 'attention_center_prior', 0.0) or 0.0)
        mask_prior = float(getattr(model_cfg, 'attention_mask_prior', 0.0) or 0.0)
        kp_prior = float(getattr(model_cfg, 'attention_kp_prior', 0.0) or 0.0)
        self._mask_prior = mask_prior
        self.attentions = nn.ModuleList()
        for i in range(4):
            if i in positions:
                red = red_map.get(attn_name, red_map.get('default', None))
                if attn_name == 'pose_prompt':
                    self.attentions.append(build_attention(attn_name, b_cfg.NUM_FILTERS[i]))
                else:
                    self.attentions.append(build_attention(attn_name, b_cfg.NUM_FILTERS[i], reduction=red, center_prior=center_prior, mask_prior=mask_prior, kp_prior=kp_prior))
            else:
                self.attentions.append(nn.Identity())
        self.norm = LayerNorm(b_cfg.NUM_FILTERS[-1], eps=1e-06, data_format='channels_first')
        drop_rate = float(getattr(model_cfg, 'dropout', 0.0) or 0.0)
        self.head = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(1), nn.Dropout(p=drop_rate), nn.Linear(b_cfg.NUM_FILTERS[-1], self.num_classes))
        self.aux_head = None
        self.pose_branch = None
        self._frozen_bn = False
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if getattr(m, 'skip_init', False):
            return
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, kp_map=None):
        mask0 = None
        if self._mask_prior > 0:
            first_attn = next((a for a in self.attentions if isinstance(a, (MotionAttention, MotionStatAttention))), None)
            if first_attn is not None:
                mask0 = first_attn._mouse_mask(x)
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            a = self.attentions[i]
            if isinstance(a, (MotionAttention, MotionStatAttention)):
                x = a(x, kp_map=kp_map, mask_map=mask0)
            else:
                x = a(x)
        return self.norm(x)

    def forward(self, x, return_aux=False, pose=None, kp_map=None):
        if isinstance(x, dict):
            x = x['video']
        if x.dim() == 4:
            x = x.unsqueeze(0)
        feats = self.forward_features(x, kp_map=kp_map)
        logits = self.head(feats)
        if kp_map is not None and self.pose_branch is not None:
            if pose is not None:
                pv = (pose[..., 2].amax(dim=(1, 2)) > 0).float().unsqueeze(1)
            else:
                pv = torch.ones(logits.shape[0], 1, device=logits.device)
            logits = logits + self.pose_branch(kp_map) * pv
        if return_aux:
            if self.aux_head is None:
                raise ValueError('aux_head 未初始化（先调用 set_aux_head）')
            return (logits, self.aux_head(feats))
        return logits

    def set_pose_branch(self, num_classes, in_channels=None, heatmap_size=64, hidden=128, dropout=0.4):
        from models.pose_branch import PoseBranch
        dev = self.head[3].weight.device
        self.pose_branch = PoseBranch(in_channels=in_channels or 27, num_classes=num_classes, heatmap_size=heatmap_size, hidden=hidden, dropout=dropout).to(dev)

    def set_aux_head(self, num_aux_classes):
        in_dim = self.head[3].in_features
        dev = self.head[3].weight.device
        self.aux_head = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(1), nn.Dropout(p=float(getattr(self.model_cfg, 'dropout', 0.0) or 0.0)), nn.Linear(in_dim, num_aux_classes)).to(dev)
        nn.init.trunc_normal_(self.aux_head[3].weight, std=0.02)
        if self.aux_head[3].bias is not None:
            nn.init.constant_(self.aux_head[3].bias, 0)

    def freeze_backbone(self):
        for p in self.downsample_layers.parameters():
            p.requires_grad = False
        for p in self.stages.parameters():
            p.requires_grad = False
        for p in self.norm.parameters():
            p.requires_grad = False
        for p in self.attentions.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = True
        self._frozen_bn = True
        self._apply_bn_eval()

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
        self._frozen_bn = False

    def _apply_bn_eval(self):
        for m in (self.downsample_layers, self.stages, self.norm, self.attentions):
            for mod in m.modules():
                if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm3d)):
                    mod.eval()

    def train(self, mode=True):
        super().train(mode)
        if mode and self._frozen_bn:
            self._apply_bn_eval()
        return self

    def set_head(self, num_classes):
        self.num_classes = num_classes
        in_dim = self.head[3].in_features
        dev = self.head[3].weight.device
        self.head[3] = nn.Linear(in_dim, num_classes).to(dev)
        nn.init.trunc_normal_(self.head[3].weight, std=0.02)
        if self.head[3].bias is not None:
            nn.init.constant_(self.head[3].bias, 0)

def load_pretrained(model, weight_path, device='cpu'):
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f'pretrained weights not found: {weight_path}')
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    pretrained = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    if isinstance(pretrained, dict) and 'state_dict' in pretrained:
        pretrained = pretrained['state_dict']
    model_state = model.state_dict()
    new_state = OrderedDict()
    matched = 0
    for prefixes in (['backbone.', 'module.backbone.', ''],):
        for (k, v) in pretrained.items():
            for pref in prefixes:
                name = k[len(pref):] if k.startswith(pref) else k
                if name in model_state and model_state[name].shape == v.shape:
                    new_state[name] = v
                    matched += 1
                    break
    if matched == 0:
        for (k, v) in pretrained.items():
            if k in model_state and model_state[k].shape == v.shape:
                new_state[k] = v
                matched += 1
    if getattr(model, 'motion_channel', False):
        stem_key = 'downsample_layers.0.0.weight'
        if stem_key in model_state and stem_key in new_state:
            v3 = new_state[stem_key]
            rep = v3.mean(dim=1, keepdim=True).repeat(1, model_state[stem_key].shape[1] - 3, 1, 1, 1)
            new_state[stem_key] = torch.cat([v3, rep], dim=1)
            matched += 1
    model.load_state_dict(new_state, strict=False)
    print(f'[TAdaConv] loaded {matched}/{len(model_state)} params from {weight_path}')
    return model

def build_model(cfg):
    full_cfg = None
    if isinstance(cfg, dict):
        if 'MODEL' in cfg:
            full_cfg = cfg
            cfg = cfg['MODEL']
        cfg = SimpleNamespace(**cfg)
    if full_cfg is not None:
        clip_cfg = full_cfg.get('CLIP') or {}
        cfg.motion_channel = bool(clip_cfg.get('motion_channel', False))
        cfg.flow_channel = bool(clip_cfg.get('flow_channel', False))
    model = TAdaConvNeXtV2WithAttention(cfg)
    w = getattr(cfg, 'init_weights', None)
    if w and os.path.exists(w):
        load_pretrained(model, w)
    return model
