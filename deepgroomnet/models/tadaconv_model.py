import os
import torch
import torch.nn as nn
from types import SimpleNamespace
from collections import OrderedDict
from tadaconv.models.module_zoo.branches.tadaconvnextv2 import TAdaConvNeXtV2Block
from tadaconv.models.module_zoo.ops.misc import LayerNorm
from deepgroomnet.models.cpam import CPAM
from deepgroomnet.models.stsa import STSA

def _official_block_cfg():
    return SimpleNamespace(VIDEO=SimpleNamespace(BACKBONE=SimpleNamespace(BRANCH=SimpleNamespace(ROUTE_FUNC_R=2, ROUTE_FUNC_K=[3, 3], HEAD_DIM=48))))

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
    for (k, v) in pretrained.items():
        name = k
        for pref in ('backbone.', 'module.backbone.', ''):
            if k.startswith(pref):
                name = k[len(pref):]
                break
        if name in model_state and model_state[name].shape == v.shape:
            new_state[name] = v
            matched += 1
    model.load_state_dict(new_state, strict=False)
    print(f'[TAdaConv] loaded {matched}/{len(model_state)} params from {weight_path}')
    return model

class TAdaConvModel(nn.Module):

    def __init__(self, num_classes=5, depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), t_stride=1, t_kernel_size=3, drop_path=0.2, use_cpam=True, use_stsa=True, in_chans=3, use_checkpoint=False):
        super().__init__()
        self.num_classes = num_classes
        self.dims = list(dims)
        self.depths = list(depths)
        self.use_cpam = use_cpam
        self.use_stsa = use_stsa
        self.use_checkpoint = use_checkpoint
        n_stages = len(self.dims)
        assert n_stages == 4, 'TAdaConvNeXtV2 固定 4 个 stage'
        blk_cfg = _official_block_cfg()
        ln_cfg = {'eps': 1e-06, 'data_format': 'channels_first'}
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(nn.Conv3d(in_chans, dims[0], kernel_size=(t_kernel_size, 4, 4), stride=(t_stride, 4, 4), padding=((t_kernel_size - 1) // 2, 0, 0)), LayerNorm(dims[0], **ln_cfg))
        self.downsample_layers.append(stem)
        for i in range(n_stages - 1):
            self.downsample_layers.append(nn.Sequential(LayerNorm(dims[i], **ln_cfg), nn.Conv3d(dims[i], dims[i + 1], kernel_size=(1, 2, 2), stride=(1, 2, 2))))
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path, sum(self.depths))]
        cur = 0
        for i in range(n_stages):
            self.stages.append(nn.Sequential(*[TAdaConvNeXtV2Block(blk_cfg, dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=1e-06) for j in range(self.depths[i])]))
            cur += self.depths[i]
        self.norm = nn.LayerNorm(dims[-1], eps=1e-06)
        self.stsa3 = STSA(dims[2]) if use_stsa else None
        self.stsa4 = STSA(dims[3]) if use_stsa else None
        self.cpam = CPAM(stage_channels=dims, num_classes=num_classes, out_ch=dims[-1]) if use_cpam else None
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(dims[-1], num_classes))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def set_head(self, num_classes, class_counts=None):
        self.num_classes = num_classes
        lin = self.head[1]
        dev = lin.weight.device
        new_lin = nn.Linear(lin.in_features, num_classes).to(dev)
        nn.init.trunc_normal_(new_lin.weight, std=0.02)
        if class_counts is not None:
            cc = torch.as_tensor(class_counts, dtype=torch.float32, device=dev).clamp(min=1.0)
            pi = cc / cc.sum()
            b = -torch.log((1.0 - pi) / (pi + 1e-08))
            new_lin.bias.data.copy_(b.clamp(-5.0, 5.0))
        else:
            nn.init.constant_(new_lin.bias, 0)
        self.head[1] = new_lin

    def set_num_classes(self, num_classes, class_counts=None):
        self.set_head(num_classes, class_counts=class_counts)
        if self.cpam is not None:
            aux = self.cpam.aux_head
            lin = aux[-1]
            dev = lin.weight.device
            new_lin = nn.Linear(lin.in_features, num_classes).to(dev)
            nn.init.trunc_normal_(new_lin.weight, std=0.02)
            if class_counts is not None:
                cc = torch.as_tensor(class_counts, dtype=torch.float32, device=dev).clamp(min=1.0)
                pi = cc / cc.sum()
                new_lin.bias.data.copy_((-torch.log((1.0 - pi) / (pi + 1e-08))).clamp(-5.0, 5.0))
            else:
                nn.init.constant_(new_lin.bias, 0)
            aux[-1] = new_lin

    def freeze_backbone(self):
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
        for i in range(min(n, 4)):
            for p in self.downsample_layers[i].parameters():
                p.requires_grad = False
            for p in self.stages[i].parameters():
                p.requires_grad = False

    def forward_features(self, x):
        stages_feat = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            for blk in self.stages[i]:
                if self.use_checkpoint and self.training:
                    x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
                else:
                    x = blk(x)
            stages_feat.append(x)
            if i == 2 and self.stsa3 is not None:
                x = self.stsa3(x)
            elif i == 3 and self.stsa4 is not None:
                x = self.stsa4(x)
        pooled = x.mean([-3, -2, -1])
        return (self.norm(pooled), stages_feat)

    def forward(self, x, return_aux=True):
        if x.dim() == 4:
            x = x.unsqueeze(0)
        x = x.permute(0, 2, 1, 3, 4) if x.dim() == 5 and x.shape[1] != 3 else x
        if x.shape[1] not in (3, 4):
            x = x.permute(0, 2, 1, 3, 4)
        (feats, stage_feats) = self.forward_features(x)
        logits = self.head(feats)
        aux_logits = None
        if self.cpam is not None:
            (_, aux_logits) = self.cpam(stage_feats)
        if return_aux:
            return (logits, aux_logits)
        return logits

    def num_parameters(self, only_trainable=False):
        return sum((p.numel() for p in self.parameters() if p.requires_grad or not only_trainable))
