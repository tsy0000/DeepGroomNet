"""DeepGroomNet - Lightning 训练模块.

- TAdaConvLightning: LightningModule, 双分类头 + 条件自适应损失 (式8-10).
- 冻结策略: LP-FT (Linear Probe -> Full Fine-Tune).
    LP 阶段: 冻结主干 (含 STSA), 只训 head + CPAM 辅助头.
    LP-FT 阶段: 冻结 Stage1-3, 解冻 Stage4 + STSA + CPAM + head.
- configure_optimizers: 分层 lr (head/新模块用高 lr, 主干低 lr).
"""
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.tadaconv_model import TAdaConvModel
from losses.auxiliary import AuxFocalLoss


class TAdaConvLightning(pl.LightningModule):
    """TAdaConvModel 的 Lightning 封装.

    Args:
        cfg (dict): 配置字典 (config.yaml 解析后).
        num_classes (int): 分类数.
        pos_weight (Tensor): 类别正样本权重.
        freeze_stages (int): 冻结前 N 个 stage (LP-FT). None = 不冻结.
    """

    def __init__(self, cfg, num_classes=5, pos_weight=None, freeze_stages=None):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.num_classes = num_classes
        self.freeze_stages = freeze_stages

        m = cfg["MODEL"]
        self.model = TAdaConvModel(
            num_classes=num_classes,
            t_stride=1,                       # T=32
            t_kernel_size=3,
            drop_path=float(m.get("drop_path", 0.2)),
            use_cpam=bool(m.get("cpam", True)),
            use_stsa=bool(m.get("stsa", True)),
        )

        self.loss_fn = AuxFocalLoss(
            num_classes=num_classes,
            pos_weight=pos_weight,
            gamma=float(cfg["LOSS"].get("binary_gamma", 1.0)),
            alpha=float(m.get("cpam_alpha", 0.9)),
            beta=float(m.get("cpam_beta", 0.1)),
            l2sp_weight=float(cfg["TRAIN"].get("l2sp_weight", 1e-3)),
        )
        # 预训练加载 (若提供)
        self._load_pretrained(cfg["MODEL"].get("pretrained"))

    def _load_pretrained(self, path):
        if not path or not os.path.exists(path):
            return
        import torch as _t
        sd = _t.load(path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        # 只加载主干匹配的键 (前缀对齐: 主干权重直接匹配模型命名)
        model_sd = self.model.state_dict()
        miss, unexp = self.model.load_state_dict(sd, strict=False)
        print(f"[TAdaConv] 预训练加载完成. missing={len(miss.missing_keys)} "
              f"unexpected={len(miss.unexpected_keys)}")

    def _freeze(self, names_containing):
        for n, p in self.model.named_parameters():
            if any(k in n for k in names_containing):
                p.requires_grad = False

    def _unfreeze(self, names_containing):
        for n, p in self.model.named_parameters():
            if any(k in n for k in names_containing):
                p.requires_grad = True

    def set_phase(self, phase):
        """phase: 'lp' (冻结主干) 或 'lpft' (冻结 Stage1-3)."""
        if phase == "lp":
            # 冻结所有主干+STSA, 只训 head + cpam
            self._freeze(["downsample_layers", "stages", "stsa", "norm"])
            self._unfreeze(["head", "cpam"])
        elif phase == "lpft":
            # 冻结 Stage1-3 (downsample_layers[0..3] + stages[0..2]), 解冻其余
            self._freeze(["downsample_layers.0", "downsample_layers.1",
                          "downsample_layers.2", "downsample_layers.3",
                          "stages.0", "stages.1", "stages.2"])
            self._unfreeze(["head", "cpam", "stsa", "stages.3", "downsample_layers"])

    def forward(self, x):
        return self.model(x, return_aux=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, aux = self.model(x, return_aux=True)
        tgt = self._to_onehot(y)
        total, d = self.loss_fn(logits, tgt, aux_logits=aux, model=self.model)
        for k, v in d.items():
            self.log(f"train/{k}", v, on_step=False, on_epoch=True)
        return total

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, aux = self.model(x, return_aux=True)
        tgt = self._to_onehot(y)
        total, d = self.loss_fn(logits, tgt, aux_logits=aux, model=self.model)
        for k, v in d.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True)
        # 记录预测用于 AUROC 计算
        self._val_logits = getattr(self, "_val_logits", [])
        self._val_targets = getattr(self, "_val_targets", [])
        self._val_logits.append(logits.detach())
        self._val_targets.append(tgt.detach())
        return total

    def on_validation_epoch_end(self):
        if not hasattr(self, "_val_logits") or not self._val_logits:
            return
        logits = torch.cat(self._val_logits)
        targets = torch.cat(self._val_targets)
        try:
            from sklearn.metrics import roc_auc_score
            # 逐类 AUROC
            auroc = roc_auc_score(targets.cpu().numpy(), logits.cpu().numpy(),
                                  average="macro", multi_class="ovr")
            self.log("val/macro_auroc", auroc, prog_bar=True, on_epoch=True)
        except Exception:
            pass
        self._val_logits = []
        self._val_targets = []

    def _to_onehot(self, y):
        return nn.functional.one_hot(y, num_classes=self.num_classes).float()

    def configure_optimizers(self):
        m = self.cfg["TRAIN"]
        head_lr_mult = float(m.get("head_lr_mult", 10.0))
        lr = float(m.get("lr", 2e-4))
        wd = float(m.get("weight_decay", 5e-4))

        # 分层: head/cpam/stsa 高 lr, 主干低 lr
        head_params, backbone_params = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in n for k in ("head", "cpam", "stsa")):
                head_params.append(p)
            else:
                backbone_params.append(p)

        opt = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": lr, "weight_decay": wd},
                {"params": head_params, "lr": lr * head_lr_mult, "weight_decay": wd},
            ]
        )
        epochs = int(m.get("epochs", 60))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        return [opt], [{"scheduler": sched, "interval": "epoch", "frequency": 1}]