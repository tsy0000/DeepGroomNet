"""训练编排: LP (Linear Probe) -> LP-FT 两阶段冻结微调."""

import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from .lightning_module import TadaPandaLightning


def build_trainer(cfg, stage):
    """构造 Lightning Trainer.

    stage: 'lp' 或 'lpft'.
    """
    t = cfg["TRAIN"]
    log_dir = cfg["OUTPUT"]["log_dir"]
    ckpt_dir = cfg["OUTPUT"]["checkpoint_dir"]
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, f"{stage}.ckpt")
    callbacks = [
        ModelCheckpoint(dirpath=ckpt_dir, filename=stage,
                        monitor="val/macro_auroc", mode="max",
                        save_top_k=1, verbose=True),
    ]
    if stage == "lpft":
        callbacks.append(EarlyStopping(monitor="val/macro_auroc", patience=12, mode="max"))

    return pl.Trainer(
        max_epochs=int(t.get("epochs", 60)) if stage == "lpft" else int(t.get("lp", {}).get("epochs", 5)),
        accelerator="auto",
        devices=1,
        precision="32-true" if not t.get("mixed_precision", False) else "16-mixed",
        callbacks=callbacks,
        default_root_dir=log_dir,
        enable_progress_bar=True,
        log_every_n_steps=1,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
    )


def run_pipeline(cfg, tr_loader, va_loader, num_classes, pos_weight):
    """LP -> LP-FT 完整训练流程.

    Returns:
        (lp_model, lpft_model): 两阶段训练后的模型.
    """
    t = cfg["TRAIN"]

    # ---- LP: 冻结主干, 只训 head + cpam ----
    lp_model = TadaPandaLightning(cfg, num_classes=num_classes, pos_weight=pos_weight,
                                  freeze_stages=None)
    lp_model.set_phase("lp")
    if t.get("lp", {}).get("enable", True):
        trainer = build_trainer(cfg, "lp")
        trainer.fit(lp_model, tr_loader, va_loader)
        print("[LP] 完成: 冻结主干, 训练 head+cpam")

    # ---- LP-FT: 冻结 Stage1-3, 解冻 Stage4+STSA+CPAM+head ----
    lpft_model = lp_model
    if t.get("lpft", {}).get("enable", True):
        lpft_model.set_phase("lpft")
        trainer = build_trainer(cfg, "lpft")
        trainer.fit(lpft_model, tr_loader, va_loader)
        print("[LP-FT] 完成: 冻结 Stage1-3, 训练 Stage4+STSA+CPAM+head")

    return lp_model, lpft_model