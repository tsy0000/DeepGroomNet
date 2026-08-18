"""Lightning 训练模块测试: 构建 + 冻结策略 + 一步训练."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from train.lightning_module import TAdaConvLightning

CFG = {
    "MODEL": {"cpam": True, "stsa": True, "cpam_alpha": 0.9, "cpam_beta": 0.1,
              "drop_path": 0.1, "pretrained": ""},
    "LOSS": {"binary_gamma": 0.5, "binary_label_smoothing": 0.05},
    "TRAIN": {"lr": 2e-4, "head_lr_mult": 10.0, "weight_decay": 5e-4,
              "epochs": 60, "l2sp_weight": 0.0},
    "OUTPUT": {"log_dir": "outputs/logs", "checkpoint_dir": "outputs/checkpoints"},
}


def test_build():
    m = TAdaConvLightning(CFG, num_classes=5, pos_weight=None)
    x = torch.randn(2, 3, 32, 224, 224)
    logits, aux = m(x)
    assert logits.shape == (2, 5)
    assert aux.shape == (2, 5)
    print(f"[OK] 构建+前向: logits={tuple(logits.shape)} aux={tuple(aux.shape)}")


def test_freeze_lp():
    m = TAdaConvLightning(CFG, num_classes=5, pos_weight=None)
    m.set_phase("lp")
    # 主干应冻结
    frozen = sum(1 for n, p in m.model.named_parameters() if not p.requires_grad)
    total = sum(1 for _ in m.model.parameters())
    # head/cpam 应可训
    head_trainable = all(p.requires_grad for n, p in m.model.named_parameters() if "head" in n)
    assert head_trainable, "head 应可训练"
    print(f"[OK] LP: 冻结 {frozen}/{total} 参数, head可训={head_trainable}")


def test_freeze_lpft():
    m = TAdaConvLightning(CFG, num_classes=5, pos_weight=None)
    m.set_phase("lpft")
    # Stage1-3 (stages.0/1/2) 应冻结
    s0_frozen = not next(m.model.stages[0].parameters()).requires_grad
    s3_trainable = next(m.model.stages[3].parameters()).requires_grad
    assert s0_frozen, "Stage1 应冻结"
    assert s3_trainable, "Stage4 应可训练"
    print(f"[OK] LP-FT: Stage1冻结={s0_frozen}, Stage4可训={s3_trainable}")


def test_train_step():
    m = TAdaConvLightning(CFG, num_classes=5, pos_weight=None)
    m.set_phase("lpft")
    x = torch.randn(2, 3, 32, 224, 224)
    y = torch.randint(0, 5, (2,))
    loss = m.training_step((x, y), 0)
    assert loss.item() > 0
    print(f"[OK] training_step loss={loss.item():.4f}")


def test_optimizer():
    m = TAdaConvLightning(CFG, num_classes=5, pos_weight=None)
    m.set_phase("lpft")
    optimizers, schedulers = m.configure_optimizers()
    assert len(optimizers) == 1
    assert len(optimizers[0].param_groups) == 2, "应有 2 个参数组 (主干/新模块)"
    print(f"[OK] 优化器: {len(optimizers[0].param_groups)} 参数组, "
          f"{len(schedulers)} scheduler")


if __name__ == "__main__":
    test_build()
    test_freeze_lp()
    test_freeze_lpft()
    test_train_step()
    test_optimizer()
    print("Lightning 训练模块全部测试通过")