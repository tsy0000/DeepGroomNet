# -*- coding: utf-8 -*-
"""打印 CPAM 与 STSA 结构 (torchinfo + 逐层形状), 输出 UTF-8 文件.

输出: outputs/figures/cpam_stsa_structure.txt
用法:  D:/Anaconda3/envs/deepgroomnet3/python.exe scripts/diag_modules.py
"""
import os, sys, io
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
import torchinfo
from models.tadaconv_model import TAdaConvModel

buf = io.StringIO()


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    buf.write(s + "\n")


def dump_module(mod, prefix):
    p(prefix + "  (结构递归)")
    for name, child in mod.named_children():
        p(f"    {name}: {type(child).__name__}  "
          f"params={sum(x.numel() for x in child.parameters()):,}")
    p("")


def main():
    model = TAdaConvModel(num_classes=5, t_stride=1, drop_path=0.0,
                          use_cpam=True, use_stsa=True)
    model.eval()
    x = torch.randn(1, 3, 32, 224, 224)

    p("=" * 74)
    p("整个模型 torchinfo (输入 1x3x32x224x224)")
    p("=" * 74)
    summary = torchinfo.summary(model, input_data=x, depth=4, verbose=0,
                                col_names=["input_size", "output_size", "num_params"])
    buf.write(summary.__repr__() + "\n\n")

    p("=" * 74)
    p("CPAM 结构 + 参数 (Comprehensive Perception Auxiliary Module)")
    p("=" * 74)
    dump_module(model.cpam, "cpam")
    for name, child in model.cpam.named_children():
        if child._get_name() in ("ModuleList", "Sequential"):
            p(f"  cpam.{name}:")
            for i, sub in enumerate(child):
                p(f"    [{i}] {type(sub).__name__}  params={sum(x.numel() for x in sub.parameters()):,}")

    p("=" * 74)
    p("STSA3 / STSA4 结构 + 参数")
    p("=" * 74)
    for tag in ("stsa3", "stsa4"):
        m = getattr(model, tag)
        p(f"{tag} (输入通道 {m.channels}):")
        dump_module(m, tag)

    # 前向核对形状
    with torch.no_grad():
        logits, aux = model(x)
    p("=" * 74)
    p(f"前向输出: logits={tuple(logits.shape)}  aux_logits={tuple(aux.shape)}")
    p("CPAM 总参数量: ", f"{sum(x.numel() for x in model.cpam.parameters()):,}")
    p("STSA3 参数量: ", f"{sum(x.numel() for x in model.stsa3.parameters()):,}")
    p("STSA4 参数量: ", f"{sum(x.numel() for x in model.stsa4.parameters()):,}")

    out = os.path.join(ROOT, "docs", "DIMENSION_TRACE.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print("\n已保存:", out)


if __name__ == "__main__":
    main()
