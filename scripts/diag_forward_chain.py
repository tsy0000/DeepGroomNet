# -*- coding: utf-8 -*-
"""diag_forward_chain.py - 打印模型完整前向传播调用链路 (模块结构 + 维度变换 + 原文核对).

直接运行 (无需参数):
  python scripts/diag_forward_chain.py

它构建 5 类 openfield 模型 (TAdaConvNeXtV2-Tiny + STSA + CPAM + 双头),
喂入 (2,3,32,224,224) 假数据, 用 forward hook 逐模块打印:
  ▶ 进入模块  in=<输入形状>   [原文: 式X / 图Y]
  ◀ 离开模块  out=<输出形状>
缩进表示调用嵌套层级, 便于核对每一级维度变换。

可选项:
  --classes N   分类数 (默认 5)
  --cpam 0/1    CPAM 开/关 (默认 1)
  --stsa 0/1    STSA 开/关 (默认 1)
  --drop-path X dropout (默认 0.0, 便于纯净核对维度)
  --out FILE    同时把输出写进 UTF-8 文本文件 (默认不写)
"""
import os, sys, argparse
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.tadaconv_model import TAdaConvModel

DEPTH = 0

# ---------------------------------------------------------------------------
# hook 构造: 进入打印 in, 离开打印 out, 缩进嵌套
# ---------------------------------------------------------------------------
def _pre(name, note):
    def pre(mod, inp):
        global DEPTH
        x = inp[0] if isinstance(inp, (tuple, list)) else inp
        if isinstance(x, (tuple, list)):          # CPAM 输入是 4 个 stage 特征 list
            shp = [tuple(t.shape) for t in x]
        else:
            shp = tuple(x.shape)
        print("  " * DEPTH + f">> {name:<22} in = {str(shp):<32} {note}")
        DEPTH += 1
    return pre

def _post(name):
    def post(mod, inp, out):
        global DEPTH
        DEPTH -= 1
        if isinstance(out, (tuple, list)):
            # CPAM 返回 (Y, aux) / 模型返回 (logits, aux)
            o = tuple(tuple(t.shape) for t in out)
        else:
            o = tuple(out.shape)
        print("  " * DEPTH + f"<< {name:<22} out = {str(o)}")
    return post

def _get(mod, path):
    cur = mod
    for p in path.split("."):
        if not p:
            continue
        cur = getattr(cur, p)
    return cur

def _reg(model, name, path, note=""):
    m = _get(model, path)
    m.register_forward_pre_hook(_pre(name, note))
    m.register_forward_hook(_post(name))

# ---------------------------------------------------------------------------
# 注册要打印的模块 (结构化层级 + 原文公式对照)
# ---------------------------------------------------------------------------
def register(model, use_cpam, use_stsa):
    _reg(model, "MODEL(根)", "", "整体 (N,C,T,H,W) -> (logits, aux)")
    _reg(model, "downsample.0.stem", "downsample_layers.0",
         "[stem] Conv3d k(3,4,4) s(1,4,4) 3->96: 224->56, T=32 保持")
    _reg(model, "downsample.1", "downsample_layers.1",
         "[下采样] (1,2,2): 96->192, 56->28")
    _reg(model, "downsample.2", "downsample_layers.2",
         "[下采样] (1,2,2): 192->384, 28->14")
    _reg(model, "downsample.3", "downsample_layers.3",
         "[下采样] (1,2,2): 384->768, 14->7")
    # 4 个 stage (每 stage 展开每个 block, 确认 T/H/W 全程不变)
    for i, nd in enumerate((3, 3, 9, 3)):
        _reg(model, f"Stage{i+1}(共{nd}块)", f"stages.{i}",
             f"[主干Stage{i+1}] TAdaConvNeXtV2 块×{nd}")
        for j in range(nd):
            _reg(model, f"  .block{j}", f"stages.{i}.{j}",
                 "TAdaConvNeXtV2Block (C,T,H,W 保持不变)")
    _reg(model, "norm", "norm", "GAP 后 LayerNorm (N,768)")
    _reg(model, "head", "head", "主头 Linear 768->5 (sigmoid 多标签)")
    if use_stsa:
        for tag in ("stsa3", "stsa4"):
            _reg(model, f"{tag}(总)", tag, "STSA 空间移位注意力 (式11-15)")
            _reg(model, f"  {tag}.mlp1", f"{tag}.mlp1",
                 "[式11] MLP1: C->3C (Linear->GELU->Linear)")
            _reg(model, f"  {tag}.sa.mlp", f"{tag}.sa.mlp",
                 "[图5a] SA: MLP2(GELU(MLP1)), 沿C维")
            _reg(model, f"  {tag}.ta.conv", f"{tag}.ta.conv",
                 "[图5b] TA: Conv3d(3,1,1) 生成 sigmoid 时间掩码")
            _reg(model, f"  {tag}.mlp2", f"{tag}.mlp2",
                 "[式15] MLP2: C->C 融合")
    if use_cpam:
        _reg(model, "CPAM(总)", "cpam", "辅助监督 (式2-7)")
        for i in range(4):
            _reg(model, f"  cpam.proj{i}", f"cpam.proj.{i}",
                 "[式2] fi: 线性投影 Ci->768")
            _reg(model, f"  cpam.mixer{i}", f"cpam.mixers.{i}",
                 "[式4] gi: (3,1,1) 时间混合")
        _reg(model, "  cpam.fusion", "cpam.fusion",
             "[式6] GAP->MLP->softmax(尺度维4) 自适应权重")
        _reg(model, "  cpam.fusion.mlp", "cpam.fusion.mlp",
             "reduction=8: 768->96->768*4")
        _reg(model, "  cpam.aux_head", "cpam.aux_head",
             "辅助头: GAP+Linear 768->5")

# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, default=5)
    ap.add_argument("--cpam", type=int, default=1)
    ap.add_argument("--stsa", type=int, default=1)
    ap.add_argument("--drop-path", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--out", default=None, help="额外写入 UTF-8 文本文件")
    args = ap.parse_args()

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        sys.stdout = open(args.out, "w", encoding="utf-8")

    torch.manual_seed(0)
    model = TAdaConvModel(
        num_classes=args.classes,
        t_stride=1, drop_path=args.drop_path,
        use_cpam=bool(args.cpam), use_stsa=bool(args.stsa),
        use_checkpoint=False,          # eval 下 hooks 直接触发
    )
    model.eval()
    print("=" * 88)
    print(f"模型: TAdaConvNeXtV2-Tiny + STSA({bool(args.stsa)}) + CPAM({bool(args.cpam)}) "
          f"+ 双头, 参数 {sum(p.numel() for p in model.parameters())/1e6:.2f}M, "
          f"输入 (N,C,T,H,W)  N={args.batch}")
    print(">>=进入模块 in=输入形状 [原文: 式/图];  <<=离开模块 out=输出形状")
    print("=" * 88)
    register(model, bool(args.cpam), bool(args.stsa))
    x = torch.randn(args.batch, 3, 32, 224, 224)
    with torch.no_grad():
        logits, aux = model(x)
    print("=" * 88)
    print(f"最终输出: logits {tuple(logits.shape)} (sigmoid 多标签), "
          f"aux_logits {tuple(aux.shape) if aux is not None else None}")
    print("原文核对要点: STSA 式11(MLP1 C->3C) 式12(split3) 式13/14(移位) "
          "图5a(SA softmax于K维) 图5b(TA sigmoid) 式15(MLP2 C->C) 图3(残差⊕)")
    print("            CPAM 式2(proj) 式3(插值对齐) 式4(mixer) 式6(softmax尺度维) 式7(加权和)")
    print("=" * 88)

if __name__ == "__main__":
    main()