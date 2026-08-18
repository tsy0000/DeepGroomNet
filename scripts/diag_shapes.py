# -*- coding: utf-8 -*-
"""维度诊断: 逐步打印 CPAM 与 STSA 模块内部每个算子的张量形状变换。

用途: 核对 CPAM / STSA 模块结构构造是否符合论文式2-15 与预期维度。
用法:  python scripts/diag_shapes.py   (CPU 即可, 用小 batch)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from models.modules.cpam import CPAM
from models.modules.stsa import STSA, spatial_shift1, spatial_shift2


def banner(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def trace_cpam(cpam, stage_feats):
    banner("CPAM 内部维度变换  (论文式2-7)")
    print(f"输入 4 个 stage 特征:")
    for i, f in enumerate(stage_feats):
        print(f"  stage{i+1} X: {tuple(f.shape)}   (N,C,T,H,W)")
    target = stage_feats[-1].shape[2:]
    print(f"\n对齐目标 (最深层空间尺寸) T,H,W = {tuple(target)}")

    proj = []
    for i, x in enumerate(stage_feats):
        p = cpam.proj[i](x)
        print(f"  式2 proj[{i}] 1x1 Conv({x.shape[1]}->{cpam.out_ch}): {tuple(x.shape)} -> {tuple(p.shape)}")
        if p.shape[2:] != tuple(target):
            p = torch.nn.functional.interpolate(
                p, size=tuple(target), mode="trilinear", align_corners=False)
            print(f"  式3 interpolate 对齐: -> {tuple(p.shape)}")
        if cpam.mixers is not None:
            p = cpam.mixers[i](p)
            print(f"  式4 mixer[{i}] (3,1,1)Conv+BN+GELU: -> {tuple(p.shape)}")
        proj.append(p)

    X = torch.stack(proj, dim=1)
    print(f"  式5 stack 沿新尺度维: -> {tuple(X.shape)}   (N, num_stages, C, T, H, W)")

    fusion = cpam.fusion
    B, height, C, T, H, W = X.shape
    feats_sum = X.sum(dim=1)
    print(f"  式6 求和统计 feats_sum: {tuple(X.shape)} -> {tuple(feats_sum.shape)}")
    pooled = fusion.avg_pool(feats_sum)
    print(f"       GAP(3d): -> {tuple(pooled.shape)}")
    mlp_out = fusion.mlp(pooled)
    print(f"       MLP (C->d->C*{height}): -> {tuple(mlp_out.shape)}")
    attn = torch.nn.functional.softmax(mlp_out.view(B, height, C, 1, 1, 1), dim=1)
    print(f"       softmax 尺度维: -> {tuple(attn.shape)}   (每个 scale 一个权重)")
    Y = torch.sum(X * attn, dim=1)
    print(f"  式7 加权求和 Y: -> {tuple(Y.shape)}   (N, C, T, H, W)")

    aux_logits = cpam.aux_head(Y)
    print(f"  aux_head (GAP+Flatten+FC): -> {tuple(aux_logits.shape)}   (N, num_classes)")
    return Y, aux_logits


def trace_stsa(stsa, x):
    banner("STSA 内部维度变换  (论文式11-15)")
    C = stsa.channels
    N_, C_, T, H, W = x.shape
    print(f"输入 x: {tuple(x.shape)}   (N,C,T,H,W), C={C}")
    chi = stsa.mlp1(x)
    print(f"  式11 mlp1 (C->3C): {tuple(x.shape)} -> {tuple(chi.shape)}")

    x1 = chi[:, :C]
    x2 = chi[:, C:2 * C]
    x3 = chi[:, 2 * C:]
    print(f"  式12 split 三路:")
    print(f"       x1: {tuple(x1.shape)}   x2: {tuple(x2.shape)}   x3: {tuple(x3.shape)}")

    sx1 = spatial_shift1(x1)
    sx2 = spatial_shift2(x2)
    print(f"  式13/14 非对称空间移位 (仅 H/W, T 不变, χ3 不变):")
    print(f"       ss1(x1): {tuple(sx1.shape)}   ss2(x2): {tuple(sx2.shape)}   x3: {tuple(x3.shape)}")
    feats = [sx1, sx2, x3]

    # SA 分支
    sa = stsa.sa
    x_stack = torch.stack(feats, dim=1)
    print(f"  SA: stack 三路 -> {tuple(x_stack.shape)}")
    fsum = x_stack.sum(dim=1)
    print(f"      求和 -> {tuple(fsum.shape)}")
    pooled = fsum.mean(dim=(2, 3, 4), keepdim=True)
    print(f"      GAP -> {tuple(pooled.shape)}")
    sa_mlp = sa.mlp(pooled)
    print(f"      MLP(C->{sa.mlp[0].out_channels}->{C*sa.groups}) -> {tuple(sa_mlp.shape)}")
    sa_attn = torch.nn.functional.softmax(sa_mlp.view(x.shape[0], sa.groups, C, 1, 1, 1), dim=1)
    print(f"      softmax(分组维) -> {tuple(sa_attn.shape)}")
    sa_out = (x_stack * sa_attn).sum(dim=1)
    print(f"      加权求和 -> {tuple(sa_out.shape)}")

    # TA 分支
    ta = stsa.ta
    diff = torch.zeros_like(x_stack)
    diff[:, :, :, 1:, :, :] = x_stack[:, :, :, 1:, :, :] - x_stack[:, :, :, :-1, :, :]
    diff3c = diff.reshape(x.shape[0], ta.groups * C, T, H, W)
    print(f"  TA: 帧差分 diff3c -> {tuple(diff3c.shape)}")
    w = ta.sigmoid(ta.conv(diff3c))
    print(f"      (3,1,1)深度卷积+sigmoid -> {tuple(w.shape)}")
    w = w.reshape(x.shape[0], ta.groups, C, T, H, W)
    w = torch.nn.functional.softmax(w, dim=3)
    print(f"      softmax(时间维T) -> {tuple(w.shape)}")
    ta_out = (x_stack * w).sum(dim=1)
    print(f"      加权求和 -> {tuple(ta_out.shape)}")

    print(f"  式15 SA+TA -> {tuple(sa_out.shape)}")
    out = stsa.mlp2(stsa.act(sa_out + ta_out))
    print(f"      mlp2 (C->C) -> {tuple(out.shape)}")
    out = x + out
    print(f"  残差 out = x + mlp2: -> {tuple(out.shape)}   (通道不变, 即插即用)")
    return out


def main():
    torch.manual_seed(0)
    N = 2
    stage_channels = [96, 192, 384, 768]
    out_ch = 768
    T, H, W = 32, 14, 14  # 模拟最深层之前各 stage 空间尺寸
    num_classes = 5

    cpam = CPAM(stage_channels, num_classes, out_ch=out_ch)
    cpam.eval()
    print(f"CPAM 参数总量: {sum(p.numel() for p in cpam.parameters()):,}")

    # 构造与真实前向一致的 4 个 stage 特征 (空间尺寸递减)
    stage_feats = [
        torch.randn(N, 96, 32, 56, 56),
        torch.randn(N, 192, 32, 28, 28),
        torch.randn(N, 384, 32, 14, 14),
        torch.randn(N, 768, 32, 7, 7),
    ]
    Y, aux = trace_cpam(cpam, stage_feats)

    banner("STSA3 / STSA4 内部维度变换")
    print(f"STSA3 参数: {sum(p.numel() for p in STSA(384).parameters()):,}   (输入 384, stage3 后)")
    print(f"STSA4 参数: {sum(p.numel() for p in STSA(768).parameters()):,}   (输入 768, stage4 后)")
    stsa3 = STSA(384)
    stsa3.eval()
    trace_stsa(stsa3, torch.randn(N, 384, 32, 14, 14))
    stsa4 = STSA(768)
    stsa4.eval()
    trace_stsa(stsa4, torch.randn(N, 768, 32, 7, 7))

    banner("一致性核对")
    print("CPAM : Y 应为 (N, 768, 32, 7, 7),  aux_logits 应为 (N, 5)")
    print(f"      实际: Y={tuple(Y.shape)}, aux={tuple(aux.shape)}")
    print("STSA3: 输入输出应同为 (N,384,32,14,14)  (残差, 即插即用)")
    print("STSA4: 输入输出应同为 (N,768,32,7,7)   (残差, 即插即用)")
    print("主干  : 见 scripts/diag_architecture.py METHOD1 的逐模块 trace")


if __name__ == "__main__":
    main()