# -*- coding: utf-8 -*-
"""
ARCHITECTURE DIAGNOSTIC (English console output)
================================================
Verify the DeepGroomNet_v3 forward call chain and module structure.

Method 1: Forward call-chain trace with tensor shapes.
Method 2: torchinfo structural summary (module presence only).
"""
import os
import sys

ROOT = r'D:\DeepGroomNet_v3'
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn

from models.tadaconv_model import TAdaConvModel


def trace_forward():
    print('=' * 72)
    print('METHOD 1 - FORWARD CALL CHAIN TRACE (shapes at each module)')
    print('=' * 72)

    model = TAdaConvModel(
        num_classes=5, t_stride=1, drop_path=0.0,
        use_cpam=True, use_stsa=True,
    )
    model.eval()

    # --- register forward hooks on key submodules (call order = hook order) ---
    hooks = []
    order = []
    seen = set()

    def make_hook(name):
        def hook(mod, inp, out):
            if name in seen:
                return
            seen.add(name)
            if isinstance(out, (tuple, list)):
                desc = ', '.join(
                    str(tuple(t.shape)) if torch.is_tensor(t) else str(t)
                    for t in out)
                order.append(f'  {name:<34} -> out: ({desc})')
            elif torch.is_tensor(out):
                order.append(f'  {name:<34} -> out: {tuple(out.shape)}')
            else:
                order.append(f'  {name:<34} -> out: {type(out).__name__}')
        return hook

    target = {
        'stem': model.downsample_layers[0],
        'downsample1': model.downsample_layers[1],
        'downsample2': model.downsample_layers[2],
        'downsample3': model.downsample_layers[3],
        'stage1': model.stages[0],
        'stage2': model.stages[1],
        'stage3': model.stages[2],
        'stage4': model.stages[3],
        'STSA3': model.stsa3,
        'STSA4': model.stsa4,
        'CPAM': model.cpam,
        'norm': model.norm,
        'head': model.head,
    }
    for name, mod in target.items():
        if mod is not None:
            hooks.append(mod.register_forward_hook(make_hook(name)))

    # --- run one forward pass with the real video layout ---
    print('\n>> Forward pass: dummy_x = torch.randn(2, 3, 32, 224, 224)  (N,C,T,H,W)\n')
    dummy_x = torch.randn(2, 3, 32, 224, 224)
    logits, aux_logits = model(dummy_x)

    # the STSA/CPAM/norm/head are called inside forward_features; hooks print in
    # actual invocation order regardless of our dict order
    print('\n--- Actual invocation order (top -> bottom) ---')
    for line in order:
        print(line)

    print('\n--- Return values ---')
    print(f'  main logits : {tuple(logits.shape)}')
    print(f'  aux  logits : '
          f'{None if aux_logits is None else tuple(aux_logits.shape)}')

    for h in hooks:
        h.remove()
    return model


def torchinfo_summary(model):
    print('\n' + '=' * 72)
    print('METHOD 2 - TORCHINFO STRUCTURAL SUMMARY (top-level overview)')
    print('=' * 72)
    from torchinfo import summary
    dummy = torch.randn(2, 3, 32, 224, 224)
    print(summary(model, input_data=dummy, depth=2, verbose=0,
                  col_names=["input_size", "output_size", "num_params"]))


def design_checklist():
    print('\n' + '=' * 72)
    print('DESIGN CHECKLIST (compare against intended design)')
    print('=' * 72)
    model = TAdaConvModel(num_classes=5, t_stride=1, use_cpam=True, use_stsa=True)
    checks = []
    # 1. stem / stages
    checks.append(('stem (Conv3d + LayerNorm) defined', model.downsample_layers[0] is not None))
    checks.append(('4 stages defined', len(model.stages) == 4))
    checks.append(('2 STSA modules defined (stsa3/stsa4)',
                   model.stsa3 is not None and model.stsa4 is not None))
    checks.append(('CPAM defined', model.cpam is not None))
    # 2. CPAM input = 4 stage features
    checks.append(('CPAM.num_stages == 4', model.cpam.num_stages == 4))
    checks.append(('CPAM has aux_head (final Linear)',
                   isinstance(model.cpam.aux_head[-1], nn.Linear)))
    # 3. head
    checks.append(('main head = Sequential(Dropout, Linear)',
                   isinstance(model.head, nn.Sequential) and isinstance(model.head[-1], nn.Linear)))
    # 4. norm
    checks.append(('final norm = LayerNorm over channel dim',
                   isinstance(model.norm, nn.LayerNorm)))
    for name, ok in checks:
        print(f'  [{"OK" if ok else "MISSING"}] {name}')


if __name__ == '__main__':
    m = trace_forward()
    torchinfo_summary(m)
    design_checklist()
    print('\nDONE')
