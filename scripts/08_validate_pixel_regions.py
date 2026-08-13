import argparse
import os
import random
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.clip_sampler import sample_frames
from data.dataset import ClipDataset
from data.mouse_mask import compute_mouse_mask
from models.attention import MotionStatAttention
from models.backbones.tadaconvnext import build_model
CLIPS_CSV = 'D:\\DeepGroomNet_v2\\outputs\\clips\\clips.csv'
CLIP_DIR = 'D:\\DeepGroomNet_v2\\outputs\\clips'

def load_config():
    return {'MODEL': {'backbone': 'TAdaConvNeXtV2_Tiny', 'num_classes': 5, 'attention': 'motion_stat', 'attention_reductions': {'motion': 16}, 'attention_positions': [0, 1, 2, 3], 'dropout': 0.4, 'init_weights': 'C:/Users/WIN10/PycharmProjects/mouseaction/DeepBehavior/TAdaConv/tadaconvnextv2_tiny_in1k_k400_79.6.pyth'}, 'CLASSES': {'openfield_keep': ['Locomote', 'Rear', 'FaceGroom', 'BodyGroom', 'background']}}

def cohens_d(a, b):
    (a, b) = (np.asarray(a, float), np.asarray(b, float))
    (na, nb) = (len(a), len(b))
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp

def body_mask(arr):
    import cv2
    gray = arr.mean(axis=3)
    body = gray.mean(axis=0)
    g8 = (body * 255).astype(np.uint8)
    (_, bw) = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_otsu = (bw / 255.0).astype(np.float32)
    (h, w) = body.shape
    (y0, y1) = (int(h * 0.2), int(h * 0.8))
    (x0, x1) = (int(w * 0.2), int(w * 0.8))
    mask_center = np.zeros((h, w), np.float32)
    mask_center[y0:y1, x0:x1] = 1.0
    mask_mbg = compute_mouse_mask(arr, use_center_union=False)
    return (mask_otsu, mask_center, mask_mbg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    torch.manual_seed(0)
    random.seed(0)
    device = torch.device(args.device)
    model = build_model(load_config()).to(device).eval()
    stat0 = [m for attn in model.attentions for m in ([attn] if not isinstance(attn, torch.nn.Sequential) else list(attn)) if isinstance(m, MotionStatAttention)][0]
    stat0._capture = True
    df = pd.read_csv(CLIPS_CSV)
    clips = {l: df[df['label'] == l]['clip'].tolist() for l in ('FaceGroom', 'BodyGroom')}
    rows = []
    with torch.no_grad():
        for (label, clip_list) in clips.items():
            random.shuffle(clip_list)
            for clip in clip_list[:args.n]:
                path = os.path.join(CLIP_DIR, clip)
                frames = np.stack(ClipDataset._read_clipfile(path))
                idx = sample_frames(0, len(frames), 16)
                arr = frames[idx].astype(np.float32) / 255.0
                (mask_otsu, mask_center, mask_mbg) = body_mask(arr)
                g = arr.mean(axis=3)
                m1 = np.abs(g[1:] - g[:-1])
                m4 = np.abs(g[4:] - g[:-4])
                m1 = np.concatenate([m1, np.zeros((3, 224, 224))], axis=0)
                m4 = np.concatenate([m4, np.zeros((4, 224, 224))], axis=0)
                r = m1.mean(axis=0) / (m4.mean(axis=0) + 0.0001)
                row = {'label': label, 'clip': clip, 'otsu_cov': mask_otsu.mean(), 'cen_cov': mask_center.mean(), 'mbg_cov': mask_mbg.mean()}
                for (tag, mk) in (('otsu', mask_otsu), ('cen', mask_center), ('mbg', mask_mbg)):
                    mi = mk > 0.5
                    row[f'{tag}_in_fast'] = r[mi].mean()
                    row[f'{tag}_out_fast'] = r[~mi].mean()
                    row[f'{tag}_in_mot'] = m1.mean(axis=0)[mi].mean()
                    row[f'{tag}_out_mot'] = m1.mean(axis=0)[~mi].mean()
                rows.append(row)
                x_arr = arr
                if int(cfg['CLIP'].get('motion_channel', 0)):
                    from data.dataset import add_motion_channel
                    x_arr = add_motion_channel(arr)
                x = torch.from_numpy(x_arr.transpose(3, 0, 1, 2)).unsqueeze(0).to(device)
                model(x)
                (r_n, mu, sg, c, w_s, mm1) = stat0._capture_cache
                fast_m = r_n[0].mean(dim=0)
                gate_m = w_s[0, 0].mean(dim=0)
                for (tag, mk) in (('otsu', mask_otsu), ('cen', mask_center), ('mbg', mask_mbg)):
                    msk = F.interpolate(torch.from_numpy(mk).view(1, 1, 224, 224), size=w_s.shape[-2:], mode='nearest').to(device).float()
                    msk = (msk > 0.5).float()
                    mi = msk[0, 0] > 0.5
                    rows[-1][f'm_{tag}_in_fast'] = fast_m[mi].mean().item()
                    rows[-1][f'm_{tag}_out_fast'] = fast_m[~mi].mean().item()
                    rows[-1][f'm_{tag}_in_gate'] = gate_m[mi].mean().item()
                    rows[-1][f'm_{tag}_out_gate'] = gate_m[~mi].mean().item()
                    rows[-1][f'gate_{tag}_cov'] = (gate_m * msk[0, 0]).sum().item() / gate_m.sum().item()
                    rows[-1][f'fast_{tag}_cov'] = (fast_m * msk[0, 0]).sum().item() / fast_m.sum().item()
    dfr = pd.DataFrame(rows)
    (fg, bg) = (dfr[dfr.label == 'FaceGroom'], dfr[dfr.label == 'BodyGroom'])
    print(f'[region] n={len(fg)}/{len(bg)}')
    print(f'  掩码覆盖度: otsu FG={fg.otsu_cov.mean():.2f} BG={bg.otsu_cov.mean():.2f} | center FG={fg.cen_cov.mean():.2f} BG={bg.cen_cov.mean():.2f} | mbg FG={fg.mbg_cov.mean():.2f} BG={bg.mbg_cov.mean():.2f}')
    print('\n== 输入级（原始像素，无模型） ==')
    for tag in ('otsu', 'cen', 'mbg'):
        for col in (f'{tag}_in_fast', f'{tag}_out_fast', f'{tag}_in_mot', f'{tag}_out_mot'):
            d = cohens_d(fg[col], bg[col])
            print(f'  {col:>12s}: FG={fg[col].mean():.4f} vs BG={bg[col].mean():.4f}  Cohen d={d:+.2f}')
    print('\n== 模块级（stage0，56×56） ==')
    for tag in ('otsu', 'cen', 'mbg'):
        for col in (f'm_{tag}_in_fast', f'm_{tag}_out_fast', f'm_{tag}_in_gate', f'm_{tag}_out_gate', f'gate_{tag}_cov', f'fast_{tag}_cov'):
            d = cohens_d(fg[col], bg[col])
            print(f'  {col:>14s}: FG={fg[col].mean():.4f} vs BG={bg[col].mean():.4f}  Cohen d={d:+.2f}')
    out = os.path.join(os.path.dirname(CLIPS_CSV), 'attention_region.csv')
    dfr.to_csv(out, index=False)
    print(f'[region] saved -> {out}')
if __name__ == '__main__':
    main()
