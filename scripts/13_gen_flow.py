import os, sys, csv
sys.path.insert(0, 'D:\\DeepGroomNet_v2\\third_party')
sys.path.insert(0, 'D:\\DeepGroomNet_v2')
import numpy as np
import torch
FLOW_ROOT = 'D:\\DeepGroomNet_v2\\outputs\\flow'
CKPT = 'D:\\pretrained_models\\200221_115158_TinyMotionNet\\checkpoint.pt'
CLIPS_CSV = 'D:\\DeepGroomNet_v2\\outputs\\clips\\clips.csv'
HEAT = 32
WINDOW = 11
OUT_IDX = 5
NUM_FRAMES = 16

def load_flow_model():
    import cv2
    from deepethogram.flow_generator.models.TinyMotionNet import TinyMotionNet
    m = TinyMotionNet(num_images=WINDOW)
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    m.load_state_dict(ck['state_dict'])
    m.eval()
    return m

def sample_indices(total, n=NUM_FRAMES):
    if total <= 0:
        return []
    if total == 1:
        return [0]
    return list(np.unique(np.linspace(0, total - 1, num=n).round().astype(int)))

def clip_flow(model, frames, device):
    import cv2
    T = len(frames)
    idxs = sample_indices(T)
    with torch.no_grad():
        outs = []
        for idx in idxs:
            lo = max(0, min(idx - OUT_IDX, T - WINDOW))
            win = frames[lo:lo + WINDOW]
            if len(win) < WINDOW:
                win = win + [win[-1]] * (WINDOW - len(win))
            arr = np.stack([cv2.resize(f, (256, 256)) for f in win]).astype(np.float32) / 255.0
            arr = np.transpose(arr, (0, 3, 1, 2))
            x = torch.from_numpy(arr.reshape(1, -1, 256, 256)).to(device)
            y = model(x)
            flow = y[0][0, OUT_IDX * 2:(OUT_IDX + 1) * 2]
            flow = torch.nn.functional.interpolate(flow.unsqueeze(0), size=(HEAT, HEAT), mode='bilinear', align_corners=False)[0]
            outs.append(flow.cpu().numpy())
    return np.stack(outs).astype(np.float16)

def read_clip_frames(path):
    import cv2
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        (ok, fr) = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames

def main():
    import time
    os.makedirs(FLOW_ROOT, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_flow_model().to(device)
    rows = list(csv.DictReader(open(CLIPS_CSV, encoding='utf-8')))
    done = set(os.listdir(FLOW_ROOT))
    todo = [r for r in rows if os.path.basename(r['clip']) + '.npz' not in done]
    print(f'clip 总数 {len(rows)}，待生成 {len(todo)}', flush=True)
    t0 = time.time()
    for (i, r) in enumerate(todo):
        src = r['clip']
        if not os.path.exists(src):
            print('  missing:', src)
            continue
        frames = read_clip_frames(src)
        if len(frames) < WINDOW:
            continue
        flow = clip_flow(model, frames, device)
        np.savez_compressed(os.path.join(FLOW_ROOT, os.path.basename(src) + '.npz'), flow=flow)
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            print(f'  {i + 1}/{len(todo)} | {el:.0f}s | ETA {el / (i + 1) * (len(todo) - i - 1) / 60:.0f}min', flush=True)
    print('done', flush=True)
if __name__ == '__main__':
    main()
