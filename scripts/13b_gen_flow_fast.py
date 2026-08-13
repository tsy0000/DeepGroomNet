import os, sys, csv, time
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
NWORKERS = 8

def sample_indices(total, n=NUM_FRAMES):
    if total <= 0:
        return []
    if total == 1:
        return [0]
    return list(np.unique(np.linspace(0, total - 1, num=n).round().astype(int)))

def _window_arr(frames, idx, T):
    import cv2
    lo = max(0, min(idx - OUT_IDX, T - WINDOW))
    win = frames[lo:lo + WINDOW]
    if len(win) < WINDOW:
        win = win + [win[-1]] * (WINDOW - len(win))
    arr = np.stack([cv2.resize(f, (256, 256)) for f in win]).astype(np.float32) / 255.0
    return np.transpose(arr, (0, 3, 1, 2)).reshape(-1, 256, 256)

def worker(rows):
    import cv2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from deepethogram.flow_generator.models.TinyMotionNet import TinyMotionNet
    m = TinyMotionNet(num_images=WINDOW)
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    m.load_state_dict(ck['state_dict'])
    m = m.to(device).eval()
    done = 0
    t0 = time.time()
    for r in rows:
        src = r['clip']
        out = os.path.join(FLOW_ROOT, os.path.basename(src) + '.npz')
        if os.path.exists(out):
            continue
        cap = cv2.VideoCapture(src)
        T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = sample_indices(T)
        need = set()
        for idx in idxs:
            lo = max(0, min(idx - OUT_IDX, T - WINDOW))
            need.update(range(lo, lo + WINDOW))
        frames = [None] * T
        for fi in sorted(need):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            (ok, fr) = cap.read()
            if ok:
                frames[fi] = fr
        cap.release()
        if any((f is None for f in frames[:T])):
            continue
        xs = np.stack([_window_arr(frames, idx, T) for idx in idxs])
        with torch.no_grad():
            y = m(torch.from_numpy(xs).to(device))
        flows = y[0][:, OUT_IDX * 2:(OUT_IDX + 1) * 2]
        flows = torch.nn.functional.interpolate(flows, size=(HEAT, HEAT), mode='bilinear', align_corners=False)
        np.savez(out, flow=flows.cpu().numpy().astype(np.float16))
        done += 1
        if done % 300 == 0:
            print(f'  worker 进度 {done} clips | {(time.time() - t0) / done:.3f}s/clip', flush=True)
    return done

def main():
    import multiprocessing as mp
    os.makedirs(FLOW_ROOT, exist_ok=True)
    rows = list(csv.DictReader(open(CLIPS_CSV, encoding='utf-8')))
    done = set(os.listdir(FLOW_ROOT))
    todo = [r for r in rows if os.path.basename(r['clip']) + '.npz' not in done]
    print(f'待生成 {len(todo)} / {len(rows)}，{NWORKERS} 进程并行', flush=True)
    t0 = time.time()
    chunks = [todo[i::NWORKERS] for i in range(NWORKERS)]
    with mp.Pool(NWORKERS) as pool:
        totals = pool.map(worker, chunks)
    print(f'完成 {sum(totals)} | 总耗时 {(time.time() - t0) / 60:.1f} min', flush=True)
if __name__ == '__main__':
    main()
