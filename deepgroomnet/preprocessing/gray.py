import subprocess
import cv2
import numpy as np

SOURCE_MEAN = 146.2030487060547
SOURCE_STD = 49.36723709106445


def gray_stats(path, samples=20):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    values = []
    for index in np.linspace(0, max(total - 1, 0), samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if ok:
            values.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).reshape(-1))
    cap.release()
    if not values:
        raise ValueError(f"cannot sample frames from {path}")
    data = np.concatenate(values)
    return float(data.mean()), float(data.std())


def encode_gray(src, dst, mu=None, sd=None, source_mean=SOURCE_MEAN, source_std=SOURCE_STD):
    if mu is None or sd is None:
        mu, sd = gray_stats(src)
    cap = cv2.VideoCapture(str(src))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
               "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
               "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
               "-pix_fmt", "yuv420p", "-an", "-threads", "2", str(dst)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray = np.clip((gray - mu) * (source_std / max(sd, 1e-6)) + source_mean, 0, 255).astype(np.uint8)
        process.stdin.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).tobytes())
        count += 1
    process.stdin.close()
    process.wait()
    cap.release()
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {dst}")
    return count
