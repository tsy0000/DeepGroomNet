# web_app.py — DeepGroomNet Mouse Behavior Prediction System
# Complete replication of the original pred_openfield.py feature set,
# adapted to the DeepGroomNet_v3 model (TAdaConvNeXtV2 + CPAM).
# Run:  streamlit run web_app.py

import streamlit as st
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)   # 项目根在 sys.path (vendored tadaconv 为顶层包)

import cv2
import torch
import numpy as np
import pandas as pd
import logging
import tempfile
import subprocess
import glob
import warnings
import io
import json
import h5py
from datetime import datetime
from collections import defaultdict
from types import SimpleNamespace
import torch.nn.functional as F

warnings.filterwarnings('ignore')

from models.tadaconv_model import TAdaConvModel

os.environ['YOLO_SETTINGS_DIR'] = tempfile.gettempdir()
from ultralytics import YOLO

# ============================================================ GLOBAL STYLING
def apply_global_styling():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', Arial, sans-serif !important; }
        p, span, label, button, input, select, .stMetricValue, .stMetricLabel,
        .stMarkdownContainer p, .stMarkdownContainer span { font-size: 14px !important; }
        h2, h3, h4, [data-testid="stTabs"] button p { font-size: 20px !important; font-weight: 700 !important; }
        #super-title-container { text-align: center !important; margin: 30px 0 !important; }
        #super-title { font-size: 52px !important; font-weight: 900 !important;
                       color: #1f77b4 !important; line-height: 1.0 !important; }
        div[data-testid="stFileUploaderDropzone"] { border: 1.5px dashed #94a3b8 !important;
            border-radius: 12px !important; padding: 12px !important; }
        div[data-testid="stFileUploaderDropzone"]:hover { border-color: #1f77b4 !important; background: #f8f9fa !important; }
        .stButton>button { border-radius: 10px !important; padding: .5rem 1.2rem !important; font-weight: 600 !important; }
        .block-container { padding-top: 1.5rem; }
    </style>
    """, unsafe_allow_html=True)

# ============================================================ CONSTANTS
CLASS_NAMES = ['Locomote', 'Rear', 'FaceGroom', 'BodyGroom', 'background']
BG_NAME = 'background'
NUM_CLASSES = len(CLASS_NAMES)
NUM_FRAMES = 32
TARGET_SIZE = (224, 224)
BOX_EXPANSION_RATIO = 1.5
MEAN = [0.45, 0.45, 0.45]
STD = [0.225, 0.225, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLE = torch.cuda.is_available()

# ============================================================ LOGGING
@st.cache_resource
def setup_logging():
    logger = logging.getLogger("BehaviorPrediction")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        console = logging.StreamHandler(); console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console)
        log_dir = os.path.join(tempfile.gettempdir(), "behavior_prediction_logs")
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"), encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(fh)
    return logger

logger = None  # 惰性初始化在 main() 内 (避免模块导入时 bare-mode ScriptRunContext 警告)

# ============================================================ DLC KEYPOINTS
def load_dlc_h5(h5_path):
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'df_with_categories' in f:
                data = f['df_with_categories']['table'][:]
                return pd.DataFrame(data)
            keys = list(f.keys())
            if keys:
                return pd.DataFrame(f[keys[0]][:])
        return None
    except Exception as e:
        st.error(f"Failed to load H5 file: {str(e)}")
        return None

def load_dlc_csv(csv_path):
    try:
        return pd.read_csv(csv_path, header=[0, 1, 2])
    except Exception:
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            st.error(f"Failed to load CSV: {str(e)}")
            return None

def extract_keypoints_from_dlc(dlc_data, frame_idx):
    try:
        if dlc_data is None:
            return None
        if isinstance(dlc_data.columns, pd.MultiIndex):
            keypoints = {}
            for col in dlc_data.columns:
                if col[2] in ['x', 'y']:
                    bp = col[1]
                    keypoints.setdefault(bp, {})[col[2]] = float(dlc_data.iloc[frame_idx][col])
            return keypoints or None
        else:
            keypoints = {}
            for col in dlc_data.columns:
                if 'x' in str(col).lower() or 'y' in str(col).lower():
                    keypoints[col] = float(dlc_data.iloc[frame_idx][col])
            return keypoints or None
    except Exception:
        return None

def draw_keypoints_on_frame(frame, keypoints, radius=3, thickness=-1):
    if keypoints is None:
        return frame
    out = frame.copy()
    h, w = frame.shape[:2]
    colors = {'nose': (0, 255, 0), 'left_eye': (255, 0, 0), 'right_eye': (0, 0, 255),
              'left_ear': (255, 255, 0), 'right_ear': (255, 0, 255), 'neck': (0, 255, 255),
              'spine1': (128, 255, 0), 'spine2': (255, 128, 0), 'tail_base': (128, 0, 255),
              'tail_mid': (255, 128, 128), 'tail_tip': (128, 128, 255)}
    for bp, coords in keypoints.items():
        if isinstance(coords, dict) and 'x' in coords and 'y' in coords:
            x, y = int(coords['x']), int(coords['y'])
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(out, (x, y), radius, colors.get(bp, (255, 255, 255)), thickness)
    return out

# ============================================================ MoSEQ
def parse_moseq_csv(csv_data):
    try:
        df = pd.read_csv(csv_data, header=0) if isinstance(csv_data, str) else csv_data
        syllable_series = df.iloc[:, 0].dropna().values.astype(int)
        if len(syllable_series) == 0:
            raise ValueError("MoSeq CSV has no valid frame data")
        segments = []
        current_syl, start = syllable_series[0], 0
        for i in range(1, len(syllable_series)):
            if syllable_series[i] != current_syl:
                segments.append({"syllable": int(current_syl), "start": int(start),
                                 "end": int(i - 1), "length": int(i - start)})
                current_syl, start = syllable_series[i], i
        segments.append({"syllable": int(current_syl), "start": int(start),
                         "end": int(len(syllable_series) - 1), "length": int(len(syllable_series) - start)})
        logger.info(f"MoSeq segments parsed: {len(segments)} segments, {len(syllable_series)} frames")
        return segments, int(len(syllable_series))
    except Exception as e:
        logger.error(f"MoSeq parsing failed: {str(e)}", exc_info=True)
        raise

# ============================================================ VIDEO / CROP / TENSOR
def read_video_frames(video_path, frame_idxs):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")
    raw = []
    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = raw[-1] if raw else np.zeros((224, 224, 3), dtype=np.uint8)
        raw.append(frame)
    cap.release()
    return raw

def get_unified_crop_box(frames, yolo_model):
    all_boxes = []
    for frame in frames:
        try:
            res = yolo_model(frame, conf=0.5, verbose=False)
            if res[0].boxes is not None and len(res[0].boxes) > 0:
                all_boxes.append(res[0].boxes.xyxy[0].cpu().numpy())
        except Exception:
            pass
    if not all_boxes:
        h, w = frames[0].shape[:2]
        return 0, 0, w, h
    all_boxes = np.array(all_boxes)
    min_x, min_y = all_boxes[:, 0].min(), all_boxes[:, 1].min()
    max_x, max_y = all_boxes[:, 2].max(), all_boxes[:, 3].max()
    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
    side = max(max_x - min_x, max_y - min_y) * BOX_EXPANSION_RATIO
    half = side / 2
    x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
    x2, y2 = int(min(frames[0].shape[1], cx + half)), int(min(frames[0].shape[0], cy + half))
    w, h = x2 - x1, y2 - y1
    if w > h:
        d = w - h; y1 = max(0, y1 - d // 2); y2 = min(frames[0].shape[0], y2 + (d - d // 2))
    else:
        d = h - w; x1 = max(0, x1 - d // 2); x2 = min(frames[0].shape[1], x2 + (d - d // 2))
    return int(x1), int(y1), int(x2), int(y2)

def pad_frames_to_16(raw_frames):
    pad = NUM_FRAMES - len(raw_frames)
    if pad > 0:
        raw_frames += [raw_frames[-1]] * pad
    return raw_frames[:NUM_FRAMES]

def split_segment_into_16f_chunks(seg):
    start, end, out = seg["start"], seg["end"], []
    for i in range(0, end - start + 1, NUM_FRAMES):
        cs = start + i
        out.append(list(range(cs, min(cs + NUM_FRAMES, end + 1))))
    return out

def process_16f_to_tensor(raw_frames, yolo_model):
    raw_frames = pad_frames_to_16(raw_frames)
    x1, y1, x2, y2 = get_unified_crop_box(raw_frames, yolo_model)
    processed = []
    for frame in raw_frames:
        cropped = frame[y1:y2, x1:x2]
        resized = cv2.resize(cropped, TARGET_SIZE, cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        processed.append(t)
    video = torch.stack(processed, dim=1).to(DEVICE)          # [C,T,H,W]
    video = (video - torch.tensor(MEAN, device=DEVICE).view(3, 1, 1, 1)) / \
            torch.tensor(STD, device=DEVICE).view(3, 1, 1, 1)
    return video.unsqueeze(0)                                 # [1,C,T,H,W]

def predict_single_chunk(model, input_tensor):
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=AMP_ENABLE):
            out = model(input_tensor)
            if isinstance(out, tuple):
                out = out[0]
            prob = torch.sigmoid(out.float())
            conf, cls_idx = torch.max(prob, dim=1)
    return cls_idx.item(), conf.item()

def load_csv_file(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to load CSV: {str(e)}")
        return None


DLC_ENV_PY = r"D:\Anaconda3\envs\deeplabcut\python.exe"
MOSSEQ_ENV_PY = r"D:\Anaconda3\envs\keypoint_moseq\python.exe"


def run_moseq_fit(video_path, keypoint_dir, kappa=1e6, latent_dim=None):
    """现场运行 Keypoint-MoSeq 拟合 (镜像 notebook: PCA -> AR-HMM -> 提取音节)。

    通过子进程调用 keypoint_moseq 环境 (scripts/moseq_postprocess.py, Agg 不弹图)。
    kappa: 音节时间尺度刚度; latent_dim: PCA 成分数 (None=自动按90%方差)。
    返回 {video_name: syllable_array}。
    """
    out_dir = os.path.join(tempfile.gettempdir(), "moseq_fit_out")
    os.makedirs(out_dir, exist_ok=True)
    script = os.path.join(PROJECT_ROOT, "scripts", "moseq_postprocess.py")
    base = os.path.splitext(os.path.basename(video_path))[0]
    # 找该视频的关键点文件 (DLC SuperAnimal 输出的 after_adapt csv / h5)
    kp_file = None
    for pattern in (f"{base}*.csv", f"{base}*.h5"):
        for cand in glob.glob(os.path.join(keypoint_dir, "**", pattern), recursive=True):
            kp_file = cand; break
        if kp_file:
            break
    if kp_file is None:
        st.error(f"未找到 {base} 的关键点文件, 请先运行关键点检测")
        logger.error(f"no keypoint file for {base} in {keypoint_dir}")
        return {}
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"   # 拟合中途绝不弹图
    args = [MOSSEQ_ENV_PY, script, video_path, kp_file, out_dir, str(kappa)]
    if latent_dim:
        args.append(str(latent_dim))
    with st.spinner("Running Keypoint-MoSeq fit (PCA + AR-HMM, Agg 无图)... may take minutes"):
        result = subprocess.run(args, capture_output=True, text=True, timeout=7200, env=env)
    if result.returncode != 0:
        st.error(f"Keypoint-MoSeq fit failed: {result.stderr[-800:]}")
        logger.error(f"moseq fit error: {result.stderr[-800:]}")
        return {}
    syl_file = os.path.join(out_dir, f"{base}_syllable.npy")
    if not os.path.exists(syl_file):
        return {}
    return {base: np.load(syl_file)}


def apply_syllable_boundaries_to_results(frame_results, syllables, video_basename):
    """用 Keypoint-MoSeq 音节边界对逐帧行为做音节分界 (只在音节切换处允许行为变化)。

    对每个音节段取段内主导行为; 若某帧当前预测与段内主导不同, 保留主导(自然边界)。
    """
    import collections
    key = None
    for k in syllables:
        if k.startswith(video_basename):
            key = k; break
    if key is None:
        return frame_results
    syl = syllables[key]
    n = min(len(syl), len(frame_results))
    out = [dict(r) for r in frame_results]
    i = 0
    while i < n:
        j = i
        while j < n and syl[j] == syl[i]:
            j += 1
        counts = collections.Counter(out[k]["pred"] for k in range(i, j))
        majority = counts.most_common(1)[0][0] if counts else out[i]["pred"]
        for k in range(i, j):
            out[k]["pred"] = majority
        i = j
    return out

def detect_keypoints_superanimal(video_path, supermodel="superanimal_topviewmouse"):
    """用 DLC SuperAnimal 检测视频关键点 (HRNet + FasterRCNN), 返回关键点 CSV 路径。

    依赖 deeplabcut 环境 (DLC_ENV_PY)。通过子进程调用, 避免污染运行环境。
    """
    out_dir = os.path.join(tempfile.gettempdir(), "dlc_keypoints")
    os.makedirs(out_dir, exist_ok=True)
    script = (
        "import deeplabcut, os, sys;"
        "deeplabcut.video_inference_superanimal("
        "[sys.argv[1]], sys.argv[2], model_name='hrnet_w32',"
        "detector_name='fasterrcnn_resnet50_fpn_v2', videotype='.mp4',"
        "video_adapt=True, scale_list=[], pcutoff=0.5)"
    )
    result = subprocess.run([DLC_ENV_PY, "-c", script, video_path, supermodel],
                            capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        st.error(f"Keypoint detection failed: {result.stderr[-800:]}")
        return None
    # 找输出的关键点 csv
    base = os.path.splitext(os.path.basename(video_path))[0]
    cands = glob.glob(os.path.join(os.path.dirname(video_path), "**", f"{base}*.csv"), recursive=True)
    cands += glob.glob(os.path.join(out_dir, "**", f"{base}*.csv"), recursive=True)
    return cands[0] if cands else None

# ============================================================ MODEL LOADING
@st.cache_resource
def load_models(tadaconv_weight_path, yolo_det_path):
    try:
        if not os.path.exists(tadaconv_weight_path):
            st.error("TAdaConv weight file not found")
            return None, None
        with st.spinner("Loading TAdaConv model..."):
            model = TAdaConvModel(num_classes=NUM_CLASSES, t_stride=1,
                                   use_cpam=True, use_stsa=True, drop_path=0.0).to(DEVICE)
            ckpt = torch.load(tadaconv_weight_path, map_location=DEVICE, weights_only=False)
            state = ckpt["model_state"] if "model_state" in ckpt else ckpt
            if isinstance(state, dict) and any(k.startswith("model.") for k in state):
                state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
            model.load_state_dict(state, strict=False)
            model.eval()
        st.success("✓ TAdaConv model loaded successfully")
        logger.info("TAdaConv model loaded successfully")
        if not os.path.exists(yolo_det_path):
            st.error("YOLO weight file not found")
            return model, None
        with st.spinner("Loading YOLO model..."):
            yolo_model = YOLO(yolo_det_path, verbose=False)
        st.success("✓ YOLO loaded successfully")
        logger.info("YOLO model loaded successfully")
        return model, yolo_model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        logger.error(f"Model loading error: {str(e)}", exc_info=True)
        return None, None

# ============================================================ PREDICTION
def predict_main(model, yolo_model, video_path, moseq_csv_path):
    cap = cv2.VideoCapture(video_path)
    video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    moseq_segments, moseq_total_frames = parse_moseq_csv(moseq_csv_path)
    if moseq_total_frames != video_total_frames:
        logger.warning(f"MoSeq frames ({moseq_total_frames}) != Video frames ({video_total_frames})")
    total_frames = min(video_total_frames, moseq_total_frames)
    frame_results = [{"frame_idx": i, "pred": None, "conf": 0.0, "syllable": -1} for i in range(total_frames)]
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    prediction_log = {"video_frames": video_total_frames, "moseq_frames": moseq_total_frames,
                      "process_frames": total_frames, "segments": len(moseq_segments),
                      "predictions": defaultdict(int)}
    for seg_idx, seg in enumerate(moseq_segments):
        s, e, syl = seg["start"], seg["end"], seg["syllable"]
        if s >= total_frames:
            break
        for i in range(s, min(e + 1, total_frames)):
            frame_results[i]["syllable"] = syl
        if seg["length"] < 8:                                   # short segment -> background
            pred_name = BG_NAME; conf = 1.0
            for i in range(s, min(e + 1, total_frames)):
                frame_results[i]["pred"] = pred_name; frame_results[i]["conf"] = conf
                prediction_log["predictions"][pred_name] += 1
            continue
        chunks = split_segment_into_16f_chunks(seg)
        status_placeholder.text(f"Processing segment [{s}-{e}] ({len(chunks)} chunks)")
        for chunk_idx, frame_idxs in enumerate(chunks):
            try:
                raw_frames = read_video_frames(video_path, frame_idxs)
                input_tensor = process_16f_to_tensor(raw_frames, yolo_model)
                cls_idx, conf = predict_single_chunk(model, input_tensor)
                pred_name = CLASS_NAMES[cls_idx]
                for idx in frame_idxs:
                    if idx < total_frames:
                        frame_results[idx]["pred"] = pred_name
                        frame_results[idx]["conf"] = conf
                        prediction_log["predictions"][pred_name] += 1
                progress = (seg_idx + chunk_idx / len(chunks)) / len(moseq_segments)
                progress_placeholder.progress(min(progress, 1.0))
            except Exception as e:
                logger.error(f"Chunk prediction failed [{s}-{e}]: {str(e)}", exc_info=True)
                for idx in frame_idxs:
                    if idx < total_frames:
                        frame_results[idx]["pred"] = BG_NAME
                        frame_results[idx]["conf"] = 0.5
                        prediction_log["predictions"][BG_NAME] += 1
    unfilled = 0
    for i in range(total_frames):
        if frame_results[i]["pred"] is None:
            frame_results[i]["pred"] = BG_NAME; frame_results[i]["conf"] = 1.0
            unfilled += 1; prediction_log["predictions"][BG_NAME] += 1
    status_placeholder.text("✓ Frame prediction completed")
    # 音节分界后处理: 段内取主导行为, 只在音节边界处变 -> 保留动作自然边界
    frame_results = apply_syllable_smoothing(frame_results, moseq_segments)
    logger.info("Prediction Statistics Report")
    for k, v in prediction_log.items():
        if k != "predictions":
            logger.info(f"  {k}: {v}")
    total_pred = sum(prediction_log["predictions"].values())
    for behavior in CLASS_NAMES:
        c = prediction_log["predictions"][behavior]
        logger.info(f"  {behavior:<20}: {c:>6} frames ({(c/total_pred*100) if total_pred else 0:.2f}%)")
    save_prediction_log(frame_results, prediction_log)
    return frame_results

def save_prediction_log(frame_results, prediction_log):
    log_dir = os.path.join(tempfile.gettempdir(), "behavior_prediction_logs")
    os.makedirs(log_dir, exist_ok=True)
    detail = os.path.join(log_dir, f"prediction_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    pd.DataFrame(frame_results).to_csv(detail, index=False, encoding='utf-8')
    summary = os.path.join(log_dir, f"prediction_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(summary, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\nBehavior Prediction Summary\n" + "=" * 70 + "\n")
        for k, v in prediction_log.items():
            if k != "predictions":
                f.write(f"{k}: {v}\n")
        f.write("-" * 70 + "\nBehavior Distribution:\n")
        total_pred = sum(prediction_log["predictions"].values())
        for b in CLASS_NAMES:
            c = prediction_log["predictions"][b]
            f.write(f"  {b:<20}: {c:>6} frames ({(c/total_pred*100) if total_pred else 0:.2f}%)\n")
        f.write("=" * 70 + "\n")
    logger.info(f"Logs saved: {detail} | {summary}")


def syllable_boundaries(syllable):
    """Keypoint-MoSeq 逐帧音节 [N] -> 切换边界 [N] 0/1 (1=允许行为切换)。"""
    syl = np.asarray(syllable)
    bd = np.zeros(len(syl), dtype=np.int32)
    bd[1:] = (np.diff(syl) != 0).astype(np.int32)
    bd[0] = 1
    return bd


def apply_syllable_smoothing(frame_results, moseq_segments):
    """音节分界后处理: 每个 MoSeq 音节段内, 用段内主导行为替换逐帧预测。

    理由: Keypoint-MoSeq 音节是"无监督运动模块", 段内应保持同一动作。
    只在音节切换(边界)处允许行为变化 -> 保留动作自然边界、消除段内抖动。
    返回新的 frame_results。
    """
    import collections
    out = [dict(r) for r in frame_results]
    for seg in moseq_segments:
        s, e = seg["start"], min(seg["end"], len(out) - 1)
        if s > e:
            continue
        counts = collections.Counter(out[i]["pred"] for i in range(s, e + 1))
        if not counts:
            continue
        majority = counts.most_common(1)[0][0]
        for i in range(s, e + 1):
            out[i]["pred"] = majority
            out[i]["conf"] = max(out[i]["conf"], 0.0)
    return out

# ============================================================ UI
def main():
    global logger
    # 必须在任何其他 st.* 之前调用
    st.set_page_config(page_title="Mouse Behavior Prediction System", page_icon="🐭",
                       layout="wide", initial_sidebar_state="expanded")
    apply_global_styling()
    if logger is None:
        logger = setup_logging()

    st.markdown("""
    <div id="super-title-container"><div id="super-title">🐭 Mouse Behavior Prediction System</div></div>
    """, unsafe_allow_html=True)

    if 'temp_video_path' not in st.session_state: st.session_state.temp_video_path = None
    if 'moseq_csv_path' not in st.session_state: st.session_state.moseq_csv_path = None
    if 'frame_results' not in st.session_state: st.session_state.frame_results = []
    if 'dlc_data' not in st.session_state: st.session_state.dlc_data = None
    if 'moseq_df' not in st.session_state: st.session_state.moseq_df = None
    if 'tadaconv_path' not in st.session_state:
        st.session_state.tadaconv_path = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "lpft_best.pt")
    if 'yolo_path' not in st.session_state:
        st.session_state.yolo_path = r"C:\Users\WIN10\PycharmProjects\mouseaction\DeepBehavior\runs\detect\train\weights\best.pt"
    if 'moseq_kappa' not in st.session_state:
        st.session_state.moseq_kappa = 1e6
    if 'moseq_latent_dim' not in st.session_state:
        st.session_state.moseq_latent_dim = None

    with st.sidebar:
        st.markdown("## ⚙️ Model Configuration")
        st.session_state.tadaconv_path = st.text_input("TAdaConv Weight Path", value=st.session_state.tadaconv_path, key="tadaconv_input")
        st.session_state.yolo_path = st.text_input("YOLO Weight Path", value=st.session_state.yolo_path, key="yolo_input")
        st.markdown("---")
        st.markdown("## 🧬 Keypoint-MoSeq 参数")
        st.session_state.moseq_kappa = st.number_input("MoSeq kappa (时间尺度刚度)", min_value=1e2, max_value=1e7,
                                                       value=float(st.session_state.moseq_kappa), key="moseq_kappa_input",
                                                       step=1e4, format="%.0f")
        st.session_state.moseq_latent_dim = st.number_input("PCA 成分数 (0=自动90%方差)", min_value=0,
                                                            value=int(st.session_state.moseq_latent_dim or 0),
                                                            key="moseq_dim_input")
        st.session_state.moseq_latent_dim = st.session_state.moseq_latent_dim or None
        st.markdown("---")
        st.markdown("## 📊 Behavior Classes")
        for c in CLASS_NAMES:
            st.caption(f"• {c}")
        st.markdown("---")
        st.markdown("## 💾 System Information")
        st.caption(f"**Compute Device**: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        st.caption(f"**PyTorch**: {torch.__version__}")
        st.markdown("---")
        st.markdown("## 📝 Log Files")
        log_dir = os.path.join(tempfile.gettempdir(), "behavior_prediction_logs")
        st.caption(f"**Location**: {log_dir}")
        if os.path.exists(log_dir):
            st.caption(f"**File Count**: {len(os.listdir(log_dir))}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📹 Video Manager", "📋 Data Preview", "🧠 Behavior Prediction", "✏️ Label Editor", "💾 Export Results"])

    with tab1:
        st.markdown("### 📹 Upload Video")
        video_file = st.file_uploader(label="", type=["mp4", "avi", "mov"], key="video_upload", label_visibility="collapsed")
        if video_file:
            temp_video_path = os.path.join(tempfile.gettempdir(), video_file.name)
            with open(temp_video_path, 'wb') as f:
                f.write(video_file.getbuffer())
            st.success("✓ Video uploaded successfully")
            st.session_state.temp_video_path = temp_video_path
            vc = cv2.VideoCapture(temp_video_path)
            total_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(vc.get(cv2.CAP_PROP_FPS))
            width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Frames", f"{total_frames}")
            col2.metric("Duration", f"{duration:.2f}s")
            col3.metric("Resolution", f"{width}×{height}")
            st.session_state.total_frames = total_frames
            st.session_state.fps = fps
            st.markdown("#### Video Preview")
            preview_frame_idx = st.slider("Select preview frame", 0, min(total_frames - 1, 299), 0, key="preview_slider")
            vc.set(cv2.CAP_PROP_POS_FRAMES, preview_frame_idx)
            ret, frame = vc.read()
            if ret:
                if st.session_state.dlc_data is not None:
                    frame = draw_keypoints_on_frame(frame, extract_keypoints_from_dlc(st.session_state.dlc_data, preview_frame_idx))
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Frame {preview_frame_idx}", width=400)
            vc.release()
        st.markdown("---")
        st.markdown("### 📋 Upload MoSeq State File")
        moseq_file = st.file_uploader(label="", type=["csv"], key="moseq_upload", label_visibility="collapsed")
        if moseq_file:
            temp_csv_path = os.path.join(tempfile.gettempdir(), moseq_file.name)
            with open(temp_csv_path, 'wb') as f:
                f.write(moseq_file.getbuffer())
            st.success("✓ MoSeq loaded successfully")
            st.session_state.moseq_csv_path = temp_csv_path
            moseq_df = load_csv_file(moseq_file)
            if moseq_df is not None:
                st.write("**Preview:**")
                st.write(moseq_df.head(10))
                st.session_state.moseq_df = moseq_df
                st.info(f"MoSeq Total Frames: **{len(moseq_df)}**")
        st.markdown("---")
        st.markdown("### 📍 DeepLabCut Keypoint File")
        keypoint_file = st.file_uploader(label="", type=["h5", "csv"], key="keypoint_upload", label_visibility="collapsed")
        if keypoint_file:
            if keypoint_file.name.endswith('.h5'):
                tmp = os.path.join(tempfile.gettempdir(), keypoint_file.name)
                with open(tmp, 'wb') as f:
                    f.write(keypoint_file.getbuffer())
                dlc_data = load_dlc_h5(tmp)
            else:
                dlc_data = load_dlc_csv(keypoint_file)
            if dlc_data is not None:
                st.success("✓ Keypoint file loaded successfully")
                st.write("**Preview:**")
                st.write(dlc_data.head(10))
                st.session_state.dlc_data = dlc_data
        st.markdown("#### 🧬 Detect Keypoints (DLC SuperAnimal)")
        if st.session_state.temp_video_path:
            if st.button("🔍 Run Keypoint Detection", key="btn_kp_detect"):
                if not os.path.exists(DLC_ENV_PY):
                    st.error("deeplabcut env not found. Install at " + DLC_ENV_PY)
                else:
                    with st.spinner("Running DLC SuperAnimal keypoint detection (may take minutes)..."):
                        csv_path = detect_keypoints_superanimal(st.session_state.temp_video_path)
                    if csv_path:
                        dlc_data = load_dlc_csv(csv_path)
                        if dlc_data is not None:
                            st.session_state.dlc_data = dlc_data
                            st.success(f"✓ Keypoints detected: {os.path.basename(csv_path)}")
                            st.write(dlc_data.head(10))
        else:
            st.caption("Upload a video first to enable keypoint detection")

    with tab2:
        st.markdown("### 📊 Data Preview")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("📹 View Video Information", key="btn_video_info"):
                if 'total_frames' in st.session_state:
                    st.info(f"**Video Information**\n- Total Frames: {st.session_state.total_frames}\n- FPS: {st.session_state.fps}")
                else:
                    st.warning("Please upload video first")
        with c2:
            if st.button("📋 View MoSeq Data", key="btn_moseq"):
                if st.session_state.moseq_df is not None:
                    st.dataframe(st.session_state.moseq_df, use_container_width=True)
                else:
                    st.warning("Please upload MoSeq file first")
        with c3:
            if st.button("📍 View Keypoint Data", key="btn_keypoint"):
                if st.session_state.dlc_data is not None:
                    st.dataframe(st.session_state.dlc_data, use_container_width=True)
                else:
                    st.warning("Please upload keypoint file first")

    with tab3:
        st.markdown("### 🧠 Behavior Prediction")
        st.info("Upload video + MoSeq, then click below")
        if st.button("🚀 Start Prediction", key="predict_btn"):
            if st.session_state.temp_video_path is None:
                st.error("Please upload video file first")
            elif st.session_state.moseq_csv_path is None:
                st.error("Please upload MoSeq CSV file first")
            else:
                with st.spinner("Loading models..."):
                    model, yolo_model = load_models(st.session_state.tadaconv_path, st.session_state.yolo_path)
                if model is None or yolo_model is None:
                    st.error("Model loading failed")
                else:
                    try:
                        frame_level_results = predict_main(model, yolo_model, st.session_state.temp_video_path, st.session_state.moseq_csv_path)
                        st.session_state.frame_results = frame_level_results
                        st.success("✓ Behavior prediction completed!")
                        # 音节分界: 现场跑 Keypoint-MoSeq 拟合 -> 音节边界 -> 段内主导行为
                        st.markdown("#### 🧬 Keypoint-MoSeq Syllable Boundary")
                        if st.session_state.dlc_data is not None and os.path.exists(MOSSEQ_ENV_PY):
                            kp_dir = os.path.join(tempfile.gettempdir(), "dlc_keypoints")
                            syllables = run_moseq_fit(
                                st.session_state.temp_video_path, kp_dir,
                                kappa=st.session_state.moseq_kappa,
                                latent_dim=st.session_state.moseq_latent_dim)
                            if syllables:
                                base = os.path.splitext(os.path.basename(st.session_state.temp_video_path))[0]
                                st.session_state.frame_results = apply_syllable_boundaries_to_results(
                                    st.session_state.frame_results, syllables, base)
                                st.success("✓ Syllable-boundary smoothing applied (preserves natural bout edges)")
                            else:
                                st.warning("No syllables produced; skipped boundary smoothing")
                        else:
                            st.caption("Upload keypoints + keypoint_moseq env to enable syllable-boundary smoothing")
                        st.markdown("#### Prediction Results Statistics")
                        pred_df = pd.DataFrame(frame_level_results)
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Total Frames Predicted", len(frame_level_results))
                        c2.metric("Average Confidence", f"{pred_df['conf'].mean():.4f}")
                        c3.metric("Highest Confidence Behavior", pred_df.loc[pred_df['conf'].idxmax(), 'pred'])
                        c4.metric("Lowest Confidence", f"{pred_df['conf'].min():.4f}")
                        st.markdown("#### Behavior Distribution")
                        st.bar_chart(pred_df['pred'].value_counts())
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")
                        logger.error(f"Prediction error: {str(e)}", exc_info=True)

    with tab4:
        st.markdown("### ✏️ Frame Label Editor")
        if len(st.session_state.frame_results) > 0:
            df_results = pd.DataFrame(st.session_state.frame_results)
            c1, c2, c3 = st.columns([1, 2, 2])
            with c1:
                frame_range = st.slider("Frame Range", 0, len(df_results) - 1, (0, min(50, len(df_results) - 1)), key="frame_range_slider")
            with c2:
                st.metric("Displayed Frames", frame_range[1] - frame_range[0] + 1)
            with c3:
                filter_class = st.selectbox("Filter", ["All"] + CLASS_NAMES, key="filter_select")
            display_df = df_results[frame_range[0]:frame_range[1] + 1].copy()
            if filter_class != "All":
                display_df = display_df[display_df['pred'] == filter_class]
            st.dataframe(display_df, use_container_width=True, height=400)
            st.markdown("#### Edit Prediction Results")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                edit_frame_idx = st.number_input("Frame ID", 0, len(df_results) - 1, 0, key="edit_frame_num")
            with c2:
                cur = st.session_state.frame_results[edit_frame_idx]
                st.info(f"Current: **{cur['pred']}**\nConfidence: **{cur['conf']:.4f}**")
            with c3:
                new_label = st.selectbox("New Label", CLASS_NAMES, key="edit_label_select")
            with c4:
                if st.button("💾 Update Prediction", key="btn_update"):
                    old = st.session_state.frame_results[edit_frame_idx]['pred']
                    st.session_state.frame_results[edit_frame_idx]['pred'] = new_label
                    st.success(f"✓ Frame {edit_frame_idx} updated: {old} → {new_label}")
        else:
            st.info("Please execute prediction first")

    with tab5:
        st.markdown("### 💾 Export Results")
        if len(st.session_state.frame_results) > 0:
            export_df = pd.DataFrame(st.session_state.frame_results)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.download_button("📥 CSV", export_df.to_csv(index=False), "behavior_predictions.csv", "text/csv")
            with c2:
                try:
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        export_df.to_excel(writer, sheet_name='Predictions', index=False)
                    buffer.seek(0)
                    st.download_button("📥 Excel", buffer, "behavior_predictions.xlsx",
                                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                except Exception:
                    st.warning("Excel export needs openpyxl")
            with c3:
                st.download_button("📥 JSON", export_df.to_json(orient='records', indent=2), "behavior_predictions.json", "application/json")
            st.markdown("---")
            st.markdown("#### 📊 Statistics")
            s1, s2, s3 = st.columns(3)
            s1.bar_chart(export_df['pred'].value_counts())
            s2.metric("Average Confidence", f"{export_df['conf'].mean():.4f}")
            s3.metric("Total Frames", len(export_df))
        else:
            st.info("Please execute prediction first")

if __name__ == "__main__":
    main()
