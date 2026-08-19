# web_app.py - DeepGroomNet Mouse Behavior Prediction System
# Pipeline: upload video -> detect keypoints (DLC SuperAnimal) -> Keypoint-MoSeq
# syllables -> TAdaConv model behavior prediction with natural-boundary smoothing.
# Run:  streamlit run web_app.py

import streamlit as st
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

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
import csv
from datetime import datetime

warnings.filterwarnings('ignore')

from models.tadaconv_model import TAdaConvModel
from postprocess.thresholds import threshold_predict, compute_background, \
    syllable_boundaries, moseq_boundary_refine

os.environ['YOLO_SETTINGS_DIR'] = tempfile.gettempdir()
from ultralytics import YOLO

# ------------------------------------------------------------ CONSTANTS
CLASS_NAMES = ['Locomote', 'Rear', 'FaceGroom', 'BodyGroom', 'background']
BG_IDX = len(CLASS_NAMES) - 1
NUM_FRAMES = 32
TARGET_SIZE = 224
BOX_EXPANSION_RATIO = 1.5
MEAN = [0.45, 0.45, 0.45]
STD = [0.225, 0.225, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DLC_ENV_PY = r"D:\Anaconda3\envs\deeplabcut\python.exe"
MOSSEQ_ENV_PY = r"D:\Anaconda3\envs\keypoint_moseq\python.exe"
DEFAULT_YOLO = r"C:\Users\WIN10\PycharmProjects\mouseaction\DeepBehavior\runs\detect\train\weights\best.pt"


def apply_global_styling():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', Arial, sans-serif !important; }
        p, span, label, button, input, select, .stMetricValue, .stMetricLabel,
        .stMarkdownContainer p, .stMarkdownContainer span { font-size: 16px !important; }
        h2, h3, h4, [data-testid="stTabs"] button p { font-size: 22px !important; font-weight: 700 !important; }
        /* keep sidebar at comfortable size */
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] button,
        [data-testid="stSidebar"] input, [data-testid="stSidebar"] select,
        [data-testid="stSidebar"] .stMarkdownContainer p,
        [data-testid="stSidebar"] .stMarkdownContainer span { font-size: 14px !important; }
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


# ------------------------------------------------------------ LOGGING
_logger = None


def setup_logging():
    global _logger
    if _logger is not None:
        return _logger
    logger = logging.getLogger("BehaviorPrediction")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                               datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console)
    _logger = logger
    return logger


# ------------------------------------------------------------ MODEL
@st.cache_resource
def load_models(tadaconv_path, yolo_path):
    logger = setup_logging()
    try:
        if not os.path.exists(tadaconv_path):
            st.error("TAdaConv weight file not found: %s" % tadaconv_path)
            return None, None
        with st.spinner("Loading TAdaConv model..."):
            model = TAdaConvModel(num_classes=NUM_CLASSES, t_stride=1,
                                  use_cpam=True, use_stsa=True, drop_path=0.0).to(DEVICE)
            ck = torch.load(tadaconv_path, map_location=DEVICE, weights_only=False)
            sd = ck["model_state"] if "model_state" in ck else ck
            model.load_state_dict(sd, strict=False)
            model.eval()
        st.success("TAdaConv model loaded.")
        logger.info("TAdaConv model loaded: %s", tadaconv_path)
        if not os.path.exists(yolo_path):
            st.error("YOLO weight file not found: %s" % yolo_path)
            return model, None
        with st.spinner("Loading YOLO model..."):
            yolo = YOLO(yolo_path, verbose=False)
        st.success("YOLO loaded.")
        logger.info("YOLO model loaded: %s", yolo_path)
        return model, yolo
    except Exception as e:
        st.error("Model loading failed: %s" % e)
        logger.error("Model loading error: %s", e, exc_info=True)
        return None, None


# ------------------------------------------------------------ VIDEO / CROP
def get_unified_crop_box(frames, yolo):
    boxes = []
    idxs = np.linspace(0, len(frames) - 1, 3).astype(int) if len(frames) > 1 else [0]
    for i in idxs:
        try:
            res = yolo(frames[i], conf=0.3, verbose=False)
            if res[0].boxes is not None and len(res[0].boxes) > 0:
                boxes.append(res[0].boxes.xyxy[0].cpu().numpy())
        except Exception:
            pass
    if not boxes:
        h, w = frames[0].shape[:2]
        return 0, 0, w, h
    boxes = np.array(boxes)
    x1, y1 = boxes[:, 0].min(), boxes[:, 1].min()
    x2, y2 = boxes[:, 2].max(), boxes[:, 3].max()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * BOX_EXPANSION_RATIO
    half = side / 2
    H, W = frames[0].shape[:2]
    x1 = int(max(0, cx - half)); y1 = int(max(0, cy - half))
    x2 = int(min(W, cx + half)); y2 = int(min(H, cy + half))
    return x1, y1, x2, y2


@torch.no_grad()
def predict_frame_probs(model, yolo, video_path, stride=8):
    """YOLO-crop sliding window -> per-frame 5-class probabilities [N,5]."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("cannot open video: %s" % video_path)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    acc = np.zeros((N, NUM_CLASSES), dtype=np.float64)
    cnt = np.zeros(N, dtype=np.int32)
    starts = list(range(0, N - NUM_FRAMES + 1, stride))
    progress = st.progress(0.0)
    for wi, s in enumerate(starts):
        frames = []
        for i in range(NUM_FRAMES):
            cap.set(cv2.CAP_PROP_POS_FRAMES, s + i)
            ok, fr = cap.read()
            frames.append(fr if ok else frames[-1])
        x1, y1, x2, y2 = get_unified_crop_box(frames, yolo)
        win = []
        for fr in frames:
            crop = fr[y1:y2, x1:x2]
            rs = cv2.resize(crop, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(rs, cv2.COLOR_BGR2RGB)
            win.append(rgb.astype(np.float32) / 255.0)
        x = torch.from_numpy(np.stack(win).transpose(0, 3, 1, 2))
        x = ((x - torch.tensor(MEAN).view(1, 3, 1, 1)) /
             torch.tensor(STD).view(1, 3, 1, 1)).unsqueeze(0).to(DEVICE)
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        p = torch.sigmoid(logits.float()).cpu().numpy()[0]
        acc[s:s + NUM_FRAMES] += p
        cnt[s:s + NUM_FRAMES] += 1
        progress.progress(min((wi + 1) / len(starts), 1.0))
    cap.release()
    cnt = np.maximum(cnt, 1)
    progress.empty()
    return acc / cnt[:, None]


# ------------------------------------------------------------ KEYPOINTS
def run_keypoint_detection(video_path):
    if not os.path.exists(DLC_ENV_PY):
        st.error("deeplabcut env not found: %s" % DLC_ENV_PY)
        return None
    script = os.path.join(PROJECT_ROOT, "scripts", "detect_keypoints.py")
    with st.spinner("Detecting keypoints (DeepLabCut SuperAnimal)... may take minutes"):
        res = subprocess.run([DLC_ENV_PY, script, video_path],
                             capture_output=True, text=True, timeout=3600)
    if res.returncode != 0:
        st.error("Keypoint detection failed: %s" % res.stderr[-800:])
        return None
    out = res.stdout.strip().splitlines()
    kp_path = out[-1] if out else "NO_OUTPUT"
    if kp_path == "NO_OUTPUT":
        st.error("No keypoint CSV produced.")
        return None
    return kp_path


def run_moseq(video_path, kp_csv, out_dir):
    if not os.path.exists(MOSSEQ_ENV_PY):
        st.error("keypoint_moseq env not found: %s" % MOSSEQ_ENV_PY)
        return None
    os.makedirs(out_dir, exist_ok=True)
    script = os.path.join(PROJECT_ROOT, "scripts", "moseq_postprocess.py")
    base = os.path.splitext(os.path.basename(video_path))[0]
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    with st.spinner("Running Keypoint-MoSeq (PCA + AR-HMM, headless)... may take minutes"):
        res = subprocess.run([MOSSEQ_ENV_PY, script, video_path, kp_csv, out_dir],
                             capture_output=True, text=True, timeout=7200, env=env)
    if res.returncode != 0:
        st.error("Keypoint-MoSeq fit failed: %s" % res.stderr[-800:])
        return None
    syl_file = os.path.join(out_dir, f"{base}_syllable.npy")
    return syl_file if os.path.exists(syl_file) else None


# ------------------------------------------------------------ MAIN UI
def main():
    global _logger
    st.set_page_config(page_title="Mouse Behavior Prediction System", page_icon="🐭",
                       layout="wide", initial_sidebar_state="expanded")
    apply_global_styling()
    logger = setup_logging()

    st.markdown('<div id="super-title-container"><div id="super-title">🐭 Mouse Behavior Prediction System</div></div>',
                unsafe_allow_html=True)

    if 'temp_video_path' not in st.session_state:
        st.session_state.temp_video_path = None
    if 'kp_csv' not in st.session_state:
        st.session_state.kp_csv = None
    if 'syllable_path' not in st.session_state:
        st.session_state.syllable_path = None
    if 'frame_results' not in st.session_state:
        st.session_state.frame_results = None
    if 'tadaconv_path' not in st.session_state:
        st.session_state.tadaconv_path = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "lpft_best.pt")
    if 'yolo_path' not in st.session_state:
        st.session_state.yolo_path = DEFAULT_YOLO

    with st.sidebar:
        st.markdown("## ⚙️ Model Configuration")
        st.session_state.tadaconv_path = st.text_input("Action Recognition Model Path",
                                                       value=st.session_state.tadaconv_path, key="tadaconv_input")
        st.session_state.yolo_path = st.text_input("Mouse Detector (YOLO) Path",
                                                   value=st.session_state.yolo_path, key="yolo_input")
        st.markdown("---")
        st.markdown("## 📊 Behavior Classes")
        for c in CLASS_NAMES:
            st.markdown("• " + c)
        st.markdown("---")
        st.markdown("## 💻 System")
        st.caption("Device: %s" % ("GPU" if torch.cuda.is_available() else "CPU"))
        st.caption("PyTorch: " + torch.__version__)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["📹 Video Upload", "🧬 Keypoint Detection", "🧬 MoSeq Detection",
         "🧠 Behavior Prediction", "✏️ Label Editor", "💾 Export"])

    # ---------- Tab 1: Video Upload ----------
    with tab1:
        st.markdown("### 📹 Upload Video")
        video_file = st.file_uploader("Select a video", type=["mp4", "avi", "mov"], key="video_upload")
        if video_file:
            temp_video_path = os.path.join(tempfile.gettempdir(), video_file.name)
            with open(temp_video_path, 'wb') as f:
                f.write(video_file.getbuffer())
            st.session_state.temp_video_path = temp_video_path
            st.success("Video uploaded: %s" % video_file.name)
            vc = cv2.VideoCapture(temp_video_path)
            total_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(vc.get(cv2.CAP_PROP_FPS))
            width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            st.session_state.total_frames = total_frames
            st.session_state.fps = fps
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Frames", f"{total_frames}")
            c2.metric("Duration", f"{duration:.2f}s")
            c3.metric("Resolution", f"{width}x{height}")
            preview_idx = st.slider("Preview frame", 0, max(0, total_frames - 1), 0, key="preview_slider")
            vc.set(cv2.CAP_PROP_POS_FRAMES, preview_idx)
            ret, frame = vc.read()
            if ret:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Frame %d" % preview_idx, width=400)
            vc.release()

    # ---------- Tab 2: Keypoint Detection ----------
    with tab2:
        st.markdown("### 🧬 Keypoint Detection (DeepLabCut SuperAnimal)")
        if st.session_state.temp_video_path is None:
            st.info("Upload a video first (tab 1).")
        else:
            if st.button("🔍 Detect Keypoints", key="btn_kp"):
                st.session_state.kp_csv = run_keypoint_detection(st.session_state.temp_video_path)
                if st.session_state.kp_csv:
                    st.success("Keypoints saved: %s" % os.path.basename(st.session_state.kp_csv))
            if st.session_state.kp_csv:
                st.markdown("#### Keypoint file")
                st.code(st.session_state.kp_csv)

    # ---------- Tab 3: MoSeq Detection ----------
    with tab3:
        st.markdown("### 🧬 MoSeq Syllable Detection (Keypoint-MoSeq)")
        if st.session_state.kp_csv is None:
            st.info("Detect keypoints first (tab 2).")
        else:
            if st.button("🧬 Run Keypoint-MoSeq (syllables)", key="btn_moseq"):
                out_dir = os.path.join(tempfile.gettempdir(), "moseq_fit_out")
                st.session_state.syllable_path = run_moseq(
                    st.session_state.temp_video_path, st.session_state.kp_csv, out_dir)
                if st.session_state.syllable_path:
                    st.success("Syllables saved: %s" % os.path.basename(st.session_state.syllable_path))
            if st.session_state.syllable_path:
                syl = np.load(st.session_state.syllable_path)
                st.info("Frames: %d, syllables: %d" % (len(syl), int(syl.max()) + 1))

    # ---------- Tab 4: Behavior Prediction ----------
    with tab4:
        st.markdown("### 🧠 Behavior Prediction")
        if st.session_state.temp_video_path is None:
            st.info("Upload a video first (tab 1).")
        else:
            if st.button("🚀 Run Prediction", key="predict_btn"):
                with st.spinner("Loading models..."):
                    model, yolo = load_models(st.session_state.tadaconv_path, st.session_state.yolo_path)
                if model is None or yolo is None:
                    st.error("Model loading failed.")
                else:
                    try:
                        P = predict_frame_probs(model, yolo, st.session_state.temp_video_path)
                        thr = {c: 0.5 for c in CLASS_NAMES}
                        pred = threshold_predict(P, thr, CLASS_NAMES)
                        pred = compute_background(pred, bg_idx=BG_IDX)
                        # Natural-boundary smoothing with MoSeq syllables
                        if st.session_state.syllable_path and os.path.exists(st.session_state.syllable_path):
                            syl = np.load(st.session_state.syllable_path)[:P.shape[0]]
                            pred = moseq_boundary_refine(pred, syllable_boundaries(syl))
                            st.success("MoSeq syllable-boundary smoothing applied.")
                        else:
                            st.caption("No syllables found; skipped boundary smoothing. Run tab 3 first.")
                        pred_arg = P.argmax(axis=1)
                        n = P.shape[0]
                        dist = pred.sum(axis=0)
                        st.markdown("#### Prediction Statistics")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Frames", n)
                        c2.metric("Behaviors detected", int((dist > 0).sum()))
                        c3.metric("Most frequent", CLASS_NAMES[int(np.argmax(dist))])
                        st.markdown("#### Behavior Distribution")
                        dist_df = pd.DataFrame({"behavior": CLASS_NAMES, "frames": dist})
                        st.bar_chart(dist_df.set_index("behavior"))
                        results = [{"frame": i, "pred": CLASS_NAMES[int(pred_arg[i])]} for i in range(n)]
                        st.session_state.frame_results = results
                        st.success("Prediction completed.")
                    except Exception as e:
                        st.error("Prediction failed: %s" % e)
                        logger.error("Prediction error: %s", e, exc_info=True)

    # ---------- Tab 5: Label Editor ----------
    with tab5:
        st.markdown("### ✏️ Frame Label Editor")
        if st.session_state.frame_results is None or len(st.session_state.frame_results) == 0:
            st.info("Run prediction first (tab 4).")
        else:
            df = pd.DataFrame(st.session_state.frame_results)
            idx = st.number_input("Frame ID", 0, len(df) - 1, 0, key="edit_frame_num")
            cur = df.iloc[idx]["pred"]
            st.info("Current: %s" % cur)
            new_label = st.selectbox("New label", CLASS_NAMES, key="edit_label_select")
            if st.button("Update", key="btn_update"):
                st.session_state.frame_results[idx]["pred"] = new_label
                st.success("Frame %d updated: %s -> %s" % (idx, cur, new_label))
            st.dataframe(df, use_container_width=True, height=300)

    # ---------- Tab 6: Export ----------
    with tab6:
        st.markdown("### 💾 Export Results")
        if st.session_state.frame_results is None or len(st.session_state.frame_results) == 0:
            st.info("Run prediction first (tab 4).")
        else:
            export_df = pd.DataFrame(st.session_state.frame_results)
            st.download_button("Download CSV", export_df.to_csv(index=False), "behavior_predictions.csv", "text/csv")
            try:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    export_df.to_excel(writer, sheet_name='Predictions', index=False)
                buffer.seek(0)
                st.download_button("Download Excel", buffer, "behavior_predictions.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception:
                st.warning("Excel export needs openpyxl.")
            st.download_button("Download JSON", export_df.to_json(orient='records', indent=2),
                               "behavior_predictions.json", "application/json")


if __name__ == "__main__":
    main()
