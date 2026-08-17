"""DeepGroomNet Web App - upload a mouse open-field video, predict behaviors.

Run:  streamlit run web_app.py
Loads the trained TAdaConvNeXtV2 + CPAM model, runs sliding-window frame-level
inference with DeepEthogram post-processing, and renders a per-frame ethogram.
"""
import os, sys, tempfile, io, warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party"))

from models.backbones.tadaconvnext import TAdaConvNeXtV2
from postprocess.thresholds import threshold_predict, compute_background, min_bout_per_behavior

st.set_page_config(page_title="DeepGroomNet", page_icon="🐭", layout="wide", initial_sidebar_state="collapsed")

# ---------------------------------------------------------------- styling
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .block-container { padding-top: 2rem; max-width: 1100px; }
  h1 { font-weight: 800; letter-spacing: -0.5px; }
  .hero { font-size: 2.4rem; font-weight: 800; margin-bottom: 0.2rem; }
  .sub { color: #6b7280; margin-bottom: 1.5rem; }
  div[data-testid="stFileUploaderDropzone"] { border: 1.5px dashed #94a3b8; border-radius: 12px; }
  div[data-testid="stFileUploaderDropzone"]:hover { border-color: #2563eb; }
  .stButton>button { border-radius: 10px; padding: 0.5rem 1.2rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = ["Locomote", "Rear", "FaceGroom", "BodyGroom", "background"]
BG_IDX = len(CLASS_NAMES) - 1
COLORS = ["#2563eb", "#f59e0b", "#22c55e", "#ef4444", "#94a3b8"]

# ---------------------------------------------------------------- model
@st.cache_resource
def load_model(cfg, ckpt_path):
    mc = {"num_classes": len(CLASS_NAMES), "cpam": cfg["MODEL"].get("cpam", True),
          "cpam_beta": 0.1, "cpam_alpha": 0.9, "dropout": 0.0}
    from types import SimpleNamespace
    model = TAdaConvNeXtV2(SimpleNamespace(**mc))
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@st.cache_resource
def load_config():
    import yaml
    return yaml.safe_load(open(os.path.join(ROOT, "configs", "config.yaml"), encoding="utf-8"))


# ---------------------------------------------------------------- inference
@torch.no_grad()
def sliding_window_probs(model, frames, device):
    """frames: [N,3,H,W] float 0..1 -> per-frame prob [N,5]."""
    num_frames = 16; stride = 1; size = 224
    mean = 0.45; std = 0.225
    xs = np.stack([cv2.resize(f, (size, size)) for f in frames]).astype(np.float32) / 255.0
    xs = np.transpose(xs, (0, 3, 1, 2))                       # [N,3,H,W]
    xs = (xs - mean) / std
    N = xs.shape[0]
    acc = np.zeros((N, 5), dtype=np.float64); cnt = np.zeros(N, dtype=np.int32)
    for s in range(0, N - num_frames + 1, stride):
        # xs[s:s+16] 是 [T,C,H,W], 转成模型需要的 [B,C,T,H,W]
        x = torch.from_numpy(xs[s:s + num_frames]).permute(1, 0, 2, 3).unsqueeze(0).to(device)
        out = model(x)
        if isinstance(out, tuple): out = out[0]
        p = torch.sigmoid(out.float()).cpu().numpy()[0]
        acc[s:s + num_frames] += p; cnt[s:s + num_frames] += 1
    cnt = np.maximum(cnt, 1)
    return acc / cnt[:, None]


def postprocess(probs, class_names):
    thresholds = {c: 0.5 for c in class_names}
    pred = threshold_predict(probs, thresholds, class_names)
    pred = compute_background(pred, bg_idx=BG_IDX)
    pred = min_bout_per_behavior(pred, np.zeros(len(probs), dtype=int), class_names,
                                 percentile=1, bg_idx=BG_IDX)
    return pred


def argmax_trace(pred):
    # one-hot -> per-frame winning class (FG/BG exclusivity respected by postprocess)
    return pred.argmax(1)


# ---------------------------------------------------------------- display
def render_ethogram(frames, pred, sample_every=8):
    """Draw per-frame ethogram as horizontal color bands."""
    N = len(frames)
    K = len(CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(12, 1.2 + 0.12 * K))
    for k in range(K):
        y = K - 1 - k
        for i in range(0, N, sample_every):
            if pred[i, k]:
                ax.add_patch(Rectangle((i, y), sample_every, 0.9, color=COLORS[k], ec="none"))
    ax.set_yticks(range(K)); ax.set_yticklabels(reversed(CLASS_NAMES), fontsize=9)
    ax.set_xlim(0, N); ax.set_ylim(-0.2, K)
    ax.set_xlabel("frame"); ax.set_xticks([])
    ax.set_title("Predicted Ethogram", fontsize=11, fontweight="bold")
    for sp in ax.spines.values(): sp.set_visible(False)
    plt.tight_layout()
    return fig


def render_distribution(pred, counts):
    fig, ax = plt.subplots(figsize=(7, 2.2))
    bars = ax.bar(CLASS_NAMES, counts, color=COLORS, alpha=0.85)
    for b, v in zip(bars, counts):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v*100:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("% frames"); ax.set_title("Behavior Distribution", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------- page
cfg = load_config()
st.markdown('<div class="hero">DeepGroomNet</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Frame-level mouse behavior recognition &middot; TAdaConvNeXtV2 + CPAM</div>',
            unsafe_allow_html=True)

ckpt_default = os.path.join(ROOT, "outputs", "checkpoints", "lpft_best.pt")
ckpt = st.text_input("Model checkpoint", value=ckpt_default if os.path.exists(ckpt_default) else "")

if not ckpt or not os.path.exists(ckpt):
    st.info("Point to a trained checkpoint to enable prediction.")
    st.stop()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(cfg, ckpt)
st.caption(f"Model loaded &middot; {device} &middot; {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

uploaded = st.file_uploader("Upload an open-field mouse video (mp4)", type=["mp4", "avi", "mov"])
if uploaded is None:
    st.stop()

tmp = os.path.join(tempfile.gettempdir(), "dgnet_" + uploaded.name)
with open(tmp, "wb") as f:
    f.write(uploaded.getbuffer())

cap = cv2.VideoCapture(tmp)
frames = []
while True:
    ok, fr = cap.read()
    if not ok: break
    frames.append(fr)
cap.release()
frames = np.stack(frames)
st.success(f"Loaded {len(frames)} frames")

with st.spinner("Predicting behaviors..."):
    probs = sliding_window_probs(model, frames, device)
    pred = postprocess(probs, CLASS_NAMES)
    counts = pred.mean(0)
    trace = argmax_trace(pred)

col1, col2 = st.columns([3, 2])
with col1:
    st.pyplot(render_ethogram(frames, pred))
with col2:
    st.pyplot(render_distribution(pred, counts))
    top = CLASS_NAMES[np.bincount(trace, minlength=len(CLASS_NAMES)).argmax()]
    st.metric("Dominant behavior", top)

buf = io.BytesIO()
np.save(buf, pred)
buf.seek(0)
st.download_button("Download per-frame predictions (.npy)", data=buf, file_name="ethogram.npy")
