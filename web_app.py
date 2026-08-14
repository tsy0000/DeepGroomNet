"""
DeepGroomNet v2 — Mouse Behavior Analysis Web App（Streamlit）。

直接运行：streamlit run web_app.py
依赖：streamlit（见 requirements.txt）

功能（Tab）：
  📹 视频与数据   —— 上传视频 + MoSeq 音节文件 + 视频信息 + 帧预览
  🧠 行为预测     —— 帧级滑窗推理（与训练同构）+ 音节投票/序列平滑（可开关）+ 统计
  📊 时间线可视化 —— ethogram 行为条带 + 置信度曲线 + 音节边界叠加（发表级图）
  📋 音节视图     —— MoSeq 音节段表 + 每音节占优行为
  🤖 LLM 报告     —— 模块D：免费 LLM（Ollama / OpenAI 兼容 API）生成行为分析报告
  💾 导出结果     —— 帧级预测 CSV / JSON

与 CLI 推理（infer.py）共用同一管线：load_model / predict_clip / 滑窗生成 / 音节投票。
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

CLASS_COLORS = {
    "background": "#9E9E9E",
    "Locomote": "#1E88E5",
    "Rear": "#F4511E",
    "FaceGroom": "#43A047",
    "BodyGroom": "#8E24AA",
}
DEFAULT_CKPT = os.path.join(ROOT, "outputs", "checkpoints", "best_model.pt")

st.set_page_config(
    page_title="DeepGroomNet v2 — Mouse Behavior Analysis",
    page_icon="🐭",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_styling():
    st.markdown("""
    <style>
        html, body { font-family: 'Segoe UI', Arial, sans-serif !important; }
        p, span, label, button, input, select, .stMetricValue, .stMetricLabel {
            font-size: 14px !important; font-weight: normal !important;
        }
        h2, h3, h4, [data-testid="stTabs"] button p { font-size: 19px !important; font-weight: 600 !important; }

        /* 主标题：参考原项目 72px 样式，双保险 */
        div[data-testid="stAppViewContainer"] span.title-main,
        span.title-main {
            all: unset !important;
            display: block !important; width: 100% !important;
            text-align: center !important;
            font-family: 'Segoe UI', Arial, sans-serif !important;
            font-size: 56px !important; font-weight: 900 !important;
            letter-spacing: -1px !important;
            background: linear-gradient(90deg,
            -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
            line-height: 1.15 !important; margin: 10px 0 4px 0 !important;
        }
        div[data-testid="stAppViewContainer"] span.title-sub {
            display: block !important; text-align: center !important;
            color:
            margin: 0 0 22px 0 !important;
        }
        div[data-testid="stAppViewContainer"] div.stMarkdown div.stMarkdownContainer span.title-main { font-size: 56px !important; }

        /* 上传组件 */
        div[data-testid="stFileUploader"] button span { display: none !important; }
        div[data-testid="stFileUploader"] button::before {
            content: "upload" !important; color:
        }
        div[data-testid="stFileUploader"] button {
            padding: 6px 16px !important; border-radius: 6px !important;
            border: 1px solid
        }
        div[data-testid="stFileUploader"] button:hover {
            background:
        }
        div[data-testid="stFileUploaderDropzone"] {
            min-height: 60px !important; padding: 10px 14px !important;
            border: 1px solid
        }
        div[data-testid="stFileUploaderDropzone"]:hover { border-color:

        /* 侧边栏 */
        [data-testid="stSidebar"] { background:
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { font-size: 16px !important; }

        /* 主按钮 */
        .stButton button[kind="primary"], .stButton button {
            border-radius: 8px !important; font-weight: 600 !important;
        }
        .stButton button:hover { border-color:

        /* 卡片化 metric */
        div[data-testid="stMetric"] {
            background:
            border-radius: 10px !important; padding: 10px 14px !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
        }
    </style>
    """, unsafe_allow_html=True)


apply_styling()

st.markdown("""
<span class="title-main">🐭 DeepGroomNet v2</span>
<span class="title-sub">Mouse Behavior Analysis · Open-Field & Home-Cage · TAdaConvNeXtV2 + MoSeq</span>
""", unsafe_allow_html=True)

SS_KEYS = ["video_path", "moseq_path", "frame_results", "fps", "video_frames",
           "video_size", "class_names", "syllables", "pred_csv_path"]
for k in SS_KEYS:
    if k not in st.session_state:
        st.session_state[k] = None
if "llm_text" not in st.session_state:
    st.session_state.llm_text = None


@st.cache_resource
def load_models(ckpt_path, cfg_yaml_path):
    """加载模型（缓存）。失败返回 (None, None, 错误信息)。"""
    try:
        cfg = yaml.safe_load(open(cfg_yaml_path, encoding="utf-8"))
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        if not os.path.exists(ckpt_path):
            return None, None, f"权重不存在: {ckpt_path}"
        from infer import load_model
        model, class_names = load_model(cfg, ckpt_path, device)
        return (cfg, model, class_names, device), None, None
    except Exception as e:
        return None, None, f"模型加载失败: {e}"


def load_syllables(csv_path, n_frames):
    """读取 MoSeq state csv（首列），长度对齐到视频帧数。"""
    df = pd.read_csv(csv_path)
    col = df.columns[0]
    syl = df[col].to_numpy().astype(np.int64)
    if len(syl) > n_frames:
        syl = syl[:n_frames]
    elif len(syl) < n_frames:
        syl = np.pad(syl, (0, n_frames - len(syl)), mode="edge")
    return syl


def run_prediction(cfg, model, class_names, video_path, syllables, device,
                   window_mode="sliding", use_vote=True, smoothing="sequence",
                   crop_mode="window", progress_cb=None):
    """帧级滑窗推理（与 CLI infer.py 同一管线）。返回 (frame_results_df, syllable_df)。"""
    from data.clip_sampler import infer_sliding_windows, infer_syllable_clips
    from infer import predict_clip
    from postprocess.crop_box import build_crop_boxer
    from postprocess.syllable_vote import apply_syllable_pipeline
    from postprocess.sequence_decode import sequence_decode
    from postprocess.temporal_smoothing import smooth

    cfg = dict(cfg)
    cfg["INFER"] = dict(cfg.get("INFER", {}))
    cfg["INFER"]["crop_mode"] = crop_mode
    try:
        crop_boxer = build_crop_boxer(cfg)
    except FileNotFoundError:
        crop_boxer = None

    nf = st.session_state.video_frames
    if window_mode == "syllable":
        if syllables is None:
            raise ValueError("syllable 模式需要上传 MoSeq 文件")
        clips = infer_syllable_clips(syllables, cfg)
    else:
        clips = infer_sliding_windows(nf, cfg)

    kp_array = kp_bodyparts = None
    kp_cfg = cfg.get("KEYPOINTS") or {}
    if kp_cfg.get("enabled", True) and kp_cfg.get("root"):
        from data.keypoints import KeypointBank
        bank = KeypointBank(kp_cfg["root"])
        mouse = os.path.basename(os.path.dirname(video_path))
        kp_bodyparts, kp_array = bank.get(mouse)
    flow_model = None
    if cfg["CLIP"].get("flow_channel", False):
        from data.flow_utils import load_flow_model
        flow_model = load_flow_model().to(device)

    frame_logits = np.zeros((nf, len(class_names)), dtype=np.float32)
    counts = np.zeros(nf, dtype=np.float32)
    for i, (s, e) in enumerate(clips):
        log = predict_clip(model, video_path, s, min(e, nf), cfg, device,
                           crop_boxer=crop_boxer, kp_array=kp_array,
                           kp_bodyparts=kp_bodyparts, flow_model=flow_model,
                           video_nframes=nf).cpu().numpy()
        seg = slice(s, min(e, nf))
        frame_logits[seg] += log
        counts[seg] += 1.0
        if progress_cb and (i % 5 == 0 or i == len(clips) - 1):
            progress_cb((i + 1) / len(clips))
    valid = counts > 0
    frame_logits[valid] /= counts[valid, None]

    pred = frame_logits.argmax(1)
    conf = frame_logits.max(1)

    if syllables is not None and use_vote:
        _, pred = apply_syllable_pipeline(pred, syllables, cfg, class_names=class_names)
    else:
        if smoothing == "sequence":
            pred = sequence_decode(pred, cfg, class_names=class_names)
        else:
            pred = smooth(pred, method=smoothing, window=cfg["POSTPROCESS"].get("vote_window", 5))

    df = pd.DataFrame({
        "frame": np.arange(nf),
        "time_s": np.round(np.arange(nf) / int(cfg["CLIP"]["fps"]), 3),
        "pred_class": [class_names[i] for i in pred],
        "confidence": conf,
    })

    syl_df = None
    if syllables is not None:
        rows, i = [], 0
        while i < nf:
            j = i + 1
            while j < nf and syllables[j] == syllables[i]:
                j += 1
            seg = pred[i:j]
            vals, cnt = np.unique(seg, return_counts=True)
            maj = vals[int(np.argmax(cnt))]
            rows.append({"syllable": int(syllables[i]), "start_frame": i, "end_frame": j - 1,
                         "length": j - i,
                         "time_s": round(i / int(cfg["CLIP"]["fps"]), 3),
                         "dominant_behavior": class_names[maj]})
            i = j
        syl_df = pd.DataFrame(rows)
    return df, syl_df


def draw_ethogram(df, syllables, class_names, fps=30):
    """ethogram 时间线：行为条带 + 置信度 + 音节边界。返回 matplotlib figure。"""
    n = len(df)
    t = df["time_s"].to_numpy()
    cmap = [CLASS_COLORS[c] for c in class_names]
    color_ids = np.array([class_names.index(c) for c in df["pred_class"]])

    fig, axes = plt.subplots(3, 1, figsize=(13, 6.5), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1, 1]})
    fig.patch.set_facecolor("white")

    ax = axes[0]
    for i, c in enumerate(class_names):
        mask = color_ids == i
        if mask.any():
            idx = np.flatnonzero(mask)
            splits = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
            for seg in splits:
                ax.add_patch(Rectangle((t[seg[0]], i + 0.08), t[seg[-1]] - t[seg[0]] + 1 / fps,
                                       0.84, facecolor=cmap[i], edgecolor="none"))
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_ylim(-0.2, len(class_names) + 0.2)
    ax.set_title("Frame-level behavior timeline (ethogram)", fontsize=13, fontweight="bold")
    ax.set_xlim(0, t[-1])

    if syllables is not None:
        bound = []
        for i in range(1, n):
            if syllables[i] != syllables[i - 1]:
                bound.append(i - 1 + 0.5)
        if bound:
            for b in bound:
                ax.axvline(b / fps, color="#424242", linestyle="--", linewidth=0.6, alpha=0.55)
            ax.text(0.99, 1.02, "dashed = MoSeq syllable boundaries",
                    transform=ax.transAxes, ha="right", fontsize=9, color="#424242")

    ax1 = axes[1]
    ax1.fill_between(t, df["confidence"], color="#1E88E5", alpha=0.35)
    ax1.plot(t, df["confidence"], color="#1E88E5", linewidth=0.9)
    ax1.set_ylim(0, 1.02)
    ax1.set_ylabel("confidence")
    ax1.set_title("Per-frame confidence", fontsize=11, fontweight="bold")

    ax2 = axes[2]
    bottom = np.zeros(len(class_names))
    for i, c in enumerate(class_names):
        frac = (color_ids == i).mean() * 100
        ax2.barh(i, frac, left=0, color=cmap[i], height=0.7)
        if frac > 3:
            ax2.text(frac + 1, i, f"{frac:.1f}%", va="center", fontsize=9)
    ax2.set_yticks(range(len(class_names)))
    ax2.set_yticklabels(class_names)
    ax2.set_xlim(0, 105)
    ax2.set_xlabel("percentage of frames (%)")
    ax2.set_title("Behavior proportion", fontsize=11, fontweight="bold")

    plt.tight_layout()
    return fig


with st.sidebar:
    st.markdown("## ⚙️ Model Configuration")
    cfg_path = st.text_input("Config (config.yaml)", value=os.path.join(ROOT, "configs", "config.yaml"))
    ckpt_path = st.text_input("Checkpoint (best_model.pt)", value=DEFAULT_CKPT)
    st.markdown("---")
    st.markdown("## 🧩 Inference Options")
    window_mode = st.selectbox("Window mode", ["sliding", "syllable"],
                               help="sliding=帧级滑窗（与训练同构，推荐）；syllable=按 MoSeq 音节整窗")
    use_vote = st.checkbox("Syllable majority vote（保留自然分界）", value=True)
    smoothing = st.selectbox("Smoothing", ["sequence", "vote", "median", "none"])
    crop_mode = st.selectbox("ROI crop（与训练 clip 同策略的 YOLO 裁剪）",
                             ["window", "static", "none"],
                             help="window=每窗检测（推荐，与 02_gen_clips 一致）；static=全视频一次；"
                                  "none=整帧 resize（仅推理视频已裁剪时）")
    st.markdown("---")
    st.markdown("## 🎨 Behavior Classes")
    for c in ["background", "Locomote", "Rear", "FaceGroom", "BodyGroom"]:
        st.markdown(f"<span style='display:inline-block;width:12px;height:12px;border-radius:3px;"
                    f"background:{CLASS_COLORS[c]};margin-right:6px'></span>{c}",
                    unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("## 💻 System")
    try:
        import torch
        dev = "GPU" if torch.cuda.is_available() else "CPU"
        st.caption(f"**Device**: {dev}  **PyTorch**: {torch.__version__}")
    except Exception:
        st.caption("**Device**: torch 未安装（请在项目 env 运行）")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📹 视频与数据", "🧠 行为预测", "📊 时间线可视化", "📋 音节视图", "🤖 LLM 报告", "💾 导出结果"])

with tab1:
    st.markdown("### 📹 上传视频")
    video_file = st.file_uploader("上传视频文件（mp4/avi/mpg）", type=["mp4", "avi", "mpg"],
                                  key="video_upload", label_visibility="collapsed")
    if video_file:
        tmp = os.path.join(tempfile.gettempdir(), "dgn_v2_" + video_file.name)
        with open(tmp, "wb") as f:
            f.write(video_file.getbuffer())
        st.session_state.video_path = tmp
        cap = cv2.VideoCapture(tmp)
        st.session_state.video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.session_state.fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        st.session_state.video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        st.success(f"✓ 视频已上传：{video_file.name}")

        c1, c2, c3 = st.columns(3)
        c1.metric("总帧数", st.session_state.video_frames)
        c2.metric("时长", f"{st.session_state.video_frames / st.session_state.fps:.1f}s")
        c3.metric("分辨率", f"{st.session_state.video_size[0]}×{st.session_state.video_size[1]}")

        st.markdown("#### 帧预览")
        preview_idx = st.slider("预览帧", 0, max(0, st.session_state.video_frames - 1), 0,
                                key="preview_slider")
        cap = cv2.VideoCapture(tmp)
        cap.set(cv2.CAP_PROP_POS_FRAMES, preview_idx)
        ok, frame = cap.read()
        cap.release()
        if ok:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                     caption=f"Frame {preview_idx} (t={preview_idx / st.session_state.fps:.1f}s)",
                     width=420)

    st.markdown("---")
    st.markdown("### 📋 MoSeq 音节文件（可选，用于自然分界 + 音节投票）")
    moseq_file = st.file_uploader("上传 MoSeq state csv（首列=syllable）", type=["csv"],
                                  key="moseq_upload", label_visibility="collapsed")
    if moseq_file:
        tmp2 = os.path.join(tempfile.gettempdir(), "dgn_v2_" + moseq_file.name)
        with open(tmp2, "wb") as f:
            f.write(moseq_file.getbuffer())
        st.session_state.moseq_path = tmp2
        st.success("✓ MoSeq 已上传")
        dfm = pd.read_csv(tmp2)
        st.info(f"MoSeq 帧数：**{len(dfm)}**（列：{list(dfm.columns)[:3]}）")
        st.dataframe(dfm.head(8), use_container_width=True)

with tab2:
    st.markdown("### 🧠 行为预测")
    st.info("帧级滑窗推理（与训练同构）→ 音节投票 → 序列平滑。全部参数在侧边栏。")
    if st.button("🚀 开始预测", key="predict_btn", type="primary"):
        if not st.session_state.video_path:
            st.error("请先上传视频")
        else:
            bundle, err, _ = load_models(ckpt_path, cfg_path)
            if err:
                st.error(err)
            else:
                cfg, model, class_names, device = bundle
                st.session_state.class_names = class_names
                syllables = None
                if st.session_state.moseq_path:
                    syllables = load_syllables(st.session_state.moseq_path,
                                               st.session_state.video_frames)
                    st.session_state.syllables = syllables
                prog = st.progress(0.0, text="推理中…")
                try:
                    df, syl_df = run_prediction(
                        cfg, model, class_names, st.session_state.video_path,
                        syllables, device, window_mode=window_mode,
                        use_vote=use_vote and syllables is not None, smoothing=smoothing,
                        crop_mode=crop_mode,
                        progress_cb=lambda p: (prog.progress(p, text=f"推理中… {p * 100:.0f}%")))
                    prog.progress(1.0, text="完成")
                    st.session_state.frame_results = df
                    st.session_state.syllable_df = syl_df
                    csv_path = os.path.join(tempfile.gettempdir(), "dgn_v2_pred.csv")
                    df.to_csv(csv_path, index=False)
                    st.session_state.pred_csv_path = csv_path
                    st.success(f"✓ 预测完成：{len(df)} 帧")
                except Exception as e:
                    prog.empty()
                    st.error(f"预测失败：{e}")

    if st.session_state.frame_results is not None:
        df = st.session_state.frame_results
        st.markdown("#### 结果统计")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("预测帧数", len(df))
        c2.metric("平均置信度", f"{df['confidence'].mean():.4f}")
        top = df.loc[df["confidence"].idxmax(), "pred_class"]
        c3.metric("最高置信行为", top)
        c4.metric("最低置信度", f"{df['confidence'].min():.4f}")

        st.markdown("#### 行为分布")
        counts = df["pred_class"].value_counts()
        bar = pd.DataFrame({"percent": counts / len(df) * 100})
        st.bar_chart(bar)

        st.markdown("#### 帧级结果（前 200 行）")
        st.dataframe(df.head(200), use_container_width=True, height=260)

with tab3:
    st.markdown("### 📊 时间线可视化（ethogram）")
    if st.session_state.frame_results is None:
        st.info("请先运行行为预测")
    else:
        df = st.session_state.frame_results
        syl = st.session_state.syllables
        fig = draw_ethogram(df, syl, st.session_state.class_names,
                            fps=st.session_state.fps)
        st.pyplot(fig)
        st.caption("虚线 = MoSeq 音节边界（自然分界）；行为条带每帧一色，便于目视校验模型输出")

with tab4:
    st.markdown("### 📋 MoSeq 音节聚合")
    if st.session_state.frame_results is None:
        st.info("请先运行行为预测")
    elif not st.session_state.moseq_path:
        st.warning("未上传 MoSeq 文件——音节视图需要音节数据")
    else:
        syl_df = st.session_state.syllable_df
        if syl_df is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("音节总数", len(syl_df))
            c2.metric("平均音节时长", f"{syl_df['length'].mean():.1f} 帧")
            c3.metric("最长音节", f"{syl_df['length'].max()} 帧")
            st.dataframe(syl_df, use_container_width=True, height=400)
            st.markdown("#### 音节→行为 转移统计")
            trans = syl_df["dominant_behavior"].value_counts()
            st.bar_chart(trans)

with tab5:
    st.markdown("### 🤖 LLM 行为分析报告（模块D，免费模型）")
    if st.session_state.frame_results is None:
        st.info("请先运行行为预测")
    else:
        cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
        provider = cfg.get("LLM", {}).get("provider", "none")
        st.caption(f"当前 LLM provider：**{provider}**"
                   + ("（ollama 本地免费；或 openai_compatible 免费额度 API）" if provider != "none"
                      else "—— 可在 config.yaml 的 LLM 段配置 ollama / openai_compatible 后启用"))
        if st.button("✨ 生成行为报告", key="llm_btn", type="primary"):
            try:
                from llm.report import generate_report
                out_md = os.path.join(tempfile.gettempdir(), "dgn_v2_report.md")
                text = generate_report(cfg, st.session_state.pred_csv_path,
                                       out_md=out_md,
                                       language=cfg.get("LLM", {}).get("language", "zh"))
                st.session_state.llm_text = text
                st.session_state.llm_md = out_md
                if text is None:
                    st.warning("LLM 未启用或调用失败——已生成纯统计输出（见下方）")
            except Exception as e:
                st.error(f"LLM 报告失败：{e}")
        if st.session_state.llm_text is not None:
            st.markdown("#### 统计")
            df = st.session_state.frame_results
            vc = df["pred_class"].value_counts()
            for c, n in vc.items():
                st.markdown(f"- **{c}**: {n} 帧 ({n / len(df) * 100:.1f}%)")
            if st.session_state.llm_text:
                st.markdown("#### LLM 分析")
                st.markdown(st.session_state.llm_text)
        elif os.path.exists(os.path.join(tempfile.gettempdir(), "dgn_v2_report.md")):
            with open(os.path.join(tempfile.gettempdir(), "dgn_v2_report.md"), encoding="utf-8") as f:
                st.markdown(f.read())

with tab6:
    st.markdown("### 💾 导出结果")
    if st.session_state.frame_results is None:
        st.info("请先运行行为预测")
    else:
        df = st.session_state.frame_results
        csv_data = df.to_csv(index=False)
        json_data = df.to_json(orient="records", indent=2)
        c1, c2 = st.columns(2)
        c1.download_button("📥 下载 CSV", csv_data, "behavior_predictions.csv", "text/csv")
        c2.download_button("📥 下载 JSON", json_data, "behavior_predictions.json",
                           "application/json")
        if st.session_state.syllable_df is not None:
            st.download_button("📥 下载音节聚合 CSV", st.session_state.syllable_df.to_csv(index=False),
                               "syllable_summary.csv", "text/csv")
        st.markdown("---")
        st.markdown("#### 帧级表格（可筛选）")
        filt = st.selectbox("按行为筛选", ["全部"] + list(df["pred_class"].unique()))
        view = df if filt == "全部" else df[df["pred_class"] == filt]
        st.dataframe(view, use_container_width=True, height=400)


if __name__ == "__main__":
    pass
