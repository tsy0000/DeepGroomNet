"""Keypoint-MoSeq 后处理: 按 keypoint-moseq.ipynb 流程对单个视频拟合, 输出逐帧音节。

镜像 notebook:
  load_keypoints -> format_data(use_bodyparts/fps/anterior/posterior)
  -> fit_pca -> print_dims_to_explain_variance(pca,0.9) [= latent_dim]
  -> init_model -> update_hypparams(kappa) -> fit_model(ar_only) -> fit_model(full)
  -> extract_results -> 逐帧 syllable

用法(keypoint_moseq 环境):
  D:/Anaconda3/envs/keypoint_moseq/python.exe scripts/moseq_postprocess.py \
      <video> <keypoint_csv> <out_dir> [kappa] [latent_dim]

输出: out_dir/<video>_syllable.npy (逐帧音节) + 打印 PCA 维度。
"""
import os, sys, json, time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    if len(sys.argv) < 4:
        print("usage: moseq_postprocess.py <video> <keypoint_csv> <out_dir> [kappa] [latent_dim]")
        sys.exit(1)
    video = os.path.basename(sys.argv[1])
    kp_csv = sys.argv[2]
    out_dir = sys.argv[3]
    kappa = float(sys.argv[4]) if len(sys.argv) > 4 else 1e6
    latent_dim = int(sys.argv[5]) if len(sys.argv) > 5 else None
    os.makedirs(out_dir, exist_ok=True)

    import keypoint_moseq as kpms

    project_dir = os.path.join(out_dir, f"moseq_proj_{os.path.splitext(video)[0]}")
    os.makedirs(project_dir, exist_ok=True)

    # ---- notebook cell 2: 建项目 (需先 setup_project 才存在 config.yml) ----
    kpms.setup_project(project_dir, deeplabcut_config=r"E:/tsy/config.yaml", overwrite=True)

    # ---- notebook cell 3: 配置 (use_bodyparts 与 notebook 一致) ----
    use_bodyparts = [
        "nose", "left_ear", "right_ear", "left_ear_tip", "right_ear_tip",
        "left_eye", "right_eye", "neck", "mid_back", "mouse_center",
        "mid_backend", "mid_backend2", "mid_backend3", "tail_base",
        "left_shoulder", "left_midside", "left_hip",
        "right_shoulder", "right_midside", "right_hip", "head_midpoint",
    ]
    kpms.update_config(
        project_dir,
        video_dir=sys.argv[1],
        anterior_bodyparts=["nose"],
        posterior_bodyparts=["tail_base"],
        use_bodyparts=use_bodyparts,
        fps=30,
    )

    # ---- notebook cell 4: 加载关键点 + 格式化 ----
    coordinates, confidences, bodyparts = kpms.load_keypoints(kp_csv, "deeplabcut")
    cfg = lambda: kpms.load_config(project_dir)
    data, metadata = kpms.format_data(coordinates, confidences, **cfg())

    # ---- notebook cell 5: PCA (找 90% 方差的 latent_dim) ----
    pca = kpms.fit_pca(**data, **cfg())
    kpms.save_pca(pca, project_dir)
    if latent_dim is None:
        evr = np.cumsum(np.asarray(pca.explained_variance_ratio_))
        latent_dim = int(np.argmax(evr >= 0.9) + 1)
        print(f"[moseq] >=90% variance explained by {latent_dim} components", flush=True)
    kpms.update_config(project_dir, latent_dim=latent_dim)

    # ---- notebook cell 9/10/11: 拟合 AR-HMM ----
    model = kpms.init_model(data, pca=pca, **cfg())
    model = kpms.update_hypparams(model, kappa=kappa)
    model_name = f"post_k{int(kappa)}_d{latent_dim}"
    num_ar, num_full = 50, 150
    t0 = time.time()
    print(f"[moseq] fitting AR-only {num_ar} iters ...", flush=True)
    model = kpms.fit_model(model, data, metadata, project_dir, model_name,
                           ar_only=True, num_iters=num_ar)[0]
    print(f"[moseq] fitting full {num_full} iters ...", flush=True)
    model = kpms.fit_model(model, data, metadata, project_dir, model_name,
                           ar_only=False, start_iter=num_ar,
                           num_iters=num_ar + num_full)[0]
    print(f"[moseq] fit done in {time.time()-t0:.0f}s", flush=True)

    # ---- notebook cell 13/15: 提取逐帧音节并保存 ----
    results = kpms.extract_results(model, metadata, project_dir, model_name)
    for key, r in results.items():
        syl = np.asarray(r["syllable"])
        out = os.path.join(out_dir, f"{os.path.splitext(video)[0]}_syllable.npy")
        np.save(out, syl)
        n_syl = int(syl.max()) + 1
        print(f"[moseq] saved {out}: {len(syl)} frames, {n_syl} syllables")
    print(f"[moseq] DONE project={project_dir}")


if __name__ == "__main__":
    main()
