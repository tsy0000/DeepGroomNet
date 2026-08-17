"""Keypoint-MoSeq 现场拟合 + 音节提取（镜像 keypoint-moseq.ipynb 流程）。

镜像 notebook 关键步骤:
  kpms.setup_project -> load_keypoints -> get_pca -> save_pca
  -> init_model -> fit(ar_only) -> update_hypparams(kappa) -> fit(full)
  -> extract_results -> 每视频逐帧音节

输出: 每个视频一个 npy (逐帧音节) + project_dir (供 kappa 扫描/可视化复现)。

用法(web_app 通过子进程调用 keypoint_moseq 环境):
  python moseq_fit_inference.py <keypoint_csv_dir> <out_dir> [kappa]
"""
import os, sys, glob, json
import numpy as np

def main():
    if len(sys.argv) < 3:
        print("usage: moseq_fit_inference.py <keypoint_dir> <out_dir> [kappa]")
        sys.exit(1)
    kp_dir, out_dir = sys.argv[1], sys.argv[2]
    kappa = float(sys.argv[3]) if len(sys.argv) > 3 else 1e5
    os.makedirs(out_dir, exist_ok=True)

    import keypoint_moseq as kpms

    project_dir = os.path.join(out_dir, "moseq_proj")
    config = lambda: kpms.load_config(project_dir)

    # 1) 扫描关键点 csv/h5 (DLC SuperAnimal 输出)
    kp_files = glob.glob(os.path.join(kp_dir, "**", "*.csv"), recursive=True)
    kp_files += glob.glob(os.path.join(kp_dir, "**", "*.h5"), recursive=True)
    if not kp_files:
        print("no keypoint files found in", kp_dir); sys.exit(1)

    # 2) 加载关键点 + 投影
    coordinates, confidences, bodyparts = kpms.load_keypoints(kp_files, "deeplabcut")
    pca = kpms.get_pca(coordinates, **config())
    kpms.save_pca(pca, project_dir)

    # 3) 拟合 AR-HMM (AR-only -> full, kappa 扫描选时间尺度)
    model = kpms.init_model(coordinates, pca=pca, **config())
    model_name = f"moseq_fit_k{int(kappa)}"
    num_ar = 50; num_full = 100
    model = kpms.update_hypparams(model, kappa=kappa)
    model = kpms.fit_model(model, coordinates, confidences, metadata=None, project_dir=project_dir,
                           model_name=model_name, ar_only=True, num_iters=num_ar)[0]
    kpms.fit_model(model, coordinates, confidences, metadata=None, project_dir=project_dir,
                   model_name=model_name, ar_only=False, start_iter=num_ar, num_iters=num_full)

    # 4) 提取逐帧音节
    results = kpms.extract_results(model, kpms.load_config(project_dir), project_dir, model_name)
    for key, r in results.items():
        syl = np.asarray(r["syllable"])
        out = os.path.join(out_dir, f"{os.path.basename(key)}.syllable.npy")
        np.save(out, syl)
        print("saved", out, len(syl), "frames,", int(syl.max()) + 1, "syllables")

if __name__ == "__main__":
    main()
