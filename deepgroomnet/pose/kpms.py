import numpy as np


def _kpms():
    try:
        import keypoint_moseq as kpms
        return kpms
    except ImportError as error:
        raise RuntimeError("keypoint_moseq is required; install it in a separate "
                           "environment (see docs/INSTALL.md)") from error


def fit_kpms(keypoint_csv, out_dir, latent_dim=10, num_states=100, kappa=1e6,
             ar_iters=50, full_iters=500, seed=0):
    kpms = _kpms()
    data, metadata = kpms.load_keypoints([str(keypoint_csv)], "deeplabcut")
    data, metadata = kpms.format_data(data)
    pca = kpms.fit_pca(**data, **{"latent_dim": latent_dim, "seed": seed})
    model = kpms.init_model(data, pca=pca, num_states=num_states, latent_dim=latent_dim,
                            seed=seed)
    model, name = kpms.fit_model(model, data, metadata, str(out_dir), "kpms",
                                 ar_only=True, num_iters=ar_iters)
    kpms.fit_model(model, data, metadata, str(out_dir), name, ar_only=False,
                   start_iter=ar_iters, num_iters=ar_iters + full_iters - 1)
    return {"project_dir": str(out_dir), "model_name": name}


def apply_kpms(project_dir, model_name, keypoint_csv, out_npy, apply_iters=200,
               tracked_keypoints=None):
    kpms = _kpms()
    model, data, metadata, _ = kpms.load_checkpoint(str(project_dir), model_name)
    results = kpms.apply_model(model, data, metadata, project_dir=str(project_dir),
                               model_name=model_name, num_iters=apply_iters, ar_only=True)
    arrays = results if isinstance(results, dict) else {model_name: results}
    syllables = np.asarray(list(arrays.values())[0], dtype=np.int32).reshape(-1)
    np.save(out_npy, syllables)
    return syllables
