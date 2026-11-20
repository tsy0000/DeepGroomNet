import numpy as np
import torch
from deepgroomnet.config import CLASS_NAMES, load_config
from deepgroomnet.models.tadaconv_model import TAdaConvModel
from deepgroomnet.postprocessing.labels import probs_to_labels, write_ethogram
from deepgroomnet.preprocessing.detector import box_at, crop_to_box
from deepgroomnet.preprocessing.video import read_video
from deepgroomnet.preprocessing.windows import window_starts


def load_model(cfg, checkpoint=None, device="cpu"):
    model = TAdaConvModel(num_classes=len(CLASS_NAMES), t_stride=1,
                          use_cpam=bool(cfg["MODEL"]["cpam"]),
                          use_stsa=bool(cfg["MODEL"]["stsa"]), drop_path=0.0)
    path = checkpoint or cfg["MODEL"].get("checkpoint")
    if path:
        state = torch.load(path, map_location="cpu", weights_only=False)
        state = state.get("model_state", state.get("state_dict", state))
        model.load_state_dict(state, strict=False)
    return model.to(device).eval()


class Segmenter:
    def __init__(self, model, cfg=None, device="cpu", boxes=None, box_key=None):
        self.cfg = cfg or load_config()
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.boxes = boxes
        self.box_key = box_key

    @classmethod
    def from_checkpoint(cls, checkpoint, cfg=None, device="cpu", boxes=None, box_key=None):
        cfg = cfg or load_config()
        return cls(load_model(cfg, checkpoint, device), cfg, device, boxes, box_key)

    def _to_tensor(self, frames):
        clip = self.cfg["CLIP"]
        size = int(clip["target_size"])
        data = frames.astype(np.float32) / 255.0
        if data.shape[1] != size:
            import cv2
            data = np.stack([cv2.resize(f, (size, size)) for f in frames]).astype(np.float32) / 255.0
        data = (data - float(clip["norm_mean"])) / float(clip["norm_std"])
        return np.transpose(data, (0, 3, 1, 2))

    def probs_from_frames(self, frames, boxes=None):
        clip = self.cfg["CLIP"]
        window = int(clip["num_input_frames"])
        stride = int(self.cfg["INFERENCE"]["stride"])
        batch_size = int(self.cfg["INFERENCE"]["batch_size"])
        size = int(clip["target_size"])
        if boxes is not None:
            frames = [crop_to_box(f, box_at(boxes, self.box_key, i), size)
                      for i, f in enumerate(frames)]
        data = self._to_tensor(np.stack(frames))
        n = len(data)
        probs = np.zeros((n, len(CLASS_NAMES)), dtype=np.float64)
        counts = np.zeros(n, dtype=np.int32)
        starts = window_starts(n, window, stride)
        with torch.no_grad():
            for offset in range(0, len(starts), batch_size):
                chunk = starts[offset:offset + batch_size]
                batch = np.stack([data[s:s + window] for s in chunk])
                tensor = torch.from_numpy(batch).permute(0, 2, 1, 3, 4).to(self.device)
                out = self.model(tensor)
                out = out[0] if isinstance(out, tuple) else out
                batch_probs = torch.sigmoid(out.float()).cpu().numpy()
                for j, s in enumerate(chunk):
                    probs[s:s + window] += batch_probs[j]
                    counts[s:s + window] += 1
        return probs / np.maximum(counts, 1)[:, None]

    def probs_from_video(self, path):
        cfg = self.cfg["CLIP"]
        frames = read_video(path, int(cfg["target_size"]))
        return self.probs_from_frames(frames)

    def run(self, path, syllables=None, thresholds=None):
        probs = self.probs_from_video(path)
        names = list(CLASS_NAMES)
        thr = thresholds if thresholds is not None else self.cfg["POSTPROCESS"]["thresholds"]
        out = probs_to_labels(probs, thr, names,
                              background_index=int(self.cfg["CLASSES"]["background_index"]),
                              smooth_window=int(self.cfg["POSTPROCESS"]["smooth_window"]),
                              syllables=syllables,
                              use_sgatf=syllables is not None,
                              sgatf_cfg={k: self.cfg["SGATF"][k] for k in
                                         ("alpha_max", "max_run", "consensus_floor",
                                          "consensus_span")})
        out["class_names"] = names
        return out

    def write_outputs(self, result, out_dir, prefix="session", fps=30.0):
        import os
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f"{prefix}_probs.npy"), result["probs"].astype(np.float32))
        np.save(os.path.join(out_dir, f"{prefix}_labels.npy"), result["labels"].astype(np.int16))
        write_ethogram(result["labels"], result["class_names"],
                       os.path.join(out_dir, f"{prefix}_ethogram.csv"))
        return out_dir
