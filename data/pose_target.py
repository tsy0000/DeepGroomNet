from __future__ import annotations
import numpy as np
SUPERANIMAL_SKELETONS = [(0, 26), (1, 26), (2, 26), (3, 1), (4, 2), (5, 26), (6, 26), (26, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 25), (19, 7), (20, 19), (21, 20), (21, 13), (22, 7), (23, 22), (24, 23), (24, 13)]
HEAD_PARTS_IDX = {0, 1, 2, 3, 4, 5, 6, 26}
BODY_PARTS_IDX = set(range(27)) - HEAD_PARTS_IDX

class PoseTargetGenerator:

    def __init__(self, sigma: float=0.6, use_score: bool=True, with_kp: bool=True, with_limb: bool=True, scaling: float=1.0) -> None:
        self.sigma = sigma
        self.use_score = use_score
        self.with_kp = with_kp
        self.with_limb = with_limb
        self.scaling = scaling
        self.eps = 0.0001
        assert self.with_kp or self.with_limb, 'At least one of "with_limb" and "with_kp" should be set as True.'

    def generate_a_heatmap(self, arr, centers, max_values) -> None:
        sigma = self.sigma
        (img_h, img_w) = arr.shape
        for (center, max_value) in zip(centers, max_values):
            if max_value < self.eps:
                continue
            (mu_x, mu_y) = (center[0], center[1])
            st_x = max(int(mu_x - 3 * sigma), 0)
            ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
            st_y = max(int(mu_y - 3 * sigma), 0)
            ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
            x = np.arange(st_x, ed_x, 1, np.float32)
            y = np.arange(st_y, ed_y, 1, np.float32)
            if not (len(x) and len(y)):
                continue
            y = y[:, None]
            patch = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / 2 / sigma ** 2)
            patch = patch * max_value
            arr[st_y:ed_y, st_x:ed_x] = np.maximum(arr[st_y:ed_y, st_x:ed_x], patch)

    def generate_a_limb_heatmap(self, arr, starts, ends, start_values, end_values) -> None:
        sigma = self.sigma
        (img_h, img_w) = arr.shape
        for (start, end, start_value, end_value) in zip(starts, ends, start_values, end_values):
            value_coeff = min(start_value, end_value)
            if value_coeff < self.eps:
                continue
            (min_x, max_x) = (min(start[0], end[0]), max(start[0], end[0]))
            (min_y, max_y) = (min(start[1], end[1]), max(start[1], end[1]))
            min_x = max(int(min_x - 3 * sigma), 0)
            max_x = min(int(max_x + 3 * sigma) + 1, img_w)
            min_y = max(int(min_y - 3 * sigma), 0)
            max_y = min(int(max_y + 3 * sigma) + 1, img_h)
            x = np.arange(min_x, max_x, 1, np.float32)
            y = np.arange(min_y, max_y, 1, np.float32)
            if not (len(x) and len(y)):
                continue
            y = y[:, None]
            x_0 = np.zeros_like(x)
            y_0 = np.zeros_like(y)
            d2_start = (x - start[0]) ** 2 + (y - start[1]) ** 2
            d2_end = (x - end[0]) ** 2 + (y - end[1]) ** 2
            d2_ab = (start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2
            if d2_ab < 1:
                self.generate_a_heatmap(arr, start[None], start_value[None])
                continue
            coeff = (d2_start - d2_end + d2_ab) / 2.0 / d2_ab
            a_dominate = coeff <= 0
            b_dominate = coeff >= 1
            seg_dominate = 1 - a_dominate - b_dominate
            position = np.stack([x + y_0, y + x_0], axis=-1)
            projection = start + np.stack([coeff, coeff], axis=-1) * (end - start)
            d2_line = position - projection
            d2_line = d2_line[:, :, 0] ** 2 + d2_line[:, :, 1] ** 2
            d2_seg = a_dominate * d2_start + b_dominate * d2_end + seg_dominate * d2_line
            patch = np.exp(-d2_seg / 2.0 / sigma ** 2)
            patch = patch * value_coeff
            arr[min_y:max_y, min_x:max_x] = np.maximum(arr[min_y:max_y, min_x:max_x], patch)

    def generate_heatmap(self, arr, kps, max_values, skeletons) -> None:
        if self.with_kp:
            num_kp = kps.shape[1]
            for i in range(num_kp):
                self.generate_a_heatmap(arr[i], kps[:, i], max_values[:, i])
        if self.with_limb:
            for (i, limb) in enumerate(skeletons):
                (start_idx, end_idx) = limb
                starts = kps[:, start_idx]
                ends = kps[:, end_idx]
                start_values = max_values[:, start_idx]
                end_values = max_values[:, end_idx]
                self.generate_a_limb_heatmap(arr[i], starts, ends, start_values, end_values)

    def gen(self, pose, size, skeletons=None):
        skeletons = skeletons or []
        (T, V, _) = pose.shape
        img_h = img_w = int(size)
        num_c = 0
        if self.with_kp:
            num_c += V
        if self.with_limb:
            num_c += len(skeletons)
        ret = np.zeros([T, num_c, img_h, img_w], dtype=np.float32)
        for t in range(T):
            kps = pose[t, :, :2][None]
            scores = pose[t, :, 2][None]
            self.generate_heatmap(ret[t], kps, scores, skeletons)
        return ret
