from __future__ import annotations
import json
import os
import torch
import torch.nn.functional as F
OPENFIELD_DEFS = {'background': '旷场箱壁附近无行为学意义的空背景区域', 'Locomote': '小鼠在旷场内行走/奔跑，整体位移明显', 'Rear': '后腿站立，前爪离地或扶壁，身体竖直伸展', 'FaceGroom': '用前爪/嘴部快速搓洗面部与头部（面部梳理），前肢与头部高频小幅运动', 'BodyGroom': '低头舔舐/啃咬躯干、腹部、后肢与尾部（身体梳理），躯干部位运动'}
HOMECAGE_DEFS = {'drink': '从饮水嘴舔水', 'eat': '进食/啃咬食物（嘴部参与）', 'groom': '梳理毛发（面部/身体），嘴爪并用', 'hang': '攀爬悬挂于笼盖，身体倒挂或悬垂', 'micromovement': '微幅运动（微小动作，如耳动、鼻嗅），无明显位移', 'rear': '后腿站立（笼内直立）', 'rest': '静止休息/蜷缩睡眠', 'walk': '行走/移动（笼内）'}
BUILTIN_OPENFIELD = [[1.0, 0.15, 0.1, 0.15, 0.1], [0.15, 1.0, 0.35, 0.2, 0.2], [0.1, 0.35, 1.0, 0.2, 0.15], [0.15, 0.2, 0.2, 1.0, 0.55], [0.1, 0.2, 0.15, 0.55, 1.0]]
BUILTIN_HOMECAGE = [[1.0, 0.55, 0.25, 0.1, 0.15, 0.1, 0.1, 0.15], [0.55, 1.0, 0.5, 0.1, 0.2, 0.1, 0.1, 0.2], [0.25, 0.5, 1.0, 0.15, 0.3, 0.15, 0.3, 0.2], [0.1, 0.1, 0.15, 1.0, 0.15, 0.5, 0.05, 0.3], [0.15, 0.2, 0.3, 0.15, 1.0, 0.2, 0.4, 0.25], [0.1, 0.1, 0.15, 0.5, 0.2, 1.0, 0.1, 0.25], [0.1, 0.1, 0.3, 0.05, 0.4, 0.1, 1.0, 0.15], [0.15, 0.2, 0.2, 0.3, 0.25, 0.25, 0.15, 1.0]]

def _symmetrize(mat):
    n = len(mat)
    for i in range(n):
        for j in range(n):
            if i != j:
                v = (mat[i][j] + mat[j][i]) / 2.0
                mat[i][j] = mat[j][i] = v
    return mat

def builtin_matrices():
    return {'openfield': torch.tensor(_symmetrize([r[:] for r in BUILTIN_OPENFIELD]), dtype=torch.float32), 'homecage': torch.tensor(_symmetrize([r[:] for r in BUILTIN_HOMECAGE]), dtype=torch.float32)}

def reorder_matrix(mat, from_labels, to_labels):
    idx = [from_labels.index(n) for n in to_labels]
    sub = mat.index_select(0, torch.as_tensor(idx, dtype=torch.long))
    return sub.index_select(1, torch.as_tensor(idx, dtype=torch.long))

def _build_prompt(labels, defs):
    items = '；'.join((f'{i}: {name} —— {defs[name]}' for (i, name) in enumerate(labels)))
    n = len(labels)
    return f'你是动物行为学专家。下面是我在旷场/笼内小鼠视频中标注的 {n} 个行为类别定义：\n{items}\n\n请给出这些行为类别两两之间的语义相似度矩阵（对称，0~1，1=完全同义，0=毫不相关；对角线固定为 1）。要求：按你对该谱系小鼠 ethogram 的领域知识给出，重点区分 FaceGroom（面部梳理）与 BodyGroom（身体梳理）这类易混淆行为：它们相似度高（同属梳理）但不能为 1。\n只输出一个 JSON 二维数组（无其他文字），第 i 行第 j 列 = 类别 i 与类别 j 的相似度：'

def _parse_matrix(text, n):
    (start, end) = (text.find('['), text.rfind(']'))
    if start == -1 or end == -1 or end <= start:
        raise ValueError('LLM 输出中未找到 JSON 数组')
    import ast
    mat = ast.literal_eval(text[start:end + 1])
    if len(mat) != n or any((len(r) != n for r in mat)):
        raise ValueError(f'矩阵维度错误: 期望 {n}x{n}，实际 {len(mat)}x{(len(mat) if mat else 0)}')
    for (i, row) in enumerate(mat):
        for (j, v) in enumerate(row):
            v = float(v)
            mat[i][j] = min(max(v, 0.0), 1.0) if i != j else 1.0
    return torch.tensor(_symmetrize(mat), dtype=torch.float32)

def llm_semantic_matrices(cfg, cache_path, force_refresh=False):
    if os.path.exists(cache_path) and (not force_refresh):
        with open(cache_path, 'r', encoding='utf-8') as f:
            cached = json.load(f)
        return {'openfield': torch.tensor(cached['openfield'], dtype=torch.float32), 'homecage': torch.tensor(cached['homecage'], dtype=torch.float32)}
    from llm.client import build_llm_client
    client = build_llm_client(cfg)
    if client is None:
        return builtin_matrices()
    out = {}
    for (key, labels, defs) in (('openfield', list(OPENFIELD_DEFS.keys()), OPENFIELD_DEFS), ('homecage', list(HOMECAGE_DEFS.keys()), HOMECAGE_DEFS)):
        try:
            text = client.complete(_build_prompt(labels, defs), system='只输出 JSON，不要输出任何解释文字。', temperature=0.1, max_tokens=3000)
            out[key] = _parse_matrix(text, len(labels))
            print(f'[semantic_prior] LLM 生成 {key} 语义矩阵 OK')
        except Exception as e:
            print(f'[semantic_prior] LLM 生成 {key} 失败（{e}），回退内置 ethogram 先验')
            out[key] = builtin_matrices()[key]
    cache = {'openfield': out['openfield'].tolist(), 'homecage': out['homecage'].tolist()}
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    return out
TYPE_LEVELS_OPENFIELD = {'background': 0, 'Locomote': 1, 'Rear': 1, 'FaceGroom': 2, 'BodyGroom': 2}
TYPE_LEVELS_HOMECAGE = {'rest': 0, 'micromovement': 0, 'walk': 1, 'rear': 1, 'hang': 1, 'drink': 2, 'eat': 2, 'groom': 2}
TYPE_GROUP_OPENFIELD = {'background': 'bg', 'Locomote': 'locomotion', 'Rear': 'rearing', 'FaceGroom': 'grooming', 'BodyGroom': 'grooming'}
TYPE_GROUP_HOMECAGE = {'rest': 'still', 'micromovement': 'still', 'walk': 'locomotion', 'rear': 'rearing', 'hang': 'climbing', 'drink': 'orofacial', 'eat': 'orofacial', 'groom': 'grooming'}

def type_matrix(labels, type_group):
    n = len(labels)
    mat = [[0.1] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j or type_group[labels[i]] == type_group[labels[j]]:
                mat[i][j] = 1.0
    return torch.tensor(mat, dtype=torch.float32)

def ose_matrix(def_mat, labels, type_group, alpha=0.5):
    s_label = type_matrix(labels, type_group).to(def_mat.dtype)
    return alpha * s_label + (1.0 - alpha) * def_mat

def oagm_propagate(base_mat, labels, levels, lam=2.0, hops=2, tau=0.5, rho=0.5):
    C = base_mat.shape[0]
    levels = torch.as_tensor([levels.get(l, 1) for l in labels], dtype=torch.float32)
    d = levels.view(1, C) - levels.view(C, 1)
    g = torch.sigmoid(-lam * d)
    w = base_mat.clamp(min=0.0) * g
    w = w / (w.sum(dim=1, keepdim=True) + 1e-06)
    H = torch.eye(C, dtype=base_mat.dtype)
    for _ in range(max(hops, 0)):
        H = H + tau * w @ H
    Hn = F.normalize(H, dim=1)
    out = Hn @ Hn.t()
    out = out / out.max()
    out = rho * base_mat + (1.0 - rho) * out
    out.fill_diagonal_(1.0)
    return out

def enhance_semantic_matrix(mat, labels, levels, type_group, use_ose=True, ose_alpha=0.5, use_oagm=True, lam=2.0, hops=2, tau=0.5):
    out = mat.clone()
    if use_ose:
        out = ose_matrix(out, labels, type_group, alpha=ose_alpha)
    if use_oagm:
        out = oagm_propagate(out, labels, levels, lam=lam, hops=hops, tau=tau)
    out = (out + out.t()) / 2.0
    out.fill_diagonal_(1.0)
    return out

def semantic_regularization_loss(head_weight, target_sim, weight=1.0):
    w = F.normalize(head_weight, dim=1)
    cos = w @ w.t()
    return weight * F.mse_loss(cos, target_sim.to(cos.device))
