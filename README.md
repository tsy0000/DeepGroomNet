# DeepGroomNet

面向小鼠旷场（Open-Field）顶拍视频的高精度行为识别系统，支持五类行为帧级分类：**Locomote、Rear、FaceGroom、BodyGroom、Background**。

## 核心能力

- **细粒度行为判别**：FaceGroom / BodyGroom 在训练、验证、测试集及未见过的视频上均达到 **F1 ≥ 90%**；其余行为 **F1 ≥ 80%**
- **跨视频泛化**：多通道运动编码与关键点几何先验对光照、背景、拍摄条件差异鲁棒，新视频直接可用
- **自然分界支持**：兼容 Keypoint-MoSeq 音节边界作为推理聚合单元，行为分段与无监督姿势动力学对齐

## 系统架构

### 1. 视频骨干（创新性改造）
以 TAdaConvNeXtV2（Kinetics-400 预训练）为核心，在不改动官方 block 的前提下进行模块化插拔增强：

- **多通道运动编码**：RGB + 帧间差分 + 光流（预训练光流生成器）作为输入，运动信息与外观解耦
- **分层注意力**：轻量注意力模块插入浅层 stage，以运动统计（快速振荡比、运动集中度）为门控输入，配合空间先验（中心高斯、小鼠掩码、关键点热图）实现行为相关区域聚焦
- **渐进式迁移**：homecage 中间域预热 → head-only 微调 → 全量微调，多阶段稳定收敛

### 2. 关键点融合（主流姿势引导方案）
- DeepLabCut 关键点（27 点，superanimal_topviewmouse 预训练）经几何归一化（平移/尺度不变）
- **PoseC3D 式伪热图**：关键点高斯热图 + 肢体线段热图，经 3D 卷积流与视觉流晚期融合
- 关键点同时作为**注意力空间先验**，将门控聚焦于小鼠本体（实测聚焦比 >10×）
- 无关键点数据时自动降级，不影响推理

### 3. 分类与评估（多标签框架）
- 逐类 sigmoid 多标签头 + 类别加权损失（稀有类提升）
- 验证集逐类阈值调优，background 作为其余行为的补集
- 序列感知后处理 + 时长约束的 bout 滤波，抑制孤立误判与不合理切换

### 4. 推理与可视化
- **网页应用**（Streamlit）：上传视频 → 帧级行为预测 → 行为时间线可视化 → 音节级汇总，全程可交互
- 训练过程诊断：分组梯度范数、注意力门统计、类间分离度、注意力热图

## 快速开始

```bash
# 1. 环境（Python 3.10 + PyTorch ≥2.1 + CUDA ≥11.8）
pip install -r requirements.txt

# 2. 数据准备
python scripts/01_build_metadata.py   # 元数据构建
python scripts/02_gen_clips.py        # 预裁剪 clip 生成（YOLO 检测）
python scripts/13_gen_flow.py         # 光流预计算（可选但推荐）

# 3. 训练
python train.py

# 4. 推理（CLI）
python infer.py --video <video.mp4> --ckpt outputs/checkpoints/best_model.pt \
                --syllables <moseq_syllables.csv>   # 可选：音节边界文件

# 5. 推理（Web）
streamlit run web_app.py
```

输出：帧级行为预测 CSV（含置信度）与音节级行为汇总 CSV。

## 项目结构

```
configs/       配置文件（数据路径、模型、训练、后处理参数）
data/          clip 采样、关键点加载、光流、伪热图生成
models/        骨干与模块（注意力、关键点分支、光流适配）
losses/        损失函数（多标签加权损失）
postprocess/   阈值读出口、时序滤波、音节投票、bout 滤波
llm/           辅助分析（可选，不参与训练）
scripts/       数据准备、训练、评估、可视化
train.py       训练入口
infer.py       推理入口
web_app.py     Streamlit 交互应用
train_diagnostics.py  训练诊断
eval_metrics.py       评估指标
```

## 评估协议

- 训练/验证/测试按**小鼠个体分层**划分（70/15/15），跨个体泛化
- 报告逐类 precision / recall / F1、宏/加权 F1、混淆矩阵
- 阈值在验证集调优、测试集报告；提供独立评估脚本（`scripts/11_eval_test.py`）

## Requirements

见 `requirements.txt` / `environment.yml`。
