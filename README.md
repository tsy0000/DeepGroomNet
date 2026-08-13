# DeepGroomNet

面向小鼠旷场（Open-Field）顶拍视频的多类行为识别系统。

## Overview

DeepGroomNet 对顶拍小鼠视频进行帧级行为分类，识别五类行为：Locomote、Rear、FaceGroom、BodyGroom 与 Background。系统以视频动作识别骨干为核心，融合关键点几何先验与光流运动编码，并提供完整的训练、评估与推理管线。

## Highlights

- **预训练视频骨干**：基于 TAdaConvNeXtV2（Kinetics-400 预训练），渐进式迁移学习（中间域预热 → head 微调 → 全量微调）
- **运动编码**：RGB + 帧差 + 光流（TinyMotionNet 预训练生成器）多通道输入
- **关键点融合**：DeepLabCut 关键点 → 伪热图（关键点 + 肢体）→ 3D CNN 分支，与视觉流晚期融合；无关键点数据时自动降级
- **多标签分类头**：逐类 sigmoid + 类别加权损失 + 验证集逐类阈值调优
- **时序后处理**：序列感知解码 + 基于时长的 bout 滤波
- **可解释性**：训练过程梯度/注意力诊断输出，注意力热图可视化

## Requirements

- Python 3.10, PyTorch ≥ 2.1, CUDA ≥ 11.8
- OpenCV, Ultralytics (YOLO 检测), NumPy, Pandas, scikit-learn
- 详细依赖见 `requirements.txt` / `environment.yml`

## Quick Start

```bash
# 1. 元数据与预裁剪 clip 生成
python scripts/01_build_metadata.py
python scripts/02_gen_clips.py

# 2. 数据审计
python scripts/07_data_audit.py

# 3. 光流预计算（可选，提升运动编码）
python scripts/13_gen_flow.py

# 4. 训练（homecage 预热 → head-only → 全量微调）
python train.py

# 5. 推理
python infer.py --video <video.mp4> --ckpt <best_model.pt>
```

输出：帧级行为预测 CSV（含置信度）与音节级行为摘要 CSV。

## Configuration

所有配置集中在 `configs/config.yaml`：

- 数据路径、clip 采样参数（窗口长度/步进/分辨率）
- 骨干与注意力参数、关键点/光流开关
- 损失与训练超参（batch、学习率、梯度裁剪、早停）
- 后处理参数（阈值调优、bout 滤波）

## Project Layout

```
configs/      配置
data/         数据加载与预处理（clip 采样、关键点、光流、热图）
models/       骨干与模块（注意力、关键点分支）
losses/       损失函数
postprocess/  推理后处理（阈值、时序滤波、音节投票）
llm/          辅助分析（可选，不参与训练）
scripts/      数据准备、训练、评估、可视化脚本
train.py      训练入口（渐进式迁移 + 多任务辅助 + 阈值评估）
infer.py      推理入口
```

## Evaluation

- 训练/验证/测试按小鼠个体分层划分（70/15/15）
- 指标：逐类 precision/recall/F1、宏/加权 F1、元素级准确率、混淆矩阵
- 阈值推断在验证集上逐类调优，于测试集报告最终结果

## License

See `LICENSE`.
