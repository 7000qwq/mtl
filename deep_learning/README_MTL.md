"""
README：多任务学习框架使用指南
"""

# 多任务学习框架：无人机轨迹预测与意图识别

完整的PyTorch深度学习框架，用于无人机飞行轨迹预测和飞行意图识别。

## 项目结构

```
├── config.py                  # 配置文件（超参数、路径、模型参数）
├── data_loader.py            # 数据加载和处理模块
├── model.py                  # 模型架构定义
├── train.py                  # 训练脚本
├── evaluation.py             # 评估指标计算
├── inference.py              # 推理和可视化
├── flight_data_random/       # 输入数据目录
│   ├── hover/
│   ├── landing/
│   ├── straight_line/
│   ├── takeoff/
│   ├── turn/
│   └── z_scan/
├── mtl_output/               # 输出目录（自动创建）
│   ├── checkpoints/          # 模型检查点
│   └── logs/                 # 日志和结果
└── README.md
```

## 核心特性

### 1. 模型架构

**共享Encoder + 两个Task Heads**

```
历史轨迹 (obs_len, 3)
    ↓
[共享Encoder - LSTM/GRU]
    ↓
  分支 ─────────────────────────────────────┬──────────────
    ├→ [Trajectory Head]  → 预测轨迹 (pred_len, 3)
    └→ [Intent Head]      → 意图分类 logits (num_intents,)
```

- **Encoder**: LSTM/GRU，支持双向编码
- **Trajectory Head**: 多层感知机，输出未来轨迹坐标
- **Intent Head**: 多层感知机，输出意图类别logits

### 2. 数据处理

**轨迹级别数据集划分**
- 避免信息泄漏：同一条轨迹的所有样本只出现在一个数据集中
- 训练/验证/测试比例：70% / 15% / 15%

**滑动窗口采样**
- 观察窗口长度：`OBS_LEN` = 20 时步
- 预测窗口长度：`PRED_LEN` = 10 时步
- 每条轨迹生成多个样本

**特征归一化**
- 使用训练集计算均值和标准差
- 验证和测试阶段保持一致

### 3. 多任务损失

```
Total Loss = weight_traj * L_trajectory + weight_intent * L_intent
```

- **轨迹损失**: MSE / L1 / Smooth L1
- **意图损失**: 交叉熵 (Cross Entropy)
- **可调权重**: 灵活平衡两个任务

### 4. 评估指标

**轨迹预测**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- ADE (Average Displacement Error)
- FDE (Final Displacement Error)

**意图识别**
- 准确率 (Accuracy)
- F1-score (加权)
- 每个类别的精确率、召回率

## 快速开始

### 步骤 1: 配置超参数

编辑 `config.py` 修改超参数：

```python
# 数据参数
OBS_LEN = 20              # 观察窗口长度
PRED_LEN = 10             # 预测窗口长度

# 模型参数
ENCODER_HIDDEN_DIM = 128
ENCODER_NUM_LAYERS = 2

# 训练参数
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
LOSS_WEIGHT_TRAJ = 1.0
LOSS_WEIGHT_INTENT = 1.0
```

### 步骤 2: 训练模型

```bash
python train.py
```

**输出**：
- `mtl_output/checkpoints/best.pt` - 最优模型
- `mtl_output/checkpoints/latest.pt` - 最新检查点
- `mtl_output/logs/training_history.json` - 训练历史
- `mtl_output/logs/test_results.json` - 测试结果

### 步骤 3: 推理和评估

```bash
python inference.py
```

**功能**：
- 在测试集上评估模型
- 每个意图类别的详细指标
- 绘制训练历史曲线
- 绘制意图识别混淆矩阵

## 模块说明

### config.py

所有配置参数的集中管理：

```python
# 意图类别
INTENT_CLASSES = ['takeoff', 'hover', 'straight_line', 'turn', 'landing', 'z_scan']

# 数据参数
OBS_LEN = 20
PRED_LEN = 10
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 模型参数
ENCODER_HIDDEN_DIM = 128
ENCODER_NUM_LAYERS = 2
TRAJ_HEAD_HIDDEN_DIMS = [128, 64]
INTENT_HEAD_HIDDEN_DIMS = [64, 32]

# 训练参数
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
```

### data_loader.py

**TrajectoryDataset**: PyTorch Dataset类
- 样本：(obs_traj, pred_traj, intent)
- 自动归一化和逆归一化

**DataManager**: 数据加载管理
- `load_data()`: 读取JSON轨迹数据
- `construct_samples()`: 滑动窗口采样
- `split_by_trajectory()`: 轨迹级别数据集划分
- `create_datasets()`: 创建PyTorch Dataset
- `prepare_data()`: 完整流程

### model.py

**Encoder**: 共享时序编码器
- 支持LSTM和GRU
- 可选双向编码

**TrajectoryHead**: 轨迹预测头
- 多层感知机
- 输出 pred_len * traj_dim 维向量

**IntentHead**: 意图分类头
- 多层感知机
- 输出 num_intents 维logits

**MultiTaskTrajectoryModel**: 完整模型
- 组合Encoder和两个Heads
- Forward返回 (pred_traj, intent_logits)

**MultiTaskLoss**: 多任务损失
- 支持多种轨迹损失 (MSE/L1/SmoothL1)
- 意图损失为交叉熵
- 加权组合

### train.py

**Trainer**: 训练器类
- `train_epoch()`: 单个epoch的训练
- `validate()`: 验证流程
- `train()`: 完整训练循环

**特性**：
- 自动计算训练指标
- 早停（early stopping）
- 最优模型保存
- 梯度裁剪
- 学习率调度

### evaluation.py

**计算函数**：
- `compute_ade()`: 平均位移误差
- `compute_fde()`: 最终位移误差
- `compute_rmse()`: 均方根误差
- `compute_mae()`: 平均绝对误差
- `compute_trajectory_metrics()`: 所有轨迹指标
- `compute_intent_metrics()`: 意图识别指标
- `evaluate_model()`: 完整评估

### inference.py

**ModelInference**: 推理工具
- `predict()`: 单条轨迹推理
- `visualize_predictions()`: 3D可视化

**可视化**：
- `plot_training_history()`: 训练曲线
- `plot_confusion_matrix()`: 意图混淆矩阵

## 高级用法

### 1. 修改数据窗口长度

```python
# config.py
OBS_LEN = 30   # 观察历史更长
PRED_LEN = 15  # 预测更远的未来
```

### 2. 调整多任务权重

```python
# config.py
LOSS_WEIGHT_TRAJ = 1.5    # 更强调轨迹预测
LOSS_WEIGHT_INTENT = 0.5  # 弱化意图分类
```

### 3. 使用不同的编码器

```python
# config.py
ENCODER_TYPE = 'gru'      # 使用GRU代替LSTM
ENCODER_BIDIRECTIONAL = True  # 双向编码
```

### 4. 自定义轨迹损失

```python
# config.py
TRAJ_LOSS_TYPE = 'smoothl1'  # 使用Smooth L1损失
```

### 5. 调整学习率策略

```python
# config.py
SCHEDULER = 'step'  # 使用Step调度器
LEARNING_RATE = 5e-4
```

## 常见问题

**Q: 数据不足怎么办？**

A: 可以调整 `OBS_LEN` 和 `PRED_LEN` 生成更多样本，或使用数据增强。

**Q: 轨迹预测效果不好？**

A: 尝试：
- 增加 `ENCODER_HIDDEN_DIM` 和层数
- 增加 `TRAJ_HEAD_HIDDEN_DIMS`
- 增加训练数据
- 调整 `LOSS_WEIGHT_TRAJ` 更强调轨迹任务

**Q: 意图识别不准确？**

A: 尝试：
- 增加 `INTENT_HEAD_HIDDEN_DIMS`
- 增加 `LOSS_WEIGHT_INTENT`
- 检查数据集是否不平衡

**Q: 模型过拟合？**

A: 尝试：
- 增加 `DROPOUT` 参数
- 增加 `WEIGHT_DECAY`
- 提前停止（早停已启用）

## 输出说明

**training_history.json**

记录每个epoch的损失和指标：
```json
{
  "train_loss": [...],
  "val_traj_rmse": [...],
  "val_intent_acc": [...]
}
```

**test_results.json**

测试集的最终评估结果：
```json
{
  "epoch": 45,
  "test_metrics": {
    "rmse": 0.0234,
    "mae": 0.0189,
    "ade": 0.0290,
    "fde": 0.0456,
    "accuracy": 0.9234,
    "f1": 0.9181
  }
}
```

## 扩展和改进

1. **改进Encoder**: 尝试Transformer或Attention机制
2. **多尺度预测**: 同时预测不同时间步长
3. **条件生成**: 基于意图生成轨迹
4. **不确定性估计**: 输出预测置信度
5. **在线学习**: 实时模型更新

## 参考文献

- Jain et al. (2016) - "Structural-RNN: Deep Learning on Spatio-Temporal Graphs"
- Gupta et al. (2018) - "Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks"
- Choi et al. (2019) - "Trajectron++: Dynamically-Feasible Trajectory Forecasting with Heterogeneous Agent Interactions"

## 许可证

MIT License

## 联系方式

如有问题或建议，请反馈。
