# 多任务学习框架重构说明

## 概述
重构实现了**时间尺度不同的对齐方式 + 同一 batch 混合两种样本**的多任务训练架构。

## 核心改动

### 1. 配置文件 (config.py)
**新增参数：**
- `FULL_TRAJ_LEN = 100`: 意图识别任务使用的完整轨迹长度
- `SAMPLE_MIX_RATIO`: 混合样本的比例配置
  - `'both': 0.5` - 同时计算traj+intent损失
  - `'traj_only': 0.3` - 只计算traj损失
  - `'intent_only': 0.2` - 只计算intent损失
- `LOSS_WEIGHT_INTENT`: 调整为0.5作为lambda_intent

### 2. 数据加载器 (data_loader.py)
**关键改动：**

#### 样本构造策略
- **轨迹预测样本**: 使用滑动窗口 (obs_len=20, pred_len=10)
- **意图识别样本**: 使用完整轨迹 (full_traj_len=100)
- 每个样本随机分配类型标签: `'both'`, `'traj_only'`, `'intent_only'`

#### 样本结构
```python
{
    'sample_type': str,         # 'both'|'traj_only'|'intent_only'
    'obs_traj': ndarray,        # (20, 3) - 仅traj/both样本有
    'pred_traj': ndarray,       # (10, 3) - 仅traj/both样本有
    'full_traj': ndarray,       # (100, 3) - 仅intent/both样本有
    'intent': int,
    'intent_label': str
}
```

#### 自定义collate_fn
```python
def custom_collate_fn(batch) -> Dict:
    """
    返回:
    - obs_traj: (N_traj, 20, 3) - 只包含有traj数据的样本
    - pred_traj: (N_traj, 10, 3)
    - full_traj: (N_intent, 100, 3) - 只包含有intent数据的样本
    - traj_indices: [int] - batch中有traj数据的索引
    - intent_indices: [int] - batch中有intent数据的索引
    """
```

### 3. 模型架构 (model.py)

#### 模型forward支持可选输入
```python
def forward(self, obs_traj=None, full_traj=None):
    """
    Args:
        obs_traj: Optional (B1, 20, 3) - 用于traj预测
        full_traj: Optional (B2, 100, 3) - 用于intent分类
    
    Returns:
        pred_traj: (B1, 10, 3) or None
        intent_logits: (B2, num_intents) or None
    """
```

#### 损失函数支持混合样本
```python
def forward(self, pred_traj=None, true_traj=None,
            intent_logits=None, intent_labels=None,
            traj_indices=None, intent_indices=None):
    """
    - 只对有traj数据的样本计算traj_loss
    - 只对有intent数据的样本计算intent_loss
    - total_loss = traj_loss + lambda_intent * intent_loss
    """
```

### 4. 训练循环 (train.py)

#### train_epoch关键逻辑
```python
# 1. 提取batch中的索引信息
traj_indices = batch['traj_indices']      # 有traj的样本
intent_indices = batch['intent_indices']  # 有intent的样本

# 2. 准备可选输入
obs_traj = batch['obs_traj'] if available else None
full_traj = batch['full_traj'] if available else None

# 3. 模型前向
pred_traj_out, intent_logits = model(obs_traj=obs_traj, full_traj=full_traj)

# 4. 分别计算损失
loss, traj_loss, intent_loss = loss_fn(
    pred_traj=pred_traj_out,
    true_traj=pred_traj_true,
    intent_logits=intent_logits,
    intent_labels=intent,
    traj_indices=traj_indices,
    intent_indices=intent_indices
)
```

## 工作机制

### 同一batch中的混合样本处理

假设batch_size=4，一个batch可能包含：
```
[
    样本0: type='both'        → 计算traj_loss + intent_loss
    样本1: type='traj_only'   → 只计算traj_loss
    样本2: type='intent_only' → 只计算intent_loss
    样本3: type='both'        → 计算traj_loss + intent_loss
]
```

经过collate_fn处理后：
```python
{
    'obs_traj': Tensor[3, 20, 3],        # 样本0,1,3
    'pred_traj': Tensor[3, 10, 3],       # 样本0,1,3
    'full_traj': Tensor[3, 100, 3],      # 样本0,2,3
    'traj_indices': [0, 1, 3],
    'intent_indices': [0, 2, 3],
    'intent': Tensor[4, 1]               # 所有样本的标签
}
```

### 损失计算流程

1. **轨迹预测损失**:
   - 输入: obs_traj (3, 20, 3) → 输出: pred_traj (3, 10, 3)
   - 与 true_traj (3, 10, 3) 计算MSE
   - 只对样本0,1,3计算

2. **意图分类损失**:
   - 输入: full_traj (3, 100, 3) → 输出: intent_logits (3, 4)
   - 从intent中提取样本0,2,3的标签
   - 计算交叉熵

3. **总损失**:
   ```python
   total_loss = 1.0 * traj_loss + 0.5 * intent_loss
   ```

## 时间尺度对齐

### 轨迹预测任务
- **观察窗口**: 20个时间步 (2秒 @ 10Hz)
- **预测窗口**: 10个时间步 (1秒 @ 10Hz)
- **用途**: 短期轨迹预测

### 意图识别任务
- **完整轨迹**: 100个时间步 (10秒 @ 10Hz)
- **用途**: 长期行为模式识别

## 优势

1. **灵活的任务权重**: 通过混合比例控制每个任务的数据量
2. **高效训练**: 在同一batch中处理两种时间尺度
3. **避免过拟合**: 随机样本类型增加数据多样性
4. **可扩展性**: 易于调整混合比例和时间窗口长度

## 使用建议

### 调整混合比例
在 `config.py` 中修改:
```python
SAMPLE_MIX_RATIO = {
    'both': 0.6,          # 增加同时训练两任务的样本
    'traj_only': 0.2,     # 减少单任务样本
    'intent_only': 0.2
}
```

### 调整损失权重
```python
LOSS_WEIGHT_TRAJ = 1.0      # 轨迹预测基准
LOSS_WEIGHT_INTENT = 0.3    # 降低意图识别权重
```

### 调整时间窗口
```python
OBS_LEN = 30              # 增加观察窗口
PRED_LEN = 15             # 增加预测窗口
FULL_TRAJ_LEN = 150       # 增加完整轨迹长度
```

## 运行

```bash
cd deep_learning
python train.py
```

训练过程中会输出每个batch的样本统计：
```
[1/100] Loss: 0.0234 | Traj: 0.0180 (24 samples) | Intent: 0.0108 (28 samples)
Epoch统计: Traj样本=768, Intent样本=896
```
