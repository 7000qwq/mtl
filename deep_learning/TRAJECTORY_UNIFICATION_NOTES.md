# 轨迹长度统一修改说明

## 修改概述

为了解决不同轨迹长度不一致的问题，通过线性插值的方式将所有原始轨迹统一到相同长度。

## 修改内容

### 1. 配置文件 (config.py)

**新增参数:**
```python
UNIFIED_TRAJ_LEN = 150  # 统一所有原始轨迹到此长度（通过插值/降采样）
```

**说明:**
- 所有从JSON文件读取的原始轨迹都会被重采样到150个时间步
- 可以根据实际需求调整此参数
- 建议设置为大于等于 `OBS_LEN + PRED_LEN` 的值

### 2. 数据加载器 (data_loader.py)

#### 修改1: 新增轨迹重采样方法

```python
def _resample_trajectory(self, trajectory: np.ndarray, target_length: int) -> np.ndarray:
    """
    通过线性插值将轨迹重采样到目标长度
    
    Args:
        trajectory: 原始轨迹 (original_len, 3)
        target_length: 目标长度
    
    Returns:
        重采样后的轨迹 (target_length, 3)
    """
```

**功能:**
- 使用 `numpy.interp` 对每个维度 (x, y, z) 分别进行线性插值
- 保持起点和终点位置不变
- 平滑地调整中间时间步的分布

#### 修改2: 更新轨迹提取方法

```python
def _extract_trajectory(self, data: Dict) -> np.ndarray:
    """从JSON数据提取轨迹序列并统一长度"""
```

**变化:**
- 提取位置信息后，立即调用 `_resample_trajectory` 统一长度
- 确保所有后续处理的轨迹长度一致为 `UNIFIED_TRAJ_LEN`

#### 修改3: 简化full_traj处理逻辑

**之前:**
```python
if len(traj) >= config.FULL_TRAJ_LEN:
    # 随机截取
else:
    # 填充
```

**现在:**
```python
if config.UNIFIED_TRAJ_LEN >= config.FULL_TRAJ_LEN:
    # 随机截取（因为所有traj长度都是UNIFIED_TRAJ_LEN）
else:
    # 填充
```

**说明:**
- 由于所有轨迹已统一到 `UNIFIED_TRAJ_LEN`，不再需要每次检查单个轨迹长度
- 逻辑更简洁，性能更好

## 优势

### 1. 数据一致性
- **之前:** 每条轨迹长度不同（50-500个时间步不等）
- **现在:** 所有轨迹统一为150个时间步
- **好处:** 
  - 批处理更高效
  - 模型输入更规范
  - 避免了填充带来的虚假数据

### 2. 信息保留
- **线性插值优势:**
  - 保持轨迹的完整性和连续性
  - 起点和终点位置精确保留
  - 轨迹形状特征得以保持
  
- **对比填充方法:**
  - 填充会在末尾添加重复点，破坏轨迹动态特征
  - 插值保持了时间序列的平滑性

### 3. 灵活性
- **短轨迹:** 通过插值增加时间步，平滑过渡
- **长轨迹:** 通过降采样减少时间步，保留关键信息
- **相同长度:** 直接返回，无需处理

## 影响的文件

### 直接修改
1. `config.py` - 新增配置参数
2. `data_loader.py` - 实现重采样逻辑

### 无需修改
以下文件无需修改，因为它们使用data_loader提供的数据：
- `train.py` - 训练脚本
- `inference.py` - 推理脚本
- `evaluation.py` - 评估脚本
- `model.py` - 模型定义
- `view_trajectory.py` - 可视化工具（处理原始JSON，不经过data_loader）

## 测试验证

创建了测试脚本 `test_trajectory_resampling.py` 用于验证：

### 测试内容
1. **单元测试:** 测试不同长度轨迹的重采样
2. **实际数据测试:** 加载真实JSON文件验证
3. **完整流程测试:** 验证整个数据加载流程
4. **可视化对比:** 生成重采样前后的对比图

### 运行测试
```bash
cd deep_learning
python test_trajectory_resampling.py
```

### 预期输出
- 所有轨迹长度统一为150
- 起点和终点位置保持不变
- 生成对比可视化图表

## 使用建议

### 1. 调整统一长度
如果需要修改统一长度，只需在 `config.py` 中调整：
```python
UNIFIED_TRAJ_LEN = 200  # 例如改为200
```

### 2. 选择合适的长度
建议考虑：
- **最小值:** 应 >= `OBS_LEN + PRED_LEN` (当前为30)
- **最大值:** 考虑计算资源和训练效率
- **推荐值:** 150-300 之间（覆盖大部分轨迹的实际长度）

### 3. 验证效果
修改后建议：
1. 运行测试脚本验证功能
2. 检查训练过程是否正常
3. 对比修改前后的模型性能

## 技术细节

### 线性插值实现
```python
# 原始时间索引: [0, 1]区间均匀分布
x_old = np.linspace(0, 1, original_len)
# 目标时间索引: [0, 1]区间均匀分布
x_new = np.linspace(0, 1, target_length)

# 对每个维度分别插值
for dim in range(3):  # x, y, z
    resampled[:, dim] = np.interp(x_new, x_old, trajectory[:, dim])
```

### 性能考虑
- **时间复杂度:** O(n × d)，n为目标长度，d为维度数(3)
- **空间复杂度:** O(n × d)
- **执行时机:** 数据加载时一次性完成，不影响训练速度

## 注意事项

1. **现有模型兼容性:**
   - 如果已有训练好的模型，重新处理数据后需要重新训练
   - 归一化参数会发生变化

2. **数据一致性:**
   - 确保训练、验证、测试集都使用相同的处理方式
   - data_loader已自动保证这一点

3. **可视化:**
   - `view_trajectory.py` 显示原始数据（未重采样）
   - 如需查看重采样后的数据，使用测试脚本的可视化功能

## 后续优化建议

1. **高级插值方法:**
   - 当前使用线性插值
   - 可考虑样条插值(cubic spline)获得更平滑的轨迹

2. **自适应长度:**
   - 根据意图类型设置不同的统一长度
   - 例如：hover用较短长度，straight_line用较长长度

3. **数据增强:**
   - 基于重采样实现时间尺度变换
   - 增加训练数据的多样性
