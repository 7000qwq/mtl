"""
多任务学习框架配置文件
"""
import os

# ============== 数据路径 ==============
DATA_DIR = 'flight_data_random'
OUTPUT_DIR = 'mtl_output'
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ============== 数据参数 ==============
UNIFIED_TRAJ_LEN = 150    # 统一所有原始轨迹到此长度（通过插值/降采样）
OBS_LEN = 20              # 观察窗口长度（历史轨迹时步数）
PRED_LEN = 10             # 预测窗口长度（未来轨迹时步数）
SAMPLE_INTERVAL = 0.1     # 采样间隔（秒），采样率10Hz对应0.1s
FULL_TRAJ_LEN = 150       # 意图识别用的完整轨迹长度（整段）

# 混合样本配置
SAMPLE_MIX_RATIO = {
    'both': 1.0,          # 同时计算traj+intent损失的样本比例
    'traj_only': 0.0,     # 只计算traj损失的样本比例
    'intent_only': 0.0    # 只计算intent损失的样本比例
}

# 轨迹特征维度（x, y, z坐标）
TRAJ_DIM = 3

# 意图类别
INTENT_CLASSES = ['hover', 'straight_line', 'turn', 'z_scan']
NUM_INTENTS = len(INTENT_CLASSES)
INTENT_TO_IDX = {intent: idx for idx, intent in enumerate(INTENT_CLASSES)}

# 数据集划分比例（轨迹级别）
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============== 模型参数 ==============
# Encoder（共享）
ENCODER_TYPE = 'lstm'     # 可选: lstm, gru, transformer
ENCODER_HIDDEN_DIM = 128
ENCODER_NUM_LAYERS = 2
ENCODER_DROPOUT = 0.2
ENCODER_BIDIRECTIONAL = False

# Trajectory Head（回归）
TRAJ_HEAD_HIDDEN_DIMS = [128, 64]
TRAJ_HEAD_DROPOUT = 0.2

# Intent Head（分类）
INTENT_HEAD_HIDDEN_DIMS = [64, 32]
INTENT_HEAD_DROPOUT = 0.2

# ============== 训练参数 ==============
BATCH_SIZE = 32

# ============== 混合采样Batch 구성 ==============
# 每个batch中轨迹预测(滑窗)样本与意图识别(整段)样本的数量。两者之和应等于 BATCH_SIZE。
TRAJ_PER_BATCH = 24
INTENT_PER_BATCH = 8

NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# 多任务学习权重
LOSS_WEIGHT_TRAJ = 1.0   # 轨迹预测损失权重
LOSS_WEIGHT_INTENT = 0.9  # 意图分类损失权重（lambda_intent）

# 轨迹损失类型
TRAJ_LOSS_TYPE = 'mse'    # 可选: mse, l1, smoothl1

# 优化器配置
OPTIMIZER = 'adam'        # 可选: adam, sgd
MOMENTUM = 0.9
SCHEDULER = 'cosine'      # 可选: cosine, step, none

# 梯度裁剪
GRAD_CLIP = 1.0

# ============== 验证与测试 ==============
VALIDATION_INTERVAL = 1   # 每N个epoch进行验证
EARLY_STOPPING_PATIENCE = 15
BEST_MODEL_METRIC = 'val_traj_rmse'  # 用于保存最优模型的指标

# ============== 硬件与日志 ==============
DEVICE = 'cuda'           # cuda 或 cpu
NUM_WORKERS = 4          # DataLoader 工作线程数
SEED = 42                # 随机数种子
VERBOSE = True           # 是否打印详细信息
LOG_INTERVAL = 10        # 每N个batch打印日志

# ============== 评估指标 ==============
# 用于计算轨迹预测指标的位置特征
EVAL_POS_FEATURES = ['x', 'y', 'z']  # 使用位置三维计算ADE/FDE
EVAL_USE_NORM = False    # 是否在计算前归一化