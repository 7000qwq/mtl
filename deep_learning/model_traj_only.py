"""
单任务模型：仅轨迹预测 (Trajectory Prediction Only Baseline)
用于论文对比实验

模型结构：Encoder + TrajectoryHead（与MTL框架完全一致）
"""
import torch
import torch.nn as nn
import config

# 直接复用MTL框架中的Encoder和TrajectoryHead
from model import Encoder, TrajectoryHead


class TrajectoryOnlyModel(nn.Module):
    """单任务轨迹预测模型：仅包含 Encoder + TrajectoryHead"""
    
    def __init__(self):
        super(TrajectoryOnlyModel, self).__init__()
        
        # 使用与MTL框架完全一致的Encoder
        self.encoder = Encoder(
            input_dim=config.TRAJ_DIM,
            hidden_dim=config.ENCODER_HIDDEN_DIM,
            num_layers=config.ENCODER_NUM_LAYERS,
            dropout=config.ENCODER_DROPOUT,
            bidirectional=config.ENCODER_BIDIRECTIONAL,
            encoder_type=config.ENCODER_TYPE
        )
        
        # 计算encoder输出维度
        encoder_output_dim = config.ENCODER_HIDDEN_DIM
        if config.ENCODER_BIDIRECTIONAL:
            encoder_output_dim *= 2
        
        # 使用与MTL框架完全一致的TrajectoryHead
        self.traj_head = TrajectoryHead(
            input_dim=encoder_output_dim,
            output_dim=config.PRED_LEN * config.TRAJ_DIM,
            hidden_dims=config.TRAJ_HEAD_HIDDEN_DIMS,
            dropout=config.TRAJ_HEAD_DROPOUT
        )
        
        self.pred_len = config.PRED_LEN
        self.traj_dim = config.TRAJ_DIM
    
    def forward(self, obs_traj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_traj: (batch_size, obs_len, traj_dim) - 历史轨迹
        
        Returns:
            pred_traj: (batch_size, pred_len, traj_dim) - 预测的未来轨迹
        """
        # Encoder
        _, hidden = self.encoder(obs_traj, return_hidden=True)
        
        # Trajectory Head
        pred_logits = self.traj_head(hidden)
        pred_traj = pred_logits.view(-1, self.pred_len, self.traj_dim)
        
        return pred_traj
    
    def get_encoder_output(self, obs_traj: torch.Tensor) -> torch.Tensor:
        """获取编码器输出（用于可视化或特征提取）"""
        _, hidden = self.encoder(obs_traj, return_hidden=True)
        return hidden


class TrajectoryLoss(nn.Module):
    """轨迹预测损失函数"""
    
    def __init__(self, loss_type: str = 'mse'):
        super(TrajectoryLoss, self).__init__()
        self.loss_type = loss_type.lower()
        
        if self.loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction='mean')
        elif self.loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='mean')
        elif self.loss_type == 'smoothl1':
            self.loss_fn = nn.SmoothL1Loss(reduction='mean')
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, pred_traj: torch.Tensor, true_traj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_traj: (batch_size, pred_len, traj_dim) - 预测轨迹
            true_traj: (batch_size, pred_len, traj_dim) - 真实轨迹
        
        Returns:
            loss: 标量损失值
        """
        return self.loss_fn(pred_traj, true_traj)


def create_traj_model() -> TrajectoryOnlyModel:
    """创建单任务轨迹预测模型"""
    return TrajectoryOnlyModel()


def create_traj_loss_fn() -> TrajectoryLoss:
    """创建轨迹预测损失函数"""
    return TrajectoryLoss(loss_type=config.TRAJ_LOSS_TYPE)
