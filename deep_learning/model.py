"""
多任务学习模型架构：共享Encoder + 两个独立Task Heads
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import config


class Encoder(nn.Module):
    """共享编码器：对历史轨迹进行时序编码"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                 dropout: float, bidirectional: bool = False, encoder_type: str = 'lstm'):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.encoder_type = encoder_type
        
        # 选择编码器类型
        if encoder_type.lower() == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        elif encoder_type.lower() == 'gru':
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, return_hidden: bool = True):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            return_hidden: 是否返回隐藏状态
        
        Returns:
            output: (batch_size, seq_len, hidden_dim * num_directions)
            hidden: 最后一步的隐藏状态或全0（根据return_hidden）
        """
        rnn_output = self.rnn(x)
        if isinstance(self.rnn, nn.LSTM):
            output, (h, c) = rnn_output
        else:
            output, h = rnn_output
        output = self.dropout(output)
        
        if return_hidden:
            # 取最后一步作为序列编码
            if self.bidirectional:
                # 对双向LSTM，拼接两个方向
                hidden = output[:, -1, :]  # (batch_size, hidden_dim * 2)
            else:
                hidden = output[:, -1, :]  # (batch_size, hidden_dim)
            return output, hidden
        else:
            return output, None


class TrajectoryHead(nn.Module):
    """轨迹预测头：回归任务"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list, dropout: float):
        super(TrajectoryHead, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)
        
        Returns:
            (batch_size, output_dim) - 预测的未来轨迹向量
        """
        return self.net(x)


class IntentHead(nn.Module):
    """意图分类头：分类任务"""
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: list, dropout: float):
        super(IntentHead, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)
        
        Returns:
            (batch_size, num_classes) - logits
        """
        return self.net(x)


class MultiTaskTrajectoryModel(nn.Module):
    """多任务学习模型：轨迹预测 + 意图分类"""
    
    def __init__(self):
        super(MultiTaskTrajectoryModel, self).__init__()
        
        # 共享Encoder
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
        
        # 轨迹预测头
        self.traj_head = TrajectoryHead(
            input_dim=encoder_output_dim,
            output_dim=config.PRED_LEN * config.TRAJ_DIM,
            hidden_dims=config.TRAJ_HEAD_HIDDEN_DIMS,
            dropout=config.TRAJ_HEAD_DROPOUT
        )
        
        # 意图分类头
        self.intent_head = IntentHead(
            input_dim=encoder_output_dim,
            num_classes=config.NUM_INTENTS,
            hidden_dims=config.INTENT_HEAD_HIDDEN_DIMS,
            dropout=config.INTENT_HEAD_DROPOUT
        )
        
        self.pred_len = config.PRED_LEN
        self.traj_dim = config.TRAJ_DIM
    
    def forward(self, obs_traj: torch.Tensor = None, full_traj: torch.Tensor = None) -> tuple:
        """
        Args:
            obs_traj: Optional (batch_size, obs_len, traj_dim) - 用于traj预测
            full_traj: Optional (batch_size, full_traj_len, traj_dim) - 用于intent分类
        
        Returns:
            pred_traj: (batch_size, pred_len, traj_dim) or None
            intent_logits: (batch_size, num_intents) or None
        """
        pred_traj = None
        intent_logits = None
        
        # 轨迹预测分支
        if obs_traj is not None:
            _, hidden_traj = self.encoder(obs_traj, return_hidden=True)
            pred_logits = self.traj_head(hidden_traj)
            pred_traj = pred_logits.view(-1, self.pred_len, self.traj_dim)
        
        # 意图分类分支
        if full_traj is not None:
            _, hidden_intent = self.encoder(full_traj, return_hidden=True)
            intent_logits = self.intent_head(hidden_intent)
        
        return pred_traj, intent_logits
    
    def get_encoder_output(self, obs_traj: torch.Tensor) -> torch.Tensor:
        """获取编码器输出（用于可视化或特征提取）"""
        _, hidden = self.encoder(obs_traj, return_hidden=True)
        return hidden


class MultiTaskLoss(nn.Module):
    """多任务学习损失函数（支持混合样本）"""
    
    def __init__(self, weight_traj: float = 1.0, weight_intent: float = 1.0, 
                 loss_type: str = 'mse'):
        super(MultiTaskLoss, self).__init__()
        self.weight_traj = weight_traj
        self.weight_intent = weight_intent
        self.loss_type = loss_type.lower()
        
        # 轨迹损失
        if self.loss_type == 'mse':
            self.traj_loss_fn = nn.MSELoss(reduction='mean')
        elif self.loss_type == 'l1':
            self.traj_loss_fn = nn.L1Loss(reduction='mean')
        elif self.loss_type == 'smoothl1':
            self.traj_loss_fn = nn.SmoothL1Loss(reduction='mean')
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        # 意图损失
        self.intent_loss_fn = nn.CrossEntropyLoss(reduction='mean')
    
    def forward(self, pred_traj: torch.Tensor = None, true_traj: torch.Tensor = None,
                intent_logits: torch.Tensor = None, intent_labels: torch.Tensor = None,
                traj_indices: List = None, intent_indices: List = None) -> tuple:
        """
        Args:
            pred_traj: Optional (N_traj, pred_len, traj_dim)
            true_traj: Optional (N_traj, pred_len, traj_dim)
            intent_logits: Optional (N_intent, num_intents)
            intent_labels: (batch_size, 1) 或 (batch_size,)
            traj_indices: batch中有traj数据的样本索引
            intent_indices: batch中有intent数据的样本索引
        
        Returns:
            total_loss: 加权总损失
            traj_loss: 轨迹预测损失（如果没有traj样本则为0）
            intent_loss: 意图分类损失（如果没有intent样本则为0）
        """
        traj_loss = torch.tensor(0.0, device=intent_labels.device if intent_labels is not None else 'cpu')
        intent_loss = torch.tensor(0.0, device=intent_labels.device if intent_labels is not None else 'cpu')
        
        # 计算轨迹预测损失
        if pred_traj is not None and true_traj is not None and len(traj_indices) > 0:
            traj_loss = self.traj_loss_fn(pred_traj, true_traj)
        
        # 计算意图分类损失
        if intent_logits is not None and intent_labels is not None and len(intent_indices) > 0:
            # 只对有intent数据的样本计算损失
            intent_labels_selected = intent_labels[intent_indices]
            if intent_labels_selected.dim() > 1:
                intent_labels_selected = intent_labels_selected.squeeze(-1)
            intent_loss = self.intent_loss_fn(intent_logits, intent_labels_selected)
        
        # 加权总损失
        total_loss = self.weight_traj * traj_loss + self.weight_intent * intent_loss
        
        return total_loss, traj_loss, intent_loss


def create_model() -> MultiTaskTrajectoryModel:
    """创建多任务模型"""
    return MultiTaskTrajectoryModel()


def create_loss_fn() -> MultiTaskLoss:
    """创建损失函数"""
    return MultiTaskLoss(
        weight_traj=config.LOSS_WEIGHT_TRAJ,
        weight_intent=config.LOSS_WEIGHT_INTENT,
        loss_type=config.TRAJ_LOSS_TYPE
    )
