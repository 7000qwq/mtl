"""
单任务模型：仅意图识别 (Intent Classification Only Baseline)
用于论文对比实验

模型结构：Encoder + IntentHead（与MTL框架完全一致）
"""
import torch
import torch.nn as nn
import config

# 直接复用MTL框架中的Encoder和IntentHead
from model import Encoder, IntentHead


class IntentOnlyModel(nn.Module):
    """单任务意图识别模型：仅包含 Encoder + IntentHead"""
    
    def __init__(self):
        super(IntentOnlyModel, self).__init__()
        
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
        
        # 使用与MTL框架完全一致的IntentHead
        self.intent_head = IntentHead(
            input_dim=encoder_output_dim,
            num_classes=config.NUM_INTENTS,
            hidden_dims=config.INTENT_HEAD_HIDDEN_DIMS,
            dropout=config.INTENT_HEAD_DROPOUT
        )
    
    def forward(self, full_traj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            full_traj: (batch_size, full_traj_len, traj_dim) - 完整轨迹
        
        Returns:
            intent_logits: (batch_size, num_intents) - 意图分类logits
        """
        # Encoder
        _, hidden = self.encoder(full_traj, return_hidden=True)
        
        # Intent Head
        intent_logits = self.intent_head(hidden)
        
        return intent_logits
    
    def get_encoder_output(self, full_traj: torch.Tensor) -> torch.Tensor:
        """获取编码器输出（用于可视化或特征提取）"""
        _, hidden = self.encoder(full_traj, return_hidden=True)
        return hidden


class IntentLoss(nn.Module):
    """意图分类损失函数"""
    
    def __init__(self):
        super(IntentLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
    
    def forward(self, intent_logits: torch.Tensor, intent_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            intent_logits: (batch_size, num_intents) - 意图分类logits
            intent_labels: (batch_size,) - 真实意图标签
        
        Returns:
            loss: 标量损失值
        """
        if intent_labels.dim() > 1:
            intent_labels = intent_labels.squeeze(-1)
        return self.loss_fn(intent_logits, intent_labels)


def create_intent_model() -> IntentOnlyModel:
    """创建单任务意图识别模型"""
    return IntentOnlyModel()


def create_intent_loss_fn() -> IntentLoss:
    """创建意图分类损失函数"""
    return IntentLoss()
