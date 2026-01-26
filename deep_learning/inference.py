"""
推理脚本：使用训练好的模型进行推理和可视化
"""
import torch
import numpy as np
import json
import os
from pathlib import Path
import config
from data_loader import get_data_manager
from model import create_model
from evaluation import evaluate_model, print_evaluation_results, get_per_class_metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


class ModelInference:
    """模型推理工具"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = create_model().to(self.device)
        self.data_manager = get_data_manager()
        
        # 加载模型
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"✓ 模型已加载: {checkpoint_path}")
    
    @torch.no_grad()
    def predict(self, obs_traj: np.ndarray) -> tuple:
        """
        单条轨迹预测
        
        Args:
            obs_traj: 观察轨迹 (obs_len, 3)
        
        Returns:
            pred_traj: 预测轨迹 (pred_len, 3)
            intent_logits: 意图logits (num_intents,)
            intent_label: 预测意图标签
        """
        # 归一化
        obs_traj_norm = (obs_traj - self.data_manager.train_dataset.norm_params['mean']) / \
                       self.data_manager.train_dataset.norm_params['std']
        
        # 转为张量
        obs_traj_tensor = torch.FloatTensor(obs_traj_norm).unsqueeze(0).to(self.device)
        
        # 推理：只使用obs_traj进行轨迹预测
        pred_traj_norm, intent_logits = self.model(obs_traj=obs_traj_tensor, full_traj=None)
        
        # 逆归一化
        pred_traj = self.data_manager.train_dataset.denormalize(pred_traj_norm[0].cpu().numpy())
        
        # 提取意图
        if intent_logits is not None:
            intent_idx = intent_logits.argmax(dim=1).item()
            intent_label = config.INTENT_CLASSES[intent_idx]
            intent_prob = torch.softmax(intent_logits[0], dim=0).cpu().numpy()
        else:
            intent_idx = -1
            intent_label = "Unknown"
            intent_prob = np.zeros(config.NUM_INTENTS)
        
        return pred_traj, intent_logits[0].cpu().numpy() if intent_logits is not None else None, intent_label, intent_prob
    
    def visualize_predictions(self, obs_traj: np.ndarray, true_pred_traj: np.ndarray = None, 
                             save_path: str = None):
        """
        可视化预测结果
        
        Args:
            obs_traj: 观察轨迹 (obs_len, 3)
            true_pred_traj: 真实预测轨迹 (pred_len, 3)，可选
            save_path: 保存路径
        """
        pred_traj, _, intent_label, intent_prob = self.predict(obs_traj)
        
        fig = plt.figure(figsize=(15, 5))
        
        # 3D 轨迹
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot(obs_traj[:, 0], obs_traj[:, 1], obs_traj[:, 2], 'b-o', label='Observed trajectory')
        ax1.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2], 'r-s', label='Predicted trajectory')
        if true_pred_traj is not None:
            ax1.plot(true_pred_traj[:, 0], true_pred_traj[:, 1], true_pred_traj[:, 2], 'g--^', label='Ground truth trajectory')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()
        ax1.set_title('3D Trajectory')
        
        # X-Y 平面
        ax2 = fig.add_subplot(132)
        ax2.plot(obs_traj[:, 0], obs_traj[:, 1], 'b-o', label='Observed trajectory')
        ax2.plot(pred_traj[:, 0], pred_traj[:, 1], 'r-s', label='Predicted trajectory')
        if true_pred_traj is not None:
            ax2.plot(true_pred_traj[:, 0], true_pred_traj[:, 1], 'g--^', label='Ground truth trajectory')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.legend()
        ax2.set_title('XY Plane')
        ax2.grid()
        
        # 意图概率
        ax3 = fig.add_subplot(133)
        colors = ['#1f77b4' if config.INTENT_CLASSES[i] == intent_label else '#d62728' 
             for i in range(len(config.INTENT_CLASSES))]
        ax3.barh(config.INTENT_CLASSES, intent_prob, color=colors)
        ax3.set_xlabel('Probability')
        ax3.set_title(f'Predicted intent: {intent_label}')
        ax3.set_xlim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 图表已保存: {save_path}")
        
        plt.show()


def plot_training_history(history_path: str, save_dir: str = None):
    """Plot training history"""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 损失
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train loss')
    ax.plot(history['val_loss'], label='Validation loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total loss')
    ax.legend()
    ax.grid()
    
    # 轨迹损失
    ax = axes[0, 1]
    ax.plot(history['train_traj_loss'], label='Train trajectory loss')
    ax.plot(history['val_traj_loss'], label='Validation trajectory loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Trajectory loss')
    ax.legend()
    ax.grid()
    
    # 意图损失
    ax = axes[0, 2]
    ax.plot(history['train_intent_loss'], label='Train intent loss')
    ax.plot(history['val_intent_loss'], label='Validation intent loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Intent classification loss')
    ax.legend()
    ax.grid()
    
    # 轨迹指标
    ax = axes[1, 0]
    ax.plot(history['val_traj_rmse'], label='RMSE')
    ax.plot(history['val_traj_mae'], label='MAE')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')
    ax.set_title('Trajectory metrics')
    ax.legend()
    ax.grid()
    
    # 轨迹 ADE/FDE
    ax = axes[1, 1]
    ax.plot(history['val_traj_fde'], label='FDE')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')
    ax.set_title('Trajectory FDE')
    ax.legend()
    ax.grid()
    
    # 意图指标
    ax = axes[1, 2]
    ax.plot(history['val_intent_acc'], label='Accuracy')
    ax.plot(history['val_intent_f1'], label='F1')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Metric value')
    ax.set_title('Intent metrics')
    ax.legend()
    ax.grid()
    
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, 'training_history.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 训练历史图表已保存: {save_path}")
    
    plt.show()


def plot_confusion_matrix(pred_labels: np.ndarray, true_labels: np.ndarray, save_dir: str = None):
    """Plot confusion matrix (trajectory-level intent classification)."""
    # 现在意图样本是“每条轨迹一个”，所以 confusion matrix 直接按轨迹计数即可
    cm = confusion_matrix(true_labels, pred_labels, labels=range(config.NUM_INTENTS))

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',           # 计数是整数
        cmap='Blues',
        xticklabels=config.INTENT_CLASSES,
        yticklabels=config.INTENT_CLASSES
    )
    plt.ylabel('True intent')
    plt.xlabel('Predicted intent')
    plt.title('Intent confusion matrix (trajectory-level)')

    plt.tight_layout()
    if save_dir:
        save_path = os.path.join(save_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 混淆矩阵已保存: {save_path}")
    plt.show()


def main():
    """主推理流程"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'best.pt')
    
    if not os.path.exists(checkpoint_path):
        print(f"✗ 未找到模型: {checkpoint_path}")
        print("请先运行 train.py 进行训练")
        return
    
    # 加载推理工具
    inference = ModelInference(checkpoint_path, device=device)
    
    # 准备数据
    print("\n准备评估数据...")
    train_loader, val_loader, test_loader = inference.data_manager.prepare_data()
    
    # 在测试集上评估
    print("\n在测试集上评估模型...")
    test_results = evaluate_model(inference.model, test_loader, device=device)
    print_evaluation_results(test_results, dataset_name="测试集")
    
    # 每个类别的详细指标
    print("\n每个意图类别的详细指标:")
    all_pred = []
    all_true = []
    with torch.no_grad():
        for batch in test_loader:
            # 提取batch数据
            intent_indices = batch['intent_indices']
            intent = batch['intent'].to(device)
            full_traj = batch['full_traj'].to(device) if batch['full_traj'] is not None else None
            
            # 只使用full_traj进行意图分类（如果有）
            _, intent_logits = inference.model(obs_traj=None, full_traj=full_traj)
            
            if intent_logits is not None and len(intent_indices) > 0:
                intent_preds = intent_logits.argmax(dim=1).cpu().numpy()
                intent_numpy = intent.cpu().numpy().flatten()  # 确保是1维数组
                intent_labels_selected = intent_numpy[intent_indices]
                all_pred.append(intent_preds)
                all_true.append(intent_labels_selected)
    
    if all_pred:
        all_pred = np.concatenate(all_pred)
        all_true = np.concatenate(all_true)
        per_class = get_per_class_metrics(all_pred, all_true)
        
        # 计算每条轨迹对应的样本数
        samples_per_traj = config.UNIFIED_TRAJ_LEN - (config.OBS_LEN + config.PRED_LEN) + 1
        
        for intent, metrics in per_class.items():
            traj_count = metrics['support'] / samples_per_traj
            print(f"\n  {intent}:")
            print(f"    精确率: {metrics['precision']:.4f}")
            print(f"    召回率: {metrics['recall']:.4f}")
            print(f"    F1:   {metrics['f1']:.4f}")
            print(f"    轨迹数: {traj_count:.1f} ({metrics['support']} 个样本)")
    else:
        print("  (没有intent分类数据)")
    
    # 绘制训练历史
    history_path = os.path.join(config.LOG_DIR, 'training_history.json')
    if os.path.exists(history_path):
        print(f"\n绘制训练历史...")
        plot_training_history(history_path, config.LOG_DIR)
    
    # 绘制混淆矩阵
    print(f"绘制混淆矩阵...")
    plot_confusion_matrix(all_pred, all_true, config.LOG_DIR)
    
    print("\n✓ 推理和评估完成\n")


if __name__ == '__main__':
    main()
