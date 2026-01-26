"""
评估模块：轨迹预测和意图识别指标计算
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch
from typing import Dict
import config


def compute_ade(pred_traj: np.ndarray, true_traj: np.ndarray) -> float:
    """
    计算平均位移误差 (Average Displacement Error, ADE)
    
    Args:
        pred_traj: 预测轨迹 (N, T, 3) 或 (T, 3)
        true_traj: 真实轨迹 (N, T, 3) 或 (T, 3)
    
    Returns:
        ADE 值
    """
    if pred_traj.ndim == 2:
        pred_traj = pred_traj[np.newaxis, ...]
        true_traj = true_traj[np.newaxis, ...]
    
    # 计算欧氏距离
    distances = np.linalg.norm(pred_traj - true_traj, axis=2)  # (N, T)
    ade = np.mean(distances)  # 对所有样本和时步平均
    return float(ade)


def compute_fde(pred_traj: np.ndarray, true_traj: np.ndarray) -> float:
    """
    计算最终位移误差 (Final Displacement Error, FDE)
    
    Args:
        pred_traj: 预测轨迹 (N, T, 3) 或 (T, 3)
        true_traj: 真实轨迹 (N, T, 3) 或 (T, 3)
    
    Returns:
        FDE 值
    """
    if pred_traj.ndim == 2:
        pred_traj = pred_traj[np.newaxis, ...]
        true_traj = true_traj[np.newaxis, ...]
    
    # 计算最后一步的距离
    last_pred = pred_traj[:, -1, :]  # (N, 3)
    last_true = true_traj[:, -1, :]  # (N, 3)
    distances = np.linalg.norm(last_pred - last_true, axis=1)  # (N,)
    fde = np.mean(distances)
    return float(fde)


def compute_rmse(pred_traj: np.ndarray, true_traj: np.ndarray) -> float:
    """
    计算均方根误差 (Root Mean Squared Error, RMSE)
    
    Args:
        pred_traj: 预测轨迹 (N, T, 3) 或 (T, 3)
        true_traj: 真实轨迹 (N, T, 3) 或 (T, 3)
    
    Returns:
        RMSE 值
    """
    mse = np.mean((pred_traj - true_traj) ** 2)
    rmse = np.sqrt(mse)
    return float(rmse)


def compute_mae(pred_traj: np.ndarray, true_traj: np.ndarray) -> float:
    """
    计算平均绝对误差 (Mean Absolute Error, MAE)
    
    Args:
        pred_traj: 预测轨迹 (N, T, 3) 或 (T, 3)
        true_traj: 真实轨迹 (N, T, 3) 或 (T, 3)
    
    Returns:
        MAE 值
    """
    mae = np.mean(np.abs(pred_traj - true_traj))
    return float(mae)


def compute_trajectory_metrics(pred_traj: np.ndarray, true_traj: np.ndarray) -> Dict[str, float]:
    """
    计算所有轨迹预测指标
    
    Args:
        pred_traj: 预测轨迹 (batch_size * seq_len, 3)
        true_traj: 真实轨迹 (batch_size * seq_len, 3)
    
    Returns:
        字典，包含所有指标
    """
    # 将平展的数组重新整形为 (batch_size, seq_len, 3)
    num_samples = len(pred_traj) // config.PRED_LEN
    pred_traj_reshaped = pred_traj[:num_samples * config.PRED_LEN].reshape(num_samples, config.PRED_LEN, -1)
    true_traj_reshaped = true_traj[:num_samples * config.PRED_LEN].reshape(num_samples, config.PRED_LEN, -1)
    
    metrics = {
        'rmse': compute_rmse(pred_traj_reshaped, true_traj_reshaped),
        'mae': compute_mae(pred_traj_reshaped, true_traj_reshaped),
        'ade': compute_ade(pred_traj_reshaped, true_traj_reshaped),
        'fde': compute_fde(pred_traj_reshaped, true_traj_reshaped),
    }
    return metrics


def compute_intent_metrics(pred_labels: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
    """
    计算所有意图识别指标
    
    Args:
        pred_labels: 预测标签 (N,)
        true_labels: 真实标签 (N,)
    
    Returns:
        字典，包含所有指标
    """
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
    
    metrics = {
        'accuracy': float(accuracy),
        'f1': float(f1)
    }
    
    return metrics


def compute_confusion_matrix(pred_labels: np.ndarray, true_labels: np.ndarray) -> np.ndarray:
    """
    计算混淆矩阵
    
    Args:
        pred_labels: 预测标签 (N,)
        true_labels: 真实标签 (N,)
    
    Returns:
        混淆矩阵
    """
    return confusion_matrix(true_labels, pred_labels)


@torch.no_grad()
def evaluate_model(model, dataloader, device: str = 'cuda') -> Dict:
    """
    在数据集上评估模型（支持混合样本）
    
    Args:
        model: 多任务学习模型
        dataloader: 数据加载器
        device: 计算设备
    
    Returns:
        包含所有评估指标的字典
    """
    model.eval()
    model.to(device)
    
    all_pred_trajs = []
    all_true_trajs = []
    all_pred_intents = []
    all_true_intents = []
    
    for batch in dataloader:
        # 提取batch数据
        traj_indices = batch['traj_indices']
        intent_indices = batch['intent_indices']
        intent = batch['intent'].to(device)
        
        # 准备模型输入
        obs_traj = batch['obs_traj'].to(device) if batch['obs_traj'] is not None else None
        pred_traj_true = batch['pred_traj'].to(device) if batch['pred_traj'] is not None else None
        full_traj = batch['full_traj'].to(device) if batch['full_traj'] is not None else None
        
        # Forward pass
        pred_traj_out, intent_logits = model(obs_traj=obs_traj, full_traj=full_traj)
        
        # 记录轨迹预测数据
        if pred_traj_out is not None and pred_traj_true is not None:
            all_pred_trajs.append(pred_traj_out.cpu().numpy())
            all_true_trajs.append(pred_traj_true.cpu().numpy())
        
        # 记录意图分类数据
        if intent_logits is not None and len(intent_indices) > 0:
            intent_preds = intent_logits.argmax(dim=1).cpu().numpy()
            intent_numpy = intent.cpu().numpy().flatten()  # 确保是1维数组
            intent_labels_selected = intent_numpy[intent_indices]
            all_pred_intents.append(intent_preds)
            all_true_intents.append(intent_labels_selected)
    
    # 汇总数据
    results = {}
    
    if all_pred_trajs:
        all_pred_trajs = np.vstack(all_pred_trajs)
        all_true_trajs = np.vstack(all_true_trajs)
        traj_metrics = compute_trajectory_metrics(all_pred_trajs, all_true_trajs)
        results.update(traj_metrics)
    else:
        results.update({'rmse': 0.0, 'mae': 0.0, 'ade': 0.0, 'fde': 0.0})
    
    if all_pred_intents:
        all_pred_intents = np.concatenate(all_pred_intents)
        all_true_intents = np.concatenate(all_true_intents)
        intent_metrics = compute_intent_metrics(all_pred_intents, all_true_intents)
        results.update(intent_metrics)
    else:
        results.update({'accuracy': 0.0, 'f1': 0.0})
    
    return results


def print_evaluation_results(results: Dict, dataset_name: str = "Dataset"):
    """
    打印评估结果
    
    Args:
        results: 评估结果字典
        dataset_name: 数据集名称
    """
    print(f"\n{'='*60}")
    print(f"{dataset_name} 评估结果")
    print(f"{'='*60}")
    
    print("\n轨迹预测指标:")
    print(f"  RMSE: {results.get('rmse', 'N/A'):.4f}")
    print(f"  MAE:  {results.get('mae', 'N/A'):.4f}")
    print(f"  ADE:  {results.get('ade', 'N/A'):.4f}")
    print(f"  FDE:  {results.get('fde', 'N/A'):.4f}")
    
    print("\n意图识别指标:")
    print(f"  准确率: {results.get('accuracy', 'N/A'):.4f}")
    print(f"  F1:    {results.get('f1', 'N/A'):.4f}")
    print(f"{'='*60}\n")


def get_per_class_metrics(pred_labels: np.ndarray, true_labels: np.ndarray) -> Dict:
    """
    计算每个类别的分类指标
    
    Args:
        pred_labels: 预测标签
        true_labels: 真实标签
    
    Returns:
        每个类别的精确率、召回率、F1
    """
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, labels=range(config.NUM_INTENTS),
        zero_division=0
    )
    
    results = {}
    for i, intent in enumerate(config.INTENT_CLASSES):
        results[intent] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
    
    return results
