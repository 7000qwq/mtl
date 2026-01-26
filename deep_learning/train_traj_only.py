"""
训练脚本：单任务轨迹预测 (Trajectory Prediction Only Baseline)
用于论文对比实验

数据使用：仅使用 data_loader.py 中的 traj_only 样本
训练参数：与MTL框架保持一致
"""
import os
import sys
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import config
from data_loader import get_data_manager, TrajectoryDataset
from model_traj_only import create_traj_model, create_traj_loss_fn
from evaluation import compute_trajectory_metrics


# 单任务轨迹预测的输出目录
TRAJ_OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, 'traj_only')
TRAJ_CHECKPOINT_DIR = os.path.join(TRAJ_OUTPUT_DIR, 'checkpoints')
TRAJ_LOG_DIR = os.path.join(TRAJ_OUTPUT_DIR, 'logs')

os.makedirs(TRAJ_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TRAJ_LOG_DIR, exist_ok=True)


def traj_only_collate_fn(batch):
    """
    专用于轨迹预测任务的collate函数
    仅处理 traj_only 样本
    """
    obs_trajs = []
    pred_trajs = []
    intent_labels = []
    
    for item in batch:
        if 'obs_traj' in item and 'pred_traj' in item:
            obs_trajs.append(item['obs_traj'])
            pred_trajs.append(item['pred_traj'])
            intent_labels.append(item['intent_label'])
    
    result = {
        'obs_traj': torch.stack(obs_trajs, dim=0) if obs_trajs else None,
        'pred_traj': torch.stack(pred_trajs, dim=0) if pred_trajs else None,
        'intent_label': intent_labels
    }
    
    return result


class TrajOnlyDataManager:
    """
    单任务轨迹预测的数据管理器
    复用MTL框架的数据加载和划分逻辑，仅筛选 traj_only 样本
    """
    
    def __init__(self):
        # 使用MTL框架的数据管理器
        self.mtl_data_manager = get_data_manager()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def prepare_data(self):
        """准备数据：复用MTL数据划分，仅筛选traj_only样本"""
        # 加载并划分数据（使用MTL框架的逻辑）
        self.mtl_data_manager.load_data()
        self.mtl_data_manager.construct_samples()
        self.mtl_data_manager.split_by_trajectory()
        
        # 筛选traj_only样本
        train_traj_samples = [s for s in self.mtl_data_manager.train_samples 
                              if s.get('sample_type') == 'traj_only']
        val_traj_samples = [s for s in self.mtl_data_manager.val_samples 
                            if s.get('sample_type') == 'traj_only']
        test_traj_samples = [s for s in self.mtl_data_manager.test_samples 
                             if s.get('sample_type') == 'traj_only']
        
        print(f"\n[TrajOnly] 筛选后的样本数量:")
        print(f"  训练集: {len(train_traj_samples)} 个 traj_only 样本")
        print(f"  验证集: {len(val_traj_samples)} 个 traj_only 样本")
        print(f"  测试集: {len(test_traj_samples)} 个 traj_only 样本\n")
        
        # 创建数据集（使用训练集计算归一化参数）
        self.train_dataset = TrajectoryDataset(
            train_traj_samples,
            norm_params=None,  # 从训练集计算
            is_train=True
        )
        
        norm_params = self.train_dataset.norm_params
        print(f"  归一化参数 (来自训练集):")
        print(f"    Mean: {norm_params['mean']}")
        print(f"    Std: {norm_params['std']}\n")
        
        self.val_dataset = TrajectoryDataset(
            val_traj_samples,
            norm_params=norm_params,
            is_train=False
        )
        self.test_dataset = TrajectoryDataset(
            test_traj_samples,
            norm_params=norm_params,
            is_train=False
        )
        
        return self.get_dataloaders()
    
    def get_dataloaders(self):
        """获取数据加载器"""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            collate_fn=traj_only_collate_fn,
            drop_last=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            collate_fn=traj_only_collate_fn
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            collate_fn=traj_only_collate_fn
        )
        
        return train_loader, val_loader, test_loader


class TrajOnlyTrainer:
    """单任务轨迹预测训练器"""
    
    def __init__(self, model, loss_fn, device, data_manager):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        self.data_manager = data_manager
        
        # 优化器（与MTL保持一致）
        if config.OPTIMIZER.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY
            )
        elif config.OPTIMIZER.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config.LEARNING_RATE,
                momentum=config.MOMENTUM,
                weight_decay=config.WEIGHT_DECAY
            )
        else:
            raise ValueError(f"Unsupported optimizer: {config.OPTIMIZER}")
        
        # 学习率调度器（与MTL保持一致）
        if config.SCHEDULER.lower() == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.NUM_EPOCHS
            )
        elif config.SCHEDULER.lower() == 'step':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=config.NUM_EPOCHS // 3,
                gamma=0.1
            )
        else:
            self.scheduler = None
        
        # 日志
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_rmse': [], 'val_mae': [], 'val_ade': [], 'val_fde': []
        }
        self.best_val_metric = float('inf')
        self.early_stop_counter = 0
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            obs_traj = batch['obs_traj']
            pred_traj_true = batch['pred_traj']
            
            if obs_traj is None or pred_traj_true is None:
                continue
            
            obs_traj = obs_traj.to(self.device)
            pred_traj_true = pred_traj_true.to(self.device)
            
            # Forward pass
            pred_traj_out = self.model(obs_traj)
            
            # 计算损失
            loss = self.loss_fn(pred_traj_out, pred_traj_true)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if config.GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRAD_CLIP)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if config.VERBOSE and (batch_idx + 1) % config.LOG_INTERVAL == 0:
                print(f"  [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    @torch.no_grad()
    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        total_loss = 0.0
        
        all_pred_trajs = []
        all_true_trajs = []
        
        for batch in val_loader:
            obs_traj = batch['obs_traj']
            pred_traj_true = batch['pred_traj']
            
            if obs_traj is None or pred_traj_true is None:
                continue
            
            obs_traj = obs_traj.to(self.device)
            pred_traj_true = pred_traj_true.to(self.device)
            
            # Forward pass
            pred_traj_out = self.model(obs_traj)
            
            # 计算损失
            loss = self.loss_fn(pred_traj_out, pred_traj_true)
            total_loss += loss.item()
            
            # 收集预测结果
            all_pred_trajs.append(pred_traj_out.cpu().numpy())
            all_true_trajs.append(pred_traj_true.cpu().numpy())
        
        # 汇总指标
        num_batches = len(val_loader)
        avg_loss = total_loss / max(num_batches, 1)
        
        # 轨迹预测指标
        if all_pred_trajs:
            all_pred_trajs = np.vstack(all_pred_trajs)
            all_true_trajs = np.vstack(all_true_trajs)
            traj_metrics = compute_trajectory_metrics(all_pred_trajs, all_true_trajs)
        else:
            traj_metrics = {'rmse': 0.0, 'mae': 0.0, 'ade': 0.0, 'fde': 0.0}
        
        return {
            'loss': avg_loss,
            **traj_metrics
        }
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        
        # 保存最新检查点
        latest_path = os.path.join(TRAJ_CHECKPOINT_DIR, 'latest.pt')
        torch.save(checkpoint, latest_path)
        
        # 保存最优模型
        if is_best:
            best_path = os.path.join(TRAJ_CHECKPOINT_DIR, 'best.pt')
            torch.save(checkpoint, best_path)
            print(f"  ✓ 保存最优模型 (epoch {epoch})")
    
    def train(self, train_loader, val_loader):
        """完整训练流程"""
        print("\n" + "="*80)
        print(f"开始单任务轨迹预测训练 (Trajectory Prediction Only Baseline)")
        print(f"设备: {self.device}")
        print(f"Epochs: {config.NUM_EPOCHS}, Batch Size: {config.BATCH_SIZE}")
        print(f"学习率: {config.LEARNING_RATE}, 优化器: {config.OPTIMIZER}")
        print(f"损失函数: {config.TRAJ_LOSS_TYPE.upper()}")
        print("="*80 + "\n")
        
        for epoch in range(1, config.NUM_EPOCHS + 1):
            print(f"Epoch {epoch}/{config.NUM_EPOCHS}")
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # 验证
            if epoch % config.VALIDATION_INTERVAL == 0:
                val_metrics = self.validate(val_loader)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_rmse'].append(val_metrics['rmse'])
                self.history['val_mae'].append(val_metrics['mae'])
                self.history['val_ade'].append(val_metrics['ade'])
                self.history['val_fde'].append(val_metrics['fde'])
                
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_metrics['loss']:.4f}")
                print(f"  RMSE: {val_metrics['rmse']:.4f}, MAE: {val_metrics['mae']:.4f}")
                print(f"  ADE: {val_metrics['ade']:.4f}, FDE: {val_metrics['fde']:.4f}\n")
                
                # 检查最优模型（使用RMSE作为指标）
                current_metric = val_metrics['rmse']
                if current_metric < self.best_val_metric:
                    self.best_val_metric = current_metric
                    self.early_stop_counter = 0
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    self.early_stop_counter += 1
                
                # 早停
                if self.early_stop_counter >= config.EARLY_STOPPING_PATIENCE:
                    print(f"\n⚠ 早停：验证指标未改善 {config.EARLY_STOPPING_PATIENCE} 个epoch")
                    break
            else:
                print(f"  Train Loss: {train_loss:.4f}\n")
            
            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 保存检查点
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        print("\n✓ 训练完成\n")
        return self.history


def main():
    # 设置随机种子
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)
    
    # 设备
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    
    # 数据加载
    print("\n" + "="*80)
    print("单任务轨迹预测 Baseline (Trajectory Prediction Only)")
    print("="*80)
    print("\n加载数据...")
    data_manager = TrajOnlyDataManager()
    train_loader, val_loader, test_loader = data_manager.prepare_data()
    
    # 创建模型和损失函数
    print("创建模型...")
    model = create_traj_model()
    loss_fn = create_traj_loss_fn()
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型参数: {total_params:,} (可训练: {trainable_params:,})\n")
    
    # 创建训练器
    trainer = TrajOnlyTrainer(model, loss_fn, device, data_manager)
    
    # 训练
    history = trainer.train(train_loader, val_loader)
    
    # 保存训练历史
    history_path = os.path.join(TRAJ_LOG_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ 训练历史已保存: {history_path}")
    
    # 在测试集上评估最优模型
    print("\n在测试集上评估最优模型...")
    best_model_path = os.path.join(TRAJ_CHECKPOINT_DIR, 'best.pt')
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = trainer.validate(test_loader)
    print("\n测试集结果:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    print(f"  ADE: {test_metrics['ade']:.4f}")
    print(f"  FDE: {test_metrics['fde']:.4f}\n")
    
    test_results = {
        'epoch': checkpoint['epoch'],
        'test_metrics': test_metrics
    }
    results_path = os.path.join(TRAJ_LOG_DIR, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"✓ 测试结果已保存: {results_path}\n")


if __name__ == '__main__':
    main()
