"""
训练脚本：多任务学习框架的训练循环
"""
import os
import sys
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import config as config
from data_loader import get_data_manager
from model import create_model, create_loss_fn
from evaluation import evaluate_model, compute_trajectory_metrics, compute_intent_metrics


class Trainer:
    """多任务学习训练器"""
    
    def __init__(self, model, loss_fn, device, data_manager):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        self.data_manager = data_manager
        
        # 优化器
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
        
        # 学习率调度器
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
            'train_loss': [], 'train_traj_loss': [], 'train_intent_loss': [],
            'val_loss': [], 'val_traj_loss': [], 'val_intent_loss': [],
            'val_traj_rmse': [], 'val_traj_mae': [], 'val_traj_fde': [],
            'val_intent_acc': [], 'val_intent_f1': []
        }
        self.best_val_metric = float('inf') if 'rmse' in config.BEST_MODEL_METRIC else 0
        self.early_stop_counter = 0
    
    def train_epoch(self, train_loader):
        """训练一个epoch（支持混合样本）"""
        self.model.train()
        total_loss = 0.0
        total_traj_loss = 0.0
        total_intent_loss = 0.0
        num_batches = 0
        num_traj_samples = 0
        num_intent_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # 提取batch数据
            traj_indices = batch['traj_indices']
            intent_indices = batch['intent_indices']
            intent = batch['intent'].to(self.device)
            
            # 准备模型输入
            obs_traj = batch['obs_traj'].to(self.device) if batch['obs_traj'] is not None else None
            pred_traj_true = batch['pred_traj'].to(self.device) if batch['pred_traj'] is not None else None
            full_traj = batch['full_traj'].to(self.device) if batch['full_traj'] is not None else None
            
            # Forward pass
            pred_traj_out, intent_logits = self.model(obs_traj=obs_traj, full_traj=full_traj)
            
            # 计算损失
            loss, traj_loss, intent_loss = self.loss_fn(
                pred_traj=pred_traj_out,
                true_traj=pred_traj_true,
                intent_logits=intent_logits,
                intent_labels=intent,
                traj_indices=traj_indices,
                intent_indices=intent_indices
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if config.GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRAD_CLIP)
            
            self.optimizer.step()
            
            # 日志
            total_loss += loss.item()
            total_traj_loss += traj_loss.item()
            total_intent_loss += intent_loss.item()
            num_batches += 1
            num_traj_samples += len(traj_indices)
            num_intent_samples += len(intent_indices)
            
            if config.VERBOSE and (batch_idx + 1) % config.LOG_INTERVAL == 0:
                print(f"  [{batch_idx + 1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} | "
                      f"Traj: {traj_loss.item():.4f} ({len(traj_indices)} samples) | "
                      f"Intent: {intent_loss.item():.4f} ({len(intent_indices)} samples)")
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_traj_loss = total_traj_loss / max(num_batches, 1)
        avg_intent_loss = total_intent_loss / max(num_batches, 1)
        
        if config.VERBOSE:
            print(f"  Epoch统计: Traj样本={num_traj_samples}, Intent样本={num_intent_samples}")
        
        return avg_loss, avg_traj_loss, avg_intent_loss
    
    @torch.no_grad()
    def validate(self, val_loader):
        """验证（支持混合样本）"""
        self.model.eval()
        total_loss = 0.0
        total_traj_loss = 0.0
        total_intent_loss = 0.0
        
        all_pred_trajs = []
        all_true_trajs = []
        all_intent_preds = []
        all_intent_labels = []
        
        for batch in val_loader:
            # 提取batch数据
            traj_indices = batch['traj_indices']
            intent_indices = batch['intent_indices']
            intent = batch['intent'].to(self.device)
            
            # 准备模型输入
            obs_traj = batch['obs_traj'].to(self.device) if batch['obs_traj'] is not None else None
            pred_traj_true = batch['pred_traj'].to(self.device) if batch['pred_traj'] is not None else None
            full_traj = batch['full_traj'].to(self.device) if batch['full_traj'] is not None else None
            
            # Forward pass
            pred_traj_out, intent_logits = self.model(obs_traj=obs_traj, full_traj=full_traj)
            
            # 计算损失
            loss, traj_loss, intent_loss = self.loss_fn(
                pred_traj=pred_traj_out,
                true_traj=pred_traj_true,
                intent_logits=intent_logits,
                intent_labels=intent,
                traj_indices=traj_indices,
                intent_indices=intent_indices
            )
            
            # 记录数据
            total_loss += loss.item()
            total_traj_loss += traj_loss.item()
            total_intent_loss += intent_loss.item()
            
            # 收集轨迹预测结果
            if pred_traj_out is not None and pred_traj_true is not None:
                all_pred_trajs.append(pred_traj_out.cpu().numpy())
                all_true_trajs.append(pred_traj_true.cpu().numpy())
            
            # 收集意图分类结果
            if intent_logits is not None and len(intent_indices) > 0:
                intent_preds = intent_logits.argmax(dim=1).cpu().numpy()
                intent_numpy = intent.cpu().numpy().flatten()  # 确保是1维数组
                intent_labels_selected = intent_numpy[intent_indices]
                all_intent_preds.append(intent_preds)
                all_intent_labels.append(intent_labels_selected)
        
        # 汇总指标
        num_batches = len(val_loader)
        avg_loss = total_loss / max(num_batches, 1)
        avg_traj_loss = total_traj_loss / max(num_batches, 1)
        avg_intent_loss = total_intent_loss / max(num_batches, 1)
        
        # 轨迹预测指标
        traj_metrics = {}
        if all_pred_trajs:
            all_pred_trajs = np.vstack(all_pred_trajs)
            all_true_trajs = np.vstack(all_true_trajs)
            traj_metrics = compute_trajectory_metrics(all_pred_trajs, all_true_trajs)
        else:
            traj_metrics = {'rmse': 0.0, 'mae': 0.0, 'fde': 0.0}
        
        # 意图分类指标
        intent_metrics = {}
        if all_intent_preds:
            all_intent_preds = np.concatenate(all_intent_preds)
            all_intent_labels = np.concatenate(all_intent_labels)
            intent_metrics = compute_intent_metrics(all_intent_preds, all_intent_labels)
        else:
            intent_metrics = {'accuracy': 0.0, 'f1': 0.0}
        
        return {
            'loss': avg_loss,
            'traj_loss': avg_traj_loss,
            'intent_loss': avg_intent_loss,
            **traj_metrics,
            **intent_metrics
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
        latest_path = os.path.join(config.CHECKPOINT_DIR, 'latest.pt')
        torch.save(checkpoint, latest_path)
        
        # 保存最优模型
        if is_best:
            best_path = os.path.join(config.CHECKPOINT_DIR, 'best.pt')
            torch.save(checkpoint, best_path)
            print(f"  ✓ 保存最优模型 (epoch {epoch})")
    
    def train(self, train_loader, val_loader):
        """完整训练流程"""
        print("\n" + "="*80)
        print(f"开始多任务学习训练")
        print(f"设备: {self.device}")
        print(f"Epochs: {config.NUM_EPOCHS}, Batch Size: {config.BATCH_SIZE}")
        print(f"学习率: {config.LEARNING_RATE}, 优化器: {config.OPTIMIZER}")
        print(f"轨迹损失权重: {config.LOSS_WEIGHT_TRAJ}, 意图损失权重: {config.LOSS_WEIGHT_INTENT}")
        print("="*80 + "\n")
        
        for epoch in range(1, config.NUM_EPOCHS + 1):
            print(f"Epoch {epoch}/{config.NUM_EPOCHS}")
            
            # 训练
            train_loss, train_traj_loss, train_intent_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_traj_loss'].append(train_traj_loss)
            self.history['train_intent_loss'].append(train_intent_loss)
            
            # 验证
            if epoch % config.VALIDATION_INTERVAL == 0:
                val_metrics = self.validate(val_loader)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_traj_loss'].append(val_metrics['traj_loss'])
                self.history['val_intent_loss'].append(val_metrics['intent_loss'])
                self.history['val_traj_rmse'].append(val_metrics['rmse'])
                self.history['val_traj_mae'].append(val_metrics['mae'])
                self.history['val_traj_fde'].append(val_metrics['fde'])
                self.history['val_intent_acc'].append(val_metrics['accuracy'])
                self.history['val_intent_f1'].append(val_metrics['f1'])
                
                print(f"  Train Loss: {train_loss:.4f} "
                      f"(Traj: {train_traj_loss:.4f}, Intent: {train_intent_loss:.4f})")
                print(f"  Val Loss: {val_metrics['loss']:.4f} "
                      f"(Traj: {val_metrics['traj_loss']:.4f}, Intent: {val_metrics['intent_loss']:.4f})")
                print(f"  轨迹预测 - RMSE: {val_metrics['rmse']:.4f}, "
                      f"MAE: {val_metrics['mae']:.4f}, FDE: {val_metrics['fde']:.4f}")
                print(f"  意图识别 - 准确率: {val_metrics['accuracy']:.4f}, "
                      f"F1: {val_metrics['f1']:.4f}\n")
                
                # 检查最优模型
                if config.BEST_MODEL_METRIC == 'val_traj_rmse':
                    current_metric = val_metrics['rmse']
                    is_better = current_metric < self.best_val_metric
                elif config.BEST_MODEL_METRIC == 'val_intent_acc':
                    current_metric = val_metrics['accuracy']
                    is_better = current_metric > self.best_val_metric
                else:
                    current_metric = val_metrics['loss']
                    is_better = current_metric < self.best_val_metric
                
                if is_better:
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
                print(f"  Train Loss: {train_loss:.4f} "
                      f"(Traj: {train_traj_loss:.4f}, Intent: {train_intent_loss:.4f})\n")
            
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
    print("\n加载数据...")
    data_manager = get_data_manager()
    train_loader, val_loader, test_loader = data_manager.prepare_data()
    
    # 创建模型和损失函数
    print("创建模型...")
    model = create_model()
    loss_fn = create_loss_fn()
    
    # 创建训练器
    trainer = Trainer(model, loss_fn, device, data_manager)
    
    # 训练
    history = trainer.train(train_loader, val_loader)
    
    # 保存训练历史
    history_path = os.path.join(config.LOG_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ 训练历史已保存: {history_path}")
    
    # 在测试集上评估最优模型
    print("\n在测试集上评估最优模型...")
    best_model_path = os.path.join(config.CHECKPOINT_DIR, 'best.pt')
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = trainer.validate(test_loader)
    print("\n测试集结果:")
    print(f"  轨迹预测 - RMSE: {test_metrics['rmse']:.4f}, "
          f"MAE: {test_metrics['mae']:.4f}, FDE: {test_metrics['fde']:.4f}")
    print(f"  意图识别 - 准确率: {test_metrics['accuracy']:.4f}, "
          f"F1: {test_metrics['f1']:.4f}\n")
    
    test_results = {
        'epoch': checkpoint['epoch'],
        'test_metrics': test_metrics
    }
    results_path = os.path.join(config.LOG_DIR, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"✓ 测试结果已保存: {results_path}\n")


if __name__ == '__main__':
    main()
