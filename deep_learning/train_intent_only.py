"""
训练脚本：单任务意图识别 (Intent Classification Only Baseline)
用于论文对比实验

数据使用：仅使用 data_loader.py 中的 intent_only 样本（每条轨迹一个）
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
from model_intent_only import create_intent_model, create_intent_loss_fn
from evaluation import compute_intent_metrics, compute_confusion_matrix


# 单任务意图识别的输出目录
INTENT_OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, 'intent_only')
INTENT_CHECKPOINT_DIR = os.path.join(INTENT_OUTPUT_DIR, 'checkpoints')
INTENT_LOG_DIR = os.path.join(INTENT_OUTPUT_DIR, 'logs')

os.makedirs(INTENT_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(INTENT_LOG_DIR, exist_ok=True)


def intent_only_collate_fn(batch):
    """
    专用于意图识别任务的collate函数
    仅处理 intent_only 样本
    """
    full_trajs = []
    intents = []
    intent_labels = []
    
    for item in batch:
        if 'full_traj' in item:
            full_trajs.append(item['full_traj'])
            intents.append(item['intent'])
            intent_labels.append(item['intent_label'])
    
    result = {
        'full_traj': torch.stack(full_trajs, dim=0) if full_trajs else None,
        'intent': torch.cat(intents, dim=0).squeeze(-1) if intents else None,
        'intent_label': intent_labels
    }
    
    return result


class IntentOnlyDataManager:
    """
    单任务意图识别的数据管理器
    复用MTL框架的数据加载和划分逻辑，仅筛选 intent_only 样本
    """
    
    def __init__(self):
        # 使用MTL框架的数据管理器
        self.mtl_data_manager = get_data_manager()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def prepare_data(self):
        """准备数据：复用MTL数据划分，仅筛选intent_only样本"""
        # 加载并划分数据（使用MTL框架的逻辑）
        self.mtl_data_manager.load_data()
        self.mtl_data_manager.construct_samples()
        self.mtl_data_manager.split_by_trajectory()
        
        # 筛选intent_only样本
        train_intent_samples = [s for s in self.mtl_data_manager.train_samples 
                                if s.get('sample_type') == 'intent_only']
        val_intent_samples = [s for s in self.mtl_data_manager.val_samples 
                              if s.get('sample_type') == 'intent_only']
        test_intent_samples = [s for s in self.mtl_data_manager.test_samples 
                               if s.get('sample_type') == 'intent_only']
        
        print(f"\n[IntentOnly] 筛选后的样本数量:")
        print(f"  训练集: {len(train_intent_samples)} 个 intent_only 样本")
        print(f"  验证集: {len(val_intent_samples)} 个 intent_only 样本")
        print(f"  测试集: {len(test_intent_samples)} 个 intent_only 样本")
        
        # 统计各类别数量
        self._print_class_distribution("训练集", train_intent_samples)
        self._print_class_distribution("验证集", val_intent_samples)
        self._print_class_distribution("测试集", test_intent_samples)
        
        # 创建数据集（使用训练集计算归一化参数）
        self.train_dataset = TrajectoryDataset(
            train_intent_samples,
            norm_params=None,  # 从训练集计算
            is_train=True
        )
        
        norm_params = self.train_dataset.norm_params
        print(f"\n  归一化参数 (来自训练集):")
        print(f"    Mean: {norm_params['mean']}")
        print(f"    Std: {norm_params['std']}\n")
        
        self.val_dataset = TrajectoryDataset(
            val_intent_samples,
            norm_params=norm_params,
            is_train=False
        )
        self.test_dataset = TrajectoryDataset(
            test_intent_samples,
            norm_params=norm_params,
            is_train=False
        )
        
        return self.get_dataloaders()
    
    def _print_class_distribution(self, split_name, samples):
        """打印类别分布"""
        class_counts = {}
        for s in samples:
            label = s['intent_label']
            class_counts[label] = class_counts.get(label, 0) + 1
        print(f"    {split_name}类别分布: {class_counts}")
    
    def get_dataloaders(self):
        """获取数据加载器"""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            collate_fn=intent_only_collate_fn,
            drop_last=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            collate_fn=intent_only_collate_fn
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            collate_fn=intent_only_collate_fn
        )
        
        return train_loader, val_loader, test_loader


class IntentOnlyTrainer:
    """单任务意图识别训练器"""
    
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
            'val_accuracy': [], 'val_f1': []
        }
        self.best_val_metric = 0.0  # 使用准确率作为指标，越大越好
        self.early_stop_counter = 0
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            full_traj = batch['full_traj']
            intent = batch['intent']
            
            if full_traj is None or intent is None:
                continue
            
            full_traj = full_traj.to(self.device)
            intent = intent.to(self.device)
            
            # Forward pass
            intent_logits = self.model(full_traj)
            
            # 计算损失
            loss = self.loss_fn(intent_logits, intent)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if config.GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRAD_CLIP)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 计算训练准确率
            preds = intent_logits.argmax(dim=1)
            correct += (preds == intent).sum().item()
            total += intent.size(0)
            
            if config.VERBOSE and (batch_idx + 1) % config.LOG_INTERVAL == 0:
                batch_acc = (preds == intent).float().mean().item()
                print(f"  [{batch_idx + 1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}, Acc: {batch_acc:.4f}")
        
        avg_loss = total_loss / max(num_batches, 1)
        train_acc = correct / max(total, 1)
        
        return avg_loss, train_acc
    
    @torch.no_grad()
    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        total_loss = 0.0
        
        all_preds = []
        all_labels = []
        
        for batch in val_loader:
            full_traj = batch['full_traj']
            intent = batch['intent']
            
            if full_traj is None or intent is None:
                continue
            
            full_traj = full_traj.to(self.device)
            intent = intent.to(self.device)
            
            # Forward pass
            intent_logits = self.model(full_traj)
            
            # 计算损失
            loss = self.loss_fn(intent_logits, intent)
            total_loss += loss.item()
            
            # 收集预测结果
            preds = intent_logits.argmax(dim=1).cpu().numpy()
            labels = intent.cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels)
        
        # 汇总指标
        num_batches = len(val_loader)
        avg_loss = total_loss / max(num_batches, 1)
        
        # 意图分类指标
        if all_preds:
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            intent_metrics = compute_intent_metrics(all_preds, all_labels)
        else:
            intent_metrics = {'accuracy': 0.0, 'f1': 0.0}
        
        return {
            'loss': avg_loss,
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
        latest_path = os.path.join(INTENT_CHECKPOINT_DIR, 'latest.pt')
        torch.save(checkpoint, latest_path)
        
        # 保存最优模型
        if is_best:
            best_path = os.path.join(INTENT_CHECKPOINT_DIR, 'best.pt')
            torch.save(checkpoint, best_path)
            print(f"  ✓ 保存最优模型 (epoch {epoch})")
    
    def train(self, train_loader, val_loader):
        """完整训练流程"""
        print("\n" + "="*80)
        print(f"开始单任务意图识别训练 (Intent Classification Only Baseline)")
        print(f"设备: {self.device}")
        print(f"Epochs: {config.NUM_EPOCHS}, Batch Size: {config.BATCH_SIZE}")
        print(f"学习率: {config.LEARNING_RATE}, 优化器: {config.OPTIMIZER}")
        print(f"类别数: {config.NUM_INTENTS}, 类别: {config.INTENT_CLASSES}")
        print("="*80 + "\n")
        
        for epoch in range(1, config.NUM_EPOCHS + 1):
            print(f"Epoch {epoch}/{config.NUM_EPOCHS}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # 验证
            if epoch % config.VALIDATION_INTERVAL == 0:
                val_metrics = self.validate(val_loader)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_accuracy'].append(val_metrics['accuracy'])
                self.history['val_f1'].append(val_metrics['f1'])
                
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"  Val Loss: {val_metrics['loss']:.4f}")
                print(f"  Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}\n")
                
                # 检查最优模型（使用准确率作为指标）
                current_metric = val_metrics['accuracy']
                if current_metric > self.best_val_metric:
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
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}\n")
            
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
    print("单任务意图识别 Baseline (Intent Classification Only)")
    print("="*80)
    print("\n加载数据...")
    data_manager = IntentOnlyDataManager()
    train_loader, val_loader, test_loader = data_manager.prepare_data()
    
    # 创建模型和损失函数
    print("创建模型...")
    model = create_intent_model()
    loss_fn = create_intent_loss_fn()
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型参数: {total_params:,} (可训练: {trainable_params:,})\n")
    
    # 创建训练器
    trainer = IntentOnlyTrainer(model, loss_fn, device, data_manager)
    
    # 训练
    history = trainer.train(train_loader, val_loader)
    
    # 保存训练历史
    history_path = os.path.join(INTENT_LOG_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ 训练历史已保存: {history_path}")
    
    # 在测试集上评估最优模型
    print("\n在测试集上评估最优模型...")
    best_model_path = os.path.join(INTENT_CHECKPOINT_DIR, 'best.pt')
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = trainer.validate(test_loader)
    print("\n测试集结果:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1-score: {test_metrics['f1']:.4f}")
    
    # 计算并保存混淆矩阵
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            full_traj = batch['full_traj']
            intent = batch['intent']
            if full_traj is None:
                continue
            full_traj = full_traj.to(device)
            intent_logits = model(full_traj)
            preds = intent_logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(intent.numpy())
    
    if all_preds:
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        conf_matrix = compute_confusion_matrix(all_preds, all_labels)
        print(f"\n混淆矩阵:")
        print(f"  类别顺序: {config.INTENT_CLASSES}")
        print(conf_matrix)
    
    test_results = {
        'epoch': checkpoint['epoch'],
        'test_metrics': test_metrics,
        'confusion_matrix': conf_matrix.tolist() if len(all_preds) > 0 else None,
        'class_names': config.INTENT_CLASSES
    }
    results_path = os.path.join(INTENT_LOG_DIR, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"\n✓ 测试结果已保存: {results_path}\n")


if __name__ == '__main__':
    main()
