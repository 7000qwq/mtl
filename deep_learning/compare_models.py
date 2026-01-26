"""
模型对比脚本：对比 MTL、Trajectory-Only Baseline、Intent-Only Baseline 三者效果
用于论文实验结果分析
"""
import torch
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import config
from data_loader import get_data_manager, TrajectoryDataset
from model import create_model
from model_traj_only import create_traj_model
from model_intent_only import create_intent_model
from evaluation import (compute_trajectory_metrics, compute_intent_metrics, 
                        compute_confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============== 路径配置 ==============
MTL_CHECKPOINT = os.path.join(config.CHECKPOINT_DIR, 'best.pt')
TRAJ_ONLY_CHECKPOINT = os.path.join(config.OUTPUT_DIR, 'traj_only', 'checkpoints', 'best.pt')
INTENT_ONLY_CHECKPOINT = os.path.join(config.OUTPUT_DIR, 'intent_only', 'checkpoints', 'best.pt')

MTL_HISTORY = os.path.join(config.LOG_DIR, 'training_history.json')
TRAJ_ONLY_HISTORY = os.path.join(config.OUTPUT_DIR, 'traj_only', 'logs', 'training_history.json')
INTENT_ONLY_HISTORY = os.path.join(config.OUTPUT_DIR, 'intent_only', 'logs', 'training_history.json')

COMPARISON_OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, 'comparison')
os.makedirs(COMPARISON_OUTPUT_DIR, exist_ok=True)


class ModelComparison:
    """三模型对比工具"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.data_manager = get_data_manager()
        
        # 模型字典
        self.models = {}
        self.results = {}
        
    def load_models(self):
        """加载三个模型"""
        print("\n" + "="*80)
        print("加载模型")
        print("="*80)
        
        # 1. MTL模型
        if os.path.exists(MTL_CHECKPOINT):
            mtl_model = create_model().to(self.device)
            checkpoint = torch.load(MTL_CHECKPOINT, map_location=self.device, weights_only=False)
            mtl_model.load_state_dict(checkpoint['model_state_dict'])
            mtl_model.eval()
            self.models['MTL'] = {
                'model': mtl_model,
                'epoch': checkpoint['epoch'],
                'type': 'multi-task'
            }
            print(f"✓ MTL模型已加载 (epoch {checkpoint['epoch']})")
        else:
            print(f"✗ MTL模型未找到: {MTL_CHECKPOINT}")
        
        # 2. Trajectory-Only Baseline
        if os.path.exists(TRAJ_ONLY_CHECKPOINT):
            traj_model = create_traj_model().to(self.device)
            checkpoint = torch.load(TRAJ_ONLY_CHECKPOINT, map_location=self.device, weights_only=False)
            traj_model.load_state_dict(checkpoint['model_state_dict'])
            traj_model.eval()
            self.models['Traj-Only'] = {
                'model': traj_model,
                'epoch': checkpoint['epoch'],
                'type': 'trajectory'
            }
            print(f"✓ Trajectory-Only Baseline已加载 (epoch {checkpoint['epoch']})")
        else:
            print(f"✗ Trajectory-Only模型未找到: {TRAJ_ONLY_CHECKPOINT}")
        
        # 3. Intent-Only Baseline
        if os.path.exists(INTENT_ONLY_CHECKPOINT):
            intent_model = create_intent_model().to(self.device)
            checkpoint = torch.load(INTENT_ONLY_CHECKPOINT, map_location=self.device, weights_only=False)
            intent_model.load_state_dict(checkpoint['model_state_dict'])
            intent_model.eval()
            self.models['Intent-Only'] = {
                'model': intent_model,
                'epoch': checkpoint['epoch'],
                'type': 'intent'
            }
            print(f"✓ Intent-Only Baseline已加载 (epoch {checkpoint['epoch']})")
        else:
            print(f"✗ Intent-Only模型未找到: {INTENT_ONLY_CHECKPOINT}")
        
        print(f"\n共加载 {len(self.models)} 个模型")
        
    def prepare_data(self):
        """准备评估数据"""
        print("\n" + "="*80)
        print("准备评估数据")
        print("="*80)
        
        self.data_manager.load_data()
        self.data_manager.construct_samples()
        self.data_manager.split_by_trajectory()
        self.data_manager.create_datasets()
        
        # 获取归一化参数
        self.norm_params = self.data_manager.train_dataset.norm_params
        
        # 筛选不同类型的测试样本
        self.test_traj_samples = [s for s in self.data_manager.test_samples 
                                   if s.get('sample_type') == 'traj_only']
        self.test_intent_samples = [s for s in self.data_manager.test_samples 
                                     if s.get('sample_type') == 'intent_only']
        
        print(f"  测试集轨迹预测样本: {len(self.test_traj_samples)}")
        print(f"  测试集意图识别样本: {len(self.test_intent_samples)}")
        
    @torch.no_grad()
    def evaluate_trajectory_prediction(self):
        """评估轨迹预测任务"""
        print("\n" + "="*80)
        print("评估轨迹预测任务")
        print("="*80)
        
        # 创建轨迹预测测试数据集
        traj_dataset = TrajectoryDataset(
            self.test_traj_samples,
            norm_params=self.norm_params,
            is_train=False
        )
        
        results = {}
        
        for model_name, model_info in self.models.items():
            model = model_info['model']
            model_type = model_info['type']
            
            # 跳过intent-only模型
            if model_type == 'intent':
                continue
            
            all_preds = []
            all_trues = []
            
            for sample in traj_dataset:
                if 'obs_traj' not in sample:
                    continue
                    
                obs_traj = sample['obs_traj'].unsqueeze(0).to(self.device)
                pred_traj_true = sample['pred_traj'].numpy()
                
                # 模型推理
                if model_type == 'multi-task':
                    pred_traj, _ = model(obs_traj=obs_traj, full_traj=None)
                else:  # trajectory
                    pred_traj = model(obs_traj)
                
                pred_traj = pred_traj[0].cpu().numpy()
                
                all_preds.append(pred_traj)
                all_trues.append(pred_traj_true)
            
            if all_preds:
                all_preds = np.stack(all_preds)
                all_trues = np.stack(all_trues)
                metrics = compute_trajectory_metrics(all_preds, all_trues)
                results[model_name] = metrics
                print(f"\n  {model_name}:")
                print(f"    RMSE: {metrics['rmse']:.4f}")
                print(f"    MAE:  {metrics['mae']:.4f}")
                print(f"    ADE:  {metrics['ade']:.4f}")
                print(f"    FDE:  {metrics['fde']:.4f}")
        
        self.results['trajectory'] = results
        return results
    
    @torch.no_grad()
    def evaluate_intent_classification(self):
        """评估意图识别任务"""
        print("\n" + "="*80)
        print("评估意图识别任务")
        print("="*80)
        
        # 创建意图识别测试数据集
        intent_dataset = TrajectoryDataset(
            self.test_intent_samples,
            norm_params=self.norm_params,
            is_train=False
        )
        
        results = {}
        confusion_matrices = {}
        
        for model_name, model_info in self.models.items():
            model = model_info['model']
            model_type = model_info['type']
            
            # 跳过trajectory-only模型
            if model_type == 'trajectory':
                continue
            
            all_preds = []
            all_trues = []
            
            for sample in intent_dataset:
                if 'full_traj' not in sample:
                    continue
                    
                full_traj = sample['full_traj'].unsqueeze(0).to(self.device)
                intent_true = sample['intent'].item()
                
                # 模型推理
                if model_type == 'multi-task':
                    _, intent_logits = model(obs_traj=None, full_traj=full_traj)
                else:  # intent
                    intent_logits = model(full_traj)
                
                intent_pred = intent_logits.argmax(dim=1).item()
                
                all_preds.append(intent_pred)
                all_trues.append(intent_true)
            
            if all_preds:
                all_preds = np.array(all_preds)
                all_trues = np.array(all_trues)
                metrics = compute_intent_metrics(all_preds, all_trues)
                cm = compute_confusion_matrix(all_preds, all_trues)
                
                results[model_name] = metrics
                confusion_matrices[model_name] = cm
                
                print(f"\n  {model_name}:")
                print(f"    Accuracy: {metrics['accuracy']:.4f}")
                print(f"    F1-score: {metrics['f1']:.4f}")
        
        self.results['intent'] = results
        self.results['confusion_matrices'] = confusion_matrices
        return results
    
    def print_comparison_table(self):
        """打印对比表格"""
        print("\n" + "="*80)
        print("模型对比汇总")
        print("="*80)
        
        # 轨迹预测对比表
        print("\n【轨迹预测任务】")
        traj_data = []
        traj_results = self.results.get('trajectory', {})
        for model_name in ['MTL', 'Traj-Only']:
            if model_name in traj_results:
                m = traj_results[model_name]
                traj_data.append([
                    model_name,
                    f"{m['rmse']:.4f}",
                    f"{m['mae']:.4f}",
                    f"{m['ade']:.4f}",
                    f"{m['fde']:.4f}"
                ])
        
        if traj_data:
            print(tabulate(traj_data, 
                          headers=['Model', 'RMSE ↓', 'MAE ↓', 'ADE ↓', 'FDE ↓'],
                          tablefmt='grid'))
        
        # 意图识别对比表
        print("\n【意图识别任务】")
        intent_data = []
        intent_results = self.results.get('intent', {})
        for model_name in ['MTL', 'Intent-Only']:
            if model_name in intent_results:
                m = intent_results[model_name]
                intent_data.append([
                    model_name,
                    f"{m['accuracy']:.4f}",
                    f"{m['f1']:.4f}"
                ])
        
        if intent_data:
            print(tabulate(intent_data,
                          headers=['Model', 'Accuracy ↑', 'F1-score ↑'],
                          tablefmt='grid'))
        
        # 计算MTL相对于baseline的改进
        print("\n【MTL相对于Baseline的提升】")
        improvements = []
        
        if 'MTL' in traj_results and 'Traj-Only' in traj_results:
            mtl = traj_results['MTL']
            baseline = traj_results['Traj-Only']
            for metric in ['rmse', 'mae', 'ade', 'fde']:
                improvement = (baseline[metric] - mtl[metric]) / baseline[metric] * 100
                improvements.append([f'Trajectory {metric.upper()}', 
                                    f"{improvement:+.2f}%",
                                    "↑ 更好" if improvement > 0 else "↓ 更差"])
        
        if 'MTL' in intent_results and 'Intent-Only' in intent_results:
            mtl = intent_results['MTL']
            baseline = intent_results['Intent-Only']
            for metric in ['accuracy', 'f1']:
                improvement = (mtl[metric] - baseline[metric]) / baseline[metric] * 100
                improvements.append([f'Intent {metric.capitalize()}',
                                    f"{improvement:+.2f}%",
                                    "↑ 更好" if improvement > 0 else "↓ 更差"])
        
        if improvements:
            print(tabulate(improvements,
                          headers=['Metric', 'Improvement', 'Direction'],
                          tablefmt='grid'))
    
    def plot_comparison_bar_chart(self, save: bool = True):
        """绘制对比柱状图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 轨迹预测对比
        ax1 = axes[0]
        traj_results = self.results.get('trajectory', {})
        if traj_results:
            metrics = ['RMSE', 'MAE', 'ADE', 'FDE']
            x = np.arange(len(metrics))
            width = 0.35
            
            models = list(traj_results.keys())
            colors = ['#2ecc71', '#3498db']  # MTL绿色，Baseline蓝色
            
            for i, model_name in enumerate(models):
                m = traj_results[model_name]
                values = [m['rmse'], m['mae'], m['ade'], m['fde']]
                bars = ax1.bar(x + i*width, values, width, label=model_name, color=colors[i])
                # 添加数值标签
                for bar, val in zip(bars, values):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=9)
            
            ax1.set_xlabel('Metrics')
            ax1.set_ylabel('Error Value')
            ax1.set_title('Trajectory Prediction Comparison')
            ax1.set_xticks(x + width/2)
            ax1.set_xticklabels(metrics)
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
        
        # 意图识别对比
        ax2 = axes[1]
        intent_results = self.results.get('intent', {})
        if intent_results:
            metrics = ['Accuracy', 'F1-score']
            x = np.arange(len(metrics))
            width = 0.35
            
            models = list(intent_results.keys())
            colors = ['#2ecc71', '#e74c3c']  # MTL绿色，Baseline红色
            
            for i, model_name in enumerate(models):
                m = intent_results[model_name]
                values = [m['accuracy'], m['f1']]
                bars = ax2.bar(x + i*width, values, width, label=model_name, color=colors[i])
                # 添加数值标签
                for bar, val in zip(bars, values):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=9)
            
            ax2.set_xlabel('Metrics')
            ax2.set_ylabel('Score')
            ax2.set_title('Intent Classification Comparison')
            ax2.set_xticks(x + width/2)
            ax2.set_xticklabels(metrics)
            ax2.set_ylim([0, 1.1])
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(COMPARISON_OUTPUT_DIR, 'comparison_bar_chart.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ 对比柱状图已保存: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrices(self, save: bool = True):
        """绘制混淆矩阵对比"""
        cms = self.results.get('confusion_matrices', {})
        if not cms:
            print("没有混淆矩阵数据")
            return
        
        n_models = len(cms)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for ax, (model_name, cm) in zip(axes, cms.items()):
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=config.INTENT_CLASSES,
                yticklabels=config.INTENT_CLASSES,
                ax=ax
            )
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'{model_name} Confusion Matrix')
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(COMPARISON_OUTPUT_DIR, 'confusion_matrices_comparison.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 混淆矩阵对比图已保存: {save_path}")
        
        plt.show()
    
    def plot_training_curves(self, save: bool = True):
        """绘制训练曲线对比"""
        histories = {}
        
        # 加载训练历史
        if os.path.exists(MTL_HISTORY):
            with open(MTL_HISTORY, 'r') as f:
                histories['MTL'] = json.load(f)
        
        if os.path.exists(TRAJ_ONLY_HISTORY):
            with open(TRAJ_ONLY_HISTORY, 'r') as f:
                histories['Traj-Only'] = json.load(f)
        
        if os.path.exists(INTENT_ONLY_HISTORY):
            with open(INTENT_ONLY_HISTORY, 'r') as f:
                histories['Intent-Only'] = json.load(f)
        
        if not histories:
            print("没有训练历史数据")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 训练损失对比
        ax = axes[0, 0]
        colors = {'MTL': '#2ecc71', 'Traj-Only': '#3498db', 'Intent-Only': '#e74c3c'}
        for name, hist in histories.items():
            if 'train_loss' in hist:
                ax.plot(hist['train_loss'], label=f'{name}', color=colors.get(name, 'gray'))
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Comparison')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 验证损失对比
        ax = axes[0, 1]
        for name, hist in histories.items():
            if 'val_loss' in hist:
                ax.plot(hist['val_loss'], label=f'{name}', color=colors.get(name, 'gray'))
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Validation Loss Comparison')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 轨迹预测RMSE对比
        ax = axes[1, 0]
        if 'MTL' in histories and 'val_traj_rmse' in histories['MTL']:
            ax.plot(histories['MTL']['val_traj_rmse'], label='MTL', color='#2ecc71')
        if 'Traj-Only' in histories and 'val_rmse' in histories['Traj-Only']:
            ax.plot(histories['Traj-Only']['val_rmse'], label='Traj-Only', color='#3498db')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('Trajectory RMSE Comparison')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 意图识别准确率对比
        ax = axes[1, 1]
        if 'MTL' in histories and 'val_intent_acc' in histories['MTL']:
            ax.plot(histories['MTL']['val_intent_acc'], label='MTL', color='#2ecc71')
        if 'Intent-Only' in histories and 'val_accuracy' in histories['Intent-Only']:
            ax.plot(histories['Intent-Only']['val_accuracy'], label='Intent-Only', color='#e74c3c')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Intent Accuracy Comparison')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(COMPARISON_OUTPUT_DIR, 'training_curves_comparison.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 训练曲线对比图已保存: {save_path}")
        
        plt.show()
    
    def export_results(self):
        """导出结果为JSON"""
        export_data = {
            'trajectory_prediction': self.results.get('trajectory', {}),
            'intent_classification': self.results.get('intent', {}),
            'model_info': {
                name: {'epoch': info['epoch'], 'type': info['type']} 
                for name, info in self.models.items()
            },
            'config': {
                'obs_len': config.OBS_LEN,
                'pred_len': config.PRED_LEN,
                'full_traj_len': config.FULL_TRAJ_LEN,
                'encoder_type': config.ENCODER_TYPE,
                'encoder_hidden_dim': config.ENCODER_HIDDEN_DIM,
                'num_epochs': config.NUM_EPOCHS,
                'batch_size': config.BATCH_SIZE,
                'learning_rate': config.LEARNING_RATE
            }
        }
        
        # 转换numpy数组为list
        for task in ['trajectory_prediction', 'intent_classification']:
            if task in export_data:
                for model_name, metrics in export_data[task].items():
                    for key, value in metrics.items():
                        if isinstance(value, np.ndarray):
                            export_data[task][model_name][key] = value.tolist()
                        elif isinstance(value, (np.float32, np.float64)):
                            export_data[task][model_name][key] = float(value)
        
        save_path = os.path.join(COMPARISON_OUTPUT_DIR, 'comparison_results.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"✓ 对比结果已导出: {save_path}")
        
        return export_data
    
    def generate_latex_table(self):
        """生成LaTeX表格（用于论文）"""
        print("\n" + "="*80)
        print("LaTeX表格（可直接复制到论文）")
        print("="*80)
        
        # 轨迹预测表格
        print("\n% Trajectory Prediction Results")
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{Trajectory Prediction Performance Comparison}")
        print("\\begin{tabular}{lcccc}")
        print("\\toprule")
        print("Model & RMSE $\\downarrow$ & MAE $\\downarrow$ & ADE $\\downarrow$ & FDE $\\downarrow$ \\\\")
        print("\\midrule")
        
        traj_results = self.results.get('trajectory', {})
        for model_name in ['MTL', 'Traj-Only']:
            if model_name in traj_results:
                m = traj_results[model_name]
                display_name = 'MTL (Ours)' if model_name == 'MTL' else 'Trajectory-Only'
                print(f"{display_name} & {m['rmse']:.4f} & {m['mae']:.4f} & {m['ade']:.4f} & {m['fde']:.4f} \\\\")
        
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\label{tab:trajectory}")
        print("\\end{table}")
        
        # 意图识别表格
        print("\n% Intent Classification Results")
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{Intent Classification Performance Comparison}")
        print("\\begin{tabular}{lcc}")
        print("\\toprule")
        print("Model & Accuracy $\\uparrow$ & F1-score $\\uparrow$ \\\\")
        print("\\midrule")
        
        intent_results = self.results.get('intent', {})
        for model_name in ['MTL', 'Intent-Only']:
            if model_name in intent_results:
                m = intent_results[model_name]
                display_name = 'MTL (Ours)' if model_name == 'MTL' else 'Intent-Only'
                print(f"{display_name} & {m['accuracy']:.4f} & {m['f1']:.4f} \\\\")
        
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\label{tab:intent}")
        print("\\end{table}")


def main():
    """主函数"""
    print("\n" + "="*80)
    print("模型对比分析工具")
    print("MTL vs Trajectory-Only Baseline vs Intent-Only Baseline")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 创建对比工具
    comparison = ModelComparison(device=device)
    
    # 加载模型
    comparison.load_models()
    
    if len(comparison.models) == 0:
        print("\n✗ 没有找到任何已训练的模型，请先运行训练脚本")
        return
    
    # 准备数据
    comparison.prepare_data()
    
    # 评估
    comparison.evaluate_trajectory_prediction()
    comparison.evaluate_intent_classification()
    
    # 打印对比表格
    comparison.print_comparison_table()
    
    # 生成LaTeX表格
    comparison.generate_latex_table()
    
    # 绘制对比图表
    print("\n生成对比图表...")
    comparison.plot_comparison_bar_chart(save=True)
    comparison.plot_confusion_matrices(save=True)
    comparison.plot_training_curves(save=True)
    
    # 导出结果
    comparison.export_results()
    
    print("\n" + "="*80)
    print("✓ 对比分析完成")
    print(f"  结果保存目录: {COMPARISON_OUTPUT_DIR}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
