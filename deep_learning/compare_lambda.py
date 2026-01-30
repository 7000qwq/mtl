"""
Lambda对比脚本：对比不同 lambda (意图损失权重) 下的 MTL 模型效果
用于论文消融实验分析

数据来源：lamda=0, lamda=0.1, lamda=0.2, ... 等文件夹中的 comparison_results.json
"""
import torch
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import config
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============== 路径配置 ==============
# Lambda 值列表（按顺序）
LAMBDA_VALUES = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# 各 lambda 文件夹的路径模板
BASE_DIR = Path(__file__).parent
LAMBDA_DIRS = {
    0: BASE_DIR / 'lamda=0',
    0.1: BASE_DIR / 'lamda=0.1',
    0.2: BASE_DIR / 'lamda=0.2',
    0.3: BASE_DIR / 'lamda=0.3',
    0.4: BASE_DIR / 'lamda=0.4',
    0.6: BASE_DIR / 'lamda=0.6',
    0.5: BASE_DIR / 'lamda=0.5',
    0.7: BASE_DIR / 'lamda=0.7',
    0.8: BASE_DIR / 'lamda=0.8',
    0.9: BASE_DIR / 'lamda=0.9',
    1.0: BASE_DIR / 'lamda=1'
}

# 输出目录
LAMBDA_COMPARISON_OUTPUT_DIR = BASE_DIR / 'lambda_comparison'
LAMBDA_COMPARISON_OUTPUT_DIR.mkdir(exist_ok=True)


class LambdaComparison:
    """Lambda消融实验对比工具"""
    
    def __init__(self):
        self.results = {}  # {lambda_value: comparison_results}
        self.baseline_traj = None  # Trajectory-Only baseline (不受lambda影响)
        self.baseline_intent = None  # Intent-Only baseline (不受lambda影响)
        
    def load_results(self):
        """加载所有lambda实验的结果"""
        print("\n" + "="*80)
        print("加载各 Lambda 实验结果")
        print("="*80)
        
        for lam in LAMBDA_VALUES:
            result_path = LAMBDA_DIRS.get(lam)
            if result_path is None:
                print(f"  ✗ Lambda={lam}: 路径未配置")
                continue
                
            json_path = result_path / 'mtl_output' / 'comparison' / 'comparison_results.json'
            
            if not json_path.exists():
                print(f"  ✗ Lambda={lam}: 结果文件不存在 ({json_path})")
                continue
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.results[lam] = data
                print(f"  ✓ Lambda={lam}: 已加载")
                
                # 提取baseline（所有lambda下baseline应该相同，取第一个即可）
                if self.baseline_traj is None and 'trajectory_prediction' in data:
                    if 'Traj-Only' in data['trajectory_prediction']:
                        self.baseline_traj = data['trajectory_prediction']['Traj-Only']
                
                if self.baseline_intent is None and 'intent_classification' in data:
                    if 'Intent-Only' in data['intent_classification']:
                        self.baseline_intent = data['intent_classification']['Intent-Only']
                        
            except Exception as e:
                print(f"  ✗ Lambda={lam}: 加载失败 ({e})")
        
        print(f"\n共加载 {len(self.results)} 个 lambda 实验结果")
        if self.baseline_traj:
            print(f"  Trajectory Baseline: RMSE={self.baseline_traj['rmse']:.4f}")
        if self.baseline_intent:
            print(f"  Intent Baseline: Accuracy={self.baseline_intent['accuracy']:.4f}")
    
    def get_mtl_metrics(self, task: str, metric: str) -> Tuple[List[float], List[float]]:
        """
        获取各lambda下MTL模型的指标值
        
        Args:
            task: 'trajectory_prediction' 或 'intent_classification'
            metric: 指标名称 (如 'rmse', 'accuracy')
        
        Returns:
            lambdas: lambda值列表
            values: 对应的指标值列表
        """
        lambdas = []
        values = []
        
        for lam in sorted(self.results.keys()):
            data = self.results[lam]
            if task in data and 'MTL' in data[task] and metric in data[task]['MTL']:
                lambdas.append(lam)
                values.append(data[task]['MTL'][metric])
        
        return lambdas, values
    
    def print_comparison_table(self):
        """打印对比表格"""
        print("\n" + "="*80)
        print("Lambda 消融实验结果汇总")
        print("="*80)
        
        # 轨迹预测对比表
        print("\n【轨迹预测任务】")
        traj_headers = ['Lambda', 'RMSE ↓', 'MAE ↓', 'ADE ↓', 'FDE ↓']
        traj_data = []
        
        # 添加Baseline行
        if self.baseline_traj:
            b = self.baseline_traj
            traj_data.append([
                'Baseline (Traj-Only)',
                f"{b['rmse']:.4f}",
                f"{b['mae']:.4f}",
                f"{b['ade']:.4f}",
                f"{b['fde']:.4f}"
            ])
        
        # 添加各lambda的MTL结果
        for lam in sorted(self.results.keys()):
            data = self.results[lam]
            if 'trajectory_prediction' in data and 'MTL' in data['trajectory_prediction']:
                m = data['trajectory_prediction']['MTL']
                traj_data.append([
                    f"λ={lam}",
                    f"{m['rmse']:.4f}",
                    f"{m['mae']:.4f}",
                    f"{m['ade']:.4f}",
                    f"{m['fde']:.4f}"
                ])
        
        if traj_data:
            print(tabulate(traj_data, headers=traj_headers, tablefmt='grid'))
        
        # 意图识别对比表
        print("\n【意图识别任务】")
        intent_headers = ['Lambda', 'Accuracy ↑', 'F1-score ↑']
        intent_data = []
        
        # 添加Baseline行
        if self.baseline_intent:
            b = self.baseline_intent
            intent_data.append([
                'Baseline (Intent-Only)',
                f"{b['accuracy']:.4f}",
                f"{b['f1']:.4f}"
            ])
        
        # 添加各lambda的MTL结果
        for lam in sorted(self.results.keys()):
            data = self.results[lam]
            if 'intent_classification' in data and 'MTL' in data['intent_classification']:
                m = data['intent_classification']['MTL']
                intent_data.append([
                    f"λ={lam}",
                    f"{m['accuracy']:.4f}",
                    f"{m['f1']:.4f}"
                ])
        
        if intent_data:
            print(tabulate(intent_data, headers=intent_headers, tablefmt='grid'))
    
    def plot_comparison_bar_chart(self, save: bool = True):
        """
        绘制对比柱状图：不同lambda下相同指标紧密贴在一起
        Baseline 也放在图中作为对比基准
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # ==================== 轨迹预测对比 ====================
        ax1 = axes[0]
        traj_metrics = ['RMSE', 'MAE', 'ADE', 'FDE']
        metric_keys = ['rmse', 'mae', 'ade', 'fde']
        
        # 准备数据：每个指标对应 [baseline, lambda=0, lambda=0.1, ...]
        lambdas_sorted = sorted(self.results.keys())
        n_lambdas = len(lambdas_sorted)
        n_groups = len(traj_metrics)
        
        # bar宽度和位置计算
        bar_width = 0.1
        group_width = (n_lambdas + 1) * bar_width + 0.15  # +1 for baseline
        
        # 颜色方案：baseline用灰色，不同lambda用渐变色
        colors_lambda = plt.cm.viridis(np.linspace(0.2, 0.9, n_lambdas))
        color_baseline = '#808080'  # 灰色
        
        for g_idx, (metric_name, metric_key) in enumerate(zip(traj_metrics, metric_keys)):
            group_start = g_idx * group_width
            
            # 绘制Baseline
            if self.baseline_traj and metric_key in self.baseline_traj:
                baseline_val = self.baseline_traj[metric_key]
                bar = ax1.bar(group_start, baseline_val, bar_width, 
                             color=color_baseline, edgecolor='black', linewidth=0.5)
                # 数值标签
                ax1.text(group_start, baseline_val + 0.005, f'{baseline_val:.3f}',
                        ha='center', va='bottom', fontsize=7, rotation=90)
            
            # 绘制各lambda的MTL结果
            for l_idx, lam in enumerate(lambdas_sorted):
                x_pos = group_start + (l_idx + 1) * bar_width
                data = self.results[lam]
                if 'trajectory_prediction' in data and 'MTL' in data['trajectory_prediction']:
                    val = data['trajectory_prediction']['MTL'].get(metric_key, 0)
                    bar = ax1.bar(x_pos, val, bar_width, 
                                 color=colors_lambda[l_idx], edgecolor='black', linewidth=0.5)
                    ax1.text(x_pos, val + 0.005, f'{val:.3f}',
                            ha='center', va='bottom', fontsize=7, rotation=90)
        
        # 设置x轴
        group_centers = [g * group_width + (n_lambdas) * bar_width / 2 for g in range(n_groups)]
        ax1.set_xticks(group_centers)
        ax1.set_xticklabels(traj_metrics)
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Error Value (lower is better)')
        ax1.set_title('Trajectory Prediction: Lambda Comparison')
        ax1.grid(axis='y', alpha=0.3)
        
        # 图例
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color_baseline, edgecolor='black', label='Baseline')]
        for l_idx, lam in enumerate(lambdas_sorted):
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=colors_lambda[l_idx], 
                                                  edgecolor='black', label=f'λ={lam}'))
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        # ==================== 意图识别对比 ====================
        ax2 = axes[1]
        intent_metrics = ['Accuracy', 'F1-score']
        intent_keys = ['accuracy', 'f1']
        n_intent_groups = len(intent_metrics)
        
        for g_idx, (metric_name, metric_key) in enumerate(zip(intent_metrics, intent_keys)):
            group_start = g_idx * group_width
            
            # 绘制Baseline
            if self.baseline_intent and metric_key in self.baseline_intent:
                baseline_val = self.baseline_intent[metric_key]
                bar = ax2.bar(group_start, baseline_val, bar_width,
                             color=color_baseline, edgecolor='black', linewidth=0.5)
                ax2.text(group_start, baseline_val + 0.005, f'{baseline_val:.3f}',
                        ha='center', va='bottom', fontsize=8, rotation=0)
            
            # 绘制各lambda的MTL结果
            for l_idx, lam in enumerate(lambdas_sorted):
                x_pos = group_start + (l_idx + 1) * bar_width
                data = self.results[lam]
                if 'intent_classification' in data and 'MTL' in data['intent_classification']:
                    val = data['intent_classification']['MTL'].get(metric_key, 0)
                    bar = ax2.bar(x_pos, val, bar_width,
                                 color=colors_lambda[l_idx], edgecolor='black', linewidth=0.5)
                    ax2.text(x_pos, val + 0.005, f'{val:.3f}',
                            ha='center', va='bottom', fontsize=8, rotation=0)
        
        # 设置x轴
        group_centers = [g * group_width + (n_lambdas) * bar_width / 2 for g in range(n_intent_groups)]
        ax2.set_xticks(group_centers)
        ax2.set_xticklabels(intent_metrics)
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Score (higher is better)')
        ax2.set_title('Intent Classification: Lambda Comparison')
        ax2.set_ylim([0.0, 1.05])  # 聚焦于高准确率区域
        ax2.grid(axis='y', alpha=0.3)
        ax2.legend(handles=legend_elements, loc='lower right', fontsize=8)
        
        plt.tight_layout()
        
        if save:
            save_path = LAMBDA_COMPARISON_OUTPUT_DIR / 'lambda_comparison_bar_chart.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Lambda对比柱状图已保存: {save_path}")
        
        plt.show()
    
    def plot_lambda_trend(self, save: bool = True):
        """绘制Lambda变化趋势图（折线图）"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # ==================== 轨迹预测趋势 ====================
        ax1 = axes[0]
        metrics = [('rmse', 'RMSE', '#e74c3c'), 
                   ('mae', 'MAE', '#3498db'),
                   ('fde', 'FDE', '#2ecc71')]
        
        for metric_key, metric_name, color in metrics:
            lambdas, values = self.get_mtl_metrics('trajectory_prediction', metric_key)
            if lambdas and values:
                ax1.plot(lambdas, values, 'o-', label=metric_name, color=color, 
                        linewidth=2, markersize=8)
                
                # 添加baseline水平线
                if self.baseline_traj and metric_key in self.baseline_traj:
                    ax1.axhline(y=self.baseline_traj[metric_key], color=color, 
                               linestyle='--', alpha=0.5, linewidth=1)
        
        ax1.set_xlabel('Lambda (λ)')
        ax1.set_ylabel('Error Value')
        ax1.set_title('Trajectory Prediction vs Lambda')
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_xticks(sorted(self.results.keys()))
        
        # ==================== 意图识别趋势 ====================
        ax2 = axes[1]
        metrics = [('accuracy', 'Accuracy', '#9b59b6'),
                   ('f1', 'F1-score', '#f39c12')]
        
        for metric_key, metric_name, color in metrics:
            lambdas, values = self.get_mtl_metrics('intent_classification', metric_key)
            if lambdas and values:
                ax2.plot(lambdas, values, 'o-', label=metric_name, color=color,
                        linewidth=2, markersize=8)
                
                # 添加baseline水平线
                if self.baseline_intent and metric_key in self.baseline_intent:
                    ax2.axhline(y=self.baseline_intent[metric_key], color=color,
                               linestyle='--', alpha=0.5, linewidth=1,
                               label=f'{metric_name} Baseline')
        
        ax2.set_xlabel('Lambda (λ)')
        ax2.set_ylabel('Score')
        ax2.set_title('Intent Classification vs Lambda')
        ax2.legend()
        ax2.grid(alpha=0.3)
        ax2.set_xticks(sorted(self.results.keys()))
        ax2.set_ylim([0.9, 1.01])
        
        plt.tight_layout()
        
        if save:
            save_path = LAMBDA_COMPARISON_OUTPUT_DIR / 'lambda_trend.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Lambda趋势图已保存: {save_path}")
        
        plt.show()
    
    def plot_tradeoff_scatter(self, save: bool = True):
        """绘制轨迹预测vs意图识别的trade-off散点图"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        lambdas_sorted = sorted(self.results.keys())
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(lambdas_sorted)))
        
        # 收集数据点
        for l_idx, lam in enumerate(lambdas_sorted):
            data = self.results[lam]
            if ('trajectory_prediction' in data and 'MTL' in data['trajectory_prediction'] and
                'intent_classification' in data and 'MTL' in data['intent_classification']):
                
                traj_rmse = data['trajectory_prediction']['MTL']['rmse']
                intent_acc = data['intent_classification']['MTL']['accuracy']
                
                ax.scatter(traj_rmse, intent_acc, c=[colors[l_idx]], s=200, 
                          edgecolor='black', linewidth=1.5, zorder=5)
                ax.annotate(f'λ={lam}', (traj_rmse, intent_acc), 
                           textcoords="offset points", xytext=(10, 5),
                           fontsize=10, fontweight='bold')
        
        # 添加baseline点
        if self.baseline_traj and self.baseline_intent:
            ax.axvline(x=self.baseline_traj['rmse'], color='gray', linestyle='--', 
                      alpha=0.7, label='Traj Baseline RMSE')
            ax.axhline(y=self.baseline_intent['accuracy'], color='gray', linestyle=':', 
                      alpha=0.7, label='Intent Baseline Acc')
        
        ax.set_xlabel('Trajectory RMSE (lower is better) →')
        ax.set_ylabel('Intent Accuracy (higher is better) →')
        ax.set_title('Multi-Task Learning Trade-off: Trajectory vs Intent')
        ax.legend(loc='lower left')
        ax.grid(alpha=0.3)
        
        # 标注最优区域
        ax.annotate('Better', xy=(0.05, 0.95), xycoords='axes fraction',
                   fontsize=12, color='green', fontweight='bold',
                   ha='left', va='top')
        
        plt.tight_layout()
        
        if save:
            save_path = LAMBDA_COMPARISON_OUTPUT_DIR / 'tradeoff_scatter.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Trade-off散点图已保存: {save_path}")
        
        plt.show()
    
    def find_best_lambda(self):
        """找出各指标下的最佳lambda"""
        print("\n" + "="*80)
        print("最佳 Lambda 分析")
        print("="*80)
        
        # 轨迹预测：越低越好
        traj_metrics = ['rmse', 'mae', 'ade', 'fde']
        print("\n【轨迹预测】(越低越好)")
        for metric in traj_metrics:
            best_lam = None
            best_val = float('inf')
            for lam, data in self.results.items():
                if 'trajectory_prediction' in data and 'MTL' in data['trajectory_prediction']:
                    val = data['trajectory_prediction']['MTL'].get(metric, float('inf'))
                    if val < best_val:
                        best_val = val
                        best_lam = lam
            if best_lam is not None:
                baseline_val = self.baseline_traj.get(metric, 0) if self.baseline_traj else 0
                improvement = (baseline_val - best_val) / baseline_val * 100 if baseline_val else 0
                print(f"  {metric.upper()}: 最佳 λ={best_lam} (值={best_val:.4f}, 相比baseline提升 {improvement:+.2f}%)")
        
        # 意图识别：越高越好
        intent_metrics = ['accuracy', 'f1']
        print("\n【意图识别】(越高越好)")
        for metric in intent_metrics:
            best_lam = None
            best_val = 0
            for lam, data in self.results.items():
                if 'intent_classification' in data and 'MTL' in data['intent_classification']:
                    val = data['intent_classification']['MTL'].get(metric, 0)
                    if val > best_val:
                        best_val = val
                        best_lam = lam
            if best_lam is not None:
                baseline_val = self.baseline_intent.get(metric, 0) if self.baseline_intent else 0
                improvement = (best_val - baseline_val) / baseline_val * 100 if baseline_val else 0
                print(f"  {metric.capitalize()}: 最佳 λ={best_lam} (值={best_val:.4f}, 相比baseline提升 {improvement:+.2f}%)")
    
    def export_results(self):
        """导出汇总结果为JSON"""
        export_data = {
            'lambda_values': sorted(self.results.keys()),
            'baseline': {
                'trajectory': self.baseline_traj,
                'intent': self.baseline_intent
            },
            'mtl_results': {}
        }
        
        for lam in sorted(self.results.keys()):
            data = self.results[lam]
            export_data['mtl_results'][str(lam)] = {
                'trajectory': data.get('trajectory_prediction', {}).get('MTL', {}),
                'intent': data.get('intent_classification', {}).get('MTL', {})
            }
        
        save_path = LAMBDA_COMPARISON_OUTPUT_DIR / 'lambda_comparison_results.json'
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Lambda对比结果已导出: {save_path}")
        
        return export_data
    
    def generate_latex_table(self):
        """生成LaTeX表格（用于论文）"""
        print("\n" + "="*80)
        print("LaTeX表格（可直接复制到论文）")
        print("="*80)
        
        # 轨迹预测表格
        print("\n% Trajectory Prediction Results with Different Lambda")
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{Trajectory Prediction Performance with Different $\\lambda$}")
        print("\\begin{tabular}{lcccc}")
        print("\\toprule")
        print("$\\lambda$ & RMSE $\\downarrow$ & MAE $\\downarrow$ & ADE $\\downarrow$ & FDE $\\downarrow$ \\\\")
        print("\\midrule")
        
        # Baseline
        if self.baseline_traj:
            b = self.baseline_traj
            print(f"Baseline & {b['rmse']:.4f} & {b['mae']:.4f} & {b['ade']:.4f} & {b['fde']:.4f} \\\\")
            print("\\midrule")
        
        # 各lambda
        for lam in sorted(self.results.keys()):
            data = self.results[lam]
            if 'trajectory_prediction' in data and 'MTL' in data['trajectory_prediction']:
                m = data['trajectory_prediction']['MTL']
                print(f"{lam} & {m['rmse']:.4f} & {m['mae']:.4f} & {m['ade']:.4f} & {m['fde']:.4f} \\\\")
        
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\label{tab:lambda_traj}")
        print("\\end{table}")
        
        # 意图识别表格
        print("\n% Intent Classification Results with Different Lambda")
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{Intent Classification Performance with Different $\\lambda$}")
        print("\\begin{tabular}{lcc}")
        print("\\toprule")
        print("$\\lambda$ & Accuracy $\\uparrow$ & F1-score $\\uparrow$ \\\\")
        print("\\midrule")
        
        # Baseline
        if self.baseline_intent:
            b = self.baseline_intent
            print(f"Baseline & {b['accuracy']:.4f} & {b['f1']:.4f} \\\\")
            print("\\midrule")
        
        # 各lambda
        for lam in sorted(self.results.keys()):
            data = self.results[lam]
            if 'intent_classification' in data and 'MTL' in data['intent_classification']:
                m = data['intent_classification']['MTL']
                print(f"{lam} & {m['accuracy']:.4f} & {m['f1']:.4f} \\\\")
        
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\label{tab:lambda_intent}")
        print("\\end{table}")


def main():
    """主函数"""
    print("\n" + "="*80)
    print("Lambda 消融实验对比分析工具")
    print("对比不同 λ (意图损失权重) 下的 MTL 模型效果")
    print("="*80)
    
    # 创建对比工具
    comparison = LambdaComparison()
    
    # 加载结果
    comparison.load_results()
    
    if len(comparison.results) == 0:
        print("\n✗ 没有找到任何 lambda 实验结果")
        return
    
    # 打印对比表格
    comparison.print_comparison_table()
    
    # 找出最佳lambda
    comparison.find_best_lambda()
    
    # 生成LaTeX表格
    comparison.generate_latex_table()
    
    # 绘制对比图表
    print("\n生成对比图表...")
    comparison.plot_comparison_bar_chart(save=True)
    comparison.plot_lambda_trend(save=True)
    comparison.plot_tradeoff_scatter(save=True)
    
    # 导出结果
    comparison.export_results()
    
    print("\n" + "="*80)
    print("✓ Lambda 消融实验对比分析完成")
    print(f"  结果保存目录: {LAMBDA_COMPARISON_OUTPUT_DIR}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
