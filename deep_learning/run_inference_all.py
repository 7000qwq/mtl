# -*- coding: utf-8 -*-
"""
一键推理计时脚本
在验证集上分别对以下三个模型进行推理，记录用时，结果保存到 mtl_output/timing/inference_times.json：
  1. Traj-Only  （单任务轨迹预测）
  2. Intent-Only（单任务意图识别）
  3. MTL lambda=0.7（多任务学习）

前置条件：先运行 run_train_all.py 完成三个模型的训练。

用法：
  cd /home/zoliang/mtl/deep_learning
  python run_inference_all.py
"""
import sys
import io
import os

# 强制 stdout/stderr 使用 UTF-8，解决终端编码为 ASCII 时无法打印中文的问题
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ['PYTHONIOENCODING'] = 'utf-8'

import time
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# 切换到脚本所在目录，确保相对路径正确
script_dir = Path(__file__).parent
os.chdir(str(script_dir))
sys.path.insert(0, str(script_dir))

import config  # noqa: E402

OUTPUT_DIR = Path('mtl_output/timing')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEP = '=' * 70
results = {}

print(f'\n{SEP}')
print(f'  推理计时 | 设备: {DEVICE}')
print(SEP)


# ─────────────────────────────────────────────────────────────────────────────
def time_val_inference(model, val_loader, forward_fn):
    """
    在验证集上运行推理并计时（仅统计 forward pass 时间）。

    forward_fn(model, batch, device) -> (n_samples_in_batch,)
    返回: (total_seconds, total_samples)
    """
    model.eval()
    model.to(DEVICE)

    # GPU Warmup：用第一个 batch 预热，避免首次 CUDA kernel 启动影响计时
    if DEVICE.type == 'cuda':
        with torch.no_grad():
            for batch in val_loader:
                forward_fn(model, batch, DEVICE)
                break
        torch.cuda.synchronize()

    total_samples = 0
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        for batch in val_loader:
            n = forward_fn(model, batch, DEVICE)
            total_samples += n

    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return elapsed, total_samples


# ─────────────────────────────────────────────────────────────────────────────
# 1. Traj-Only
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{SEP}')
print('  推理: Traj-Only 模型 (验证集)')
print(SEP)

try:
    from train_traj_only import TrajOnlyDataManager
    from model_traj_only import create_traj_model

    ckpt_path = 'mtl_output/traj_only/checkpoints/best.pt'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'模型文件不存在: {ckpt_path}\n'
                                '请先运行 run_train_all.py 完成训练')

    print('  加载数据...')
    traj_dm = TrajOnlyDataManager()
    _, traj_val_loader, _ = traj_dm.prepare_data()

    print('  加载模型...')
    traj_model = create_traj_model()
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    traj_model.load_state_dict(ckpt['model_state_dict'])
    print(f'  ✓ 已加载模型: {ckpt_path}')

    def traj_forward(model, batch, device):
        obs_traj = batch['obs_traj']
        if obs_traj is None:
            return 0
        obs_traj = obs_traj.to(device)
        model(obs_traj)
        return obs_traj.size(0)

    elapsed, n_samples = time_val_inference(traj_model, traj_val_loader, traj_forward)
    ms_per = elapsed / n_samples * 1000 if n_samples > 0 else 0
    results['traj_only'] = {
        'model': 'Traj-Only (单任务轨迹预测)',
        'total_seconds': round(elapsed, 6),
        'num_samples': n_samples,
        'ms_per_sample': round(ms_per, 4)
    }
    print(f'  总用时: {elapsed:.4f}s | 样本数: {n_samples} | 每样本: {ms_per:.3f}ms')

except Exception as e:
    print(f'  ✗ Traj-Only 推理失败: {e}')
    results['traj_only'] = {'error': str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Intent-Only
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{SEP}')
print('  推理: Intent-Only 模型 (验证集)')
print(SEP)

try:
    from train_intent_only import IntentOnlyDataManager
    from model_intent_only import create_intent_model

    ckpt_path = 'mtl_output/intent_only/checkpoints/best.pt'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'模型文件不存在: {ckpt_path}\n'
                                '请先运行 run_train_all.py 完成训练')

    print('  加载数据...')
    intent_dm = IntentOnlyDataManager()
    _, intent_val_loader, _ = intent_dm.prepare_data()

    print('  加载模型...')
    intent_model = create_intent_model()
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    intent_model.load_state_dict(ckpt['model_state_dict'])
    print(f'  ✓ 已加载模型: {ckpt_path}')

    def intent_forward(model, batch, device):
        full_traj = batch['full_traj']
        if full_traj is None:
            return 0
        full_traj = full_traj.to(device)
        model(full_traj)
        return full_traj.size(0)

    elapsed, n_samples = time_val_inference(intent_model, intent_val_loader, intent_forward)
    ms_per = elapsed / n_samples * 1000 if n_samples > 0 else 0
    results['intent_only'] = {
        'model': 'Intent-Only (单任务意图识别)',
        'total_seconds': round(elapsed, 6),
        'num_samples': n_samples,
        'ms_per_sample': round(ms_per, 4)
    }
    print(f'  总用时: {elapsed:.4f}s | 样本数: {n_samples} | 每样本: {ms_per:.3f}ms')

except Exception as e:
    print(f'  ✗ Intent-Only 推理失败: {e}')
    results['intent_only'] = {'error': str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 3. MTL (lambda=0.7)
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{SEP}')
print('  推理: MTL 模型 (lambda=0.7, 验证集)')
print(SEP)

try:
    from data_loader import get_data_manager
    from model import create_model

    ckpt_path = 'mtl_output/checkpoints/best.pt'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'模型文件不存在: {ckpt_path}\n'
                                '请先运行 run_train_all.py 完成训练')

    print('  加载数据...')
    mtl_dm = get_data_manager()
    _, mtl_val_loader, _ = mtl_dm.prepare_data()

    print('  加载模型...')
    mtl_model = create_model()
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    mtl_model.load_state_dict(ckpt['model_state_dict'])
    print(f'  ✓ 已加载模型: {ckpt_path}')

    def mtl_forward(model, batch, device):
        obs_traj = batch.get('obs_traj')
        full_traj = batch.get('full_traj')
        obs_traj = obs_traj.to(device) if obs_traj is not None else None
        full_traj = full_traj.to(device) if full_traj is not None else None
        model(obs_traj=obs_traj, full_traj=full_traj)
        # 统计本 batch 的总样本数（traj样本 + intent样本）
        n = (obs_traj.size(0) if obs_traj is not None else 0) + \
            (full_traj.size(0) if full_traj is not None else 0)
        return n

    elapsed, n_samples = time_val_inference(mtl_model, mtl_val_loader, mtl_forward)
    ms_per = elapsed / n_samples * 1000 if n_samples > 0 else 0
    results['mtl_lambda07'] = {
        'model': 'MTL 多任务学习 (lambda=0.7)',
        'lambda': 0.7,
        'total_seconds': round(elapsed, 6),
        'num_samples': n_samples,
        'ms_per_sample': round(ms_per, 4)
    }
    print(f'  总用时: {elapsed:.4f}s | 样本数: {n_samples} | 每样本: {ms_per:.3f}ms')

except Exception as e:
    print(f'  ✗ MTL 推理失败: {e}')
    results['mtl_lambda07'] = {'error': str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 保存结果 & 打印汇总
# ─────────────────────────────────────────────────────────────────────────────
results['_meta'] = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'device': str(DEVICE),
    'note': '仅统计模型 forward pass 时间（已做 GPU warmup）'
}

output_path = OUTPUT_DIR / 'inference_times.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f'\n{SEP}')
print('  验证集推理用时汇总')
print(SEP)
for key in ('traj_only', 'intent_only', 'mtl_lambda07'):
    info = results.get(key, {})
    if 'error' not in info:
        print(f'  ✓ {info.get("model",""):35s} '
              f'总计: {info["total_seconds"]:8.4f}s  '
              f'样本: {info["num_samples"]:6d}  '
              f'每样本: {info["ms_per_sample"]:.3f}ms')
    else:
        print(f'  ✗ {key:35s} 失败: {info["error"]}')
print(SEP)
print(f'\n  结果已保存: {output_path}\n')
