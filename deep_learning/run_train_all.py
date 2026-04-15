# -*- coding: utf-8 -*-
"""
一键训练脚本
顺序训练以下三个模型，记录各自训练用时，结果保存到 mtl_output/timing/training_times.json：
  1. Traj-Only  （单任务轨迹预测）
  2. Intent-Only（单任务意图识别）
  3. MTL lambda=0.7（多任务学习，意图损失权重=0.7）

用法：
  cd /home/zoliang/mtl/deep_learning
  python run_train_all.py
"""
import subprocess
import sys
import io
import os
import time

# 强制 stdout/stderr 使用 UTF-8，解决终端编码为 ASCII 时无法打印中文的问题
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ['PYTHONIOENCODING'] = 'utf-8'  # 子进程也继承此设置
import json
from pathlib import Path
from datetime import datetime

# 切换到脚本所在目录，确保相对路径正确
os.chdir(Path(__file__).parent)

OUTPUT_DIR = Path('mtl_output/timing')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEP = '=' * 70
results = {}


def run_and_time(label, cmd, env=None):
    print(f'\n{SEP}')
    print(f'  开始训练: {label}')
    print(f'  命令: {" ".join(cmd)}')
    print(f'{SEP}')
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env)
    elapsed = time.perf_counter() - t0
    if proc.returncode == 0:
        print(f'\n  ✓ {label} 训练完成，用时: {elapsed:.1f}s ({elapsed/60:.2f}min)')
    else:
        print(f'\n  ✗ {label} 训练失败 (returncode={proc.returncode})')
    return elapsed, proc.returncode


# ───────────── 1. Traj-Only ─────────────
elapsed, rc = run_and_time(
    'Traj-Only',
    [sys.executable, 'train_traj_only.py']
)
results['traj_only'] = {
    'model': 'Traj-Only (单任务轨迹预测)',
    'seconds': round(elapsed, 3),
    'minutes': round(elapsed / 60, 4),
    'returncode': rc
}

# ───────────── 2. Intent-Only ─────────────
elapsed, rc = run_and_time(
    'Intent-Only',
    [sys.executable, 'train_intent_only.py']
)
results['intent_only'] = {
    'model': 'Intent-Only (单任务意图识别)',
    'seconds': round(elapsed, 3),
    'minutes': round(elapsed / 60, 4),
    'returncode': rc
}

# ───────────── 3. MTL lambda=0.7 ─────────────
env = os.environ.copy()
env['MTL_LAMBDA'] = '0.7'
elapsed, rc = run_and_time(
    'MTL (lambda=0.7)',
    [sys.executable, 'train.py'],
    env=env
)
results['mtl_lambda07'] = {
    'model': 'MTL 多任务学习 (lambda=0.7)',
    'lambda': 0.7,
    'seconds': round(elapsed, 3),
    'minutes': round(elapsed / 60, 4),
    'returncode': rc
}

# ───────────── 保存结果 ─────────────
results['_meta'] = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'device': 'see training logs'
}

output_path = OUTPUT_DIR / 'training_times.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# ───────────── 打印汇总 ─────────────
print(f'\n{SEP}')
print('  训练用时汇总')
print(SEP)
for key in ('traj_only', 'intent_only', 'mtl_lambda07'):
    info = results[key]
    status = '✓' if info['returncode'] == 0 else '✗'
    print(f'  {status} {info["model"]:30s}  {info["seconds"]:8.1f}s  '
          f'({info["minutes"]:.2f} min)')
print(f'{SEP}')
print(f'\n  结果已保存: {output_path}\n')
