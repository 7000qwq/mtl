"""
数据加载模块：JSON读取、样本构造、归一化处理（支持混合样本）
"""
import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import config

class MixedTaskBatchSampler(torch.utils.data.Sampler[List[int]]):
    """在同一个batch里按固定比例混合两类样本：
    - traj_only 样本用于轨迹预测（滑窗）
    - intent_only 样本用于意图识别（整段）

    说明：
    - sampler 只负责“取哪些 index 组成一个 batch”
    - collate_fn 会把一个 batch 内不同类型样本分别堆叠成 obs_traj/pred_traj 和 full_traj
    """
    def __init__(self, dataset: Dataset, traj_per_batch: int, intent_per_batch: int, drop_last: bool = True, seed: int = 42):
        self.dataset = dataset
        self.traj_per_batch = int(traj_per_batch)
        self.intent_per_batch = int(intent_per_batch)
        self.batch_size = self.traj_per_batch + self.intent_per_batch
        self.drop_last = drop_last
        self.rng = np.random.RandomState(seed)

        # 预先收集两类样本在 dataset 中的索引
        self.traj_indices = [i for i, s in enumerate(dataset.samples) if s.get('sample_type') == 'traj_only']
        self.intent_indices = [i for i, s in enumerate(dataset.samples) if s.get('sample_type') == 'intent_only']

        if self.traj_per_batch <= 0 or self.intent_per_batch <= 0:
            raise ValueError("traj_per_batch 和 intent_per_batch 都必须 > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须 > 0")
        if len(self.traj_indices) == 0 or len(self.intent_indices) == 0:
            raise ValueError("dataset 中必须同时包含 traj_only 与 intent_only 样本，才能进行混合采样。")

    def __iter__(self):
        # 每个 epoch 重新打乱
        traj = self.traj_indices.copy()
        intent = self.intent_indices.copy()
        self.rng.shuffle(traj)
        self.rng.shuffle(intent)

        t_ptr = 0
        i_ptr = 0

        # 估算 epoch 内 batch 数：用“更稀缺的那类”决定
        num_batches = min(len(traj) // self.traj_per_batch, len(intent) // self.intent_per_batch)
        if not self.drop_last:
            num_batches = max(num_batches, 1)

        for _ in range(num_batches):
            # 如果不够就重新洗牌续上（保证无限供应的感觉，但仍然是无放回分段采样）
            if t_ptr + self.traj_per_batch > len(traj):
                self.rng.shuffle(traj)
                t_ptr = 0
            if i_ptr + self.intent_per_batch > len(intent):
                self.rng.shuffle(intent)
                i_ptr = 0

            batch_idx = traj[t_ptr:t_ptr + self.traj_per_batch] + intent[i_ptr:i_ptr + self.intent_per_batch]
            t_ptr += self.traj_per_batch
            i_ptr += self.intent_per_batch

            self.rng.shuffle(batch_idx)  # batch 内也打乱，避免类型顺序偏置
            yield batch_idx

    def __len__(self):
        return min(len(self.traj_indices) // self.traj_per_batch, len(self.intent_indices) // self.intent_per_batch)

class TrajectoryDataset(Dataset):
    """轨迹数据集（支持混合样本）"""
    def __init__(self, samples: List[Dict], norm_params: Dict = None, 
                 is_train: bool = True):
        """
        Args:
            samples: 样本列表，每个元素为 {
                'sample_type': str ('both'|'traj_only'|'intent_only'),
                'obs_traj': ndarray (obs_len, 3) - 对traj任务
                'pred_traj': ndarray (pred_len, 3) - 对traj任务
                'full_traj': ndarray (full_traj_len, 3) - 对intent任务
                'intent': int,
                'intent_label': str
            }
            norm_params: 归一化参数 {'mean': ndarray, 'std': ndarray}
            is_train: 是否为训练集（用于确定是否计算归一化参数）
        """
        self.samples = samples
        self.is_train = is_train
        
        # 计算或使用提供的归一化参数
        if norm_params is None:
            self.norm_params = self._compute_norm_params()
        else:
            self.norm_params = norm_params
    
    def _compute_norm_params(self) -> Dict:
        """从训练样本计算归一化参数"""
        if not self.samples:
            return {'mean': np.zeros(config.TRAJ_DIM), 'std': np.ones(config.TRAJ_DIM)}
        
        all_trajs = []
        for sample in self.samples:
            # 收集所有可用的轨迹数据
            if 'obs_traj' in sample and sample['obs_traj'] is not None:
                all_trajs.append(sample['obs_traj'])
            if 'pred_traj' in sample and sample['pred_traj'] is not None:
                all_trajs.append(sample['pred_traj'])
            if 'full_traj' in sample and sample['full_traj'] is not None:
                all_trajs.append(sample['full_traj'])
        
        if not all_trajs:
            return {'mean': np.zeros(config.TRAJ_DIM), 'std': np.ones(config.TRAJ_DIM)}
        
        all_trajs = np.vstack(all_trajs)
        mean = np.mean(all_trajs, axis=0)
        std = np.std(all_trajs, axis=0)
        std = np.where(std < 1e-6, 1.0, std)  # 避免除以零
        
        return {'mean': mean, 'std': std}
    
    def normalize(self, traj: np.ndarray) -> np.ndarray:
        """归一化轨迹"""
        return (traj - self.norm_params['mean']) / self.norm_params['std']
    
    def denormalize(self, traj: np.ndarray) -> np.ndarray:
        """逆归一化轨迹"""
        return traj * self.norm_params['std'] + self.norm_params['mean']
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        result = {
            'sample_type': sample['sample_type'],
            'intent': torch.LongTensor([sample['intent']]),
            'intent_label': sample['intent_label']
        }
        
        # 根据样本类型添加对应的数据
        if sample['sample_type'] in ['both', 'traj_only']:
            result['obs_traj'] = torch.FloatTensor(self.normalize(sample['obs_traj']))
            result['pred_traj'] = torch.FloatTensor(self.normalize(sample['pred_traj']))
        
        if sample['sample_type'] in ['both', 'intent_only']:
            result['full_traj'] = torch.FloatTensor(self.normalize(sample['full_traj']))
        
        return result


class DataManager:
    """数据管理器：加载、处理、分割数据"""
    
    def __init__(self):
        self.raw_trajectories = {}  # {'intent': [traj1, traj2, ...]}
        self.samples = []
        self.train_samples = []
        self.val_samples = []
        self.test_samples = []
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.rng = np.random.RandomState(config.SEED)  # 用于样本类型随机分配
    
    def _resample_trajectory(self, trajectory: np.ndarray, target_length: int) -> np.ndarray:
        """
        通过线性插值将轨迹重采样到目标长度
        
        Args:
            trajectory: 原始轨迹 (original_len, 3)
            target_length: 目标长度
        
        Returns:
            重采样后的轨迹 (target_length, 3)
        """
        if len(trajectory) == target_length:
            return trajectory
        
        original_len = len(trajectory)
        # 原始时间索引
        x_old = np.linspace(0, 1, original_len)
        # 目标时间索引
        x_new = np.linspace(0, 1, target_length)
        
        # 对每个维度分别进行插值
        resampled = np.zeros((target_length, trajectory.shape[1]), dtype=np.float32)
        for dim in range(trajectory.shape[1]):
            resampled[:, dim] = np.interp(x_new, x_old, trajectory[:, dim])
        
        return resampled
    
    def load_data(self) -> None:
        """从 flight_data_random 加载所有轨迹数据"""
        print("[DataManager] 开始加载轨迹数据...")
        
        intent_dirs = [d for d in Path(config.DATA_DIR).iterdir() 
                      if d.is_dir() and d.name in config.INTENT_CLASSES]
        
        for intent_dir in intent_dirs:
            intent_name = intent_dir.name
            self.raw_trajectories[intent_name] = []
            
            json_files = sorted(intent_dir.glob('*.json'))
            print(f"  意图 '{intent_name}': {len(json_files)} 条轨迹")
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    trajectory = self._extract_trajectory(data)
                    if trajectory is not None:
                        self.raw_trajectories[intent_name].append({
                            'data': trajectory,
                            'filename': json_file.name
                        })
                except Exception as e:
                    print(f"    警告: 读取 {json_file.name} 失败 - {e}")
        
        total = sum(len(v) for v in self.raw_trajectories.values())
        print(f"✓ 成功加载 {total} 条轨迹\n")
    
    def _extract_trajectory(self, data: Dict) -> np.ndarray:
        """从JSON数据提取轨迹序列并统一长度"""
        try:
            trajectory = data.get('trajectory', [])
            if not trajectory:
                return None
            
            positions = []
            for record in trajectory:
                pos = record.get('position', {})
                x = pos.get('x', 0.0)
                y = pos.get('y', 0.0)
                z = pos.get('z', 0.0)
                positions.append([x, y, z])
            
            if len(positions) < config.OBS_LEN + config.PRED_LEN:
                return None
            
            trajectory_array = np.array(positions, dtype=np.float32)
            
            # 统一轨迹长度：通过插值/降采样到UNIFIED_TRAJ_LEN
            trajectory_unified = self._resample_trajectory(trajectory_array, config.UNIFIED_TRAJ_LEN)
            
            return trajectory_unified
        except Exception as e:
            return None
    
    def construct_samples(self) -> None:
        """构造样本（推荐）：轨迹预测用滑窗样本；意图识别每条轨迹只取1个整段样本。

        目标：
        - traj_only: 对每条轨迹生成大量 (obs_traj -> pred_traj) 的滑窗样本
        - intent_only: 对每条轨迹仅生成 1 个 full_traj 样本（避免用滑窗重复放大意图监督）
        """
        print("[DataManager] 构造样本：滑窗轨迹预测 + 每轨迹一个意图样本 ...")
        self.samples = []

        traj_count = 0
        intent_count = 0

        for intent_name, trajs in self.raw_trajectories.items():
            intent_idx = config.INTENT_TO_IDX[intent_name]

            for traj_data in trajs:
                traj = traj_data['data']  # (UNIFIED_TRAJ_LEN, traj_dim)
                filename = traj_data['filename']

                # 1) 轨迹预测：滑动窗口（生成 traj_only 样本）
                max_start = config.UNIFIED_TRAJ_LEN - config.OBS_LEN - config.PRED_LEN
                if max_start >= 0:
                    # 兼容：旧版 config.SAMPLE_INTERVAL 可能是秒(float)，range 需要 int
                    stride = getattr(config, "WINDOW_STRIDE", None)
                    if stride is None:
                        stride = getattr(config, "SAMPLE_STRIDE", None)
                    if stride is None:
                        stride = getattr(config, "SAMPLE_INTERVAL", 1)

                    # 将 stride 归一成“点步长”（int）
                    if isinstance(stride, float):
                        # 旧语义通常是秒；在不知道真实采样率的情况下，最安全的向前兼容是至少步长为 1
                        stride = max(1, int(round(stride)))
                    else:
                        stride = max(1, int(stride))

                    for start in range(0, max_start + 1, stride):
                        obs = traj[start:start + config.OBS_LEN]
                        pred = traj[start + config.OBS_LEN:start + config.OBS_LEN + config.PRED_LEN]
                        self.samples.append({
                            'sample_type': 'traj_only',
                            'obs_traj': obs,
                            'pred_traj': pred,
                            'full_traj': None,
                            'intent': intent_idx,
                            'intent_label': intent_name,
                            'traj_filename': filename
                        })
                        traj_count += 1

                # 2) 意图识别：每条轨迹只取 1 个整段 full_traj（生成 intent_only 样本）
                #    traj 已统一到 UNIFIED_TRAJ_LEN；full_traj 长度为 FULL_TRAJ_LEN
                if config.UNIFIED_TRAJ_LEN >= config.FULL_TRAJ_LEN:
                    max_full_start = config.UNIFIED_TRAJ_LEN - config.FULL_TRAJ_LEN
                    full_start = self.rng.randint(0, max_full_start + 1)
                    full_traj = traj[full_start:full_start + config.FULL_TRAJ_LEN]
                else:
                    padding_len = config.FULL_TRAJ_LEN - config.UNIFIED_TRAJ_LEN
                    padding = np.tile(traj[-1:], (padding_len, 1))
                    full_traj = np.vstack([traj, padding])

                self.samples.append({
                    'sample_type': 'intent_only',
                    'obs_traj': None,
                    'pred_traj': None,
                    'full_traj': full_traj,
                    'intent': intent_idx,
                    'intent_label': intent_name,
                    'traj_filename': filename
                })
                intent_count += 1

        print(f"✓ 共构造 {len(self.samples)} 个样本：traj_only={traj_count}, intent_only={intent_count}")
        if len(self.samples) > 0:
            print(f"  - traj_only: {traj_count} ({traj_count/len(self.samples)*100:.1f}%)")
            print(f"  - intent_only: {intent_count} ({intent_count/len(self.samples)*100:.1f}%)")
        print()

    def split_by_trajectory(self) -> None:
        """按轨迹级别划分数据集（避免信息泄漏）"""
        print("[DataManager] 按轨迹级别划分数据集...")
        
        # 获取唯一轨迹
        unique_trajs = {}
        for intent_name, trajs in self.raw_trajectories.items():
            for traj_data in trajs:
                filename = traj_data['filename']
                if filename not in unique_trajs:
                    unique_trajs[filename] = intent_name
        
        filenames = list(unique_trajs.keys())
        
        # 第一次分割：训练集 vs (验证+测试)
        train_files, temp_files = train_test_split(
            filenames,
            test_size=(config.VAL_RATIO + config.TEST_RATIO),
            random_state=config.SEED
        )
        
        # 第二次分割：验证集 vs 测试集
        val_size = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
        val_files, test_files = train_test_split(
            temp_files,
            test_size=1 - val_size,
            random_state=config.SEED
        )
        
        train_set = set(train_files)
        val_set = set(val_files)
        test_set = set(test_files)
        
        self.train_samples = [s for s in self.samples if s['traj_filename'] in train_set]
        self.val_samples = [s for s in self.samples if s['traj_filename'] in val_set]
        self.test_samples = [s for s in self.samples if s['traj_filename'] in test_set]
        
        print(f"  训练集: {len(self.train_samples)} 个样本 ({len(train_files)} 条轨迹)")
        print(f"  验证集: {len(self.val_samples)} 个样本 ({len(val_files)} 条轨迹)")
        print(f"  测试集: {len(self.test_samples)} 个样本 ({len(test_files)} 条轨迹)\n")
    
    def create_datasets(self) -> None:
        """创建 PyTorch Dataset 对象"""
        print("[DataManager] 创建 PyTorch 数据集...")
        
        # 从训练集计算归一化参数
        self.train_dataset = TrajectoryDataset(
            self.train_samples,
            norm_params=None,  # 自动计算
            is_train=True
        )
        
        # 使用训练集的归一化参数
        norm_params = self.train_dataset.norm_params
        self.val_dataset = TrajectoryDataset(
            self.val_samples,
            norm_params=norm_params,
            is_train=False
        )
        self.test_dataset = TrajectoryDataset(
            self.test_samples,
            norm_params=norm_params,
            is_train=False
        )
        
        print(f"  归一化参数 (来自训练集):")
        print(f"    Mean: {norm_params['mean']}")
        print(f"    Std: {norm_params['std']}\n")
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """获取数据加载器"""
        train_loader = DataLoader(
            self.train_dataset,
            batch_sampler=MixedTaskBatchSampler(
                self.train_dataset,
                traj_per_batch=getattr(config, 'TRAJ_PER_BATCH', config.BATCH_SIZE // 2),
                intent_per_batch=getattr(config, 'INTENT_PER_BATCH', config.BATCH_SIZE - config.BATCH_SIZE // 2),
                drop_last=True,
                seed=config.SEED
            ),
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        return train_loader, val_loader, test_loader
    
    def prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """完整数据处理流程"""
        self.load_data()
        self.construct_samples()
        self.split_by_trajectory()
        self.create_datasets()
        return self.get_dataloaders()


def get_data_manager() -> DataManager:
    """获取数据管理器实例"""
    return DataManager()


def custom_collate_fn(batch: List[Dict]) -> Dict:
    """
    自定义collate函数，处理混合样本的batch
    
    Args:
        batch: 样本列表
    
    Returns:
        包含以下字段的字典：
        - sample_types: List[str] - 每个样本的类型
        - obs_traj: Optional[Tensor] - (N_traj, obs_len, 3)
        - pred_traj: Optional[Tensor] - (N_traj, pred_len, 3)
        - full_traj: Optional[Tensor] - (N_intent, full_traj_len, 3)
        - traj_indices: List[int] - batch中哪些位置有traj数据
        - intent_indices: List[int] - batch中哪些位置有intent数据
        - intent: Tensor - (batch_size,)
        - intent_label: List[str]
    """
    sample_types = [item['sample_type'] for item in batch]
    intent = torch.cat([item['intent'] for item in batch], dim=0).squeeze(-1)  # (batch_size,)
    intent_label = [item['intent_label'] for item in batch]
    
    # 收集traj相关数据
    traj_indices = []
    obs_trajs = []
    pred_trajs = []
    for i, item in enumerate(batch):
        if item['sample_type'] in ['both', 'traj_only']:
            traj_indices.append(i)
            obs_trajs.append(item['obs_traj'])
            pred_trajs.append(item['pred_traj'])
    
    # 收集intent相关数据
    intent_indices = []
    full_trajs = []
    for i, item in enumerate(batch):
        if item['sample_type'] in ['both', 'intent_only']:
            intent_indices.append(i)
            full_trajs.append(item['full_traj'])
    
    result = {
        'sample_types': sample_types,
        'intent': intent,
        'intent_label': intent_label,
        'traj_indices': traj_indices,
        'intent_indices': intent_indices
    }
    
    if obs_trajs:
        result['obs_traj'] = torch.stack(obs_trajs, dim=0)
        result['pred_traj'] = torch.stack(pred_trajs, dim=0)
    else:
        result['obs_traj'] = None
        result['pred_traj'] = None
    
    if full_trajs:
        result['full_traj'] = torch.stack(full_trajs, dim=0)
    else:
        result['full_traj'] = None
    
    return result
