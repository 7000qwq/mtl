import airsim
import json
import math
import os
import random
import time
from datetime import datetime

# ============== 配置 ==============
INTENT_TYPE = 'straight_line'  # 可选: takeoff, hover, straight_line, turn, landing, z_scan
RUNS = 20                       # 每种意图重复次数
DATA_DIR = './flight_data_random'
SAMPLE_RATE = 10               # 采样频率 Hz
SAMPLE_INTERVAL = 1.0 / SAMPLE_RATE

# 风随机参数
WIND_SPEED_RANGE = (0.0, 15.0)  # m/s
WIND_VERTICAL_RANGE = (-2.0, 2.0)

# 起始位置随机范围 (NED)
START_POS_RANGE = {
    'x': (-10.0, 10.0),
    'y': (-10.0, 10.0),
    'z': (-12.0, -6.0)  # 负值表示向下（高度）
}

# 控制噪声参数
VEL_NOISE_STD = 0.2    # 高斯噪声标准差 m/s
VEL_NOISE_DT = 0.2     # 每次带噪控制命令持续时间 s

# ============== 工具函数 ==============

def sample_random_start():
    return [
        random.uniform(*START_POS_RANGE['x']),
        random.uniform(*START_POS_RANGE['y']),
        random.uniform(*START_POS_RANGE['z'])
    ]


def apply_random_wind(client):
    speed = random.uniform(*WIND_SPEED_RANGE)
    yaw = random.uniform(0, 2 * math.pi)
    vertical = random.uniform(*WIND_VERTICAL_RANGE)
    wind = airsim.Vector3r(
        speed * math.cos(yaw),
        speed * math.sin(yaw),
        vertical
    )
    client.simSetWind(wind)
    return wind


def clear_wind(client):
    client.simSetWind(airsim.Vector3r(0, 0, 0))


def record_state(client, target_pos=None, command_type=None, meta=None):
    state = client.getMultirotorState()
    imu = client.getImuData()
    gps = client.getGpsData()
    baro = client.getBarometerData()

    record = {
        'timestamp': state.timestamp,
        'position': {
            'x': state.kinematics_estimated.position.x_val,
            'y': state.kinematics_estimated.position.y_val,
            'z': state.kinematics_estimated.position.z_val
        },
        'velocity': {
            'x': state.kinematics_estimated.linear_velocity.x_val,
            'y': state.kinematics_estimated.linear_velocity.y_val,
            'z': state.kinematics_estimated.linear_velocity.z_val
        },
        'acceleration': {
            'x': state.kinematics_estimated.linear_acceleration.x_val,
            'y': state.kinematics_estimated.linear_acceleration.y_val,
            'z': state.kinematics_estimated.linear_acceleration.z_val
        },
        'orientation': {
            'w': state.kinematics_estimated.orientation.w_val,
            'x': state.kinematics_estimated.orientation.x_val,
            'y': state.kinematics_estimated.orientation.y_val,
            'z': state.kinematics_estimated.orientation.z_val
        },
        'angular_velocity': {
            'x': imu.angular_velocity.x_val,
            'y': imu.angular_velocity.y_val,
            'z': imu.angular_velocity.z_val
        },
        'gps_velocity': {
            'x': gps.gnss.velocity.x_val,
            'y': gps.gnss.velocity.y_val,
            'z': gps.gnss.velocity.z_val
        } if gps.is_valid else None,
        'altitude': baro.altitude,
        'landed_state': state.landed_state,
        'target_position': target_pos,
        'command_type': command_type,
        'meta': meta or {}
    }
    return record


def save_run(intent_type, run_number, trajectory, metadata):
    intent_dir = os.path.join(DATA_DIR, intent_type)
    os.makedirs(intent_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(intent_dir, f'{intent_type}_{timestamp}_run{run_number}.json')
    with open(filename, 'w') as f:
        json.dump({
            'intent_type': intent_type,
            'run_number': run_number,
            'sample_rate': SAMPLE_RATE,
            'metadata': metadata,
            'trajectory': trajectory
        }, f, indent=2)
    print(f'✓ 数据已保存: {filename}')


# ============== 通用准备 ==============

def arm_and_takeoff(client):
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()


def move_to_random_start(client):
    start_pos = sample_random_start()
    print(f'移动到随机起始点: {start_pos}')
    client.moveToPositionAsync(start_pos[0], start_pos[1], start_pos[2], 3).join()
    return start_pos


def noisy_velocity_command(client, base_vx, base_vy, base_vz, duration_s, label, trajectory, meta):
    """按段执行带噪声的速度控制并记录"""
    t0 = time.time()
    while time.time() - t0 < duration_s:
        vx = base_vx + random.gauss(0, VEL_NOISE_STD)
        vy = base_vy + random.gauss(0, VEL_NOISE_STD)
        vz = base_vz + random.gauss(0, VEL_NOISE_STD)
        client.moveByVelocityAsync(vx, vy, vz, VEL_NOISE_DT).join()
        record = record_state(
            client,
            command_type=label,
            meta={**(meta or {}), 'cmd_vx': vx, 'cmd_vy': vy, 'cmd_vz': vz}
        )
        trajectory.append(record)
        time.sleep(max(0.0, SAMPLE_INTERVAL - VEL_NOISE_DT))


# ============== 意图实现 ==============

def intent_takeoff(client, meta):
    trajectory = []
    arm_and_takeoff(client)
    start_pos = move_to_random_start(client)
    meta.update({'start_pos': start_pos})
    # 记录起飞后的稳定段
    t0 = time.time()
    while time.time() - t0 < 5:
        trajectory.append(record_state(client, command_type='takeoff', meta=meta))
        time.sleep(SAMPLE_INTERVAL)
    return trajectory


def intent_hover(client, meta):
    trajectory = []
    arm_and_takeoff(client)
    start_pos = move_to_random_start(client)
    meta.update({'start_pos': start_pos})
    client.hoverAsync().join()
    t0 = time.time()
    while time.time() - t0 < 10:
        trajectory.append(record_state(client, command_type='hover', meta=meta))
        time.sleep(SAMPLE_INTERVAL)
    return trajectory


def intent_straight_line(client, meta):
    trajectory = []
    arm_and_takeoff(client)
    start_pos = move_to_random_start(client)
    meta.update({'start_pos': start_pos})
    
    # 随机设置三个轴的速度
    vx = random.uniform(-5.0, 5.0)
    vy = random.uniform(-5.0, 5.0)
    vz = random.uniform(-1.0, 1.0)  # z轴速度范围较小
    duration = random.uniform(8, 12)
    
    meta.update({'base_vx': vx, 'base_vy': vy, 'base_vz': vz, 'duration': duration})
    print(f'直线飞行: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}, 持续{duration:.1f}秒')
    
    # 使用带噪声的速度控制
    noisy_velocity_command(
        client, 
        base_vx=vx, 
        base_vy=vy, 
        base_vz=vz, 
        duration_s=duration, 
        label='straight_line', 
        trajectory=trajectory, 
        meta=meta
    )
    
    # 停止后记录稳定态
    t0 = time.time()
    while time.time() - t0 < 3:
        trajectory.append(record_state(client, command_type='post_line', meta=meta))
        time.sleep(SAMPLE_INTERVAL)
    return trajectory


def intent_turn(client, meta):
    trajectory = []
    arm_and_takeoff(client)
    start_pos = move_to_random_start(client)
    meta.update({'start_pos': start_pos})
    # 第一段 Y 方向
    noisy_velocity_command(client, base_vx=0, base_vy=5, base_vz=0, duration_s=5, label='turn_leg1', trajectory=trajectory, meta=meta)
    # 转弯后 X 方向
    noisy_velocity_command(client, base_vx=5, base_vy=0, base_vz=0, duration_s=5, label='turn_leg2', trajectory=trajectory, meta=meta)
    return trajectory


def intent_landing(client, meta):
    trajectory = []
    arm_and_takeoff(client)
    start_pos = move_to_random_start(client)
    meta.update({'start_pos': start_pos})
    # 悬停一段再降落
    t0 = time.time()
    while time.time() - t0 < 3:
        trajectory.append(record_state(client, command_type='pre_land', meta=meta))
        time.sleep(SAMPLE_INTERVAL)
    client.landAsync().join()
    t1 = time.time()
    while time.time() - t1 < 5:
        trajectory.append(record_state(client, command_type='landing', meta=meta))
        time.sleep(SAMPLE_INTERVAL)
    return trajectory


def intent_z_scan(client, meta):
    """Z 字形扫描：多次转折的连续z字形路径，z方向保持一致"""
    trajectory = []
    arm_and_takeoff(client)
    start_pos = move_to_random_start(client)
    meta.update({'start_pos': start_pos})

    # 随机化扫描参数
    num_turns = random.randint(4, 8)  # z字转折次数（4-8次）
    leg_length = random.uniform(15, 25)  # 每段的长度
    leg_spacing = random.uniform(3, 6)   # 每段之间的间距
    base_z = start_pos[2]  # 基准高度
    z_noise_std = 0.15  # z方向抖动标准差
    speed = 5.0

    # 构建z字形航点
    waypoints = [start_pos]
    current_x = start_pos[0]
    current_y = start_pos[1]
    direction = 1  # 1表示正向，-1表示反向
    
    for i in range(num_turns):
        # 前进一段
        current_x += leg_length * direction
        z_offset = random.gauss(0, z_noise_std)  # 微小抖动
        waypoints.append([current_x, current_y, base_z + z_offset])
        
        # 如果不是最后一段，则横移
        if i < num_turns - 1:
            current_y += leg_spacing
            z_offset = random.gauss(0, z_noise_std)
            waypoints.append([current_x, current_y, base_z + z_offset])
            direction *= -1  # 改变方向
    
    meta.update({
        'num_turns': num_turns,
        'leg_length': leg_length,
        'leg_spacing': leg_spacing,
        'base_z': base_z,
        'waypoints': waypoints
    })
    
    print(f'Z字形扫描: {num_turns}次转折, 航段长度={leg_length:.1f}m, 间距={leg_spacing:.1f}m')
    
    # 依次飞向各航点
    for idx in range(1, len(waypoints)):
        start_wp = waypoints[idx - 1]
        end_wp = waypoints[idx]
        
        # 计算这段的方向和速度
        dx = end_wp[0] - start_wp[0]
        dy = end_wp[1] - start_wp[1]
        dz = end_wp[2] - start_wp[2]
        dist = max(1e-3, math.sqrt(dx * dx + dy * dy + dz * dz))
        vx = speed * dx / dist
        vy = speed * dy / dist
        vz = speed * dz / dist
        duration = dist / speed
        
        # 使用带噪声的速度控制
        label = f'z_leg{idx}'
        noisy_velocity_command(
            client,
            base_vx=vx,
            base_vy=vy,
            base_vz=vz,
            duration_s=duration,
            label=label,
            trajectory=trajectory,
            meta={**meta, 'leg_target': end_wp, 'leg_index': idx}
        )

    # 扫描结束后短暂稳定记录
    client.hoverAsync().join()
    t0 = time.time()
    while time.time() - t0 < 3:
        trajectory.append(record_state(client, command_type='post_z_scan', meta=meta))
        time.sleep(SAMPLE_INTERVAL)

    return trajectory


INTENT_MAP = {
    'takeoff': intent_takeoff,
    'hover': intent_hover,
    'straight_line': intent_straight_line,
    'turn': intent_turn,
    'landing': intent_landing,
    'z_scan': intent_z_scan
}


# ============== 主流程 ==============

def run_once(intent_type, run_number):
    client = airsim.MultirotorClient()
    client.confirmConnection()

    if intent_type not in INTENT_MAP:
        raise ValueError(f'无效意图: {intent_type}')

    # 随机风
    wind_vec = apply_random_wind(client)
    meta = {
        'intent_type': intent_type,
        'run_number': run_number,
        'wind': {'x': wind_vec.x_val, 'y': wind_vec.y_val, 'z': wind_vec.z_val},
        'vel_noise_std': VEL_NOISE_STD
    }

    trajectory = []
    try:
        intent_fn = INTENT_MAP[intent_type]
        trajectory = intent_fn(client, meta)
    finally:
        clear_wind(client)
        client.armDisarm(False)
        client.enableApiControl(False)
    save_run(intent_type, run_number, trajectory, meta)


def main():
    print(f'随机数据集生成，意图: {INTENT_TYPE}, 次数: {RUNS}')
    for i in range(1, RUNS + 1):
        print(f'\n--- 运行 {i}/{RUNS} ---')
        run_once(INTENT_TYPE, i)


if __name__ == '__main__':
    main()
