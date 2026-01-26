import airsim
import json
import os
from datetime import datetime
import time

# ==================== 配置部分 ====================
# 选择要生成的意图类型：'takeoff', 'hover', 'straight_line', 'turn', 'landing'
INTENT_TYPE = 'takeoff'

# 数据保存目录
DATA_DIR = './flight_data'
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# 采样频率（Hz）
SAMPLE_RATE = 10
SAMPLE_INTERVAL = 1.0 / SAMPLE_RATE

# ==================== 数据记录函数 ====================
def record_state(client, target_pos=None, command_type=None):
    """记录当前无人机状态"""
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
        'command_type': command_type
    }
    return record

def save_trajectory(trajectory_data, intent_type, run_number):
    """保存轨迹数据到JSON文件"""
    # 创建目录结构
    intent_dir = os.path.join(DATA_DIR, intent_type)
    os.makedirs(intent_dir, exist_ok=True)
    
    # 生成文件名
    filename = os.path.join(intent_dir, f'{intent_type}_{TIMESTAMP}_run{run_number}.json')
    
    # 保存数据
    with open(filename, 'w') as f:
        json.dump({
            'intent_type': intent_type,
            'timestamp': TIMESTAMP,
            'run_number': run_number,
            'sample_rate': SAMPLE_RATE,
            'trajectory': trajectory_data
        }, f, indent=2)
    
    print(f'✓ 数据已保存: {filename}')
    return filename

# ==================== 意图1: 起飞 ====================
def takeoff_intent(client):
    """起飞意图：从地面起飞到指定高度"""
    print('\n=== 起飞意图 ===')
    trajectory = []
    
    print('准备起飞...')
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    
    print('开始起飞')
    client.takeoffAsync().join()
    
    # 记录起飞过程中的状态（持续5秒）
    start_time = time.time()
    while time.time() - start_time < 5:
        record = record_state(client, target_pos=[0, 0, -10], command_type='takeoff')
        trajectory.append(record)
        time.sleep(SAMPLE_INTERVAL)
    
    return trajectory

# ==================== 意图2: 悬停 ====================
def hover_intent(client):
    """悬停意图：保持当前位置"""
    print('\n=== 悬停意图 ===')
    trajectory = []
    
    # 先飞到一个高度
    client.takeoffAsync().join()
    time.sleep(1)
    
    print('开始悬停')
    client.hoverAsync().join()
    
    # 记录悬停过程（持续10秒）
    start_time = time.time()
    current_pos = client.getMultirotorState().kinematics_estimated.position
    target_pos = [current_pos.x_val, current_pos.y_val, current_pos.z_val]
    
    while time.time() - start_time < 10:
        record = record_state(client, target_pos=target_pos, command_type='hover')
        trajectory.append(record)
        time.sleep(SAMPLE_INTERVAL)
    
    return trajectory

# ==================== 意图3: 直线飞行 ====================
def straight_line_intent(client):
    """直线飞行意图：从A点飞到B点"""
    print('\n=== 直线飞行意图 ===')
    trajectory = []
    
    # 先起飞
    client.takeoffAsync().join()
    time.sleep(1)
    
    # 定义飞行参数
    start_pos = [0, 0, -10]
    end_pos = [20, 0, -10]
    velocity = 5  # m/s
    
    print(f'从 {start_pos} 直线飞向 {end_pos}，速度 {velocity} m/s')
    
    # 开始飞行
    start_time = time.time()
    client.moveToPositionAsync(end_pos[0], end_pos[1], end_pos[2], velocity).join()
    
    # 在飞行过程中采样（由于已经.join()，这里记录到达后的状态）
    while time.time() - start_time < 15:
        record = record_state(client, target_pos=end_pos, command_type='straight_line')
        trajectory.append(record)
        time.sleep(SAMPLE_INTERVAL)
    
    return trajectory

# ==================== 意图4: 转弯 ====================
def turn_intent(client):
    """转弯意图：改变方向飞行"""
    print('\n=== 转弯意图 ===')
    trajectory = []
    
    # 先起飞到合适高度
    client.takeoffAsync().join()
    time.sleep(1)
    
    # 第一段：往Y正方向飞
    print('第一段: 飞向 (0, 20, -10)')
    client.moveToPositionAsync(0, 20, -10, 5).join()
    
    # 记录转弯过程
    print('第二段: 转弯飞向 (20, 20, -10)')
    start_time = time.time()
    client.moveToPositionAsync(20, 20, -10, 5).join()
    
    # 记录转弯中和转弯后的状态
    while time.time() - start_time < 15:
        record = record_state(client, target_pos=[20, 20, -10], command_type='turn')
        trajectory.append(record)
        time.sleep(SAMPLE_INTERVAL)
    
    return trajectory

# ==================== 意图5: 降落 ====================
def landing_intent(client):
    """降落意图：从空中降落到地面"""
    print('\n=== 降落意图 ===')
    trajectory = []
    
    # 先起飞
    client.takeoffAsync().join()
    time.sleep(2)
    
    print('开始降落')
    client.landAsync().join()
    
    # 记录降落过程（持续10秒）
    start_time = time.time()
    while time.time() - start_time < 10:
        record = record_state(client, target_pos=[0, 0, 0], command_type='landing')
        trajectory.append(record)
        time.sleep(SAMPLE_INTERVAL)
    
    return trajectory

# ==================== 主程序 ====================
def main():
    # 连接客户端
    client = airsim.MultirotorClient()
    client.confirmConnection()
    
    # 意图操作映射
    intent_operations = {
        'takeoff': takeoff_intent,
        'hover': hover_intent,
        'straight_line': straight_line_intent,
        'turn': turn_intent,
        'landing': landing_intent
    }
    
    # 检查意图类型是否有效
    if INTENT_TYPE not in intent_operations:
        print(f'❌ 意图类型 "{INTENT_TYPE}" 不存在')
        print(f'可用的意图: {list(intent_operations.keys())}')
        return
    
    print(f'当前生成数据集: {INTENT_TYPE}')
    print(f'采样频率: {SAMPLE_RATE} Hz')
    print(f'数据将保存到: {DATA_DIR}/{INTENT_TYPE}/')
    
    try:
        # 执行对应的意图操作
        operation = intent_operations[INTENT_TYPE]
        trajectory = operation(client)
        
        # 保存数据
        if trajectory:
            save_trajectory(trajectory, INTENT_TYPE, 1)
            print(f'✓ 成功记录 {len(trajectory)} 个数据点')
        else:
            print('❌ 没有记录到数据')
    
    except Exception as e:
        print(f'❌ 错误: {e}')
    
    finally:
        # 降落并断开连接
        print('\n清理中...')
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        print('✓ 已断开连接')

if __name__ == '__main__':
    main()
