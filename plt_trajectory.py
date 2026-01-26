import sys

import time
import airsim
import pygame
import numpy as np
import asyncio
import csv

# 初始化Pygame
pygame.init()
screen = pygame.display.set_mode((320, 240))
pygame.display.set_caption('keyboard ctrl')
screen.fill((0, 0, 0))

vehicle_name = "AmphiVehicleDrone"
AirSim_client = airsim.MultirotorClient()
AirSim_client.confirmConnection()
AirSim_client.enableApiControl(True, vehicle_name=vehicle_name)
AirSim_client.armDisarm(True, vehicle_name=vehicle_name)
AirSim_client.takeoffAsync(vehicle_name=vehicle_name).join()

# 基础的控制速度(m/s)
vehicle_velocity = 2.0
# 设置临时加速比例
speedup_ratio = 10.0
# 用来设置临时加速
speedup_flag = False

# 基础的偏航速率
vehicle_yaw_rate = 5.0

# 记录无人机的位置，用于绘制轨迹
trajectory = []
last_position = None

# 创建CSV文件并写入表头
with open('waypoints.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['x', 'y', 'z'])  # 表头


async def draw_trajectory():
    while True:
        if len(trajectory) > 1:
            AirSim_client.simPlotLineStrip(trajectory, color_rgba=[1.0, 0.0, 0.0, 1.0], thickness=10.0, is_persistent=True)
        await asyncio.sleep(1.2)  # 每1.2秒绘制一次


async def main():
    global last_position

    while True:
        yaw_rate = 0.0
        velocity_x = 0.0
        velocity_y = 0.0
        velocity_z = 0.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        scan_wrapper = pygame.key.get_pressed()

        # 按下空格键加速10倍
        if scan_wrapper[pygame.K_SPACE]:
            scale_ratio = speedup_ratio
        else:
            scale_ratio = 1.0

        # 根据 'A' 和 'D' 按键来设置偏航速率变量
        if scan_wrapper[pygame.K_a] or scan_wrapper[pygame.K_d]:
            yaw_rate = (scan_wrapper[pygame.K_d] - scan_wrapper[pygame.K_a]) * scale_ratio * vehicle_yaw_rate

        # 根据 'UP' 和 'DOWN' 按键来设置pitch轴速度变量(NED坐标系，x为机头向前)
        if scan_wrapper[pygame.K_UP] or scan_wrapper[pygame.K_DOWN]:
            velocity_x = (scan_wrapper[pygame.K_UP] - scan_wrapper[pygame.K_DOWN]) * scale_ratio * vehicle_velocity

        # 根据 'LEFT' 和 'RIGHT' 按键来设置roll轴速度变量(NED坐标系，y为正右方)
        if scan_wrapper[pygame.K_LEFT] or scan_wrapper[pygame.K_RIGHT]:
            velocity_y = -(scan_wrapper[pygame.K_LEFT] - scan_wrapper[pygame.K_RIGHT]) * scale_ratio * vehicle_velocity

        # 根据 'W' 和 'S' 按键来设置z轴速度变量(NED坐标系，z轴向上为负)
        if scan_wrapper[pygame.K_w] or scan_wrapper[pygame.K_s]:
            velocity_z = -(scan_wrapper[pygame.K_w] - scan_wrapper[pygame.K_s]) * scale_ratio * vehicle_velocity

        # 设置速度控制以及设置偏航控制
        AirSim_client.moveByVelocityBodyFrameAsync(vx=velocity_x, vy=velocity_y, vz=velocity_z, duration=1,
                                                   yaw_mode=airsim.YawMode(True, yaw_or_rate=yaw_rate), vehicle_name=vehicle_name)

        # 获取无人机的位置并记录到轨迹中
        position = AirSim_client.getMultirotorState(vehicle_name=vehicle_name).kinematics_estimated.position
        current_position = airsim.Vector3r(position.x_val, position.y_val, position.z_val)

        # 仅在位置发生显著变化时记录点
        if last_position is None or (abs(current_position.x_val - last_position.x_val) > 0.1 or
                                     abs(current_position.y_val - last_position.y_val) > 0.1 or
                                     abs(current_position.z_val - last_position.z_val) > 0.1):
            trajectory.append(current_position)
            last_position = current_position

            # 将航点数据写入CSV文件
            with open('waypoints.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([current_position.x_val, current_position.y_val, current_position.z_val])

        if scan_wrapper[pygame.K_ESCAPE]:
            pygame.quit()
            sys.exit()

        await asyncio.sleep(0.4)  # 每次循环等待400毫秒


# 启动异步绘制任务和主任务
loop = asyncio.get_event_loop()
loop.create_task(draw_trajectory())
loop.run_until_complete(main())