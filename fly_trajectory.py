import csv

import time
import airsim

# 初始化AirSim客户端
vehicle_name = "AmphiVehicleDrone"
AirSim_client = airsim.MultirotorClient()
AirSim_client.confirmConnection()
AirSim_client.enableApiControl(True, vehicle_name=vehicle_name)
AirSim_client.armDisarm(True, vehicle_name=vehicle_name)
AirSim_client.takeoffAsync(vehicle_name=vehicle_name).join()

# 从CSV文件中读取航点数据
waypoints = []
with open('waypoints.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # 跳过表头
    for row in reader:
        x, y, z = map(float, row)
        waypoints.append(airsim.Vector3r(x, y, z))

# 绘制航点轨迹，用红色线条表示
AirSim_client.simPlotLineStrip(waypoints, color_rgba=[1.0, 0.0, 0.0, 1.0], thickness=10.0, is_persistent=True)
time.sleep(2)  # 等待几秒钟以便查看轨迹

# 让无人机根据航点数据飞行
for point in waypoints:
    AirSim_client.moveToPositionAsync(point.x_val, point.y_val, point.z_val, 10.0, vehicle_name=vehicle_name).join()
    time.sleep(0.1)  # 等待1秒以确保位置已经到达

# 飞行完成，降落
AirSim_client.landAsync(vehicle_name=vehicle_name).join()
AirSim_client.armDisarm(False, vehicle_name=vehicle_name)
AirSim_client.enableApiControl(False, vehicle_name=vehicle_name)
print("Flight completed and drone has landed.")

