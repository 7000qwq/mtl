import airsim
# connect to the AirSim simulator
client = airsim.MultirotorClient()

client.confirmConnection()
# get control
client.enableApiControl(True)
# unlock
client.armDisarm(True)

# Async methods returns Future. Call join() to wait for task to complete.
client.takeoffAsync().join()

client.moveToZAsync(-10, 3).join()               # 以1m/s的速度飞到10米高度
client.moveToPositionAsync(5, 0, -10, 3).join()  # 以1m/s的速度飞到(5, 0, -10)处
client.moveToPositionAsync(5, 5, -10, 3).join()
client.moveToPositionAsync(0, 5, -10, 3).join()
client.moveToPositionAsync(0, 0, -10, 3).join()

client.landAsync().join()

# lock
client.armDisarm(False)
# release control
client.enableApiControl(False)