import airsim
import pprint

def print_state(client):
    state = client.getMultirotorState()
    s = pprint.pformat(state)
    print("state: %s" % s)

    imu_data = client.getImuData()
    s = pprint.pformat(imu_data)
    print("imu_data: %s" % s)

    barometer_data = client.getBarometerData()
    s = pprint.pformat(barometer_data)
    print("barometer_data: %s" % s)

    magnetometer_data = client.getMagnetometerData()
    s = pprint.pformat(magnetometer_data)
    print("magnetometer_data: %s" % s)

    gps_data = client.getGpsData()
    s = pprint.pformat(gps_data)
    print("gps_data: %s" % s)

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

print_state(client)
print('Takeoff')
client.takeoffAsync().join()

while True:
    print_state(client)
    print('Go to (-10, 10, -10) at 5 m/s')
    client.moveToPositionAsync(-10, 10, -10, 5).join()
    client.hoverAsync().join()
    print_state(client)
    print('Go to (0, 10, 0) at 5 m/s')
    client.moveToPositionAsync(0, 10, 0, 5).join()

'''
state: <MultirotorState> {   'collision': <CollisionInfo> {   'has_collided': False,
    'impact_point': <Vector3r> {   'x_val': 0.0,
    'y_val': 0.0,
    'z_val': 0.0},
    'normal': <Vector3r> {   'x_val': 0.0,
    'y_val': 0.0,
    'z_val': 0.0},
    'object_id': -1,
    'object_name': '',
    'penetration_depth': 0.0,
    'position': <Vector3r> {   'x_val': 0.0,
    'y_val': 0.0,
    'z_val': 0.0},
    'time_stamp': 0},
    'gps_location': <GeoPoint> {   'altitude': 132.059326171875,
    'latitude': 47.641376817144526,
    'longitude': -122.14003182238743},
    'kinematics_estimated': <KinematicsState> {   'angular_acceleration': <Vector3r> {   'x_val': 0.6964532732963562,     
    'y_val': 18.89827537536621,
    'z_val': -0.001737329876050353},
    'angular_velocity': <Vector3r> {   'x_val': -0.07973320037126541,
    'y_val': -1.4991852045059204,
    'z_val': -0.0025795092806220055},
    'linear_acceleration': <Vector3r> {   'x_val': 1.2079765796661377,
    'y_val': -0.004894399084150791,
    'z_val': 5.393001079559326},
    'linear_velocity': <Vector3r> {   'x_val': -3.590069532394409,
    'y_val': -0.007754978258162737,
    'z_val': -2.618924379348755},
    'orientation': <Quaternionr> {   'w_val': 0.9916215538978577,
    'x_val': -0.006011309567838907,
    'y_val': -0.12343932688236237,
    'z_val': 0.037594329565763474},
    'position': <Vector3r> {   'x_val': -10.148991584777832,
    'y_val': 10.007292747497559,
    'z_val': -10.067095756530762}},
    'landed_state': 1,
    'rc_data': <RCData> {   'is_initialized': False,
    'is_valid': False,
    'left_z': 0.0,
    'pitch': 0.0,
    'right_z': 0.0,
    'roll': 0.0,
    'switches': 0,
    'throttle': 0.0,
    'timestamp': 0,
    'vendor_id': '',
    'yaw': 0.0},
    'timestamp': 1765205776830175744}
imu_data: <ImuData> {   'angular_velocity': <Vector3r> {   'x_val': -0.07745076715946198,
    'y_val': -1.4984149932861328,
    'z_val': -0.0037158315535634756},
    'linear_acceleration': <Vector3r> {   'x_val': 0.12458030879497528,
    'y_val': 0.0026923026889562607,
    'z_val': -4.57196044921875},
    'orientation': <Quaternionr> {   'w_val': 0.9916215538978577,
    'x_val': -0.006011309567838907,
    'y_val': -0.12343932688236237,
    'z_val': 0.037594329565763474},
    'time_stamp': 1765205776830175744}
barometer_data: <BarometerData> {   'altitude': 132.11061096191406,
    'pressure': 99747.3203125,
    'qnh': 1013.25,
    'time_stamp': 1765205776818175488}
magnetometer_data: <MagnetometerData> {   'magnetic_field_body': <Vector3r> {   'x_val': 0.325639009475708,
    'y_val': 0.014844315126538277,
    'z_val': 0.29610684514045715},
    'magnetic_field_covariance': [   ],
    'time_stamp': 1765205776818175488}
gps_data: <GpsData> {   'gnss': <GnssReport> {   'eph': 0.10000111907720566,
    'epv': 0.10000111907720566,
    'fix_type': 3,
    'geo_point': <GeoPoint> {   'altitude': 131.33143615722656,
    'latitude': 47.64138431850293,
    'longitude': -122.14003180348091},
    'time_utc': 1765205776602171,
    'velocity': <Vector3r> {   'x_val': -3.6850922107696533,
    'y_val': -0.0039038187824189663,
    'z_val': -3.474977493286133}},
    'is_valid': True,
    'time_stamp': 1765205776602170880}

'''