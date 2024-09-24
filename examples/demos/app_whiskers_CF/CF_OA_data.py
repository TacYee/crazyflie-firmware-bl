import logging
import sys
from threading import Thread
import time
import os
import argparse
from datetime import datetime
from pathlib import Path
import motioncapture
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from FileLogger import FileLogger
from cflib.utils import uri_helper
import math

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

# The host name or ip address of the mocap system
host_name = '192.168.209.81'

# The type of the mocap system
# Valid options are: 'vicon', 'optitrack', 'optitrack_closed_source', 'qualisys', 'nokov', 'vrpn', 'motionanalysis'
mocap_system_type = 'optitrack'

body_name = "Chaoxiang"  # Replace with your actual body name


# True: send position and orientation; False: send position only
send_full_pose = True

#obtain position and rotation from mocap
class MocapWrapper(Thread):
    def __init__(self, body_name, file_logger):
        super().__init__()
        self.body_name = body_name
        self.file_logger = file_logger
        self.on_pose = None
        self._stay_open = True
        self.data_dict = {
            "pos_x": 0.0, 
            "pos_y": 0.0, 
            "pos_z": 0.0, 
            "yaw": 0.0  # 只保留 yaw
        }
        self.start()

    def quaternion_to_yaw(self, x, y, z, w):
        """从四元数计算 yaw"""
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return yaw

    def run(self):
        mc = motioncapture.connect(mocap_system_type, {'hostname': host_name})
        while self._stay_open:
            mc.waitForNextFrame()
            for name, obj in mc.rigidBodies.items():
                if name == self.body_name:
                    if self.on_pose:
                        pos = obj.position
                        rot = obj.rotation
                        # 计算 yaw
                        yaw = self.quaternion_to_yaw(rot[0], rot[1], rot[2], rot[3])
                        # 更新数据字典
                        self.data_dict = {
                            "pos_x": pos[0],
                            "pos_y": pos[1],
                            "pos_z": pos[2],
                            "yaw": yaw
                        }
                    time.sleep(0.02)  # 20ms

    def sync_mocap_data(self):
        """将外部 mocap 数据同步到 FileLogger 中"""
        self.file_logger.registerData("mocap", self.data_dict)

def unlock_drone(cf):
    cf.param.set_value('app.stateOuterLoop', '1')

def stop_drone(cf):
    cf.param.set_value('app.stateOuterLoop', '2')

def apply_mlp(cf):
    cf.param.set_value('app.statemlp', '1')

def set_initial_params(cf, MIN_THRESHOLD1, MAX_THRESHOLD1, MIN_THRESHOLD2, MAX_THRESHOLD2, maxSpeed, maxTurnRate):
    cf.param.set_value('app.MIN_THRESHOLD1', str(MIN_THRESHOLD1)) 
    cf.param.set_value('app.MAX_THRESHOLD1', str(MAX_THRESHOLD1))
    cf.param.set_value('app.MIN_THRESHOLD2', str(MIN_THRESHOLD2))
    cf.param.set_value('app.MAX_THRESHOLD2', str(MAX_THRESHOLD2))
    cf.param.set_value('app.maxSpeed', str(maxSpeed))
    cf.param.set_value('app.maxTurnRate', str(maxTurnRate))

def get_filename():
    fileroot = args["fileroot"] 
        
    if args["filename"] is not None:
        name = args["filename"] + ".csv"
        fname = os.path.normpath(os.path.join(
            os.getcwd(), fileroot, name))
        i = 0
        while os.path.isfile(fname):
            i = i + 1
            name = args["filename"] + "_" + str(i) + ".csv"
            fname = os.path.normpath(os.path.join(
                os.getcwd(), fileroot, name))

    else:
        # get relevant arguments
        # keywords = args["keywords"]
        # estimator = args["estimator"]
        # uwb = args["uwb"]
        # optitrack = args["optitrack"]
        # trajectory = args["trajectory"]

        # Date
        date = datetime.today().strftime(r"%Y-%m-%d+%H:%M:%S")

        # Additional keywords
        # if keywords is not None:
        #     keywords = "+" + "+".join(keywords)
        # else:
        #     keywords = ""

        # Options
        # if optitrack == "logging":
        #     options = f"optitracklog"
        # elif optitrack == "state":
        #     options = f"optitrackstate"
        # else:
        options = f""

        # Join
        name = "{}+{}.csv".format(date, options)
        fname = os.path.normpath(os.path.join(os.getcwd(), fileroot, name))
    return fname

def setup_logger():
    # 创建文件和日志配置
    Path(args["fileroot"]).mkdir(exist_ok=True)
    log_file = get_filename()

    print(f"Log location: {log_file}")

    # 初始化 FileLogger
    logconfig = args["logconfig"]
    flogger = FileLogger(cf, logconfig, log_file)

    # 启用 CF 类型的日志配置
    flogger.enableConfig("state")
    flogger.enableConfig("whisker1")
    flogger.enableConfig("whisker2")
    flogger.enableConfig("PreWhisker1")
    flogger.enableConfig("PreWhisker2")
    flogger.enableConfig("StateOuterLoop")
    flogger.enableConfig("orientation")

    # 启动 mocap 数据同步
    flogger.enableConfig("mocap")  # 启用 mocap 配置
    mocap_logger = MocapWrapper(body_name, flogger)

    # 启动日志记录
    flogger.start()

    # 每 20ms 同步一次 mocap 数据
    while flogger.is_connected:
        mocap_logger.sync_mocap_data()
        time.sleep(0.02)  # 20ms

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--fileroot", type=str, required=True)
    parser.add_argument("--logconfig", type=str, required=True)
    parser.add_argument("--filename", type=str, default=None)
    args = vars(parser.parse_args())
    
    # Initialize the low-level drivers (don't list the debug drivers)
    cflib.crtp.init_drivers()

    cf = Crazyflie(rw_cache='./cache')
    with SyncCrazyflie(URI, cf=cf) as scf:
        cf.platform.send_arming_request(True)
        filelogger=setup_logger()
        keep_flying = True
        time.sleep(3)
        set_initial_params(scf.cf, 30.0, 100.0, 50.0, 130.0, 0.2, 25.0)
        unlock_drone(scf.cf)
        print("start flying!")
        try:
            while keep_flying:
                command = input("Enter 's' to stop the drone: ").strip().lower()
                if command == 's':
                    print("Stop command received!")
                    keep_flying = False

            # 当 keep_flying 变为 False，开始执行停止逻辑
            while not keep_flying:
                stop_drone(scf.cf)  # 不断调用 stop_drone 函数
                print("Stopping the drone...")
                time.sleep(0.1)  # 给些延时，避免过度调用
        except KeyboardInterrupt:
            print('Demo terminated!')
