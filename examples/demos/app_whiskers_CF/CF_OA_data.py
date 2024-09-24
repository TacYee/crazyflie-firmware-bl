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
    def __init__(self, body_name):
        Thread.__init__(self)

        self.body_name = body_name
        self.on_pose = None
        self._stay_open = True

        self.start()

    def close(self):
        self._stay_open = False

    def run(self):
        mc = motioncapture.connect(mocap_system_type, {'hostname': host_name})
        while self._stay_open:
            mc.waitForNextFrame()
            for name, obj in mc.rigidBodies.items():
                if name == self.body_name:
                    if self.on_pose:
                        pos = obj.position
                        #call back on pose
                        self.on_pose([pos[0], pos[1], pos[2], obj.rotation])
                    time.sleep(0.02)

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
    # Create directory if not there
    Path(args["fileroot"]).mkdir(exist_ok=True)
        
    # Create filename from options and date
    log_file = get_filename()

    print(f"Log location: {log_file}")


    # Logger setup
    logconfig = args["logconfig"]
    flogger = FileLogger(cf, logconfig, log_file)
    # flogger3 = FileLogger(cf, logconfig, log_file3)

    # Enable log configurations based on system setup:
    # Defaults
    # flogger.enableConfig("attitude")
    # flogger.enableConfig("gyros")
    # flogger.enableConfig("acc")
    flogger.enableConfig("state")
    flogger.enableConfig("whisker1")
    flogger.enableConfig("whisker2")
    flogger.enableConfig("PreWhisker1")
    flogger.enableConfig("PreWhisker2")
    flogger.enableConfig("orientation")
    flogger.enableConfig("mocap") 

    # # UWB
    # if args["uwb"] == "twr":
    #     flogger.enableConfig("twr")
    # elif args["uwb"] == "tdoa":
    #     print("Needs custom TDoA logging in firmware!")
        # For instance, see here: https://github.com/Huizerd/crazyflie-firmware/blob/master/src/utils/src/tdoa/tdoaEngine.c
        # flogger.enableConfig("tdoa")
    # Flow
    if args["flow"]:
        flogger.enableConfig("laser")
    #     flogger.enableConfig("flow")
    # OptiTrack
    # if args["optitrack"] != "none":
    #     flogger.enableConfig("kalman")
    flogger.start()
    # flogger2.start()
    # flogger3.start()
    # # Estimator
    # if args["estimator"] == "kalman":
    #     flogger.enableConfig("kalman")
    # Initialize MocapWrapper to log position and rotation
    mocap_wrapper = MocapWrapper(body_name)

    # Define a callback to register position and rotation data into the logger
    def handle_pose(pose_data):
        # Register mocap data in the FileLogger
        data_dict = {
            "pos_x": pose_data[0],  # X position
            "pos_y": pose_data[1],  # Y position
            "pos_z": pose_data[2],  # Z position
            "rot_x": pose_data[3][0],  # Rotation X (Quaternion)
            "rot_y": pose_data[3][1],  # Rotation Y (Quaternion)
            "rot_z": pose_data[3][2],  # Rotation Z (Quaternion)
            "rot_w": pose_data[3][3]   # Rotation W (Quaternion)
        }
        flogger.registerData("mocap", data_dict)

    # Register the pose callback
    mocap_wrapper.on_pose = handle_pose

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
