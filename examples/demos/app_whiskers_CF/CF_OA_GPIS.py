import logging
import sys
import time
import os
import argparse
from datetime import datetime
from pathlib import Path

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from FileLogger import FileLogger
from cflib.utils import uri_helper

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

def unlock_drone(cf):
    cf.param.set_value('app.stateOuterLoop', '1')

def stop_drone(cf):
    cf.param.set_value('app.stateOuterLoop', '2')

def apply_mlp(cf):
    cf.param.set_value('app.statemlp', '1')
def apply_kf(cf):
    cf.param.set_value('app.statekf', '1')
def data_collection_mode(cf):
    cf.param.set_value('app.statemlp', '2')

def exploration_mode(cf):
    cf.param.set_value('app.stateexp', '1')

def exploration_mode(cf):
    cf.param.set_value('app.stateexp', '1')

def exploration_mode_gpis(cf):
    cf.param.set_value('app.stategpis', '1')

def set_initial_params(cf, MIN_THRESHOLD1, MAX_THRESHOLD1, MIN_THRESHOLD2, MAX_THRESHOLD2, maxSpeed, maxTurnRate):
    cf.param.set_value('app.MIN_THRESHOLD1', str(MIN_THRESHOLD1)) 
    cf.param.set_value('app.MAX_THRESHOLD1', str(MAX_THRESHOLD1))
    cf.param.set_value('app.MIN_THRESHOLD2', str(MIN_THRESHOLD2))
    cf.param.set_value('app.MAX_THRESHOLD2', str(MAX_THRESHOLD2))
    cf.param.set_value('app.maxSpeed', str(maxSpeed))
    cf.param.set_value('app.maxTurnRate', str(maxTurnRate))

def post_cal_params(cf, scale_1, scale_2, offset_1, offset_2):
    cf.param.set_value('app.scale_1', str(scale_1)) 
    cf.param.set_value('app.scale_2', str(scale_2))
    cf.param.set_value('app.offset_1', str(offset_1))
    cf.param.set_value('app.offset_2', str(offset_2))

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
    flogger.enableConfig("MLPOUTPUT")
    flogger.enableConfig("StateOuterLoop")
    flogger.enableConfig("GPISLABEL")
    # flogger.enableConfig("motor")
    # flogger.enableConfig("otpos")
    flogger.enableConfig("orientation")
    flogger.enableConfig("laser")
    flogger.enableConfig("KFOUTPUT")

    # # UWB
    # if args["uwb"] == "twr":
    #     flogger.enableConfig("twr")
    # elif args["uwb"] == "tdoa":
    #     print("Needs custom TDoA logging in firmware!")
        # For instance, see here: https://github.com/Huizerd/crazyflie-firmware/blob/master/src/utils/src/tdoa/tdoaEngine.c
        # flogger.enableConfig("tdoa")
    # Flow
    # flogger.enableConfig("laser")
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
        set_initial_params(scf.cf, 30.0, 80.0, 30.0, 80.0, 0.2, 25.0)
        post_cal_params(scf.cf, 1.0492, 1.6958, -53.2529, -119.2660)
        keep_flying = True
        time.sleep(3)
        unlock_drone(scf.cf)
        apply_mlp(scf.cf)
        apply_kf(scf.cf)
        exploration_mode(scf.cf)
        exploration_mode_gpis(scf.cf)
        print("start flying!")
        try:
            while keep_flying:
                command = input("Enter 's' to stop the drone: ").strip().lower()
                if command == 's':
                    print("Stop command received!")
                    keep_flying = False

            while not keep_flying:
                stop_drone(scf.cf) 
                print("Stopping the drone...")
                time.sleep(0.1) 
        except KeyboardInterrupt:
            print('Demo terminated!')
