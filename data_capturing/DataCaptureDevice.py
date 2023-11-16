import cv2
import numpy as np
from pygame import mixer
from typing import NamedTuple
import yaml
import sys
sys.path.append("../")

import ogl_viewer.viewer as gl

try:
    from pyk4a import PyK4A
except ImportError:
    print("WARNING: PyK4A to support azure_kinect import failed!")
    pass

try:
    import pyzed.sl as sl
except ImportError:
    print("WARNING: PyZED.sl to support azure_kinect import failed!")
    pass

import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from utils.camera_utils import write_static_intrinsic, write_static_distortion
from utils.frame_utils import write_bgr, write_depth

CALIBRATION_KEY_START = 'c'
CALIBRATION_KEY_END = 'p'
CAPTURE_KEY_START = 'd'
QUIT_KEY = 'q'

class DataCaptureDevice():
    def __init__(self, device_type):
        self.device_type = device_type
        if self.device_type == "azure_kinect":
            self.k4a = PyK4A()
        elif self.device_type == "zed_2":
            self.zed = sl.Camera()
        else:
            print("Device type is not recognized: available types are \"azure_kinect\", \"zed_2\".")
            raise NotImplementedError
    def start_camera(self):
        if self.device_type == "azure_kinect":
            self.k4a.start()
        elif self.device_type == "zed_2":
            init = sl.InitParameters(depth_mode=sl.DEPTH_MODE.ULTRA,
                                 coordinate_units=sl.UNIT.MILLIMETER,
                                 coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP,
                                 camera_resolution=sl.RESOLUTION.HD2K)
            status = self.zed.open(init)
            if status != sl.ERROR_CODE.SUCCESS:
                print(repr(status))
                exit()
            else:
                print("ZED Camera ON...")
    def get_capture(self):
        if self.device_type == "azure_kinect":
            return self.k4a.get_capture()
        elif self.device_type == "zed_2":
            runtime_parameters = sl.RuntimeParameters(enable_fill_mode=False)
            # point cloud and depth are aligned on the left image
            left_image = sl.Mat() # H * W * 4 (R,G,B,A)
            # point_cloud = sl.Mat() # H * W * 4 (X,Y,Z,?) (mm, as specified above in init)
            depth = sl.Mat() # H * W (mm)
            confidence_map = sl.Mat() # H * W
            if not (self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS):
                return None
            self.zed.retrieve_image(left_image, sl.VIEW.LEFT)
            color_timestamp_usec = self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_microseconds()
            self.zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            self.zed.retrieve_measure(confidence_map, sl.MEASURE.CONFIDENCE)
            color = np.copy(left_image.get_data()) # get numpy array
            transformed_depth = np.copy(np.round(depth.get_data()).astype(np.uint16))
            depth_confidence = np.copy(np.round(confidence_map.get_data()).astype(np.uint16))
            depth_confidence = 100 - depth_confidence # pixels with an original value of 100 were least trusted
            CaptureData = NamedTuple('CaptureData', color=np.ndarray, color_timestamp_usec=int, transformed_depth=np.ndarray, depth_confidence=np.ndarray)
            return CaptureData(color, color_timestamp_usec, transformed_depth, depth_confidence)

    def stop_camera(self):
        if self.device_type == "azure_kinect":
            self.k4a.stop()
        elif self.device_type == "zed_2":
            self.zed.close()