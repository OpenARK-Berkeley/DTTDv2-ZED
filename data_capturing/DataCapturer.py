"""
Based on Weiyu and Seth's code to capture a scene.
A scene consists of a calibration phase and a data capture scene.
The calibration phase should involve a sequnce of frames with the camera observing the ARUCO marker.
The data capture phase should follow the calibration phase in one continuous frame stream.

Press (c) to start calibration phase. 
Press (d) to start data capture phase.
Press (q) to quit.

"""

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
            runtime_parameters = sl.RuntimeParameters(enable_fill_mode=True)
            # point cloud and depth are aligned on the left image
            left_image = sl.Mat() # H * W * 4 (R,G,B,A)
            point_cloud = sl.Mat() # H * W * 4 (X,Y,Z,?) (mm, as specified above in init)
            depth = sl.Mat() # H * W (mm)
            if not (self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS):
                return None
            self.zed.retrieve_image(left_image, sl.VIEW.LEFT)
            self.zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)            
            color_timestamp_usec = self.zed.get_timestamp(sl.TIME_REFERENCE.CURRENT).get_microseconds()
            color = np.copy(left_image.get_data()) # get numpy array
            transformed_depth = np.copy(np.round(depth.get_data()).astype(np.uint16))
            CaptureData = NamedTuple('CaptureData', color=np.ndarray, color_timestamp_usec=int, transformed_depth=np.ndarray)
            return CaptureData(color, color_timestamp_usec, transformed_depth)
            
    def stop_camera(self):
        if self.device_type == "azure_kinect":
            self.k4a.stop()
        elif self.device_type == "zed_2":
            self.zed.close()

class DataCapturer():
    def __init__(self, scene_dir, camera_name, device_type="azure_kinect"):
        self.scene_dir = scene_dir
        if not os.path.isdir(scene_dir):
            os.mkdir(scene_dir)
        else:
            print("Warning! Scene already exists. May overwrite data.")

        self.camera_name = camera_name
        self.capture_device = DataCaptureDevice(device_type=device_type)
        #camera needs to exist
        dir_path = os.path.dirname(os.path.realpath(__file__))
        assert(os.path.isdir(os.path.join(dir_path, "..", "cameras", camera_name)))

    def collect_and_write_data(self, data_file, frame_id, frames_dir):
        capture = self.capture_device.get_capture()
        cur_timestamp = capture.color_timestamp_usec / 1000000.

        color_image = capture.color[:,:,:3]
        depth_image = capture.transformed_depth

        color_image = np.ascontiguousarray(color_image)

        cv2.imshow("Color Image", color_image)
        cv2.imshow("Depth Image", depth_image)

        write_bgr(frames_dir, frame_id, color_image, "png")
        write_depth(frames_dir, frame_id, depth_image)

        data_file.write("{0},{1}\n".format(frame_id, cur_timestamp))

    def start_capture(self):   

        # state
        calibration_start_frame_id = -1
        calibration_end_frame_id = -1
        capture_start_frame_id = -1
        frame_id = 0

        # start camera
        self.capture_device.start_camera()

        # output
        data_file = open(os.path.join(self.scene_dir, "camera_data.csv"), "w")
        data_file.write("Frame,Timestamp\n")

        meta_file = open(os.path.join(self.scene_dir, "camera_time_break.csv"), "w")
        meta_file.write("Calibration Start ID,Calibration End ID,Capture Start ID\n")

        scene_meta_file = os.path.join(self.scene_dir, "scene_meta.yaml")
        scene_meta = {}
        scene_meta['cam_scale'] = 0.001
        scene_meta['camera'] = self.camera_name
        with open(scene_meta_file, "w") as f:
            yaml.dump(scene_meta, f)

        write_static_intrinsic(self.camera_name, self.scene_dir, raw=True)
        write_static_distortion(self.camera_name, self.scene_dir, raw=True)

        frames_dir = os.path.join(self.scene_dir, "data_raw")
        if not os.path.isdir(frames_dir):
            os.mkdir(frames_dir)

        camera_poses_dir = os.path.join(self.scene_dir, "camera_poses")
        if not os.path.isdir(camera_poses_dir):
            os.mkdir(camera_poses_dir)

        cv2.namedWindow("Color Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth Image", cv2.WINDOW_NORMAL)

        #play sound when state transitions
        mixer.init() 
        sound_file = os.path.join(dir_path, "sound.mp3")
        sound=mixer.Sound(sound_file)

        while True:
            if cv2.waitKey(1) == ord(CALIBRATION_KEY_START) and calibration_start_frame_id == -1: 
                print("Calibration start frame_id is {}".format(frame_id))
                sound.play()
                calibration_start_frame_id = frame_id
            elif cv2.waitKey(1) == ord(CALIBRATION_KEY_END) and calibration_end_frame_id == -1:
                if calibration_start_frame_id == -1:
                    print("Please start calibration phase before ending it.")
                    continue
                print("Calibration end frame_id is {}".format(frame_id))
                sound.play()
                calibration_end_frame_id = frame_id
            elif cv2.waitKey(1) == ord(CAPTURE_KEY_START) and capture_start_frame_id == -1:
                if calibration_end_frame_id == -1:
                    print("Please perform a calibration phase prior to capturing data")
                    continue
                print("Capturing start img_id is {}".format(frame_id))
                sound.play()
                capture_start_frame_id = frame_id
            elif cv2.waitKey(1) == ord(QUIT_KEY):
                print("Finished Capture")
                if calibration_start_frame_id == -1 or calibration_end_frame_id == -1 or capture_start_frame_id == -1:
                    print("Warning!! Didn't perform a calibration and capture phase.")
                break

            if (calibration_start_frame_id > -1 and calibration_end_frame_id == -1) or capture_start_frame_id > -1:
                self.collect_and_write_data(data_file, frame_id, frames_dir)
                frame_id += 1

        meta_file.write("{0},{1},{2}\n".format(calibration_start_frame_id, calibration_end_frame_id, capture_start_frame_id))

        meta_file.close()
        data_file.close()
        self.capture_device.stop_camera()

        print("All data saved!")
        print("Run process_data to perform synchronization and extract capture phase.")
        print("Additionally, please fill scene_meta.yaml with the objects that are in this scene.")

    @staticmethod
    def capture_single_frame(capture_device: DataCaptureDevice):
        # start Azure Kinect camera
        capture_device.start_camera()

        print("Press enter to capture.")

        while True:
            capture = capture_device.get_capture()

            color_image = capture.color[:,:,:3]
            depth_image = capture.transformed_depth

            color_image = np.ascontiguousarray(color_image)

            cv2.imshow("Color Image", color_image)
            cv2.imshow("Depth Image", depth_image)

            # Enter
            if cv2.waitKey(1) == 13:
                break
        capture_device.stop_camera()
        return color_image, depth_image
