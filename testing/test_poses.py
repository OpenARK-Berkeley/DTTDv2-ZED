import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

import os, sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from calculate_extrinsic.CameraOptiExtrinsicCalculator import CameraOptiExtrinsicCalculator
from data_processing.CameraPoseSynchronizer import CameraPoseSynchronizer
from utils.affine_utils import invert_affine, rotvec_trans_from_affine_matrix
from utils.camera_utils import load_intrinsics, load_distortion, load_extrinsics
from utils.frame_utils import calculate_aruco_from_bgr_and_depth, load_bgr, load_depth
from utils.pose_dataframe_utils import convert_pose_df_to_dict

def main():
    scene_dir = os.path.join(dir_path, "..", "extrinsics_scenes/2022-06-17-17-56-49")
    frames_dir = os.path.join(scene_dir, "data")
    pose_csvs = ["camera_poses_synchronized.csv", "camera_poses_synchronized_1.csv", "camera_poses_synchronized_2.csv"]
    pose_csvs = [os.path.join(scene_dir, "camera_poses", c) for c in pose_csvs]

    scene_metadata_file = os.path.join(scene_dir, "scene_meta.yaml")
    with open(scene_metadata_file, 'r') as file:
        scene_metadata = yaml.safe_load(file)

    camera_name = scene_metadata["camera"]
    camera_intrinsic_matrix = load_intrinsics(camera_name)
    camera_distortion_coefficients = load_distortion(camera_name)
    extrinsic = load_extrinsics(camera_name)

    cam_scale = scene_metadata["cam_scale"]

    cps = CameraPoseSynchronizer()

    poses = [cps.load_from_file(f) for f in pose_csvs]
    poses = [convert_pose_df_to_dict(pdf) for pdf in poses]

    frameids = sorted(list(poses[0].keys()))

    coec = CameraOptiExtrinsicCalculator()
    opti_to_aruco = invert_affine(coec.get_aruco_to_opti_transform())

    for i in range(len(poses)):
        cv2.namedWindow("Poses_{0}".format(i))
        cv2.resizeWindow("Poses_{0}".format(i), 640, 360)

    for frame_id in frameids:
        
        opti_poses = [p[frame_id] for p in poses]

        frame = load_bgr(frames_dir, frame_id)
        frames = [np.copy(frame) for _ in range(len(poses))]

        sensor_to_optis = [opti_pose @ invert_affine(extrinsic) for opti_pose in opti_poses]
        sensor_to_arucos = [opti_to_aruco @ sensor_to_opti for sensor_to_opti in sensor_to_optis]
        aruco_to_sensors = [invert_affine(sensor_to_aruco) for sensor_to_aruco in sensor_to_arucos]

        rvecs_and_tvecs = [rotvec_trans_from_affine_matrix(aruco_to_sensor) for aruco_to_sensor in aruco_to_sensors]

        for (rvec, tvec), frame in zip(rvecs_and_tvecs, frames):
            cv2.aruco.drawAxis(frame, camera_intrinsic_matrix, camera_distortion_coefficients, rvec, tvec / 9, 0.01)  # Draw Axis

        frames = [cv2.resize(f, (640, 360)) for f in frames]

        for i, frame in enumerate(frames):
            cv2.imshow("Poses_{0}".format(i), frame)

        cv2.waitKey(15)

    for i in range(len(poses)):
        cv2.destroyWindow("Poses_{0}".format(i))

    diffs_every_frame = []

    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()

    for frame_id in frameids:
        opti_poses = [p[frame_id] for p in poses]

        sensor_to_optis = [opti_pose @ invert_affine(extrinsic) for opti_pose in opti_poses]
        sensor_to_arucos = [opti_to_aruco @ sensor_to_opti for sensor_to_opti in sensor_to_optis]
        aruco_to_sensors = [invert_affine(sensor_to_aruco) for sensor_to_aruco in sensor_to_arucos]

        frame = load_bgr(frames_dir, frame_id)
        depth = load_depth(frames_dir, frame_id)

        aruco_pose = calculate_aruco_from_bgr_and_depth(frame, depth, cam_scale, camera_intrinsic_matrix, camera_distortion_coefficients, aruco_dict, parameters)

        if aruco_pose:  # If there are markers found by detector

            rvec, tvec, _ = aruco_pose

            rvec, tvec = rvec.squeeze(), tvec.squeeze()

            camera_sensor_to_opti_transform = coec.calculate_camera_to_opti_transform(rvec, tvec)

            camera_sensor_to_aruco = opti_to_aruco @ camera_sensor_to_opti_transform
            aruco_to_sensor = invert_affine(camera_sensor_to_aruco)

            diffs = [(np.arccos((np.trace(aruco_to_sensor[:3,:3] @ np.linalg.inv(p[:3,:3])) - 1.) / 2.), np.linalg.norm(aruco_to_sensor[:3,3] - p[:3,3])) for p in aruco_to_sensors]

            diffs_every_frame.append(diffs)

    rot_diffs = [[r for r, t in f] for f in diffs_every_frame]
    trans_diffs = [[t for r, t in f] for f in diffs_every_frame]
    rot_diffs = np.array(rot_diffs)
    trans_diffs = np.array(trans_diffs)

    print(rot_diffs.shape, trans_diffs.shape)

    for i in range(len(poses)):
        r = rot_diffs[:,i]
        plt.plot(np.arange(len(r)), r, label="rot diffs {0}".format(i), linewidth=1)

    plt.yscale('log')
    plt.legend()
    plt.show()
    plt.clf()

    for i in range(len(poses)):
        t = trans_diffs[:,i]
        plt.plot(np.arange(len(t)), t, label="trans diffs {0}".format(i), linewidth=1)

    plt.yscale('log')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()