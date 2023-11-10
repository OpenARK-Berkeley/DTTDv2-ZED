import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import yaml

import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from utils.affine_utils import invert_affine, affine_matrix_from_rotvec_trans, average_quaternion, rotvec_trans_from_affine_matrix
from utils.camera_utils import load_frame_intrinsics, load_frame_distortions, write_extrinsics
from utils.frame_utils import calculate_aruco_from_bgr_and_depth, load_bgr, load_depth

class CameraOptiExtrinsicCalculator():
    def __init__(self):

        dir_path = os.path.dirname(os.path.realpath(__file__))

        # solve for aruco rotation using marker positions
        with open(os.path.join(dir_path, "aruco_corners.yaml")) as f:
            aruco_corners = yaml.safe_load(f)

        aruco_x = np.array(aruco_corners['quad1']) - np.array(aruco_corners['quad2']) + np.array(aruco_corners['quad4']) - np.array(aruco_corners['quad3'])
        aruco_y = np.array(aruco_corners['quad2']) - np.array(aruco_corners['quad3']) + np.array(aruco_corners['quad1']) - np.array(aruco_corners['quad4'])
        aruco_x /= np.linalg.norm(aruco_x, keepdims=True)
        aruco_y /= np.linalg.norm(aruco_y, keepdims=True)
        aruco_z = np.cross(aruco_x, aruco_y)

        aruco_x = np.expand_dims(aruco_x, -1).T
        aruco_y = np.expand_dims(aruco_y, -1).T
        aruco_z = np.expand_dims(aruco_z, -1).T

        opti_to_aruco = np.vstack((aruco_x, aruco_y, aruco_z))

        print(np.linalg.det(opti_to_aruco))
        print(opti_to_aruco.T @ opti_to_aruco)

        # Round since there is probably error in opti marker position leading to non-orthogonal aruco_x and aruco_y
        opti_to_aruco = R.from_matrix(opti_to_aruco).as_matrix()

        aruco_center = np.mean(np.array([aruco_corners['quad1'], aruco_corners['quad2'], aruco_corners['quad3'], aruco_corners['quad4']]), axis=0)

        self.aruco_center = aruco_center

        self.aruco_to_opti_rot = np.linalg.inv(opti_to_aruco)

    def get_aruco_to_opti_transform(self):
        aruco_to_opti_rot = self.aruco_to_opti_rot #aruco -> opti (rot (3x3))
        aruco_to_opti = np.eye(4)
        aruco_to_opti[:3,:3] = aruco_to_opti_rot

        #take aruco center, offset by 3 millimeters in aruco downwards direction (-Z)
        aruco_to_opti[:3,3] = self.aruco_center - aruco_to_opti_rot @ np.array([0., 0., 0.003])

        return aruco_to_opti

    def calculate_camera_to_opti_transform(self, rot_vec, trans):
        aff = affine_matrix_from_rotvec_trans(rot_vec, trans) #aruco -> camera
        aruco_to_opti = self.get_aruco_to_opti_transform()
        opti_to_aruco = invert_affine(aruco_to_opti)
        affine_transform_opti = aff @ opti_to_aruco #1. opti -> aruco 2. aruco -> camera = opti -> camera
        return invert_affine(affine_transform_opti) #camera -> opti

    # Let’s denote the ARUCO marker as Ar, the OptiTrack system as Opti, the Camera internal sensor as Se,
    # and the outer frame of the camera with reflective bulbs as Fr.
    # Here’s the calibration process (we want to get the extrinsics, which is the Se->Fr transformation here.
    # What we have before the calculation are Opti->Fr, Opti->Ar, and the camera raw data)
    # 1. We synchronize the Opti->Fr with the camera raw data using the timestamps.
    # 2. For each frame, we calculate the Ar->Se with the opencv API.
    # 3. For each frame, we now have Opti->Fr, Opti->Ar, Ar->Se. With matrix cascading and inversion, we can have Se->Fr for each frame.
    # 4. An assumption is taken that the start frame has the most accurate prediction of Se->Fr, we denote this prediction as Se_0->Fr_0.
    #    We then calculate the difference between Fr_i->Se_i @ Se_0 -> Fr_0, which should ideally be an identity matrix. We eliminate some of the
    #    frames with too high difference in either rotation or translation.
    # 5. We calculate the average translation and rotation (quaternion) within the filtered frames, and take that as our resulting extrinsics.
    # 6. When we plot the videos, we plot two videos.
    #   1) The first is only prediction using opencv API, which is entirely stable and accurate, representing the Ar->Se.
    #   2) The second is utilizing the outer path, which is inv(Se->Fr->Opti->Ar) for each frame. This is the unstable one.
    #
    # Notice that On the first frame of calibration (when starting to rotate around the object (after pressing 'd')),
    #  there were some reflective bulbs on the camera frame that were not correctly detected by the OptiTrack systems.
    #  This can result in a bad result per the assumption in step 4.
    def calculate_extrinsic(self, scene_dir, synchronized_poses, write_to_file=False):

        scene_metadata_file = os.path.join(scene_dir, "scene_meta.yaml")
        with open(scene_metadata_file, 'r') as file:
            scene_metadata = yaml.safe_load(file)

        camera_name = scene_metadata["camera"]
        camera_intrinsics_dict = load_frame_intrinsics(scene_dir, raw=False)
        camera_distortion_coefficients_dict = load_frame_distortions(scene_dir, raw=False)

        cam_scale = scene_metadata["cam_scale"]

        frames_dir = os.path.join(scene_dir, "data")

        virtual_camera_to_opti_transforms = []
        camera_sensor_to_opti_transforms = []

        total_frames_skipped = 0

        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters_create()

        for frame_id, opti_pose in tqdm(synchronized_poses.items(), total=len(synchronized_poses), desc="Finding ARUCO"):

            frame = load_bgr(frames_dir, frame_id, "png")
            depth = load_depth(frames_dir, frame_id)

            aruco_pose = calculate_aruco_from_bgr_and_depth(frame, depth, cam_scale, camera_intrinsics_dict[frame_id], camera_distortion_coefficients_dict[frame_id], aruco_dict, parameters)

            if aruco_pose:  # If there are markers found by detector

                rvec, tvec, _ = aruco_pose

                rvec, tvec = rvec.squeeze(), tvec.squeeze()
                # using inv(opti->aruco->camera)
                camera_sensor_to_opti_transform = self.calculate_camera_to_opti_transform(rvec, tvec)

                camera_sensor_to_opti_transforms.append(camera_sensor_to_opti_transform)
                # using inv(opti->camera_frame)
                virtual_camera_to_opti_transforms.append(opti_pose)

            else:
                total_frames_skipped += 1
                print("frame {0} did not successfully find ARUCO marker. Therefore it has been skipped. Total Frames Skipped: {1}".format(frame_id, total_frames_skipped))

        #extrinsic from virtual to camera sensor
        extrinsics = []

        for camera_to_opti, virtual_to_opti in zip(camera_sensor_to_opti_transforms, virtual_camera_to_opti_transforms):
            extrinsics.append(invert_affine(camera_to_opti) @ virtual_to_opti)

        extrinsics = np.array(extrinsics)

        #Assumption, Extrinsic 0 is mostly correct (please verify)
        #If Extrinsic 0 has swapped axis problem, then this will fail
        extrinsics_filtered = [extrinsics[0]]

        #make sure the extrinsics are relatively similar
        rotation_diffs = []
        translation_diffs = []

        rotation_diffs_filtered = []
        translation_diffs_filtered = []

        rotation_skipped = 0
        translation_skipped = 0

        good_frames = [0]

        for x in range(1, len(extrinsics)):
            rotation_diff = np.arccos((np.trace(extrinsics[x][:3,:3] @ np.linalg.inv(extrinsics_filtered[-1][:3,:3])) - 1.) / 2.)

            if np.isnan(rotation_diff):
                rotation_diff = 1.

            translation_diff = np.linalg.norm(extrinsics[x][:3,3] - extrinsics_filtered[-1][:3,3])

            if rotation_diff > 0.05:
                rotation_skipped += 1
            elif translation_diff > 0.04:
                translation_skipped += 1
            else:
                good_frames.append(x)
                extrinsics_filtered.append(extrinsics[x])
                rotation_diffs_filtered.append(rotation_diff)
                translation_diffs_filtered.append(translation_diff)

            rotation_diffs.append(rotation_diff)
            translation_diffs.append(translation_diff)

        print("Skipped {0} for large rotation diff. Skipped {1} for large translation diff".format(rotation_skipped, translation_skipped))

        if len(rotation_diffs) > 0:
            print("Diffs before filtering")
            print("rotation diffs (min, max, mean)", np.min(rotation_diffs), np.max(rotation_diffs), np.mean(rotation_diffs))
            print("translation diffs (min, max, mean)", np.min(translation_diffs), np.max(translation_diffs), np.mean(translation_diffs))

        if len(rotation_diffs_filtered) > 0:
            print("Diffs after filtering")
            print("rotation diffs (min, max, mean)", np.min(rotation_diffs_filtered), np.max(rotation_diffs_filtered), np.mean(rotation_diffs_filtered))
            print("translation diffs (min, max, mean)", np.min(translation_diffs_filtered), np.max(translation_diffs_filtered), np.mean(translation_diffs_filtered))

        extrinsics = np.array(extrinsics_filtered)
        print("Remaining extrinsics: {0}".format(len(extrinsics)))

        plt.plot(np.arange(0, len(rotation_diffs)), rotation_diffs, label="rotation")
        plt.plot(np.arange(0, len(translation_diffs)), translation_diffs, label="translation")
        plt.legend()
        plt.show()

        """
        Compute average extrinsic
        """

        translation = np.mean(extrinsics[:,:3,3], axis=0)

        extrinsic_rots = extrinsics[:,:3,:3]
        extrinsic_quats = R.from_matrix(extrinsic_rots).as_quat()

        quat = average_quaternion(extrinsic_quats)

        extrinsic = np.eye(4)
        extrinsic[:3,:3] = R.from_quat(quat).as_matrix()
        extrinsic[:3,3] = translation

        """
        Evaluate extrinsic
        """

        cv2.namedWindow("computed extrinsic validation")
        cv2.namedWindow("original ARUCO")
        opti_to_aruco = invert_affine(self.get_aruco_to_opti_transform())

        for frame_id, opti_pose in synchronized_poses.items():
            frame = load_bgr(frames_dir, frame_id, "png")
            frame_2 = np.copy(frame)

            sensor_to_opti = opti_pose @ invert_affine(extrinsic)
            sensor_to_aruco = opti_to_aruco @ sensor_to_opti
            aruco_to_sensor = invert_affine(sensor_to_aruco)

            rvec, tvec = rotvec_trans_from_affine_matrix(aruco_to_sensor)

            cv2.aruco.drawAxis(frame, camera_intrinsics_dict[frame_id], camera_distortion_coefficients_dict[frame_id], rvec, tvec / 9, 0.01)  # Draw Axis

            #NOTE: Here, we use cv2.aruco.estimatePoseSingleMarkers instead of our own aruco pose function in order to get a less influenced visualization
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
            parameters = cv2.aruco.DetectorParameters_create()
            corners, ids, rejected_img_points = cv2.aruco.detectMarkers(frame_2, aruco_dict, parameters=parameters, cameraMatrix=camera_intrinsics_dict[frame_id], distCoeff=camera_distortion_coefficients_dict[frame_id])

            if np.all(ids is not None):  # If there are markers found by detector

                #only should have 1 marker placed near origin of opti
                assert(len(ids) == 1)
                corners = corners[0]  # Iterate in markers
                # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners, 0.02, camera_intrinsics_dict[frame_id], camera_distortion_coefficients_dict[frame_id])

                #the aruco estimator assumes the marker is of size 5cmx5cm, we are using a marker of size 15cmx15cm
                rvec = rvec.squeeze()
                tvec = tvec.squeeze()

                cv2.aruco.drawAxis(frame_2, camera_intrinsics_dict[frame_id], camera_distortion_coefficients_dict[frame_id], rvec, tvec, 0.01)  # Draw Axis

            cv2.imshow("original ARUCO", frame_2)
            cv2.imshow("computed extrinsic validation", frame)
            cv2.waitKey(30)

        if write_to_file:
            print("saving extrinsic for camera: {0}".format(camera_name))
            write_extrinsics(camera_name, extrinsic)

        return extrinsic
