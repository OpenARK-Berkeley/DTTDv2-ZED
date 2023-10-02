import argparse
import cv2
import numpy as np
import copy
from tqdm import tqdm
import yaml

import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from utils.constants import AZURE_KINECT_COLOR_HEIGHT, AZURE_KINECT_COLOR_WIDTH, SCENES_DIR, IPHONE_COLOR_HEIGHT, IPHONE_COLOR_WIDTH
from utils.frame_utils import load_bgr, load_label

def main():
    parser = argparse.ArgumentParser(description='Generate semantic labeling and meta labeling')
    parser.add_argument('scene_name', type=str, help='scene directory (contains scene_meta.yaml and data (frames) and camera_poses)')

    args = parser.parse_args()

    scene_dir = os.path.join(SCENES_DIR, args.scene_name)
    frames_dir = os.path.join(scene_dir, "data")

    scene_metadata_file = os.path.join(scene_dir, "scene_meta.yaml")
    with open(scene_metadata_file, 'r') as file:
        scene_metadata = yaml.safe_load(file)

    out_arr = []

    num_frames = scene_metadata["num_frames"]

    color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255), (123, 10, 265), (245, 163, 101), (100, 100, 178)]

    for frame_id in tqdm(range(num_frames), total=num_frames):
        color_img = load_bgr(frames_dir, frame_id, "jpg")
        label = load_label(frames_dir, frame_id)

        new_label_viz = copy.deepcopy(color_img)
        label = label > 0

        new_label_viz[:, :, 2] = label[:,:]*255 + (1-label[:,:])*color_img[:,:,2]

        out = np.zeros((color_img.shape[0], color_img.shape[1] + new_label_viz.shape[1], 3)).astype(np.uint8)

        out[:,:color_img.shape[1],:] = color_img
        out[:,color_img.shape[1]:,:] = new_label_viz

        out_arr.append(out)

    size = (IPHONE_COLOR_WIDTH * 2, IPHONE_COLOR_HEIGHT)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out_vid = cv2.VideoWriter(os.path.join(dir_path, '..', 'demos' , '{0}.mp4'.format(args.scene_name)), fourcc, 15, size)

    for i in range(len(out_arr)):
        out_vid.write(out_arr[i])

    out_vid.release()

if __name__ == "__main__":
    main()
