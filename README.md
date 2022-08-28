# DTTD: Digital-Twin Tracking Dataset Official Repository

## Table of Content
- [DTTD: Digital-Twin Tracking Dataset Official Repository](#dttd-digital-twin-tracking-dataset-official-repository)
	- [Table of Content](#table-of-content)
	- [Overview](#overview)
	- [Requirements](#requirements)
	- [Code Structure](#code-structure)
	- [Datasets](#datasets)
	- [Training](#training)
	- [Evaluation](#evaluation)
		- [Evaluation on YCB\_Video Dataset](#evaluation-on-ycb_video-dataset)
		- [Evaluation on LineMOD Dataset](#evaluation-on-linemod-dataset)
	- [Results](#results)
	- [Trained Checkpoints](#trained-checkpoints)
	- [Tips for your own dataset](#tips-for-your-own-dataset)
	- [Citations](#citations)
	- [License](#license)
- [OUR DATASET REPO](#our-dataset-repo)
	- [Dataset Structure](#dataset-structure)
	- [Data Collection Pipeline](#data-collection-pipeline)
		- [Configuration \& Setup](#configuration--setup)
		- [Record Data (`tools/capture_data.py`)](#record-data-toolscapture_datapy)
		- [Check if the Extrinsic file exists](#check-if-the-extrinsic-file-exists)
		- [Process iPhone Data (if iPhone Data)](#process-iphone-data-if-iphone-data)
		- [Process Extrinsic Data to Calculate Extrinsic (If extrinsic scene)](#process-extrinsic-data-to-calculate-extrinsic-if-extrinsic-scene)
		- [Process Data (If data scene)](#process-data-if-data-scene)
	- [Minutia](#minutia)
	- [Best Scene Collection Practices](#best-scene-collection-practices)
- [Todo: Model Evaluation for this dataset](#todo-model-evaluation-for-this-dataset)
	- [DTTD: Digital-Twin Tracking Dataset](#dttd-digital-twin-tracking-dataset)
	- [Objects and Scenes data:](#objects-and-scenes-data)
- [How to run](#how-to-run)
- [Minutia](#minutia-1)
- [Best Scene Collection Practices](#best-scene-collection-practices-1)
- [Todo: Model Evaluation for this dataset](#todo-model-evaluation-for-this-dataset-1)

## Overview

This repository is the implementation code of the paper "DenseFusion: 6D Object Pose Estimation by Iterative Dense Fusion"([arXiv](https://arxiv.org/abs/1901.04780), [Project](https://sites.google.com/view/densefusion), [Video](https://www.youtube.com/watch?v=SsE5-FuK5jo)) by Wang et al. at [Stanford Vision and Learning Lab](http://svl.stanford.edu/) and [Stanford People, AI & Robots Group](http://pair.stanford.edu/). The model takes an RGB-D image as input and predicts the 6D pose of the each object in the frame. This network is implemented using [PyTorch](https://pytorch.org/) and the rest of the framework is in Python. Since this project focuses on the 6D pose estimation process, we do not specifically limit the choice of the segmentation models. You can choose your preferred semantic-segmentation/instance-segmentation methods according to your needs. In this repo, we provide our full implementation code of the DenseFusion model, Iterative Refinement model and a vanilla SegNet semantic-segmentation model used in our real-robot grasping experiment. The ROS code of the real robot grasping experiment is not included.


## Requirements

* Python 2.7/3.5/3.6 (If you want to use Python2.7 to run this repo, please rebuild the `lib/knn/` (with PyTorch 0.4.1).)
* [PyTorch 0.4.1](https://pytorch.org/) ([PyTroch 1.0 branch](<https://github.com/j96w/DenseFusion/tree/Pytorch-1.0>))
* PIL
* scipy
* numpy
* pyyaml
* logging
* matplotlib
* CUDA 7.5/8.0/9.0 (Required. CPU-only will lead to extreme slow training speed because of the loss calculation of the symmetry objects (pixel-wise nearest neighbour loss).)

## Code Structure
* **datasets**
	* **datasets/ycb**
		* **datasets/ycb/dataset.py**: Data loader for YCB_Video dataset.
		* **datasets/ycb/dataset_config**
			* **datasets/ycb/dataset_config/classes.txt**: Object list of YCB_Video dataset.
			* **datasets/ycb/dataset_config/train_data_list.txt**: Training set of YCB_Video dataset.
			* **datasets/ycb/dataset_config/test_data_list.txt**: Testing set of YCB_Video dataset.
	* **datasets/linemod**
		* **datasets/linemod/dataset.py**: Data loader for LineMOD dataset.
		* **datasets/linemod/dataset_config**: 
			* **datasets/linemod/dataset_config/models_info.yml**: Object model info of LineMOD dataset.
* **replace_ycb_toolbox**: Replacement codes for the evaluation with [YCB_Video_toolbox](https://github.com/yuxng/YCB_Video_toolbox).
* **trained_models**
	* **trained_models/ycb**: Checkpoints of YCB_Video dataset.
	* **trained_models/linemod**: Checkpoints of LineMOD dataset.
* **lib**
	* **lib/loss.py**: Loss calculation for DenseFusion model.
	* **lib/loss_refiner.py**: Loss calculation for iterative refinement model.
	* **lib/transformations.py**: [Transformation Function Library](https://www.lfd.uci.edu/~gohlke/code/transformations.py.html).
    * **lib/network.py**: Network architecture.
    * **lib/extractors.py**: Encoder network architecture adapted from [pspnet-pytorch](https://github.com/Lextal/pspnet-pytorch).
    * **lib/pspnet.py**: Decoder network architecture.
    * **lib/utils.py**: Logger code.
    * **lib/knn/**: CUDA K-nearest neighbours library adapted from [pytorch_knn_cuda](https://github.com/chrischoy/pytorch_knn_cuda).
* **tools**
	* **tools/_init_paths.py**: Add local path.
	* **tools/eval_ycb.py**: Evaluation code for YCB_Video dataset.
	* **tools/eval_linemod.py**: Evaluation code for LineMOD dataset.
	* **tools/train.py**: Training code for YCB_Video dataset and LineMOD dataset.
* **experiments**
	* **experiments/eval_result**
		* **experiments/eval_result/ycb**
			* **experiments/eval_result/ycb/Densefusion_wo_refine_result**: Evaluation result on YCB_Video dataset without refinement.
			* **experiments/eval_result/ycb/Densefusion_iterative_result**: Evaluation result on YCB_Video dataset with iterative refinement.
		* **experiments/eval_result/linemod**: Evaluation results on LineMOD dataset with iterative refinement.
	* **experiments/logs/**: Training log files.
	* **experiments/scripts**
		* **experiments/scripts/train_ycb.sh**: Training script on the YCB_Video dataset.
		* **experiments/scripts/train_linemod.sh**: Training script on the LineMOD dataset.
		* **experiments/scripts/eval_ycb.sh**: Evaluation script on the YCB_Video dataset.
		* **experiments/scripts/eval_linemod.sh**: Evaluation script on the LineMOD dataset.
* **download.sh**: Script for downloading YCB_Video Dataset, preprocessed LineMOD dataset and the trained checkpoints.


## Datasets

This work is tested on two 6D object pose estimation datasets:

* [YCB_Video Dataset](https://rse-lab.cs.washington.edu/projects/posecnn/): Training and Testing sets follow [PoseCNN](https://arxiv.org/abs/1711.00199). The training set includes 80 training videos 0000-0047 & 0060-0091 (choosen by 7 frame as a gap in our training) and synthetic data 000000-079999. The testing set includes 2949 keyframes from 10 testing videos 0048-0059.

* [LineMOD](http://campar.in.tum.de/Main/StefanHinterstoisser): Download the [preprocessed LineMOD dataset](https://drive.google.com/drive/folders/19ivHpaKm9dOrr12fzC8IDFczWRPFxho7) (including the testing results outputted by the trained vanilla SegNet used for evaluation).

Download YCB_Video Dataset, preprocessed LineMOD dataset and the trained checkpoints (You can modify this script according to your needs.):
```	
./download.sh
```

## Training

* YCB_Video Dataset:
	After you have downloaded and unzipped the YCB_Video_Dataset.zip and installed all the dependency packages, please run:
```	
./experiments/scripts/train_ycb.sh
```
* LineMOD Dataset:
	After you have downloaded and unzipped the Linemod_preprocessed.zip, please run:
```	
./experiments/scripts/train_linemod.sh
```
**Training Process**: The training process contains two components: (i) Training of the DenseFusion model. (ii) Training of the Iterative Refinement model. In this code, a DenseFusion model will be trained first. When the average testing distance result (ADD for non-symmetry objects, ADD-S for symmetry objects) is smaller than a certain margin, the training of the Iterative Refinement model will start automatically and the DenseFusion model will then be fixed. You can change this margin to have better DenseFusion result without refinement but it's inferior than the final result after the iterative refinement. 

**Checkpoints and Resuming**: After the training of each 1000 batches, a `pose_model_current.pth` / `pose_refine_model_current.pth` checkpoint will be saved. You can use it to resume the training. After each testing epoch, if the average distance result is the best so far, a `pose_model_(epoch)_(best_score).pth` /  `pose_model_refiner_(epoch)_(best_score).pth` checkpoint will be saved. You can use it for the evaluation.

**Notice**: The training of the iterative refinement model takes some time. Please be patient and the improvement will come after about 30 epoches.


* vanilla SegNet:
	Just run:
```
cd vanilla_segmentation/
python train.py --dataset_root=./datasets/ycb/YCB_Video_Dataset
```
To make the best use of the training set, several data augementation techniques are used in this code:

(1) A random noise is added to the brightness, contrast and saturation of the input RGB image with the `torchvision.transforms.ColorJitter` function, where we set the function as `torchvision.transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)`.

(2) A random pose translation noise is added to the training set of the pose estimator, where we set the range of the translation noise to 3cm for both datasets.

(3) For the YCB_Video dataset, since the synthetic data is not contain background. We randomly select the real training data as the background. In each frame, we also randomly select two instances segmentation clips from another synthetic training image to mask at the front of the input RGB-D image, so that more occlusion situations can be generated.

## Evaluation

### Evaluation on YCB_Video Dataset
For fair comparison, we use the same segmentation results of [PoseCNN](https://rse-lab.cs.washington.edu/projects/posecnn/) and compare with their results after ICP refinement.
Please run:
```
./experiments/scripts/eval_ycb.sh
```
This script will first download the `YCB_Video_toolbox` to the root folder of this repo and test the selected DenseFusion and Iterative Refinement models on the 2949 keyframes of the 10 testing video in YCB_Video Dataset with the same segmentation result of PoseCNN. The result without refinement is stored in `experiments/eval_result/ycb/Densefusion_wo_refine_result` and the refined result is in `experiments/eval_result/ycb/Densefusion_iterative_result`.

After that, you can add the path of `experiments/eval_result/ycb/Densefusion_wo_refine_result/` and `experiments/eval_result/ycb/Densefusion_iterative_result/` to the code `YCB_Video_toolbox/evaluate_poses_keyframe.m` and run it with [MATLAB](https://www.mathworks.com/products/matlab.html). The code `YCB_Video_toolbox/plot_accuracy_keyframe.m` can show you the comparsion plot result. You can easily make it by copying the adapted codes from the `replace_ycb_toolbox/` folder and replace them in the `YCB_Video_toolbox/` folder. But you might still need to change the path of your `YCB_Video Dataset/` in the `globals.m` and copy two result folders(`Densefusion_wo_refine_result/` and `Densefusion_iterative_result/`) to the `YCB_Video_toolbox/` folder.


### Evaluation on LineMOD Dataset
Just run:
```
./experiments/scripts/eval_linemod.sh
```
This script will test the models on the testing set of the LineMOD dataset with the masks outputted by the trained vanilla SegNet model. The result will be printed at the end of the execution and saved as a log in `experiments/eval_result/linemod/`.


## Results

* YCB_Video Dataset:

Quantitative evaluation result with ADD-S metric compared to other RGB-D methods. `Ours(per-pixel)` is the result of the DenseFusion model without refinement and `Ours(iterative)` is the result with iterative refinement.

<p align="center">
	<img src ="assets/result_ycb.png" width="600" />
</p>

**Important!** Before you use these numbers to compare with your methods, please make sure one important issus: One difficulty for testing on the YCB_Video Dataset is how to let the network to tell the difference between the object `051_large_clamp` and `052_extra_large_clamp`. The result of all the approaches in this table uses the same segmentation masks released by PoseCNN without any detection priors, so all of them suffer a performance drop on these two objects because of the poor detection result and this drop is also added to the final overall score. If you have added detection priors to your detector to distinguish these two objects, please clarify or do not copy the overall score for comparsion experiments.

* LineMOD Dataset:

Quantitative evaluation result with ADD metric for non-symmetry objects and ADD-S for symmetry objects(eggbox, glue) compared to other RGB-D methods. High performance RGB methods are also listed for reference.

<p align="center">
	<img src ="assets/result_linemod.png" width="500" />
</p>

The qualitative result on the YCB_Video dataset.

<p align="center">
	<img src ="assets/compare.png" width="600" />
</p>

## Trained Checkpoints
You can download the trained DenseFusion and Iterative Refinement checkpoints of both datasets from [Link](https://drive.google.com/drive/folders/19ivHpaKm9dOrr12fzC8IDFczWRPFxho7).

## Tips for your own dataset
As you can see in this repo, the network code and the hyperparameters (lr and w) remain the same for both datasets. Which means you might not need to adjust too much on the network structure and hyperparameters when you use this repo on your own dataset. Please make sure that the distance metric in your dataset should be converted to meter, otherwise the hyperparameter w need to be adjusted. Several useful tools including [LabelFusion](https://github.com/RobotLocomotion/LabelFusion) and [sixd_toolkit](https://github.com/thodan/sixd_toolkit) has been tested to work well. (Please make sure to turn on the depth image collection in LabelFusion when you use it.)


## Citations
Please cite [DenseFusion](https://sites.google.com/view/densefusion) if you use this repository in your publications:
```
@article{wang2019densefusion,
  title={DenseFusion: 6D Object Pose Estimation by Iterative Dense Fusion},
  author={Wang, Chen and Xu, Danfei and Zhu, Yuke and Mart{\'\i}n-Mart{\'\i}n, Roberto and Lu, Cewu and Fei-Fei, Li and Savarese, Silvio},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```

## License
Licensed under the [MIT License](LICENSE)




# OUR DATASET REPO


## Dataset Structure
https://docs.google.com/spreadsheets/d/1weyPvCyxU82EIokMlGhlK5uEbNt9b8a-54ziPpeGjRo/edit?usp=sharing

Final dataset output:
 * `objects` folder
 * `scenes` folder certain data:
 	 * `scenes/<scene name>/data/` folder
 	 * `scenes/<scene name>/scene_meta.yaml` metadata
 * `toolbox` folder

## What you Need to Collect your own Data
 1. OptiTrack Motion Capture system with Motive tracking software
	* This doesn't have to be running on the same computer as the other sensors. We will export the tracked poses to a CSV file.
	* Create a rigid body to track a camera's OptiTrack markers, give the rigid body the same name that is passed into `tools/capture_data.py`
 2. Microsoft Azure Kinect
	* We interface with the camera using Microsoft's K4A SDK: https://github.com/microsoft/Azure-Kinect-Sensor-SDK
 3. iPhone 12 Pro / iPhone 13
	* Please build the project in `iphone_app/` in XCode and install on the mobile device.

## Data Collection Pipeline
### Configuration & Setup
  1. Place ARUCO marker somewhere visible
  2. Place markers on the corners of the aruco marker, we use this to compute the (aruco -> opti) transform
  3. Place marker positions into `calculate_extrinsic/aruco_corners.yaml`, labeled under keys: `quad1`, `quad2`, `quad3`, and `quad4`.

### Record Data (`tools/capture_data.py`)
  1. Data collection
      * If extrinsic scene, data collection phase should be spent observing ARUCO marker, run `tools/capture_data.py --extrinsic`
  2. Example data collection scene (not extrinsic): `python tools/capture_data.py --scene_name test az_camera1`

#### Data Recording Process
  1. Start the OptiTrack recording
  2. Synchronization Phase
	  1. Press `c` to begin recording data
	  2. Observe the ARUCO marker in the scene and move the camera in different trajectories to build synchronization data
	  3. Press `p` when finished
  3. Data Capturing Phase
      1. Press `d` to begin recording data
	  2. If extrinsic scene, observe the ARUCO marker.
	  3. If data collection scene, observe objects to track
	  4. Press `q` when finished
  4. Stop OptiTrack recording
  5. Export OptiTrack recording to a CSV file with 60Hz report rate.
  6. Move tracking CSV file to `<scene name>/camera_poses/camera_pose.csv`
		 
### Process iPhone Data (if iPhone Data)
  1. Convert iPhone data formats to Kinect data formats (`tools/process_iphone_data.py`)
		* This tool converts everything to common image names, formats, and does distortion parameter fitting
  2. Continue with step 5 or 6 depending on whether computing an extrinsic or capturing scene data

### Process Extrinsic Data to Calculate Extrinsic (If extrinsic scene)
  1. Clean raw opti poses (`tools/process_data.py --extrinsic`) 
  2. Sync opti poses with frames (`tools/process_data.py --extrinsic`)
  3. Calculate camera extrinsic (`tools/calculate_camera_extrinsic.py`)
  4. Output will be placed in `cameras/<camera name>/extrinsic.txt`

### Process Data (If data scene)
  1. Clean raw opti poses (`tools/process_data.py`) <br>
	 Example: <code>python tools/process_data.py --scene_name [SCENE_NAME]</code>
  2. Sync opti poses with frames (`tools/process_data.py`) <br>
	 Example: <code>python tools/process_data.py --scene_name [SCENE_NAME]</code>
  3. Manually annotate first frame object poses (`tools/manual_annotate_poses.py`)
	 	 1. Modify (`[SCENE_NAME]/scene_meta.yml`) by adding (`objects`) field to the file according to objects and their corresponding ids.<br>
			Example: `python tools/manual_annotate_poses.py test`
  4. Recover all frame object poses and verify correctness (`tools/generate_scene_labeling.py`) <br>
	 Example: <code>python tools/generate_scene_labeling.py --fast [SCENE_NAME]</code>
	 1. Generate semantic labeling (`tools/generate_scene_labeling.py`)<br>
	 Example: <code>python /tools/generate_scene_labeling.py [SCENE_NAME]</code>
	 2. Generate per frame object poses (`tools/generate_scene_labeling.py`)<br>
	 Example: <code>python tools/generate_scene_labeling.py [SCENE_NAME]</code>

## Minutia
 * Extrinsic scenes have their color images inside of `data` stored as `png`. This is to maximize performance. Data scenes have their color images inside of `data` stored as `jpg`. This is necessary so the dataset remains usable.
 * iPhone spits out `jpg` raw color images, while Azure Kinect skips out `png` raw color images.

## Best Scene Collection Practices
 * Good synchronization phase by observing ARUCO marker, for Azure Kinect keep in mind interference from OptiTrack system.
 * Don't have objects that are in our datasets in the background. Make sure they are out of view!
 * Minimize number of extraneous ARUCO markers/APRIL tags that appear in the scene.
 * Stay in the yellow area for best OptiTrack tracking.
 * Move other cameras out of area when collecting data to avoid OptiTrack confusion.
 * Run `manual_annotate_poses.py` on all scenes after collection in order to archive extrinsic.
 * We want to keep the data anonymized. Avoid school logos and members of the lab appearing in frame.
 * Perform 90-180 revolution around objects, one way. Try to minimize stand-still time.

# Todo: Model Evaluation for this dataset
Select SOTA pose estimation & image segmentation models and perform evaluations according to certain metrics.

## DTTD: Digital-Twin Tracking Dataset

Basically, we take the optitrack which tracks markers, and we need to render the models in those poses, and then collect GT semantic segmentation data like that.

## Objects and Scenes data:
https://docs.google.com/spreadsheets/d/1weyPvCyxU82EIokMlGhlK5uEbNt9b8a-54ziPpeGjRo/edit?usp=sharing

Final dataset output:
 * `objects` folder
 * `cameras` folder
 * `scenes` folder certain data:
 	 * `scenes/<scene number>/data/` folder
 	 * `scenes/<scene number>/scene_meta.yaml` metadata
 * `toolbox` folder

# How to run
 1. Setup
	 1. Place ARUCO marker near origin (doesn't actually matter where it is anymore, but makes sense to be near OptiTrack origin)
	 2. Calibrate Opti (if you want, don't need to do this everytime, or else extrinsic changes)
	 3. Place markers on the corners of the aruco marker, use this to compute the aruco -> opti transform
	 	 * Place marker positions into `calculate_extrinsic/aruco_corners.yaml`
 2. Record Data (`tools/capture_data.py`)
     1. ARUCO Calibration
	 2. Data collection
	 	 * If extrinsic scene, data collection phase should be spent observing ARUCO marker
	 3. Example: <code>python tools/capture_data.py --scene_name official_test az_camera</code>
		 
 3. Check if the Extrinsic file exists
	 1. If Extrinsic file doesn't exist, then you need to calculate Extrinsic through Step 4
	 2. Otherwise, process data through Step 5 to generate groundtruth labels
 4. Process iPhone Data (if iPhone Data)
	1. Convert iPhone data formats to Kinect data formats (`tools/process_iphone_data.py`)
		* This tool converts everything to common image names, formats, and does distortion parameter fitting
	2. Continue with step 5 or 6 depending on whether computing an extrinsic or capturing scene data
 5. Process Extrinsic Data to Calculate Extrinsic (If extrinsic scene)
	 1. Clean raw opti poses (`tools/process_data.py`) 
	 2. Sync opti poses with frames (`tools/process_data.py`)
	 3. Calculate camera extrinsic (`tools/calculate_camera_extrinsic.py`)
 6. Process Data (If data scene)
	 1. Clean raw opti poses (`tools/process_data.py`) <br>
	 Example: <code>python tools/process_data.py --scene_name [SCENE_NAME]</code>
	 2. Sync opti poses with frames (`tools/process_data.py`) <br>
	 Example: <code>python tools/process_data.py --scene_name [SCENE_NAME]</code>
	 3. Manually annotate first frame object poses (`tools/manual_annotate_poses.py`)
	 	 1. Modify (`[SCENE_NAME]/scene_meta.yml`) by adding (`objects`) field to the file according to objects and their corresponding ids.<br>
			Example: <code>python tools/manual_annotate_poses.py official_test</code>
	 4. Recover all frame object poses and verify correctness (`tools/generate_scene_labeling.py`) <br>
	 Example: <code>python tools/generate_scene_labeling.py --fast [SCENE_NAME]</code>
	 5. Generate semantic labeling (`tools/generate_scene_labeling.py`)<br>
	 Example: <code>python /tools/generate_scene_labeling.py [SCENE_NAME]</code>
	 6. Generate per frame object poses (`tools/generate_scene_labeling.py`)<br>
	 Example: <code>python tools/generate_scene_labeling.py [SCENE_NAME]</code>


# Minutia
 * Extrinsic scenes have their color images inside of `data` stored as `png`. This is to maximize performance. Data scenes have their color images inside of `data` stored as `jpg`. This is necessary so the dataset remains usable.
 * iPhone spits out `jpg` raw color images, while Azure Kinect skips out `png` raw color images.

# Best Scene Collection Practices
 * Good synchronization phase by observing ARUCO marker, for Azure Kinect keep in mind interference from OptiTrack system.
 * Don't have objects that are in our datasets in the background. Make sure they are out of view!
 * Minimize number of extraneous ARUCO markers/APRIL tags that appear in the scene.
 * Stay in the yellow area for best OptiTrack tracking.
 * Move other cameras out of area when collecting data to avoid OptiTrack confusion.
 * Run `manual_annotate_poses.py` on all scenes after collection in order to archive extrinsic.
 * We want to keep the data anonymized. Avoid school logos and members of the lab appearing in frame.
 * Perform 90-180 revolution around objects, one way. Try to minimize stand-still time.


# Todo: Model Evaluation for this dataset
Select SOTA pose estimation & image segmentation models and perform evaluations according to certain metrics.
