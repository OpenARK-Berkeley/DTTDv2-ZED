import yaml

import os, sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

class ConfigRandLA:
    def __init__(self):
        self.k_n = 32  # KNN
        self.num_layers = 2  # Number of layers
        self.num_points = 1024  # Number of input points
        self.num_classes = 13  # Number of valid classes

        self.sub_sampling_ratio = [4, 4]  # sampling ratio of random sampling at each layer
        self.d_out = [32, 64]  # feature dimension
        self.num_sub_points = [self.num_points // 4, self.num_points // 16]

    def load_cfg(self, cfg):
        self.__dict__.update(**cfg)

class Config(yaml.YAMLObject):

    def __init__(self):
        self.iteration = 2 #number of refine iterations
        self.nepoch = 160 #total number of epochs to train

        self.use_normals = True #use normals for pcld features
        self.use_colors = True #use colors for pcld features

        self.batch_size = 4
        self.workers = 8

        self.decay_margin = 0.015
        self.refine_margin = 0.000015 #refine margin should be less than or equal to decay margin (want decay to trigger first)
        self.refine_epoch = 100 #epoch to start refining if refine margin isn't ever reached

        self.lr = 0.0001
        self.lr_scheduler = "exponential"

        self.w = 0.015
        self.w_rate = 0.3 #w decay at decay_margin

        self.noise_trans = 0.001 #amount of XYZ noise added to each point independently

        self.add_front_aug = False #add random objects as occlusions
        self.symm_rotation_aug = False #add random rotation to GT around axis of symmetry

        self.batch_norm = True #global batch norm switch

        #DO NOT USE BASIC FUSION WITH POINTNET, since the DenseFusion is the pointnet
        self.basic_fusion = False #perform a basic fusion (cat) of depth and cnn features instead of dense fusion

        self.rndla_cfg = ConfigRandLA()

        #one of ["pointnet", "pointnet2", "randlanet"]
        self.pcld_encoder = "pointnet"

        self.resnet = "resnet18"
        self.pretrained_cnn = True
        self.pretrained_model_dir = "pretrained_models/"

        self.use_confidence = True #use confidence regression vs. standard voting

        self.fill_depth = False #use hole filling algorithm to fill depth in 2D

        self.blur_depth = False #run a gaussian kernel over depth map in 2D after fill_depth (probably shouldn't since fill depth already has a blur)

        self.use_hrnet_labels = False #train and test with hrnet labels
        self.hrnet_labels_path = "labels_hrnet"

    def load_cfg(self, cfg):
        rndla_cfg = cfg.pop("rndla_cfg")
        self.rndla_cfg.load_cfg(rndla_cfg)
        self.__dict__.update(**cfg)

#dataset specific configs
#currently, only YCB tested :) TODO: FIX LINEMOD AND CUSTOM
#num_objects is number of types of objects in dataset
#num_points is number of points randomly sampled from mask
#repeat_epoch is number of times training is repeated on an epoch

class AKIPConfig(Config):
    def __init__(self):
        super().__init__()

        self.root = "E:/adam/6d-pose-dataset-training/6d-pose-dataset/toolbox/root"
        self.num_objects = 16
        self.num_points = 1024
        self.outf = 'trained_models/akip'
        self.log_dir = 'experiments/logs/akip'
        self.repeat_epoch = 1

class YCBConfig(Config):
    def __init__(self):
        super().__init__()
        self.dataset = "ycb"
        self.root = "./datasets/ycb/YCB_Video_Dataset"

        self.num_objects = 21 #number of object classes in the dataset
        self.num_points = 2048 #number of points on the input pointcloud
        self.outf = 'trained_models/ycb' #folder to save trained models
        self.log_dir = 'experiments/logs/ycb' #folder to save logs
        self.repeat_epoch = 1 #number of repeat times for one epoch training

class LinemodConfig(Config):
    def __init__(self):
        super().__init__()
        self.num_objects = 13
        self.num_points = 500
        self.outf = 'trained_models/linemod'
        self.log_dir = 'experiments/logs/linemod'
        self.repeat_epoch = 20

class CustomConfig(Config):
    def __init__(self):
        super().__init__()
        self.num_objects = 1
        self.nu_points = 500
        self.outf = 'trained_models/custom'
        self.log_dir = 'experiments/logs/custom'
        self.repeat_epoch = 1

def noop(self, *args, **kw):
    pass

def write_config(cfg, file):
    yaml.emitter.Emitter.process_tag = noop
    with open(file, "w") as f:
        yaml.dump(cfg, f)

def load_config(cfg, file):
    with open(file, "r") as f:
        data = yaml.safe_load(f)

    cfg.load_cfg(data)
    return cfg

def main():
    cfg = YCBConfig()
    write_config(cfg, "test_config.yaml")

if __name__ == "__main__":
    main()