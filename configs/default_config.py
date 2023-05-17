import torch
from yacs.config import CfgNode as CN
import sys


gettrace = getattr(sys, "gettrace", None)

_C = CN()


##########
# SYSTEM #
##########

# Combined with the model name to derive the experiment_root where a lot of files that are generated
#during training are stored
_C.experiment_dir = "./experiments"

# Switch to train instead of evaluation mode
_C.eval_only = False 

# width of the progress bar
_C.tqdm_length = 100

_C.system = CN()

# only train for a few iterations and then stop
_C.system.debugger = True if gettrace() else False

_C.system.device = "cuda" if torch.cuda.is_available() else "cpu"
_C.system.num_workers = 0 if gettrace() else 4 

# seed random, numpy and torch
_C.system.seed = 1


###########
# DATASET #
###########

_C.dataset = CN()
_C.dataset.reinforce_memory = 0.5
_C.dataset.root = "/path/to/dataset"
_C.dataset.image_size = 64
_C.dataset.total_frames = 20
_C.dataset.seen_frames = 10
_C.dataset.unseen_frames = 10

# Limit train dataset to a random sample of `length` elements
_C.dataset.demo = CN()
_C.dataset.demo.enable = False
_C.dataset.demo.length = 2000

#########
# MODEL #
#########

_C.model = CN()

# model name is set to the name of the config file later
_C.model.name = "do not set here"

# Number of groups for GroupNorm in ResNet and TemporalDynamic
_C.model.group_norm_groups = 4

# activation function used by DCB
_C.model.activation_ = "lrelu"

# model width used by most of the components
_C.model.width = 128  # *******

# number of Conv3D layers in DCB
_C.model.conv_3d_depth = 2

# Taylor order used by TemporalDynamic
_C.model.taylor_order_r = 3  # *******

# ResNet model depth, defining the structure of ResNet layers (hardcoded for specific values of 
# model depth)
_C.model.resnet = CN()
_C.model.resnet.model_depth = 18

#####################
# RESUME EXPERIMENT #
#####################

# whether to resume a previous experiment
_C.model.resume = False  

# experiment to resume, -1 means last experiment with the same `model.name`
_C.model.resume_experiment_num = -1

# model state path for resuming, leave "" for auto-detection
_C.model.model_state_path = ""

# epoch to resume from, -1 means last saved epoch
_C.model.resume_epoch = -1


###########
# TRAINER #
###########

_C.trainer = CN()

# number of training epochs
_C.trainer.num_epochs = 10001

# DataLoader batch size used for the validation set and plotting
_C.trainer.sampling_batch_size = 10

# training batch size
_C.trainer.batch_size = 8

_C.trainer.log_interval = 5

# checkpoint every n epochs
_C.trainer.save_interval = 20

# periodically add histogram over parameters and their gradients to Tensorboard
_C.trainer.enable_histogram = False

# optimizer config, either [Adam, SGD, AdamW]
_C.trainer.optimizer = CN()
_C.trainer.optimizer.type = "adam"
_C.trainer.optimizer.lr = 0.0001
_C.trainer.optimizer.weight_decay = 0.0
_C.trainer.optimizer.enable_gradient_clip = False
_C.trainer.optimizer.gradient_clip_value = 0

# learning rate scheduler config, either ReduceLROnPlateau or CyclicLR
_C.trainer.scheduler = CN()
_C.trainer.scheduler.type = "plateau"
_C.trainer.scheduler.config = CN()
_C.trainer.scheduler.config.base_lr = 0.001
_C.trainer.scheduler.config.max_lr = 0.001
_C.trainer.scheduler.config.step_size_up = 2000
_C.trainer.scheduler.config.mode = ""
_C.trainer.scheduler.config.cycle_momentum = False
_C.trainer.scheduler.patience = 0
_C.trainer.scheduler.factor = 0.0
_C.trainer.scheduler.metric = "all_ssim_metric" 
_C.trainer.scheduler.mode = ""
_C.trainer.scheduler.use_train = True



def get_config_defaults():
    return _C.clone()
