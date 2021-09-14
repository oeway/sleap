import os
# fix issue related to UDA runtime implicit initialization on GPU:0 failed. Status: device kernel image is invalid
# os.environ['CUDA_VISIBLE_DEVICES'] = ""
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import sleap
import tensorflow as tf
from sleap.nn.config import *

# from sleap.io.dataset import Labels
# labels_train = Labels.load_imjoy(filename='/data/wei/actin-comet-tail/train/manifest.json')
# labels_valid = Labels.load_imjoy(filename='/data/wei/actin-comet-tail/valid/manifest.json')
# print(labels)

# Initialize the default training job configuration.
cfg = TrainingJobConfig()

# Update paths to training data we just downloaded.
cfg.data.labels.training_labels = "/data/wei/actin-comet-tail/train/manifest.json"
cfg.data.labels.validation_labels = "/data/wei/actin-comet-tail/valid/manifest.json"

# Preprocesssing and training parameters.
cfg.data.instance_cropping.center_on_part = "5"
cfg.optimization.augmentation_config.rotate = True
cfg.optimization.augmentation_config.random_flip = 'horizontal'
cfg.optimization.augmentation_config.contrast = True
cfg.optimization.augmentation_config.scale = True
cfg.optimization.augmentation_config.uniform_noise = True
cfg.optimization.augmentation_config.brightness = True
cfg.optimization.augmentation_config.gaussian_noise = True
cfg.optimization.hard_keypoint_mining.online_mining = False
cfg.optimization.epochs = 10000 # This is the maximum number of training rounds.

# These configures the actual neural network and the model type:
cfg.model.backbone.unet = UNetConfig(
    filters=16,
    output_stride=4
)
cfg.model.heads.centered_instance = CenteredInstanceConfmapsHeadConfig(
    anchor_part="5",
    sigma=1.5,
    output_stride=4
)

# Setup how we want to save the trained model.
cfg.outputs.run_name = "baseline_model.topdown"
cfg.outputs.tensorboard.write_logs = True
cfg.outputs.tensorboard.architecture_graph = True

trainer = sleap.nn.training.Trainer.from_config(cfg)
trainer.setup()

trainer.train()