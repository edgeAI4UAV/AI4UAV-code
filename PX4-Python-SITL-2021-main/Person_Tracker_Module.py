import numpy as np
import os
import sys
from loguru import logger
from torch.nn import Module
import torch
# sys.path.append('../PX4-Python-SITL-2021-main/Tracker_Modules/videoanalyst')
from Tracker_Framework.videoanalyst.config.config import cfg, specify_task
from Tracker_Framework.videoanalyst.model import builder as model_builder
from Tracker_Framework.videoanalyst.pipeline import builder as pipeline_builder
from Tracker_Framework.videoanalyst.pipeline.tracker_impl.siamfcpp_track import SiamFCppTracker
from Tracker_Framework.videoanalyst.pipeline.utils.bbox import xywh2xyxy
from Tracker_Framework.videoanalyst.utils.image import ImageFileVideoStream, ImageFileVideoWriter
from Tracker_Framework.videoanalyst.utils.visualization import VideoWriter


workspace_dir = 'PX4-Python-SITL-2021-main'

# Pose estimation framework directory
pose_api_dir = 'Tracker_Framework'

tracker_vot_config_path = os.path.join(os.getcwd(), pose_api_dir, 'siamfcpp_vot_config')

tracker_uav123_config_path = os.path.join(os.getcwd(), pose_api_dir, 'siamfcpp_uav123_config')

def load_tracker_config(tracker_name: str) -> object:
    tracker_config_filename = tracker_name + '.yaml'
    tracker_config_fullpath = os.path.join(tracker_vot_config_path, tracker_config_filename)
    logger.info(
        "Load experiment configuration at: %s" % tracker_config_fullpath)
    root_cfg = cfg
    root_cfg.merge_from_file(tracker_config_fullpath)
    return root_cfg


def load_tracker_model(tracker_config: object) -> Module:
    """

    Returns:
        object: 
    """
    # resolve config
    root_cfg = tracker_config.test
    task, task_cfg = specify_task(root_cfg)
    task_cfg.freeze()
    window_name = task_cfg.exp_name
    # build model
    tracker_model = model_builder.build(task, task_cfg.model)
    # build pipeline
    pipeline = pipeline_builder.build(task, task_cfg.pipeline, tracker_model)
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pipeline.set_device(dev)
    return pipeline
