import torch
import torchvision.models as models
import torch.nn as nn
import argparse
import torchvision
import torch.onnx
import numpy as np
from loguru import logger
from torch2trt import TRTModule, torch2trt
from collections import OrderedDict

from torch2trt import torch2trt
import torch.nn.functional as F
import onnxoptimizer
from onnxsim import simplify
from torch.onnx import OperatorExportTypes
import io
import onnx
from videoanalyst.config.config import cfg, specify_task
from videoanalyst.model import builder as model_builder
from videoanalyst.pipeline import builder as pipeline_builder
from videoanalyst.utils import complete_path_wt_root_in_cfg
from videoanalyst.config.config import cfg as root_cfg

import argparse
import os.path as osp


def to_numpy(tensor):
    return tensor.detach().cpu().numpy(
    ) if tensor.requires_grad else tensor.cpu().numpy()


def export_siamfcpp_fea_trt(task_cfg, parsed_args):
    """ export phase "feature" (basemodel/c_z_k/r_z_k) to trt model
    """
    model = model_builder.build("track", task_cfg.model)
    model = model.eval().cuda()
    model.phase = "feature"
    x = torch.randn(1, 3, 127, 127).cuda()
    fea = model(x)
    output_path = parsed_args['output'] + "_fea.trt"
    logger.info("start cvt pytorch model")
    model_trt = torch2trt(model, [x], use_onnx=True, max_workspace_size=1<<15)
    logger.info("save trt model to {}".format(output_path))
    torch.save(model_trt.state_dict(), output_path)
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(output_path))
    trt_out = model_trt(x)
    np.testing.assert_allclose(to_numpy(fea[0]),
                               to_numpy(trt_out[0]),
                               rtol=1e-03,
                               atol=1e-05)
    logger.info("test accuracy ok")


def export_siamfcpp_track_fea_trt(task_cfg, parsed_args):
    """ export phase "freeze_track_fea" (basemodel/c_x/r_x) to trt model
    """
    model = model_builder.build("track", task_cfg.model)
    model.eval().cuda()
    model.phase = "freeze_track_fea"
    search_im = torch.randn(1, 3, 303, 303).cuda()
    fea = model(search_im)
    output_path = parsed_args['output'] + "_track_fea.trt"
    logger.info("start cvt pytorch model")
    model_trt = torch2trt(model, [search_im], use_onnx=True, max_workspace_size=1<<15)
    torch.save(model_trt.state_dict(), output_path)
    logger.info("save trt model to {}".format(output_path))
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(output_path))
    trt_outs = model_trt(search_im)
    np.testing.assert_allclose(to_numpy(fea[0]),
                               to_numpy(trt_outs[0]),
                               rtol=1e-03,
                               atol=1e-05)
    np.testing.assert_allclose(to_numpy(fea[1]),
                               to_numpy(trt_outs[1]),
                               rtol=1e-03,
                               atol=1e-05)
    logger.info("test accuracy ok")


if __name__ == '__main__':
    parser = {
        'config': '/home/velere/Documents/EdgeAI4UAV/Human_Re_Identification/video_analyst-master/experiments/siamfcpp/test/vot/siamfcpp_googlenet.yaml',
        'output': 'siamfcpp_googlenet'}
    parsed_args = parser

    # experiment config
    # exp_cfg_path = osp.realpath(parsed_args['config'])
    root_cfg.merge_from_file(parsed_args['config'])
    logger.info("Load experiment configuration at: %s" % parsed_args['config'])

    # resolve config
    # root_cfg = complete_path_wt_root_in_cfg(root_cfg, ROOT_PATH)
    root_cfg = root_cfg.test
    task, task_cfg = specify_task(root_cfg)
    task_cfg.freeze()
    export_siamfcpp_fea_trt(task_cfg, parsed_args)
    export_siamfcpp_track_fea_trt(task_cfg, parsed_args)

# def template(self, z):
#     zf = self.backbone(z)
#     if cfg.MASK.MASK:
#         zf = zf[-1]
#     if cfg.ADJUST.ADJUST:
#         zf = self.neck(zf)
#     self.zf = zf


