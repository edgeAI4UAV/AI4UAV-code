# -*- coding: utf-8 -*
import argparse
import os.path as osp
import time

import cv2
from loguru import logger
import imutils
from imutils.video import FPS

import torch


from videoanalyst.config.config import cfg, specify_task
from videoanalyst.model import builder as model_builder
from videoanalyst.pipeline import builder as pipeline_builder
from videoanalyst.pipeline.utils.bbox import xywh2xyxy
from videoanalyst.utils.image import ImageFileVideoStream, ImageFileVideoWriter
from videoanalyst.utils.visualization import VideoWriter
import os
from Hardware_Monitor_And_Metrics import ModelPerformanceProfiler

font_size = 0.5
font_width = 1


class ImageReader(object):
    def __init__(self, file_names, json_filenames):
        self.file_names = file_names
        self.json_names = json_filenames
        self.max_idx = len(file_names) - 1

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        # print(f'Paths for img: {self.file_names[self.idx]}'
        #       f'Paths for json: {self.json_names[self.idx]}')
        img_orig = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        # img_orig = cv2.resize(img_orig, (1088, 608))
        self.idx = self.idx + 1

        # with open(self.json_names[self.idx]) as jsonfile:
        #     # data = json.load(jsonfile)
        #     annot_json = json.load(jsonfile)
        human_pose_points = self.json_names[self.idx]
        # human_pose_points = [self.json_names['bbox_left'][self.idx], self.json_names['bbox_top'][self.idx],
        #                      self.json_names['bbox_width'][self.idx], self.json_names['bbox_height'][self.idx]]

        return img_orig, human_pose_points


def make_parser():
    # parser = argparse.ArgumentParser(
    #     description="press s to select the target box,\n \
    #                     then press enter or space to confirm it or press c to cancel it,\n \
    #                     press c to stop track and press q to exit program")

    parser = {
        'config': '/home/velere/Documents/EdgeAI4UAV/Human_Re_Identification/video_analyst-master/experiments/siamfcpp/test/vot/siamfcpp_alexnet_votlt.yaml',
        'device': 'cuda:0'}
    # parser.add_argument(
    #     "-cfg",
    #     "--config",
    #     default="experiments/siamfcpp/test/got10k/siamfcpp_alexnet-got.yaml",
    #     type=str,
    #     help='experiment configuration')

    return parser


def resize_bbox(bbox, in_size, out_size):
    """Resize bounding boxes according to image resize.
    Args:
        bbox (~numpy.ndarray): See the table below.
        in_size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        out_size (tuple): A tuple of length 2. The height and the width
            of the image after resized.
    .. csv-table::
        :header: name, shape, dtype, format
        :obj:`bbox`, ":math:`(R, 4)`", :obj:`float32`, \
        ":math:`(y_{min}, x_{min}, y_{max}, x_{max})`"
    Returns:
        ~numpy.ndarray:
        Bounding boxes rescaled according to the given image shapes.
    """
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[0] = int(y_scale * bbox[0])
    bbox[2] = int(y_scale * bbox[2])
    bbox[1] = int(x_scale * bbox[1])
    bbox[3] = int(x_scale * bbox[3])
    return bbox


def main(args):
    imgs_path = '/home/velere/Documents/EdgeAI4UAV/Human_Re_Identification/video_analyst-master/dataset/person20'
    annots_path = '/home/velere/Documents/EdgeAI4UAV/Human_Re_Identification/video_analyst-master/dataset/person20.txt'
    log_dir_path = '/home/velere/Documents/EdgeAI4UAV/Human_Re_Identification/deep-person-reid-master/log_dir'
    out_img_path = '/home/velere/Documents/EdgeAI4UAV/Human_Re_Identification/deep-person-reid-master/predicted_Images'

    images_files = os.listdir(imgs_path)
    images_files.sort()
    model_name = 'bagtricks_S50'

    images_files_paths = [os.path.join(imgs_path, img_file) for img_file in images_files]
    custom_model_profiler = ModelPerformanceProfiler(model_name, log_dir_path, out_img_path, annots_path)
    #
    gt_skeleton_img_points = custom_model_profiler.read_gt_bbox_and_return(human_class_num=1)
    transformed_bbox_cordss = custom_model_profiler.get_image_boxes_and_convert_xywh2xyxy_coords(
        image_file_num=0, all_flag=True)

    frame_provider = ImageReader(images_files_paths, transformed_bbox_cordss)

    root_cfg = cfg
    root_cfg.merge_from_file(args['config'])
    logger.info("Load experiment configuration at: %s" % args['config'])

    # resolve config
    root_cfg = root_cfg.test
    task, task_cfg = specify_task(root_cfg)
    task_cfg.freeze()
    window_name = task_cfg.exp_name
    # build model
    model = model_builder.build(task, task_cfg.model)
    # build pipeline
    pipeline = pipeline_builder.build(task, task_cfg.pipeline, model)
    dev = torch.device(args['device'])
    pipeline.set_device(dev)
    # init_box = None
    # template = None
    # if len(args.init_bbox) == 4:
    #     init_box = args.init_bbox

    # video_name = "untitled"
    # vw = None
    # # loop over sequence
    # frame_idx = 0  # global frame index
    frames_counter = 0
    fps = FPS().start()

    for orig_img, gt_bbox in frame_provider:
        key = 255
        frame = imutils.resize(orig_img, width=800)
        resized_box = resize_bbox(bbox=gt_bbox, in_size=orig_img.shape[0:2], out_size=frame.shape[0:2])
        if frames_counter == 0:
            init_box = resized_box
            (xx, yy, ww, hh) = [int(v) for v in init_box]
            pipeline.init(frame, init_box)
        rect_pred = pipeline.update(frame)
        resized_box2 = tuple(map(int, (
            resized_box[0], resized_box[1], resized_box[0] + resized_box[2], resized_box[1] + resized_box[3])))
        # gt_poses_bbox.append([resized_box2[0], resized_box2[1], resized_box2[2], resized_box2[3]])
        (x, y, w, h) = [int(v) for v in rect_pred]
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)

        fps.update()
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = make_parser()
    args = parser
    main(args)
