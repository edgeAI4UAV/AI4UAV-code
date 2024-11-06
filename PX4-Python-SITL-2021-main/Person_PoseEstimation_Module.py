import numpy as np
# from PoseEstimation_Framework.models.with_mobilenet import PoseEstimationWithMobileNet
# from PoseEstimation_Framework.modules.keypoints import extract_keypoints, group_keypoints
# from PoseEstimation_Framework.modules.load_state import load_state
# from PoseEstimation_Framework.modules.pose import Pose, track_poses

from mmpose.apis import (inference_top_down_pose_model, init_pose_model)
from mmpose.core import Smoother

import onnxruntime as ort
import torch
import cv2
import math


# img_mean = np.array([128, 128, 128], np.float32)
# img_scale = np.float32(1 / 256)
# pad_value = (0, 0, 0)
# net_input_height_size = 256
# stride = 8
# num_keypoints = 18
#
#
# def load_pose_estimation_model(classic_or_onnx: str, model_path: str):
#     if classic_or_onnx == 'Classic':
#         net = PoseEstimationWithMobileNet()
#         dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#
#         checkpoint = torch.load(model_path, map_location=dev)
#         load_state(net, checkpoint)
#         if torch.cuda.is_available():
#             return net.cuda()
#         else:
#             return net
#     else:
#         net = ort.InferenceSession(
#             model_path,
#             providers=[
#                 'CUDAExecutionProvider'
#             ],
#         )
#         return net
#
#
# def pose_estimation_data_preprocess(img: np.ndarray, inference_engine: str):
#     height, width, _ = img.shape
#     scale = net_input_height_size / height  # max(height, width)
#
#     scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
#     # scaled_img = pose_estimation_data_normalize(scaled_img)
#     min_dims = [net_input_height_size, max(scaled_img.shape[0], net_input_height_size)]
#     padded_img, pad = pose_estimation_data_pad_width(scaled_img, min_dims)
#     cv2.imshow('Padded_Img', padded_img)
#     cv2.waitKey(0)
#     tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
#
#     if inference_engine == 'onnx':
#         return tensor_img.numpy(), scale, pad
#     else:
#         return tensor_img.cuda(), scale, pad
#
#
# def pose_estimation_data_normalize(img: np.ndarray) -> np.ndarray:
#     img = np.array(img, dtype=np.float32)
#     img = (img - img_mean) * img_scale
#     return img
#
#
# def pose_estimation_data_pad_width(img: np.ndarray, min_dims: list):
#     h, w, _ = img.shape
#     h = min(min_dims[0], h)
#     min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
#     min_dims[1] = max(min_dims[1], w)
#     min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
#     pad = []
#     pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
#     pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
#     pad.append(int(min_dims[0] - h - pad[0]))
#     pad.append(int(min_dims[1] - w - pad[1]))
#     padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
#                                     cv2.BORDER_CONSTANT, value=0)
#     return padded_img, pad
#
#
# def convert_to_coco_format(pose_entries, all_keypoints):
#     coco_keypoints = []
#     scores = []
#     for n in range(len(pose_entries)):
#         if len(pose_entries[n]) == 0:
#             continue
#         keypoints = [0] * 17 * 3
#         to_coco_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
#         person_score = pose_entries[n][-2]
#         position_id = -1
#         for keypoint_id in pose_entries[n][:-2]:
#             position_id += 1
#             if position_id == 1:  # no 'neck' in COCO
#                 continue
#
#             cx, cy, score, visibility = 0, 0, 0, 0  # keypoint not found
#             if keypoint_id != -1:
#                 cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
#                 cx = cx + 0.5
#                 cy = cy + 0.5
#                 visibility = 1
#             keypoints[to_coco_map[position_id] * 3 + 0] = cx
#             keypoints[to_coco_map[position_id] * 3 + 1] = cy
#             keypoints[to_coco_map[position_id] * 3 + 2] = visibility
#         coco_keypoints.append(keypoints)
#         scores.append(person_score * max(0, (pose_entries[n][-1] - 1)))  # -1 for 'neck'
#     return coco_keypoints, scores
#
#
# def pose_estimation_post_processing(heatmaps: np.ndarray, pafs: np.ndarray, pad: list, scale: float):
#     total_keypoints_num = 0
#     all_keypoints_by_type = []
#
#     for kpt_idx in range(num_keypoints):  # 19th for bg
#         total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
#                                                  total_keypoints_num)
#
#     pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
#     coco_keypoints, scores = convert_to_coco_format(pose_entries, all_keypoints)
#     # for kpt_id in range(all_keypoints.shape[0]):
#     #     all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / 4 - pad[1]) / scale  # 4 is upsample_ratio
#     #     all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / 4 - pad[0]) / scale  # 4 is upsample_ratio
#     # current_poses = []
#     # temp_list = []
#     # for n in range(len(pose_entries)):
#     #     if len(pose_entries[n]) == 0:
#     #         continue
#     #     pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
#     #     for kpt_id in range(num_keypoints):
#     #         if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
#     #             pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
#     #             pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
#     #
#     #     pred_pose = Pose(pose_keypoints, pose_entries[n][18])
#     #     current_poses.append(pred_pose)
#     #     # pred_pose.draw(human_img)
#     #     # cv2.imshow("TEst2", human_img)
#     #     # cv2.waitKey(0)
#     #     # temp_metric = custom_model_profiler.pck_metric(y_true=gt_pose_for_metrics,
#     #     #                                                y_pred=pred_pose.keypoints, thr=0.50)
#     #
#     #     temp_list.append(pred_pose.confidence)
#     #
#     # return current_poses[temp_list.index(np.max(temp_list))]
#     return coco_keypoints, pose_entries

def load_pose_estimation_model(classic_or_onnx: str, model_path: str, config_path: str, smooth_cfg_path: str):
    if classic_or_onnx == 'onnx':
        raise NotImplementedError
    else:
        dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        pose_model = init_pose_model(config_path, model_path, device=dev)
        # pose_model = MMDataParallel(pose_model, device_ids=[0])
        dataset = pose_model.cfg.data['test']['type']
        dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
        dataset_info['flip_pairs'] = None
        smoother = Smoother(filter_cfg=smooth_cfg_path, keypoint_dim=2)

        return pose_model, dataset, dataset_info, smoother
