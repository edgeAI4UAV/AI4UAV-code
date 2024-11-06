import math

# ~ import torchinfo
import numpy as np
import pandas as pd
import cv2
# ~ import albumentations
import os
# ~ import pynvml
# ~ from ptflops import get_model_complexity_info
import torch
import contextlib
import sys
import json
import shutil


def calc_iou_individual(pred_box, gt_box):
    """Calculate IoU of single predicted and ground truth box

    Args:
        pred_box (list of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (list of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

    Returns:
        float: value of the IoU for the two boxes.

    Raises:
        AssertionError: if the box is obviously malformed
    """
    x1_t, y1_t, x2_t, y2_t = gt_box
    x1_p, y1_p, x2_p, y2_p = pred_box

    if (x1_p > x2_p) or (y1_p > y2_p):
        raise AssertionError(
            "Prediction box is malformed? pred box: {}".format(pred_box))
    if (x1_t > x2_t) or (y1_t > y2_t):
        raise AssertionError(
            "Ground Truth box is malformed? true box: {}".format(gt_box))

    if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
        return 0.0

    far_x = np.min([x2_t, x2_p])
    near_x = np.max([x1_t, x1_p])
    far_y = np.min([y2_t, y2_p])
    near_y = np.max([y1_t, y1_p])

    inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
    true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
    pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    iou = inter_area / (true_box_area + pred_box_area - inter_area)
    return iou


class ModelPerformanceProfiler(object):

    def __init__(self, model_name, save_path, image_save_path, gt_boxes_path):
        self.skeleton_points_data = None
        self.human_pose_points = None
        # ~ pynvml.nvmlInit()
        # 'nose', 'neck', 'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri', 'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank', 'r_eye', 'l_eye', 'r_ear', 'l_ear'
        # self.human_pose_points = {"nose": [], "neck": [], 'rightShoulder': [], 'rightElbow': [],
        #                           'rightWrist': [], 'leftShoulder': [], 'leftElbow': [], 'leftWrist': [],
        #                           'rightHip': [], 'rightKnee': [], 'rightAnkle': [], 'leftHip': [],
        #                           'leftKnee': [], 'leftAnkle': [], "rightEye": [], "leftEye": [],
        #                           'rightEar': [], 'leftEar': [], 'video_image_name': []}
        self.human_pose_name_points = ["nose", "leftEye", "rightEye", 'leftEar', 'rightEar',
                                       'leftShoulder', 'rightShoulder', 'leftElbow', 'rightElbow',
                                       'leftWrist', 'rightWrist', 'leftHip', 'rightHip', 'leftKnee', 'rightKnee',
                                       'leftAnkle', 'rightAnkle',
                                       'video_img_name']  # "neck", 'rightEar', 'leftEar', "nose",

        self.bbox_dict = {'bbox_left': [], 'bbox_top': [], 'bbox_width': [], 'bbox_height': []}

        # 'person', 'nose',
        # 'left_eye', 'right_eye',
        # 'left_ear', 'right_ear',
        # 'left_shoulder', 'right_shoulder',
        # 'left_elbow', 'right_elbow',
        # 'left_wrist', 'right_wrist',
        # 'left_hip', 'right_hip',
        # 'left_knee', 'right_knee',
        # 'left_ankle', 'right_ankle'
        # self.model = model
        self.model_name = model_name
        self.save_path = save_path
        self.image_save_path = image_save_path
        self.gt_skeleton_files = gt_boxes_path
        # self.gt_skeleton_files = os.listdir(gt_boxes_path)
        # self.gt_skeleton_files.sort()
        self.gt_skeleton_files_path = gt_boxes_path
        # ~ self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.mW_to_W = 1e3

        # os.makedirs(image_save_path, exist_ok=True)
        # save_path = os.path.join(os.getcwd(), save_folder)

    def get_image_boxes_and_convert_xywh2xyxy_coords(self, image_file_num, all_flag=False):

        bbox_cords = []
        if all_flag:
            for x1, y1, w, h in zip(self.bbox_dict['bbox_left'], self.bbox_dict['bbox_top'],
                                    self.bbox_dict['bbox_width'], self.bbox_dict['bbox_height']):
                # intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
                intbox = [x1, y1, w, h]
                bbox_cords.append(intbox)

        else:

            image_index_in_annots = [idx for idx, item in enumerate(self.bbox_dict['frame_index']) if
                                     item == image_file_num]

            for indexxx in image_index_in_annots:
                x1 = self.bbox_dict['bbox_left'][indexxx]
                y1 = self.bbox_dict['bbox_top'][indexxx]
                w = self.bbox_dict['bbox_width'][indexxx]
                h = self.bbox_dict['bbox_height'][indexxx]
                intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))

                bbox_cords.append(intbox)

        return bbox_cords

    def create_save_dir(self):
        if os.path.exists(self.save_path) or os.path.exists(self.image_save_path):
            shutil.rmtree(self.save_path)
            shutil.rmtree(self.image_save_path)
            os.mkdir(self.save_path)
            os.mkdir(self.image_save_path)
        else:
            os.mkdir(self.save_path)
            os.mkdir(self.image_save_path)

        return

    # ~ def measure_gpu_power(self):
        # ~ gpu_power_measure = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / self.mW_to_W
        # ~ return gpu_power_measure

    # ~ def measure_gpu_mem_clock(self):
        # ~ gpu_memory_clock_measure = pynvml.nvmlDeviceGetPcieThroughput(self.gpu_handle,
                                                                      # ~ pynvml.NVML_CLOCK_MEM) / 10e3
        # ~ return gpu_memory_clock_measure

    def read_gt_bbox_and_return(self, human_class_num: int):
        annotations_bbox = open(self.gt_skeleton_files)
        annotations_bboxes = annotations_bbox.readlines()
        for idx, bbox_info in enumerate(annotations_bboxes):
            annot_bbox_info = bbox_info.split(',')
            annot_bbox_info = [i if i == 'NaN' else float(i) for i in annot_bbox_info]
            if len(annot_bbox_info) == 8:
                cx = np.mean(annot_bbox_info[0::2])
                cy = np.mean(annot_bbox_info[1::2])
                x1 = min(annot_bbox_info[0::2])
                x2 = max(annot_bbox_info[0::2])
                y1 = min(annot_bbox_info[1::2])
                y2 = max(annot_bbox_info[1::2])
                A1 = np.linalg.norm(
                    [x1 - x2 for (x1, x2) in zip(annot_bbox_info[0:2], annot_bbox_info[2:4])]) * np.linalg.norm(
                    [x1 - x2 for (x1, x2) in zip(annot_bbox_info[2:4], annot_bbox_info[4:6])])
                A2 = (x2 - x1) * (y2 - y1)
                s = np.sqrt(A1 / A2)
                w = s * (x2 - x1) + 1
                h = s * (y2 - y1) + 1
                new_coords = [int(x1), int(y1), int(w), int(h)]
                [self.bbox_dict[key].append(int(var)) for var, key in zip(new_coords, self.bbox_dict.keys())]
            elif len(annot_bbox_info) == 4:

                if 'NaN' in annot_bbox_info:
                    new_coords = [0, 0, 0, 0]
                    [self.bbox_dict[key].append(int(var)) for var, key in zip(new_coords, self.bbox_dict.keys())]
                else:
                    x = annot_bbox_info[0]
                    y = annot_bbox_info[1]
                    w = annot_bbox_info[2]
                    h = annot_bbox_info[3]
                    cx = x + w / 2
                    cy = y + h / 2
                    new_coords = [int(x), int(y), int(w), int(h)]
                    [self.bbox_dict[key].append(int(var)) for var, key in zip(new_coords, self.bbox_dict.keys())]
            else:
                [self.bbox_dict[key].append(int(var)) for var, key in zip(annot_bbox_info, self.bbox_dict.keys())]

        # self.bbox_dict = {k: [x for i, x in enumerate(v) if self.bbox_dict['object_category'][i] == human_class_num]
        #                  for k, v in self.bbox_dict.items()}

        return self.bbox_dict

    def read_gt_and_return(self):
        self.skeleton_points_data = []

        for skeleton_file in self.gt_skeleton_files:
            skeleton_file_full_path = os.path.join(self.gt_skeleton_files_path, skeleton_file)
            self.human_pose_points = {}
            with open(skeleton_file_full_path) as jsonfile:
                # data = json.load(jsonfile)
                annot_json = json.load(jsonfile)
                # print(annot_json)

                # skeleton_points_data.append(annot_json)
                for joint in self.human_pose_name_points:
                    if joint in annot_json.keys():
                        self.human_pose_points[joint] = annot_json[joint]
                    else:
                        self.human_pose_points[joint] = list(annot_json.values())[0]
                self.skeleton_points_data.append(self.human_pose_points)

        return self.skeleton_points_data

    # ~ def export_model_summary(self, model):

        # ~ file_path = '{}_macs_and_params.log'.format(self.model_name)
        # ~ filerep = open(file_path, "w", encoding="utf-8")
        # ~ # model.forward = model.forward_dummy
        # ~ macs, params = get_model_complexity_info(model, (3,) + tuple([256, 128]), as_strings=true,
                                                 # ~ print_per_layer_stat=true, verbose=true,
                                                 # ~ input_constructor=none,
                                                 # ~ ost=filerep)
        # ~ filerep.close()
        # ~ with open('{}.log'.format(self.model_name), 'w', encoding="utf-8") as f2:
            # ~ report = torchinfo.summary(model, device='cuda')
            # ~ f2.write(str(report))
            # ~ # f2.write(outptt_report)
            # ~ # f2.close()
        # ~ return

    @staticmethod
    def model_fidelity_metrics_results(gt_boxes, pred_boxes, iou_thr):
        """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.

        Args:
            gt_boxes (list of list of floats): list of locations of ground truth
                objects as [xmin, ymin, xmax, ymax]
            pred_boxes (list): dict of dicts of 'boxes' (formatted like `gt_boxes`)
                and 'scores'
            iou_thr (float): value of IoU to consider as threshold for a
                true prediction.

        Returns:
            dict: true positives (int), false positives (int), false negatives (int)
        """

        all_pred_indices = range(len(pred_boxes))
        all_gt_indices = range(len(gt_boxes))
        if len(all_pred_indices) == 0:
            tp = 0
            fp = 0
            fn = len(gt_boxes)
            return {'true_pos': tp, 'true_neg': 0.0,
                    'false_pos': fp, 'false_neg': fn,
                    'acc': 0.0, 'false_positive_rate': 0.0, 'sensitivity': 0.0,
                    'specificity': 0.0, 'false_negative_rate': 0.0, 'precision': 0.0,
                    'recall': 0.0}
        if len(all_gt_indices) == 0:
            tp = 0
            fp = len(pred_boxes)
            fn = 0
            return {'true_pos': tp, 'true_neg': 0.0,
                    'false_pos': fp, 'false_neg': fn,
                    'acc': 0.0, 'false_positive_rate': 0.0, 'sensitivity': 0.0,
                    'specificity': 0.0, 'false_negative_rate': 0.0, 'precision': 0.0,
                    'recall': 0.0}

        gt_idx_thr = []
        pred_idx_thr = []
        ious = []
        for ipb, pred_box in enumerate(pred_boxes):
            print(f"metrics for: {ipb}/{len(pred_boxes)}")
            for igb, gt_box in enumerate(gt_boxes):
                iou = calc_iou_individual(pred_box, gt_box)
                if iou > iou_thr:
                    gt_idx_thr.append(igb)
                    pred_idx_thr.append(ipb)
                    ious.append(iou)

        args_desc = np.argsort(ious)[::-1]
        if len(args_desc) == 0:
            # No matches
            tp = 0
            fp = len(pred_boxes)
            fn = len(gt_boxes)
        else:
            gt_match_idx = []
            pred_match_idx = []
            for idx in args_desc:
                print(f"metrics 2 for: {idx}/{len(args_desc)}")

                gt_idx = gt_idx_thr[idx]
                pr_idx = pred_idx_thr[idx]
                # If the boxes are unmatched, add them to matches
                if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                    gt_match_idx.append(gt_idx)
                    pred_match_idx.append(pr_idx)
            tp = len(gt_match_idx)
            fp = len(pred_boxes) - len(pred_match_idx)
            fn = len(gt_boxes) - len(gt_match_idx)
            tn = (len(gt_boxes) + len(gt_boxes)) - (fp + fn + tp)
            accuracy = np.round(len(gt_match_idx) / len(gt_boxes))

            FPR = np.round(fp / (fp + tn), 2)
            sensitivity = np.round(tp / (tp + fn), 2)
            specificity = np.round(tn / (tn + fp), 2)
            FNR = np.round(fn / (tp + fn), 2)

            try:
                precision = np.round(tp / (tp + fp), 2)
            except ZeroDivisionError:
                precision = 0.0
            try:
                recall = np.round(tp / (tp + fp), 2)
            except ZeroDivisionError:
                recall = 0.0

        return {'true_pos': tp, 'true_neg': tn,
                'false_pos': fp, 'false_neg': fn,
                'acc': accuracy, 'false_positive_rate': FPR, 'sensitivity': sensitivity,
                'specificity': specificity, 'false_negative_rate': FNR, 'precision': precision,
                'recall': recall}

    def export_hardware_metrics(self, dict_results_list):
        mean_dict = {}
        for key in dict_results_list[0].keys():
            # if key == 'number_of_test_frames':
            #     mean_dict[key] = dict_results_list[-1][key]
            # else:
            mean_dict[key] = np.round(np.mean([d[key] for d in dict_results_list]), 2)

        df = pd.DataFrame({key: pd.Series(val) for key, val in mean_dict.items()})
        df.to_csv(f"{self.model_name}_HARDWARE.csv")
        return

    def export_dict(self, dict_exp, name):
        df = pd.DataFrame({key: pd.Series(val) for key, val in dict_exp.items()})
        df.to_csv(name)
        return

    def export_model_resources(self, model_report):
        text_file = open(f"{self.model_name}_hardware_profile.txt", "w")

        # write string to file
        text_file.write(model_report.key_averages().table(sort_by="cuda_memory_usage", row_limit=-1))

        # close file
        text_file.close()
        return

    @staticmethod
    def draw_and_export_prediction_images(gt_image, pred_image, image_save_path):
        x, y, w, h = 0, 0, 200, 100
        # Create background rectangle with color
        # cv2.putText(img=gt_image, text="Ground Truth", org=(x + int(w / 20), y + int(h / 2.0)),
        #             fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 255, 0), thickness=4)
        #
        # cv2.putText(img=pred_image, text="Predicted", org=(x + int(w / 20), y + int(h / 2.0)),
        #             fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 255, 0), thickness=4)

        Hori = np.concatenate((pred_image, gt_image), axis=1)
        # cv2.imshow("Test", Hori)
        # cv2.waitKey(0)
        cv2.imwrite(image_save_path, Hori)
        return

    @staticmethod
    def draw_bbox(image, bboxes):
        if type(bboxes[0]) == list:
            for boxx in bboxes:
                # x1 = int(boxx[0])
                # y1 = int(boxx[1])
                # x2 = int(boxx[2])
                # y2 = int(boxx[3])
                #
                intbox = tuple(map(int, (boxx[0], boxx[1], boxx[0] + boxx[2], boxx[1] + boxx[3])))
                xmin = intbox[0]
                ymin = intbox[1]
                xmax = intbox[2]
                ymax = intbox[3]
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=1)
            # cv2.rectangle(image, (int(boxx[0]), int(boxx[1])), (int(boxx[2]), int(boxx[3])), color=(0, 255, 0), thickness=1)
        else:
            # x1 = int(bboxes[0])
            # y1 = int(bboxes[1])
            # x2 = int(bboxes[2])
            # y2 = int(bboxes[3])

            intbox = tuple(map(int, (bboxes[0], bboxes[1], bboxes[0] + bboxes[2], bboxes[1] + bboxes[3])))
            xmin = intbox[0]
            ymin = intbox[1]
            xmax = intbox[2]
            ymax = intbox[3]
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=1)
            # cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)

        # cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)
        return image

    @staticmethod
    def get_bbox(keypoints):
        bboxx = []
        found_keypoints = np.zeros((np.count_nonzero(keypoints[:, 0] != -1), 2), dtype=np.int32)
        found_kpt_id = 0
        for kpt_id in range(17):
            if keypoints[kpt_id, 0] == -1:
                continue
            found_keypoints[found_kpt_id] = keypoints[kpt_id]
            found_kpt_id += 1
        bbox = cv2.boundingRect(found_keypoints)
        bboxx.append(bbox[0])
        bboxx.append(bbox[1])
        bboxx.append(bbox[2] + bbox[0])
        bboxx.append(bbox[3] + bbox[1])
        return bboxx

    @staticmethod
    def calculate_gt_box_diagonal(box_points):
        box_point_A = [box_points[0], box_points[1]]
        box_point_B = [box_points[2], box_points[3]]

        differences = np.array(box_point_B) - np.array(box_point_A)
        distance = np.sqrt(np.dot(differences.T, differences))  # / 100
        return distance

    # @staticmethod
    # def resize_bbox(bbox, in_size, out_size):
    #     """Resize bounding boxes according to image resize.
    #     Args:
    #         bbox (~numpy.ndarray): See the table below.
    #         in_size (tuple): A tuple of length 2. The height and the width
    #             of the image before resized.
    #         out_size (tuple): A tuple of length 2. The height and the width
    #             of the image after resized.
    #     .. csv-table::
    #         :header: name, shape, dtype, format
    #         :obj:`bbox`, ":math:`(R, 4)`", :obj:`float32`, \
    #         ":math:`(y_{min}, x_{min}, y_{max}, x_{max})`"
    #     Returns:
    #         ~numpy.ndarray:
    #         Bounding boxes rescaled according to the given image shapes.
    #     """
    #     y_scale = float(out_size[0]) / float(in_size[0])
    #     x_scale = float(out_size[1]) / float(in_size[1])
    #     bbox[0] = int(y_scale * bbox[0])
    #     bbox[2] = int(y_scale * bbox[2])
    #     bbox[1] = int(x_scale * bbox[1])
    #     bbox[3] = int(x_scale * bbox[3])
    #     return bbox
    #
    # def resize_image(self, img_arr, bboxes, new_size):
    #     """
    #     :param img_arr: original image as a numpy array
    #     :param bboxes: bboxes as numpy array where each row is 'x_min', 'y_min', 'x_max', 'y_max', "class_id"
    #     :param h: resized height dimension of image
    #     :param w: resized weight dimension of image
    #     :return: dictionary containing {image:transformed, bboxes:['x_min', 'y_min', 'x_max', 'y_max', "class_id"]}
    #     """
    #     oh, ow = img_arr.shape[:2]
    #     height = new_size
    #     width = new_size
    #
    #     if oh > height or ow > width:
    #         # shrinking image algorithm
    #         interp = cv2.INTER_AREA
    #     else:
    #         # stretching image algorithm
    #         interp = cv2.INTER_CUBIC
    #
    #     ratio = ow / oh
    #     w = width
    #     h = round(w / ratio)
    #     if h > height:
    #         h = height
    #         w = round(oh * ratio)
    #     pad_bottom = abs(height - h)
    #     pad_right = abs(width - w)
    #     # bboxes.append(1)
    #     # create resize transform pipeline
    #     # transform = albumentations.Compose(
    #     #     [albumentations.LongestMaxSize(max_size=new_size, interpolation=1, always_apply=True),
    #     #     # [albumentations.Resize(height=new_size, width=new_size, always_apply=True),
    #     #      albumentations.PadIfNeeded(min_height=h, min_width=w, border_mode=0, value=(0, 0, 0))])
    #     # # , albumentations.BboxParams(format='yolo')
    #     # transformed = transform(image=img_arr)
    #     scaled_img = cv2.resize(img_arr, (w, h), interpolation=interp)
    #     padded_img = cv2.copyMakeBorder(
    #         scaled_img, 0, pad_bottom, 0, pad_right, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    #
    #     checked_coords = []
    #
    #     if type(bboxes[0]) == int:
    #         new_box = self.resize_bbox(bbox=bboxes, in_size=np.shape(img_arr)[:2], out_size=(224, 224))
    #         # transformed['bboxes'] = new_box
    #         return padded_img, new_box
    #     else:
    #         for bbox in bboxes:
    #             new_box = self.resize_bbox(bbox=list(bbox), in_size=np.shape(img_arr)[:2], out_size=(h, w))
    #             # if new_box[2] <= new_box[0] or new_box[3] <= new_box[1]:
    #             #     continue
    #             # elif new_box[0] >= w or new_box[2] >= w:
    #             #     continue
    #             # elif new_box[3] >= h or new_box[1] >= h:
    #             #     continue
    #             # else:
    #             #     checked_coords.append(new_box)
    #             checked_coords.append(new_box)
    #         # transformed['bboxes'] = checked_coords
    #         return padded_img, new_box

    def oks_metric(self, y_true_index, y_pred):
        y_true = self.skeleton_points_data[y_true_index]
        # Object Keypoint Similarity
        visibility = (np.ones((17, 1)) * 1).astype(int)
        # You might want to set these global constant
        # outside the function scope
        y_true = list(y_true.values())[:-1]
        # y_true += [[0, 0]]
        y_true = np.array(y_true, dtype=np.int32)
        y_pred = np.array(y_pred, dtype=np.int32)
        KAPPA = np.array([1] * len(y_true))
        # The object scale
        # You might need a dynamic value for the object scale
        SCALE = 20.0

        # Compute the L2/Euclidean Distance
        distances = np.linalg.norm(y_pred - y_true, axis=-1)
        # Compute the exponential part of the equation
        exp_vector = np.exp(-(distances ** 2) / (2 * (SCALE ** 2) * (KAPPA ** 2)))
        # The numerator expression
        numerator = np.dot(exp_vector, visibility.astype(bool).astype(int))
        # The denominator expression
        denominator = np.sum(visibility.astype(bool).astype(int))
        return numerator / denominator

    def pck_metric(self, y_pred, y_true_index, thr):
        y_true = self.skeleton_points_data[y_true_index]

        y_true = list(y_true.values())[:-1]
        # y_true += [[0, 0]]
        y_true = np.array(y_true, dtype=np.int32)

        y_pred = np.array(y_pred, dtype=np.int32)

        num_points, _ = y_pred.shape
        results = np.full(num_points, 0, dtype=np.float32)
        thrs = []

        # for i in range(num_imgs):
        for i in range(num_points):
            differences = np.array(y_pred[i]) - np.array(y_true[i])
            distance = np.sqrt(np.dot(differences.T, differences)) / 100
            if distance <= thr:
                results[i] = 1

        thrs = np.array(thrs)

        mean_points = np.mean(results, axis=0)
        mean_all = np.mean(mean_points)

        return mean_points

    def pdj_metric(self, y_pred, y_true_index):

        fraction = 0.05
        y_true = self.skeleton_points_data[y_true_index]

        y_true = list(y_true.values())[:-1]
        # y_true += [[0, 0]]
        y_true = np.array(y_true, dtype=np.int32)
        y_true_bbox = self.get_bbox(keypoints=y_true)
        diagonal_bbox = self.calculate_gt_box_diagonal(box_points=y_true_bbox)
        y_pred = np.array(y_pred, dtype=np.int32)

        num_points, _ = y_pred.shape
        results = np.full(num_points, 0, dtype=np.float32)
        thrs = []

        # for i in range(num_imgs):
        for i in range(num_points):
            differences = np.array(y_pred[i]) - np.array(y_true[i])
            distance = np.sqrt(np.dot(differences.T, differences))
            if distance <= fraction * diagonal_bbox:
                results[i] = 1

        mean_points = np.mean(results)

        return mean_points
