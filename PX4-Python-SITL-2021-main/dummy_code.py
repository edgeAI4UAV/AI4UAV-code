import cv2
from Hardware_Monitor_And_Metrics import ModelPerformanceProfiler
from imutils.video import FPS
import os
import imutils
from Person_Tracker_Module import load_tracker_config, load_tracker_model
from Person_PoseEstimation_Module import load_pose_estimation_model
from loguru import logger
import pandas as pd 
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result, inference_bottom_up_pose_model, process_mmdet_results, get_track_id)
from mmpose.core import Smoother
from time import time

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

pose_api_dir = 'PoseEstimation_Framework'
pose_estimation_model_config = 'alexnet_coco_256x192.py'
pose_estimation_model = 'alexnet_coco_256x192.pth'
pose_estimation_filter = '_base_/filters/one_euro.py'

VISION_SYSTEM_PARAMETERS = {
    'tracker_name': 'siamfcpp_alexnet_votlt', #  siamfcpp_alexnet_votlt
    'person_init_frame_coords_filepath': os.path.join(os.getcwd(), 'person_bbox_coordinates.json'),
    'person_init_frame_filepath': os.path.join(os.getcwd(), 'init_frame.png'),
    'pose_estimation_models_weights': os.path.join(os.getcwd(), pose_api_dir, pose_estimation_model),
    'pose_estimation_models_config': os.path.join(os.getcwd(),  pose_api_dir, pose_estimation_model_config),
    'pose_estimation_smooth_config': os.path.join(os.getcwd(),  pose_api_dir, pose_estimation_filter)
}

params = {'image_height': None, 'image_width': None, 'resized_height': None, 'resized_width': None,
          'x_ax_pos': None, 'y_ax_pos': None, 'cent_rect_half_width': None, 'cent_rect_half_height': None,
          'cent_rect_p1': None, 'cent_rect_p2': None, 'scaling_factor': None, 'min_tgt_radius': None}
          
class VisionSystem:
    def __init__(self, vision_system_params: dict):
        logger.info("Load tracker config...")
        tracker_config = load_tracker_config(tracker_name=vision_system_params['tracker_name'])
        logger.info("Load and build tracker model...")
        self.siamfcpp_tracker = load_tracker_model(tracker_config=tracker_config)

        logger.success("Person tracking algorithm successfully loaded...")
        logger.info("Person tracking algorithm initialization...")

        init_human_person_coordinates = pd.read_json(vision_system_params['person_init_frame_coords_filepath'],
                                                     typ='series')
        init_human_person_image = cv2.imread(vision_system_params['person_init_frame_filepath'], 1)

        init_box = [init_human_person_coordinates['x_pixel_coordinate'],
                    init_human_person_coordinates['y_pixel_coordinate'],
                    init_human_person_coordinates['bbox_width_coordinates'],
                    init_human_person_coordinates['bbox_height_coordinates']]


        logger.info("Load Pose Estimation Model...")
        self.pose_estimation_model, self.dataset, self.dataset_info, self.smoother = load_pose_estimation_model(
            classic_or_onnx='Classic',
            model_path=vision_system_params['pose_estimation_models_weights'],
            config_path=vision_system_params['pose_estimation_models_config'],
            smooth_cfg_path=vision_system_params['pose_estimation_smooth_config'])

        logger.success("Person pose estimation model successfully loaded...")

	
    def initialize_tracker(self, init_human_person_image, init_box):
        self.siamfcpp_tracker.init(init_human_person_image, init_box)
        logger.info("Person tracking model initialized...")
        
    def get_target_coordinates(self, frame):
        """ Detects a target by using color range segmentation and returns its 'camera pixel' coordinates."""
        logger.info("Person track execution started...")

        # Resize the image frame for the detection process, if needed
        if params['scaling_factor'] != 1:
            dimension = (params['resized_width'], params['resized_height'])
            #frame = cv2.resize(frame, dimension, interpolation=cv2.INTER_AREA)

        pred_bbox = self.siamfcpp_tracker.update(frame)
        pred_bbox_confidence = self.siamfcpp_tracker.get_track_score()
        logger.info(f"Tracker confidence score: {pred_bbox_confidence} with predicted bounding box: {pred_bbox}")
        selected_person_pose = []
        if pred_bbox_confidence > 0.60:

            roi_for_pose = frame[int(pred_bbox[1]): int(pred_bbox[1] + pred_bbox[3]),
                           int(pred_bbox[0]): int(pred_bbox[0] + pred_bbox[2])]

            pred_bbox = [int(val) for val in pred_bbox]
            pose_bbox = [{'bbox': pred_bbox}]
            current_poses = []
            pose_results, returned_outputs = inference_top_down_pose_model(
                self.pose_estimation_model,
                frame,
                pose_bbox,
                bbox_thr=None,
                format='xywh',
                dataset=self.dataset,
                dataset_info=self.dataset_info,
                return_heatmap=False
            )
            pose_results = self.smoother.smooth(pose_results)
            try:
                for pose_point in pose_results:
                    human_pose_results = pose_point['keypoints'][:, :2]
                    human_pose_results = [list(map(int, human_pose_res)) for human_pose_res in human_pose_results]
                    current_poses.append(human_pose_results)
            except IndexError:
                bbox_xmin = pred_bbox[0]
                bbox_ymin = pred_bbox[1]
                bbox_xmax = pred_bbox[0] + pred_bbox[2]
                bbox_ymax = pred_bbox[1] + pred_bbox[3]
                # person_bbox_center = [int((pred_bbox[0] + pred_bbox[2]) / 2), int((pred_bbox[1] + pred_bbox[3]) / 2)]
                person_bbox_center = [int((bbox_xmin + bbox_xmax) / 2), int((bbox_ymin + bbox_ymax) / 2)]
                # person_bbox_center = [int(pred_bbox[0] + pred_bbox[2]), int(pred_bbox[1] + pred_bbox[3])]
                cv2.circle(frame, (person_bbox_center[0], person_bbox_center[1]), 1, (255, 0, 255), -1)

            try:

                selected_person_pose = current_poses[0]
                for keypoints in selected_person_pose:
                     cv2.circle(frame, (int(keypoints[0]), int(keypoints[1])),
                                3, (255, 0, 255), -1)

                left_shoulder = selected_person_pose[5]
                right_shoulder = selected_person_pose[6]

                chest_center = (int((left_shoulder[0] / 2) + (right_shoulder[0] / 2)),
                                int((left_shoulder[1] / 2) + (right_shoulder[1] / 2)))

                person_bbox_center = chest_center
                # logger.info(
                #     f" Left shoulder point: {left_shoulder},"
                #     f" Right shoulder point: {right_shoulder}, Chest point: {chest_center}")
                # cv2.circle(frame, (chest_center[0], chest_center[1]),
                #            3, (255, 0, 255), -1)
                # cv2.circle(frame, (left_shoulder[0], left_shoulder[1]),
                #            3, (255, 0, 255), -1)
                # cv2.circle(frame, (right_shoulder[0], right_shoulder[1]),
                #            3, (255, 0, 255), -1)
            except IndexError:
                bbox_xmin = pred_bbox[0]
                bbox_ymin = pred_bbox[1]
                bbox_xmax = pred_bbox[0] + pred_bbox[2]
                bbox_ymax = pred_bbox[1] + pred_bbox[3]
                # person_bbox_center = [int((pred_bbox[0] + pred_bbox[2]) / 2), int((pred_bbox[1] + pred_bbox[3]) / 2)]
                person_bbox_center = [int((bbox_xmin + bbox_xmax) / 2), int((bbox_ymin + bbox_ymax) / 2)]
                # person_bbox_center = [int(pred_bbox[0] + pred_bbox[2]), int(pred_bbox[1] + pred_bbox[3])]
                # cv2.circle(frame, (person_bbox_center[0], person_bbox_center[1]), 1, (255, 0, 255), -1)

            # # logger.info(
            # #     f"Person pose: {selected_person_pose}")
            #
            
            (x, y, w, h) = [int(v) for v in pred_bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)
            cv2.imshow("PredPose", frame)
            cv2.waitKey(1)
            # image_center_x = int(frame.shape[1] // 2)
            # image_center_y = int(frame.shape[0] // 2)
            # h, w = frame.shape[0:2]
            # bbox_xmin = pred_bbox[0]
            # bbox_ymin = pred_bbox[1]
            # bbox_xmax = pred_bbox[0] + pred_bbox[2]
            # bbox_ymax = pred_bbox[1] + pred_bbox[3]
            # person_bbox_center = [int((bbox_xmin + bbox_xmax) / 2), int((bbox_ymin + bbox_ymax) / 2)]

            # cv2.imshow("Test Point", frame)
            # cv2.waitKey(0)
            # dY, dX = h // 2 - person_bbox_center[0], person_bbox_center[1] - w // 2
            logger.success("Person track and pose successfully execution finished...")
            return {'width': None, #person_bbox_center[1], person_bbox_center[0]} person_bbox_center[0] person_bbox_center[1]
                    'height': None}, frame, pred_bbox, selected_person_pose
        else:
            person_bbox_center = [None, None]
            selected_person_pose = None
            logger.error("Person track and pose not found...")
            return {'width': person_bbox_center[1],
                    'height': person_bbox_center[0]}, frame, pred_bbox, selected_person_pose
    
def main():
	
	imgs_path = '/home/user/Downloads/Tracker_Dataset/group3'
	annots_path = '/home/user/Downloads/Tracker_Dataset/group3.txt'
	log_dir_path = '/home/user/Downloads/Tracker_Dataset/log_dir'
	out_img_path = '/home/user/Downloads/Tracker_Dataset/predicted_Images'
	
	model_name = "Test_Model"
	vision_sub_system = VisionSystem(vision_system_params=VISION_SYSTEM_PARAMETERS)

	images_files = os.listdir(imgs_path)
	images_files.sort()
    
	images_files_paths = [os.path.join(imgs_path, img_file) for img_file in images_files]
	custom_model_profiler = ModelPerformanceProfiler(model_name, log_dir_path, out_img_path, annots_path)
    #
	gt_skeleton_img_points = custom_model_profiler.read_gt_bbox_and_return(human_class_num=1)
	transformed_bbox_cordss = custom_model_profiler.get_image_boxes_and_convert_xywh2xyxy_coords(
        image_file_num=0, all_flag=True)

	frame_provider = ImageReader(images_files_paths, transformed_bbox_cordss)
    
	frames_counter = 0
	fps = FPS().start()
	# ~ times = []
	for orig_img, gt_bbox in frame_provider:
		start_time = time()

		key = 255
		frame = imutils.resize(orig_img, width=600)
		resized_box = resize_bbox(bbox=gt_bbox, in_size=orig_img.shape[0:2], out_size=frame.shape[0:2])
		if frames_counter == 0:
			init_box = resized_box
			vision_sub_system.initialize_tracker(frame, init_box)
		
		# ~ cv2.imshow("FRAME", frame)
		# ~ cv2.waitKey(0)
		tgt_cam_coord, frame, person_bbox, person_pose = vision_sub_system.get_target_coordinates(frame)
		# ~ times.append(time() - t1)
		fps.update()
		fps.stop()

		logger.info(f"--Original image width, height: {params['image_width']}, {params['image_height']}")
		logger.info(f"--Target camera coordinates: {tgt_cam_coord}")
		# ~ cv2.imshow("Frame", frame)
		# ~ cv2.waitKey(1)
		frames_counter += 1
		fps.update()
		fps.stop()
		print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
		print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
		print("FPS: ", 1.0 / (time() - start_time))


if __name__ == "__main__":
    main()
