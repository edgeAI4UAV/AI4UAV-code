import pyzed as sl




# Create a ZED camera object
zed = sl.Camera()

# Set SVO path for playback
input_path = "/home/user/Downloads/PX4-Python-SITL-2021-main/Test_Video.svo"
init_parameters = sl.InitParameters()
init_parameters.set_from_svo_file(input_path)

# Open the ZED
zed = sl.Camera()
err = zed.open(init_parameters)

svo_image_left = sl.Mat()
svo_image_right = sl.Mat()

while not exit_app:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        # Read side by side frames stored in the SVO
        zed.retrieve_image(svo_image_left, sl.VIEW.LEFT)
        zed.retrieve_image(svo_image_right, sl.VIEW.RIGHT)
        
        image_ocv_left = svo_image_left.get_data()
        image_ocv_right = svo_image_right.get_data()
        
        cv2.imshow("LEFT", image_ocv_left)
        cv2.imshow("RIGHT", image_ocv_right)
        cv2.waitKey(0)
        # Get frame count
        svo_position = zed.get_svo_position();
    elif zed.grab() == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
        print("SVO end has been reached. Looping back to first frame")
        zed.set_svo_position(0)

