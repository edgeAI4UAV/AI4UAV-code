#!/usr/bin/env python
# ------ To run the code with a real drone connected to the PC via telemetry modules ------ #
# await drone.connect(system_address="serial:///dev/ttyUSB0:57600")
# CAUTION: Color range segmentation is not the best approach to detect and track objects
#          This example is just a didactic proof of concept. Don't run it with a real drone
#          unless you have good experience flying real drones and know what you are doing.
# ----------------------------------------------------------------------------------------- #

from mavsdk import System
from mavsdk.offboard import (OffboardError, PositionNedYaw)
from mavsdk.gimbal import GimbalMode, ControlMode
# from cvzone.PoseModule import PoseDetector
import asyncio
import cv2
import numpy as np
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst


# detector = PoseDetector()

# Constants
PERCENT_CENTER_RECT = 0.05  # For calculating the center rectangle's size
PERCENT_TARGET_RADIUS = 0.1 * PERCENT_CENTER_RECT  # Minimum target radius to follow
HOVERING_ALTITUDE = 3.0  # Altitude in meters to which the drone will perform its tasks
NUM_FILT_POINTS = 50  # Number of filtering points for the Moving Average Filter
DESIRED_IMAGE_HEIGHT = 480  # A smaller image makes the detection less CPU intensive

# A dictionary of two empty buffers (arrays) for the Moving Average Filter
filt_buffer = {'width': [], 'height': []}

# A dictionary of general parameters derived from the camera image size,
# which will be populated later with the 'get_image_params' function
params = {'image_height': None, 'image_width': None, 'resized_height': None, 'resized_width': None,
          'x_ax_pos': None, 'y_ax_pos': None, 'cent_rect_half_width': None, 'cent_rect_half_height': None,
          'cent_rect_p1': None, 'cent_rect_p2': None, 'scaling_factor': None, 'min_tgt_radius': None}


class Video():
    """BlueRov video capture class constructor

    Attributes:
        port (int): Video UDP port
        video_codec (string): Source h264 parser
        video_decode (string): Transform YUV (12bits) to BGR (24bits)
        video_pipe (object): GStreamer top-level pipeline
        video_sink (object): Gstreamer sink element
        video_sink_conf (string): Sink configuration
        video_source (string): Udp source ip and port
    """

    def __init__(self, port=5600):
        """Summary

        Args:
            port (int, optional): UDP port
        """

        Gst.init(None)  #

        self.port = port
        self._frame = None

        # [Software component diagram](https://www.ardusub.com/software/components.html)
        # UDP video stream (:5600)
        self.video_source = 'udpsrc port={}'.format(self.port)
        # [Rasp raw image](http://picamera.readthedocs.io/en/release-0.7/recipes2.html#raw-image-capture-yuv-format)
        # Cam -> CSI-2 -> H264 Raw (YUV 4-4-4 (12bits) I420)
        self.video_codec = ' ! application/x-rtp, payload=96 ! rtph264depay ! h264parse ! avdec_h264'
        # Python don't have nibble, convert YUV nibbles (4-4-4) to OpenCV standard BGR bytes (8-8-8)
        self.video_decode = \
            '! decodebin ! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert'
        # Create a sink to get data
        self.video_sink_conf = \
            '! appsink emit-signals=true sync=false max-buffers=2 drop=true'

        self.video_pipe = None
        self.video_sink = None

        self.run()

    def start_gst(self, config=None):
        """ Start gstreamer pipeline and sink
        Pipeline description list e.g:
            [
                'videotestsrc ! decodebin', \
                '! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert',
                '! appsink'
            ]

        Args:
            config (list, optional): Gstreamer pileline description list
        """

        if not config:
            config = \
                [
                    'videotestsrc ! decodebin',
                    '! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert',
                    '! appsink'
                ]

        command = ''.join(config)
        print(command)
        self.video_pipe = Gst.parse_launch(command)
        self.video_pipe.set_state(Gst.State.PLAYING)
        self.video_sink = self.video_pipe.get_by_name('appsink0')

    @staticmethod
    def gst_to_opencv(sample):
        """Transform byte array into np array

        Args:
            sample (TYPE): Description

        Returns:
            TYPE: Description
        """
        buf = sample.get_buffer()
        caps = sample.get_caps()
        array = np.ndarray(
            (
                caps.get_structure(0).get_value('height'),
                caps.get_structure(0).get_value('width'),
                3
            ),
            buffer=buf.extract_dup(0, buf.get_size()), dtype=np.uint8)
        return array

    def frame(self):
        """ Get Frame

        Returns:
            iterable: bool and image frame, cap.read() output
        """
        return self._frame

    def frame_available(self):
        """Check if frame is available

        Returns:
            bool: true if frame is available
        """
        return type(self._frame) != type(None)

    def run(self):
        """ Get frame to update _frame
        """

        self.start_gst(
            [
                self.video_source,
                self.video_codec,
                self.video_decode,
                self.video_sink_conf
            ])

        self.video_sink.connect('new-sample', self.callback)

    def callback(self, sink):
        sample = sink.emit('pull-sample')
        new_frame = self.gst_to_opencv(sample)
        self._frame = new_frame

        return Gst.FlowReturn.OK


class Drone:
    @staticmethod
    async def fly():
        # destination_longitude = 40.929480
        # destination_latitude = 24.381135
        global current_longitude, current_latitude, absolute_altitude, altitude, Distance

        # Get a reference to a 'System' object, which represents the drone,
        # and open a connection to it. The system address used here is the default
        # address for a simulated drone running in the same machine machine where
        # this code will run (localhost)
        uav = System()
        await uav.connect(system_address="udp://:14540")  # Connects to the UAV
        # Asynchronously poll the connection state until receiving an 'is_connected' confirmation
        print("Establishing Connection...")

        # Check CONNECTION & if there is a positive connection Feedback = CONNECTED
        async for state in uav.core.connection_state():
            if state.is_connected:
                print("UAV target UUID: {state.uuid}")  # Prints the UUID of the UAV to which the system connected
                print(f"-- Connected to drone!")
                break

        async for gps_info in uav.telemetry.gps_info():
            print(f"GPS info: {gps_info}")
            break
        print("Establishing GPS lock on UAV..")
        # Checks the gps Connection via telemetry health command
        print("Waiting for drone to have a global position estimate...")
        async for health in uav.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                print("-- Global position estimate OK")
                break

        print("Fetching amsl altitude at home location....")
        async for terrain in uav.telemetry.home():
            absolute_altitude = terrain.absolute_altitude_m
            print(absolute_altitude)
            break

        async for position in uav.telemetry.position():
            print(position)
            altitude = position.relative_altitude_m
            current_latitude = position.latitude_deg
            current_longitude = position.longitude_deg
            break

        async for flight_mode in uav.telemetry.flight_mode():
            print(f"Flight mode: {flight_mode}")
            break

        # distance
        async for distance in uav.telemetry.distance_sensor():
            for i in range(2):
                print(f"distance %d: {distance.current_distance_m}" % i)
            break

        print("Taking control of gimbal")
        await uav.gimbal.take_control(ControlMode.PRIMARY)

        print("Look down")
        await uav.gimbal.set_pitch_and_yaw(-35, 0)
        await asyncio.sleep(2)

        # Activate the drone motors
        print("Arming UAV")
        await uav.action.arm()

        # print("Taking Off")
        # await uav.action.takeoff()
        await asyncio.sleep(2)

        # Send an initial position to the drone before changing to "offboard" flight mode
        print("-- Setting initial setpoint")
        await uav.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 0.0))

        print("-- Starting offboard")
        try:
            await uav.offboard.start()
        except OffboardError as error:
            print(f"Starting offboard mode failed with error code: {error._result.result}")
            print("-- Disarming")
            await uav.action.disarm()
            return

        # Variables to store the NED coordinates, plus Yaw angle. Default pose:
        N_coord = 0
        E_coord = 0
        D_coord = -HOVERING_ALTITUDE  # The drone will always detect and track at HOVERING_ALTITUDE
        yaw_angle = 0  # Drone always points to North

        # Make the drone go (take off) to the default pose
        await uav.offboard.set_position_ned(PositionNedYaw(N_coord, E_coord, D_coord, yaw_angle))
        await asyncio.sleep(4)  # Give the drone time to gain altitude

        video = Video()
        while True:
            async for distance in uav.telemetry.distance_sensor():
                # print(f"distance %d: {distance.current_distance_m}" % i)
                Distance = distance.current_distance_m
                break
            # Wait for the next frame
            if not video.frame_available():
                continue
            frame = video.frame()
            await get_image_params(frame)
            print(f"-- Original image width, height: {params['image_width']}, {params['image_height']}")
            cv2.imshow("Frame", frame)
            cv2.waitKey(0)
            # Get the target coordinates (if any target was detected)
            tgt_cam_coord, frame, contour = await get_target_coordinates(frame)

            # If a target was found, filter their coordinates
            if tgt_cam_coord['width'] is not None and tgt_cam_coord['height'] is not None:
                # Apply Moving Average filter to target camera coordinates
                tgt_filt_cam_coord = await moving_average_filter(tgt_cam_coord)
                status = "Detect Target"
                # if Distance > 2000 and Distance < 4000:
                # N_coord = N_coord - 1
                # else:
                # N_coord = N_coord + 1
                # await uav.offboard.set_position_ned(PositionNedYaw(N_coord, E_coord, D_coord, yaw_angle))


            # No target was found, set target camera coordinates to the Cartesian origin,
            # so the drone doesn't move
            else:
                # The Cartesian origin is where the x and y Cartesian axes are located
                # in the image, in pixel units
                tgt_cam_coord = {'width': params['y_ax_pos'],
                                 'height': params['x_ax_pos']}  # Needed just for drawing objects
                tgt_filt_cam_coord = {'width': params['y_ax_pos'], 'height': params['x_ax_pos']}
                status = "No Target"
                '''
                    if E_coord >= 15:
                        E_coord = E_coord - 1
                    elif E_coord < 15 and E_coord >= 0:
                        E_coord = E_coord + 1
                    else:
                        E_coord = E_coord
                    '''
                # E_coord = E_coord + 1
                # await uav.offboard.set_position_ned(PositionNedYaw(N_coord, E_coord, D_coord, yaw_angle))
            # Convert from camera coordinates to Cartesian coordinates (in pixel units)
            tgt_cart_coord = {'x': (tgt_filt_cam_coord['width'] - params['y_ax_pos']),
                              'y': (params['x_ax_pos'] - tgt_filt_cam_coord['height'])}

            # Compute scaling conversion factor from camera coordinates in pixel units
            # to Cartesian coordinates in meters
            COORD_SYS_CONV_FACTOR = 0.02

            # If the target is outside the center rectangle, compute North and East coordinates
            if abs(tgt_cart_coord['x']) > params['cent_rect_half_width'] or \
                    abs(tgt_cart_coord['y']) > params['cent_rect_half_height']:
                # Compute North, East coordinates applying "camera pixel" to Cartesian conversion factor
                E_coord = tgt_cart_coord['x'] * COORD_SYS_CONV_FACTOR
                N_coord = tgt_cart_coord['y'] * COORD_SYS_CONV_FACTOR
                # D_coord, yaw_angle don't change
            # print(N_coord, E_coord, D_coord, yaw_angle)
            '''
                yaw_angle = math.degrees(
                    math.atan((destination_longitude - current_longitude) / (destination_latitude - current_latitude)))
                if (destination_latitude != current_latitude) and (destination_latitude < current_latitude):
                    yaw_angle += 180
                '''
            # Command the drone to the current NED + Yaw pose
            await uav.offboard.set_position_ned(PositionNedYaw(N_coord, E_coord, D_coord, yaw_angle))
            # Draw objects over the detection image frame just for visualization
            frame = await draw_objects(tgt_cam_coord, tgt_filt_cam_coord, frame, contour)

            # draw the status text on the frame
            north_label = "y:"
            east_label = "x:"
            distance_label = "Distance:"
            cv2.putText(frame, str(north_label), (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, str(east_label), (5, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, str(distance_label), (5, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, str(N_coord), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, str(E_coord), (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            if status == "Detect Target":
                cv2.putText(frame, status, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                cv2.putText(frame, status, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, str(distance.current_distance_m), (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # detect all human body
            # frame = detector.findPose(frame)
            # lmList, bboxInfo = detector.findPosition(frame, bboxWithHands=True)

            cv2.imshow('Human Detector', frame)

            # Catch aborting key from computer keyboard
            key = cv2.waitKey(1) & 0xFF
            # If the 'q' key is pressed, break the 'while' infinite loop
            if key == ord("q"):
                # After leaving the infinite loop, return the drone to home before ending the script
                print("-- Return to launch...")
                await uav.action.return_to_launch()
                print("NOTE: check the drone has landed already before running again this script.")
                # await asyncio.sleep(10)
                # print("-- Disarming")
                # await uav.action.disarm()
                # await asyncio.sleep(2)
                cv2.destroyAllWindows()
                break


async def get_image_params(frame):
    params['image_height'], params['image_width'], _ = frame.shape

    # Compute the scaling factor to scale the image to a desired size
    if params['image_height'] != DESIRED_IMAGE_HEIGHT:
        params['scaling_factor'] = round((DESIRED_IMAGE_HEIGHT / params['image_height']), 2)  # Rounded scaling factor

    else:
        params['scaling_factor'] = 1

    # print("params['scaling_factor']: ", params['scaling_factor'])

    # Compute resized width and height and resize the image
    params['resized_width'] = int(params['image_width'] * params['scaling_factor'])
    params['resized_height'] = int(params['image_height'] * params['scaling_factor'])
    dimension = (params['resized_width'], params['resized_height'])
    frame = cv2.resize(frame, dimension, interpolation=cv2.INTER_AREA)

    # Compute the center rectangle's half width and half height
    params['cent_rect_half_width'] = round(
        params['resized_width'] * (0.5 * PERCENT_CENTER_RECT))  # Use half percent (0.5)
    params['cent_rect_half_height'] = round(
        params['resized_height'] * (0.5 * PERCENT_CENTER_RECT))  # Use half percent (0.5)

    # Compute the minimum target radius to follow. Smaller detected targets will be ignored
    params['min_tgt_radius'] = round(params['resized_width'] * PERCENT_TARGET_RADIUS)

    # Compute the position for the X and Y Cartesian coordinates in camera pixel units
    params['x_ax_pos'] = int(params['resized_height'] / 2 - 1)
    params['y_ax_pos'] = int(params['resized_width'] / 2 - 1)

    # Compute two points: p1 in the upper left and p2 in the lower right that will be used to
    # draw the center rectangle iin the image frame
    params['cent_rect_p1'] = (params['y_ax_pos'] - params['cent_rect_half_width'],
                              params['x_ax_pos'] - params['cent_rect_half_height'])
    params['cent_rect_p2'] = (params['y_ax_pos'] + params['cent_rect_half_width'],
                              params['x_ax_pos'] + params['cent_rect_half_height'])

    return


async def get_target_coordinates(frame):
    """ Detects a target by using color range segmentation and returns its 'camera pixel' coordinates."""

    # Use the 'threshold_inRange.py' script included with the code to get
    # your own bounds with any color
    # To detect a blue target:
    HSV_LOWER_BOUND = (107, 119, 41)
    HSV_UPPER_BOUND = (124, 255, 255)
    # To detect a red target:
    HSV_LOWER_BOUND = (160, 50, 50)
    HSV_UPPER_BOUND = (180, 255, 255)

    # Resize the image frame for the detection process, if needed
    if params['scaling_factor'] != 1:
        dimension = (params['resized_width'], params['resized_height'])
        frame = cv2.resize(frame, dimension, interpolation=cv2.INTER_AREA)

    # Blur the image to remove high frequency content
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)

    # Change color space from BGR to HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Histogram equalisation to minimize the effect of variable lighting
    # hsv[:, :, 0] = cv2.equalizeHist(hsv[:, :, 0]) # on the H-channel
    # hsv[:, :, 1] = cv2.equalizeHist(hsv[:, :, 1]) # on the S-channel
    # hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2]) # on the V-channel

    # Get a mask with all the pixels inside our defined color boundaries
    mask = cv2.inRange(hsv, HSV_LOWER_BOUND, HSV_UPPER_BOUND)

    # Erode and dilate to remove small blobs
    mask = cv2.erode(mask, None, iterations=3)
    mask = cv2.dilate(mask, None, iterations=3)

    # Find all contours in the masked image
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # _ , contours, _  = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Centroid coordinates to be returned:
    cX = None
    cY = None

    # To save the larges contour, presumably the detected object
    largest_contour = None

    # Check if at least one contour was found
    if len(contours) > 0:
        # Get the largest contour of all posibly detected
        largest_contour = max(contours, key=cv2.contourArea)

        # Compute the radius of an enclosing circle around the largest contour
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)

        # Compute centroid only if contour radius is larger than 0.5 half the center rectangle
        if radius > params['min_tgt_radius']:
            # Compute contour raw moments
            M = cv2.moments(largest_contour)
            # Get the contour's centroid
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

    # Return centroid coordinates (camera pixel units), the analized frame and the largest contour
    return {'width': cX, 'height': cY}, frame, largest_contour


async def moving_average_filter(coord):
    """ Applies Low-Pass Moving Average Filter to a pair of (x, y) coordinates."""

    # Append new coordinates to filter buffers
    filt_buffer['width'].append(coord['width'])
    filt_buffer['height'].append(coord['height'])

    # If the filters were full already with a number of NUM_FILT_POINTS values,
    # discard the oldest value (FIFO buffer)
    if len(filt_buffer['width']) > NUM_FILT_POINTS:
        filt_buffer['width'] = filt_buffer['width'][1:]
        filt_buffer['height'] = filt_buffer['height'][1:]

    # Compute filtered camera coordinates
    N = len(filt_buffer['width'])  # Get the number of values in buffers (will be < NUM_FILT_POINTS at the start)

    # Sum all values for each coordinate
    w_sum = sum(filt_buffer['width'])
    h_sum = sum(filt_buffer['height'])
    # Compute the average
    w_filt = int(round(w_sum / N))
    h_filt = int(round(h_sum / N))

    # Return filtered coordinates as a dictionary
    return {'width': w_filt, 'height': h_filt}


async def draw_objects(cam_coord, filt_cam_coord, frame, contour):
    """ Draws visualization objects from the detection process.
    Position coordinates of every object are always in 'camera pixel' units"""

    # Draw the Cartesian axes
    cv2.line(frame, (0, params['x_ax_pos']), (params['resized_width'], params['x_ax_pos']), (0, 128, 255), 1)
    cv2.line(frame, (params['y_ax_pos'], 0), (params['y_ax_pos'], params['resized_height']), (0, 128, 255), 1)
    cv2.circle(frame, (params['y_ax_pos'], params['x_ax_pos']), 1, (255, 255, 255), -1)

    # Draw the center (tolerance) rectangle
    cv2.rectangle(frame, params['cent_rect_p1'], params['cent_rect_p2'], (0, 178, 255), 1)

    # Draw the detected object's contour, if any
    cv2.drawContours(frame, contour, -1, (0, 255, 0), 2)

    # Compute Cartesian coordinates of unfiltered detected object's centroid
    x_cart_coord = cam_coord['width'] - params['y_ax_pos']
    y_cart_coord = params['x_ax_pos'] - cam_coord['height']

    # Compute Cartesian coordinates of filtered detected object's centroid
    x_filt_cart_coord = filt_cam_coord['width'] - params['y_ax_pos']
    y_filt_cart_coord = params['x_ax_pos'] - filt_cam_coord['height']

    # Draw unfiltered centroid as a red dot, including coordinate values
    cv2.circle(frame, (cam_coord['width'], cam_coord['height']), 5, (0, 0, 255), -1)
    cv2.putText(frame, str(x_cart_coord) + ", " + str(y_cart_coord),
                (cam_coord['width'] + 25, cam_coord['height'] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Draw filtered centroid as a blue dot, including coordinate values
    cv2.circle(frame, (filt_cam_coord['width'], filt_cam_coord['height']), 5, (255, 30, 30), -1)
    cv2.putText(frame, str(x_filt_cart_coord) + ", " + str(y_filt_cart_coord),
                (filt_cam_coord['width'] + 25, filt_cam_coord['height'] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 30, 30), 1)

    return frame  # Return the image frame with all drawn objects


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(Drone.fly())
