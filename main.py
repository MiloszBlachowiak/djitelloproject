from djitellopy import Tello
import cv2
import keyboard
from enum import Enum

# from __future__ import print_function  # Python 2/3 compatibility
import cv2
import numpy as npss
import sys

""" sources
- https://automaticaddison.com/how-to-detect-aruco-markers-using-opencv-and-python/
- https://www.pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/
"""

# https://stackoverflow.com/questions/72372154/how-to-make-stream-from-tello-sdk-2-0-captured-by-opencv-with-cv2-videocapture
tello = Tello()

aruco_dictionary_to_use = "DICT_ARUCO_ORIGINAL"

# The different ArUco dictionaries built into the OpenCV library.
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
}

ARUCO_MOVEMENT_DICT = {
    13: tello.move_forward,
    26: tello.move_left,
    39: tello.move_right

}

class Direction(Enum):
    HOVER = 0
    FORWARD = 1
    BACKWARDS = 2
    LEFT = 3
    RIGHT = 4
    ROTATE_LEFT = 5
    ROTATE_RIGHT = 6
    UP = 7
    DOWN = 8

# tello initialize connection and query battery level
tello.connect()
print(tello.query_battery())
# print(tello.get_temperature())

# takeoff
tello.takeoff()
tello.move_up(30)

# start video streaming
tello.streamon()

def main():
    # Check that we have a valid ArUco marker
    if ARUCO_DICT.get(aruco_dictionary_to_use, None) is None:
        print("[ERR] ArUCo tag of '{}' is not supported".format(aruco_dictionary_to_use))
        sys.exit(0)

    # Load the ArUco dictionary
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_dictionary_to_use])
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    while True:
        frame = tello.get_frame_read().frame

        # Detect ArUco markers in the video frame
        (corners, ids, rejected) = detector.detectMarkers(frame)

        # Check that at least one ArUco marker was detected
        if len(corners) > 0:
            ids = ids.flatten()
            for (marker_corner, marker_id) in zip(corners, ids):

                # # Extract the marker corners
                corners = marker_corner.reshape((4, 2))
                (top_left, top_right, bottom_right, bottom_left) = corners

                # Convert the (x,y) coordinate pairs to integers
                top_right = (int(top_right[0]), int(top_right[1]))
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
                top_left = (int(top_left[0]), int(top_left[1]))

                # Draw the bounding box of the ArUco detection
                cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
                cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
                cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
                cv2.line(frame, bottom_left, top_left, (0, 255, 0), 2)

                # Calculate and draw the center of the ArUco marker
                center_x = int((top_left[0] + bottom_right[0]) / 2.0)
                center_y = int((top_left[1] + bottom_right[1]) / 2.0)
                cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

                # Draw the ArUco marker ID on the video frame
                # The ID is always located at the top_left of the ArUco marker
                cv2.putText(frame, str(marker_id),
                            (top_left[0], top_left[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)
                
                # perform movement
                if (marker_id in ARUCO_MOVEMENT_DICT.keys()):
                    ARUCO_MOVEMENT_DICT[marker_id](30)
                

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    print(__doc__)
    main()
