import numpy as np
import math
import cv2
import sys

def angle_to_rot(ax, ay, az):
    sx, sy, sz = np.sin(ax), np.sin(ay), np.sin(az)
    cx, cy, cz = np.cos(ax), np.cos(ay), np.cos(az)
    Rx = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))
    Ry = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
    Rz = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))
    # Apply X rotation first, then Y, then Z
    return Rz @ Ry @ Rx

def main():
    # Read images from a video file in the current folder.
    video_capture = cv2.VideoCapture("input.mp4")     # Open video capture object
    got_image, img = video_capture.read()       # Make sure we can read video
    if not got_image:
        print("Cannot read video source")
        sys.exit()

    image_height, image_width = img.shape[:2]

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videoWriter = cv2.VideoWriter("output.mp4", fourcc=fourcc, fps=30.0, frameSize=(image_width, image_height))

    # Camera params
    Fx = 675
    Fy = 675
    f = 675
    x_c = 320
    y_c = 240

    marker_length = 2 # inches
    dist_coeff = np.zeros((1,4))

    K = np.array([
        [f, 0, x_c],
        [0, f, y_c],
        [0, 0, 1]]).astype(float)

    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

    while True:
        got_image, img = video_capture.read()
        if not got_image:
            videoWriter.release()
            sys.exit()

        corners, ids, _ = cv2.aruco.detectMarkers(
            image=img,
            dictionary=arucoDict
        )
        if ids is not None:
            img = cv2.aruco.drawDetectedMarkers(
            image=img, corners=corners, ids=ids, borderColor=(0, 0, 255))

            rvecs,tvecs,_ = cv2.aruco.estimatePoseSingleMarkers(
                corners=corners, markerLength=marker_length,
                cameraMatrix=K, distCoeffs=dist_coeff
            )

            # Get the pose of the first detected marker with respect to the camera.
            rvec_m_c = rvecs[0]                 # This is a 1x3 rotation vector
            tm_c = tvecs[0]                     # This is a 1x3 translation vector

            img = cv2.aruco.drawAxis(
                image=img, cameraMatrix=K, distCoeffs=dist_coeff,
                rvec=rvec_m_c, tvec=tm_c, length=marker_length
            )


        img_output = img.copy()
        videoWriter.write(img_output)
            
        cv2.imshow("window", img)

        cv2.waitKey(30)

if __name__ == "__main__":
    main()