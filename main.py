import numpy as np
import math
import cv2
import sys
import config

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
    video_capture = cv2.VideoCapture(0)     # Open video capture object
    got_image, img = video_capture.read()       # Make sure we can read video
    if not got_image:
        print("Cannot read video source")
        sys.exit()

    image_height, image_width = img.shape[:2]
    
    dist_coeff = np.zeros((1,4))

    K = np.array([
        [config.f, 0, config.x_c],
        [0, config.f, config.y_c],
        [0, 0, 1]]).astype(float)

    points = np.array([[0, 0, 0, 1]]).T

    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)

    while True:
        got_image, img = video_capture.read()

        corners, ids, _ = cv2.aruco.detectMarkers(
            image=img,
            dictionary=arucoDict
        )
        if ids is not None:
            if config.DEBUG:
                img = cv2.aruco.drawDetectedMarkers(
                image=img, corners=corners, ids=ids, borderColor=(0, 0, 255))

            rvecs,tvecs,_ = cv2.aruco.estimatePoseSingleMarkers(
                corners=corners, markerLength=config.marker_length,
                cameraMatrix=K, distCoeffs=dist_coeff
            )

            i = 0
            for id in ids:
                # Get the pose of the detected marker with respect to the camera.
                rvec_m_c = rvecs[i]                 # This is a 1x3 rotation vector
                tm_c = tvecs[i]                     # This is a 1x3 translation vector

                if config.DEBUG:
                    img = cv2.aruco.drawAxis(
                        image=img, cameraMatrix=K, distCoeffs=dist_coeff,
                        rvec=rvec_m_c, tvec=tm_c, length=config.marker_length
                    )

                R_m_p = np.array([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]])

                t_m_p = np.array([[0, 0, -config.height]]).T

                R_m_c = cv2.Rodrigues(rvec_m_c)[0]

                H_M_C = np.block([[R_m_c, tm_c.T], [0,0,0,1]])

                H_M_P= np.block([[R_m_p, t_m_p], [0,0,0,1]])

                H_P_M = np.linalg.inv(H_M_P)

                H_P_C =  H_M_C @ H_P_M

                Mext = H_P_C[0:3, :]

                p = K @ Mext @ points
                p = p / p[2]

                p = p.T.astype(int)

                print(f"x: {p[0][0]}, y: {p[0][1]}")

                # Do something with mext to project image
                text_offset = 15
                if (id == config.H_ID):
                    cv2.circle(img, (p[0][0], p[0][1]), config.radius, (0, 0, 255), -1)
                    cv2.putText(img, "H", (p[0][0] - text_offset, p[0][1] + text_offset), cv2.FONT_HERSHEY_PLAIN, 3, color=(0,0,0), thickness=2)
                elif (id == config.C_ID):
                    cv2.circle(img, (p[0][0], p[0][1]), config.radius, (0, 255, 0), -1)
                    cv2.putText(img, "C", (p[0][0] - text_offset, p[0][1] + text_offset), cv2.FONT_HERSHEY_PLAIN, 3, color=(0,0,0), thickness=2)
                elif (id == config.N_ID):
                    cv2.circle(img, (p[0][0], p[0][1]), config.radius, (0, 255, 255), -1)
                    cv2.putText(img, "N", (p[0][0] - text_offset, p[0][1] + text_offset), cv2.FONT_HERSHEY_PLAIN, 3, color=(0,0,0), thickness=2)
                elif (id == config.O_ID):
                    cv2.circle(img, (p[0][0], p[0][1]), config.radius, (255, 0, 0), -1)
                    cv2.putText(img, "O", (p[0][0] - text_offset, p[0][1] + text_offset), cv2.FONT_HERSHEY_PLAIN, 3, color=(0,0,0), thickness=2)

                i += 1

        cv2.imshow('Window', img)



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.waitKey(30)
    
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()