from inspect import isdatadescriptor
import numpy as np
import math
import cv2
import sys
import config
import itertools
from sklearn.cluster import DBSCAN

def angle_to_rot(ax, ay, az):
    sx, sy, sz = np.sin(ax), np.sin(ay), np.sin(az)
    cx, cy, cz = np.cos(ax), np.cos(ay), np.cos(az)
    Rx = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))
    Ry = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
    Rz = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))
    # Apply X rotation first, then Y, then Z
    return Rz @ Ry @ Rx

def euc_distance(tvec1, tvec2):
    return ((tvec1[0] - tvec2[0])**2 + (tvec1[1] - tvec2[1])**2 + (tvec1[2] - tvec2[2])**2)**0.5

def calc_center(points, rvecs, tvecs):

    t = np.zeros((len(points), 1, 3))
    r = np.zeros((len(points), 1, 3))
    i = 0
    for p in points:
        t[i] = tvecs[p][0]
        r[i] = rvecs[p][0]
        i += 1

    rvec = np.mean(r, axis=0)
    tvec = np.mean(t, axis=0)

    return rvec, tvec

def calc_position(rvec_m_c, tm_c):
    points = np.array([[0, 0, 0, 1]]).T

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

    p = config.K @ Mext @ points
    p = p / p[2]

    p = p.T.astype(int)
    return p

def draw_element(img, id, rvec_m_c, tm_c):
    if config.DEBUG:
        img = cv2.aruco.drawAxis(
            image=img, cameraMatrix=config.K, distCoeffs=config.dist_coeff,
            rvec=rvec_m_c, tvec=tm_c, length=config.marker_length
        )

    p = calc_position(rvec_m_c, tm_c)

    # print(f"x: {p[0][0]}, y: {p[0][1]}")

    # Do something with mext to project image
    
    if (id == config.H_ID):
        cv2.circle(img, (p[0][0], p[0][1]), config.radius, (0, 0, 255), -1)
        cv2.putText(img, "H", (p[0][0] - config.text_offset, p[0][1] + config.text_offset), cv2.FONT_HERSHEY_PLAIN, 3, color=(0,0,0), thickness=2)
    elif (id == config.C_ID):
        cv2.circle(img, (p[0][0], p[0][1]), config.radius, (0, 255, 0), -1)
        cv2.putText(img, "C", (p[0][0] - config.text_offset, p[0][1] + config.text_offset), cv2.FONT_HERSHEY_PLAIN, 3, color=(0,0,0), thickness=2)
    elif (id == config.N_ID):
        cv2.circle(img, (p[0][0], p[0][1]), config.radius, (0, 255, 255), -1)
        cv2.putText(img, "N", (p[0][0] - config.text_offset, p[0][1] + config.text_offset), cv2.FONT_HERSHEY_PLAIN, 3, color=(0,0,0), thickness=2)
    elif (id == config.O_ID):
        cv2.circle(img, (p[0][0], p[0][1]), config.radius, (255, 0, 0), -1)
        cv2.putText(img, "O", (p[0][0] - config.text_offset, p[0][1] + config.text_offset), cv2.FONT_HERSHEY_PLAIN, 3, color=(0,0,0), thickness=2)

def draw_molecule(img, molecule, points, rvecs, tvecs):
    rvec, tvec = calc_center(points, rvecs, tvecs)

    p = calc_position(rvec, tvec)

    if (molecule == "H2O"):
        cv2.circle(img, (p[0][0], p[0][1]), config.mol_radius, (127, 0, 255), -1)
        cv2.putText(img, "H2O", (p[0][0] - config.mol_text_offset, p[0][1] + config.text_offset), cv2.FONT_HERSHEY_PLAIN, 3, color=(0,0,0), thickness=2)
        cv2.putText(img, "Water", (p[0][0] - 80, p[0][1] - config.mol_name_offset), cv2.FONT_HERSHEY_PLAIN, 3, color=(255, 255, 255), thickness=2)
    elif (molecule == "CO2"):
        cv2.circle(img, (p[0][0], p[0][1]), config.mol_radius, (255, 127, 0), -1)
        cv2.putText(img, "CO2", (p[0][0] - config.mol_text_offset, p[0][1] + config.text_offset), cv2.FONT_HERSHEY_PLAIN, 3, color=(0,0,0), thickness=2)
        cv2.putText(img, "Carbon-Dioxide", (p[0][0] - 150, p[0][1] - config.mol_name_offset), cv2.FONT_HERSHEY_PLAIN, 3, color=(255, 255, 255), thickness=2)
    elif (molecule == "N2"):
        cv2.circle(img, (p[0][0], p[0][1]), config.mol_radius, (0, 255, 255), -1)
        cv2.putText(img, "N2", (p[0][0] - config.mol_text_offset, p[0][1] + config.text_offset), cv2.FONT_HERSHEY_PLAIN, 3, color=(0,0,0), thickness=2)
        cv2.putText(img, "Nitrogen Gas", (p[0][0] - 150, p[0][1] - config.mol_name_offset), cv2.FONT_HERSHEY_PLAIN, 3, color=(255, 255, 255), thickness=2)
    elif (molecule == "O3"):
        cv2.circle(img, (p[0][0], p[0][1]), config.mol_radius, (255, 0, 0), -1)
        cv2.putText(img, "O3", (p[0][0] - config.mol_text_offset, p[0][1] + config.text_offset), cv2.FONT_HERSHEY_PLAIN, 3, color=(0,0,0), thickness=2)
        cv2.putText(img, "Ozone", (p[0][0] - 80, p[0][1] - config.mol_name_offset), cv2.FONT_HERSHEY_PLAIN, 3, color=(255, 255, 255), thickness=2)
    elif (molecule == "O2"):
        cv2.circle(img, (p[0][0], p[0][1]), config.mol_radius, (255, 0, 0), -1)
        cv2.putText(img, "O2", (p[0][0] - config.mol_text_offset, p[0][1] + config.text_offset), cv2.FONT_HERSHEY_PLAIN, 3, color=(0,0,0), thickness=2)
        cv2.putText(img, "Oxygen", (p[0][0] - 80, p[0][1] - config.mol_name_offset), cv2.FONT_HERSHEY_PLAIN, 3, color=(255, 255, 255), thickness=2)
    elif (molecule == "NH4"):
        cv2.circle(img, (p[0][0], p[0][1]), config.mol_radius, (0, 64, 255), -1)
        cv2.putText(img, "NH4", (p[0][0] - config.mol_text_offset, p[0][1] + config.text_offset), cv2.FONT_HERSHEY_PLAIN, 3, color=(0,0,0), thickness=2)
        cv2.putText(img, "Ammonium", (p[0][0] - 100, p[0][1] - config.mol_name_offset), cv2.FONT_HERSHEY_PLAIN, 3, color=(255, 255, 255), thickness=2)
    elif (molecule == "NH3"):
        cv2.circle(img, (p[0][0], p[0][1]), config.mol_radius, (0, 96, 255), -1)
        cv2.putText(img, "NH3", (p[0][0] - config.mol_text_offset, p[0][1] + config.text_offset), cv2.FONT_HERSHEY_PLAIN, 3, color=(0,0,0), thickness=2)
        cv2.putText(img, "Ammonia", (p[0][0] - 100, p[0][1] - config.mol_name_offset), cv2.FONT_HERSHEY_PLAIN, 3, color=(255, 255, 255), thickness=2)
    elif (molecule == "CH4"):
        cv2.circle(img, (p[0][0], p[0][1]), config.mol_radius, (0, 64, 255), -1)
        cv2.putText(img, "CH4", (p[0][0] - config.mol_text_offset, p[0][1] + config.text_offset), cv2.FONT_HERSHEY_PLAIN, 3, color=(0,0,0), thickness=2)
        cv2.putText(img, "Methane", (p[0][0] - 100, p[0][1] - config.mol_name_offset), cv2.FONT_HERSHEY_PLAIN, 3, color=(255, 255, 255), thickness=2)
    elif (molecule == "H2O2"):
        cv2.circle(img, (p[0][0], p[0][1]), config.mol_radius, (127, 0, 127), -1)
        cv2.putText(img, "H2O2", (p[0][0] - config.mol_text_offset, p[0][1] + config.text_offset), cv2.FONT_HERSHEY_PLAIN, 3, color=(0,0,0), thickness=2)
        cv2.putText(img, "Hydrogen Peroxide", (p[0][0] - 180, p[0][1] - config.mol_name_offset), cv2.FONT_HERSHEY_PLAIN, 3, color=(255, 255, 255), thickness=2)


def main():
    # Read images from a video file in the current folder.
    video_capture = cv2.VideoCapture(1)     # Open video capture object
    got_image, img = video_capture.read()       # Make sure we can read video
    if not got_image:
        print("Cannot read video source")
        sys.exit()

    if config.record:
        image_height, image_width = img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        videoWriter = cv2.VideoWriter("output.mp4", fourcc=fourcc, fps=30.0, frameSize=(image_width, image_height))


    while True:
        got_image, img = video_capture.read()

        corners, ids, _ = cv2.aruco.detectMarkers(
            image=img,
            dictionary=config.arucoDict
        )

        if ids is not None:
            if config.DEBUG:
                img = cv2.aruco.drawDetectedMarkers(
                image=img, corners=corners, ids=ids, borderColor=(0, 0, 255))

            rvecs,tvecs,_ = cv2.aruco.estimatePoseSingleMarkers(
                corners=corners, markerLength=config.marker_length,
                cameraMatrix=config.K, distCoeffs=config.dist_coeff
            )

            # Remove un-recognized ids:
            end = len(ids)
            i = 0
            while i < end:
                if ids[i][0] not in config.aruco_id_map:
                    if config.DEBUG:
                        print(f"Warning: unrecognized id, {ids[i]}")
                    ids = np.delete(ids, i, 0)
                    rvecs = np.delete(rvecs, i, 0)
                    tvecs = np.delete(tvecs, i, 0)
                    end -= 1
                else:
                    i += 1

            if ids.size > 0:
                squeezed = np.squeeze(tvecs, axis=1)

                clusters = DBSCAN(eps=config.dist_threshold, min_samples=1).fit(squeezed)
                cluster_labels = clusters.labels_
                num_clusters = max(clusters.labels_) + 1

                # Build the list of molecules
                molecules = {}
                
                i = 0
                for cluster_id in cluster_labels:
                    if cluster_id not in molecules:
                        molecules[cluster_id] = []
                    molecules[cluster_id].append(i)
                    i += 1

                for mol in molecules:
                    element_sum = {"H": 0, "C": 0, "N": 0, "O": 0}
                    for element in molecules[mol]:
                        element_sum[config.aruco_id_map[ids[element][0]]] += 1
                    is_molecule = False
                    for m in config.molecules:
                        if config.molecules[m] == element_sum:
                            if config.DEBUG:
                                print(m)
                            
                            draw_molecule(img, m, molecules[mol], rvecs, tvecs)
                            is_molecule = True
                    if not is_molecule:
                        for element in molecules[mol]:
                            draw_element(img, ids[element], rvecs[element], tvecs[element])

        if config.record:
            img_output = img.copy()
            videoWriter.write(img_output)
        cv2.imshow('Window', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.waitKey(30)
    
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()