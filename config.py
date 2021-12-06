import numpy as np
import cv2

DEBUG = False
record = False

# Aruco ID numbers
H_ID = 76
C_ID = 571
N_ID = 658
O_ID = 888

aruco_id_map = {H_ID: "H", C_ID: "C", N_ID: "N", O_ID: "O"}

# Aruco tag length
marker_length = 1.77165 # inches

# Camera Parameters
Fx = 675
Fy = 675
f = 675
x_c = 320
y_c = 240

K = np.array([
        [f, 0, x_c],
        [0, f, y_c],
        [0, 0, 1]]).astype(float)

dist_coeff = np.zeros((1,4))

arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)

# Element parameters
height = 3
radius = 25
mol_radius = 60
text_offset = 15
mol_text_offset = 45
mol_name_offset = 80

dist_threshold = 3

# Molecules
num_elements = 4

molecules = {"H2O": {"H": 2, "C": 0, "N": 0, "O": 1},
             "CO2": {"H": 0, "C": 1, "N": 0, "O": 2},
             "N2": {"H": 0, "C": 0, "N": 2, "O": 0},
             "O3": {"H": 0, "C": 0, "N": 0, "O": 3},
             "NH3": {"H": 3, "C": 0, "N": 1, "O": 0},
             "NH4": {"H": 4, "C": 0, "N": 1, "O": 0},
             "O2": {"H": 0, "C": 0, "N": 0, "O": 2},
             "CH4": {"H": 4, "C": 1, "N": 0, "O": 0},
             "H2O2": {"H": 2, "C": 0, "N": 0, "O": 2},}