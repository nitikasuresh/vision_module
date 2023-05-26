import cv2
# print("cv2: ", cv2.__version__)
# assert cv2.__version__[0] == '3'

import numpy as np
import pickle
import os
import glob

CHECKERBOARD = (4,4)
# subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
# calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW

objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)
print("\nobjp shape: ", objp.shape)

_img_shape = None
objpoints = [] #3d points in world
imgpoints = [] #2s points in image
corners = []
# corners = np.zeros(1, 16, 2)

# images = None # list of images to import

cv2.namedWindow('image')

# bind function to the window
def draw_circle(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x,y
        # print("mouseX, mouseY: ", mouseX, mouseY)
        cv2.circle(gray,(x,y),100,(255,0,0),-1)
        corners.append([mouseX, mouseY])


# cv2.setMouseCallback('image', draw_circle)

for i in range(10): # 10
    img = cv2.imread('Fisheye_Imgs/img' + str(i) + '.jpg')
    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # # find corners of chess board
    # ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    # if ret == True:
    #     objpoints.append(objp)
    #     # cv2.cornerSubPix(gray, corners, (3,3), (-1,-1), subpix_criteria)
    #     imgpoints.append(corners)

    # mannually select corners of sensor grid upper left to bottom right
    # cv2.imshow('image', gray)
    # cv2.setMouseCallback('image', draw_circle)
    # k = cv2.waitKey(0)
    # if k == 27:
    #     cv2.destroyAllWindows()
    objpoints.append(objp)
    ## check the points and verify they work well
    # imgpoints.append(np.expand_dims(np.array(corners[16*i:16*(i+1)]), axis=0).astype(np.float32))

# print("len objpoints: ", len(objpoints))
print("objpoints: ", objpoints[0].shape)
# print("imgpoints: ", imgpoints)


# with open('Fisheye_Imgs/corners.pickle', 'wb') as f:
#     pickle.dump(imgpoints, f)

with open('Fisheye_Imgs/corners10.pickle', 'rb') as f:
    imgpoints = pickle.load(f)

print("imgpoints: ", imgpoints[0].shape)

# # plot circles at selected corners to verify everything should work
# for i in range(10): #10
#     img = cv2.imread('Fisheye_Imgs/img' + str(i) + '.jpg')
#     if _img_shape == None:
#         _img_shape = img.shape[:2]
#     else:
#         assert _img_shape == img.shape[:2]
    
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     corners = imgpoints[i]
#     print("corners: ", corners.shape)
#     for j in range(corners.shape[1]):
#         x = int(corners[:,j,0][0])
#         y = int(corners[:,j,1][0])
#         cv2.circle(gray, (x, y), 5, (0, 0, 255), 10)
#     cv2.imshow('image', gray)
#     k = cv2.waitKey(0)
#     if k == 27:
#         cv2.destroyAllWindows()


N_OK = len(objpoints)
K = np.zeros((3,3))
D = np.zeros((4,1))
rvecs = [np.zeros((1,1,3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1,1,3), dtype=np.float64) for i in range(N_OK)]
rms, _, _, _, _ = cv2.fisheye.calibrate(objpoints,
                                        imgpoints,
                                        gray.shape[::-1],
                                        K,
                                        D,
                                        rvecs,
                                        tvecs,
                                        calibration_flags,
                                        (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 30, 1e-6))

print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("K= ", K)
print("D= ", D)

DIM = _img_shape[::-1]

img_path = 'Fisheye_Imgs/test1.jpg' # specific image to undistort
img = cv2.imread(img_path)
h,w = img.shape[:2]
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K,D,np.eye(3),K,DIM, cv2.CV_16SC2)
undistort_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
cv2.imshow("undistorted", undistort_img)
cv2.waitKey(10000)
cv2.destroyAllWindows()