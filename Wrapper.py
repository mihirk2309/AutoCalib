import numpy as np
import cv2
import matplotlib
import os
import glob
import scipy


def get_corners(images):
    corners_all = []
    for i, img in enumerate(images):
        # print("i = ", i)
        img_copy = img.copy()
        ret, corners = cv2.findChessboardCorners(img, (9, 6), flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        # corners = cv2.cornerSubPix(img_copy, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
        corners_all.append(corners)
        if ret:
            # print(corners)
            fnl = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            # cv2.imshow("fnl", fnl)
            # cv2.waitKey(0)
        else:
            print("No Checkerboard Found")

    return corners_all

def get_homography(images, corners_all, world_points):

    H = []
    for i, (img, corners) in enumerate(zip(images, corners_all)):
        # print("i = ", i)
        img_corners= np.array([[corners[0][0]],
                           [corners[8][0]],
                           [corners[53][0]],
                           [corners[45][0]]])
        H_img,_ = cv2.findHomography(world_points, img_corners)
        H.append(H_img)

    return H

def make_V(H,i,j):
    v = np.array([[H[0][i]*H[0][j]],
                  [H[0][i]*H[1][j] + H[0][j]*H[1][i]],
                  [H[1][i]*H[1][j]],
                  [H[2][i]*H[0][j] + H[0][i]*H[2][j]],
                  [H[2][i]*H[1][j] + H[1][i]*H[2][j]],
                  [H[2][i]*H[2][j]]])
    return v

def get_intrinsic_matrix(images, H):
    V = []
    for i, (img, h) in enumerate(zip(images, H)):
        # print("i = ", i)
        v1_2 = make_V(H[i],0,1)
        v1_1 = make_V(H[i],0,0)
        v2_2 = make_V(H[i],1,1)
        v_img = np.vstack([np.transpose(v1_2), np.transpose(v1_1-v2_2)])
        V.append(v_img)
        # print(v_img)

    V = np.resize(np.array(V), (26,6))
    # print("V: = ",V)
    _,_,V = np.linalg.svd(V)
    b = V[:][5]
    B = np.zeros([3,3])
    B[0][0]=b[0]
    B[0][1]=b[1]
    B[1][0]=b[1]
    B[0][2]=b[3]
    B[2][0]=b[3]
    B[1][1]=b[2]
    B[1][2]=b[4]
    B[2][1]=b[4]
    B[2][2]=b[5]

    v_0 = (B[0][1]*B[0][2] - B[0][0]*B[1][2])/(B[0][0]*B[1][1]-B[0][1]**2)
    lambda_ = (B[2][2]-((B[0][2]**2)+v_0*(B[0][1]*B[0][2] - B[0][0]*B[1][2]))/B[0][0])
    alpha = np.sqrt(lambda_/B[0][0])
    beta = np.sqrt(((lambda_*B[0][0])/(B[0][0]*B[1][1] - B[0][1]**2)))
    gamma = -(B[0][1])*(alpha**2)*beta/lambda_
    u_0 = (gamma*v_0/beta) - (B[0][2]*alpha**2)/lambda_

    # print("v_0: = ", v_0)
    # print("u_0: = ", u_0)
    # print("Lambda: = ", lambda_)
    # print("Alpha: = ", alpha)
    # print("Beta: = ", beta)
    # print("gamma: = ", gamma)

    A = np.array([[alpha, gamma, u_0],
                  [0,   beta,  v_0],
                  [0,   0,      1]])

    # print("B: = ",B)
    return A, lambda_

def get_extrinsics(A, H):

    R_final = []
    Rt_final = []
    t_final = []
    for i, h in enumerate(H):
        # print("i = ", i)
        h1 = np.array([h[0][0], h[1][0], h[2][0]])
        h2 = np.array([h[0][1], h[1][1], h[2][1]])
        h3 = np.array([h[0][2], h[1][2], h[2][2]])
        # print("h: = ", h2)

        # c = np.linalg.norm(np.linalg.inv(A)*h2)
        l = 1/np.linalg.norm(np.matmul(np.linalg.inv(A),h2))
        r1 = l*np.matmul(np.linalg.inv(A),h1)
        r2 = l*np.matmul(np.linalg.inv(A),h2)
        r3 = np.cross(r1, r2)
        t = l*np.matmul(np.linalg.inv(A),h3)
        # print("r1: = ", r1)
        # print("r2: = ", r2)
        # print("r3: = ", r3)
        # print("t: = ", t)

        R = np.transpose(np.vstack([r1,r2,r3]))
        Rt = np.transpose(np.vstack([r1,r2,r3,t]))
        R_final.append(R)
        Rt_final.append(Rt)
        t_final.append(t)
        # print("R: = ", Rt)

    # print(t_final)
    return R_final, t_final, Rt_final

def loss_fn(A, Rt, world_points):
    

def main():

    print('Camera Calibration Begin ----------')
    images = [cv2.imread(file) for file in glob.glob("./Calibration_Imgs/*.jpg")]

    world_points = np.array([[21.5, 21.5], [21.5*9, 21.5], [21.5*9, 21.5*6], [21.5, 21.5*6]], dtype='float32')

    corners_all = get_corners(images)
    H = get_homography(images, corners_all, world_points)
    # print("H: = ", H)
    A, lambda_ = get_intrinsic_matrix(images, H)
    # print("A: = ", A)
    # print("lambda: = ", lambda_)
    R, t, Rt = get_extrinsics(A, H)
    # print("Rt: = ", Rt)
    # print(len(corners_all[0]))
    
    


if __name__ == '__main__':
    main()