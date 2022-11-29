import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scipy.optimize import least_squares

def corner_detection(img, **kwargs):

    sigma = kwargs.pop("sigma", 0.33)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Canny edge detector
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    median_intensity = np.median(blurred)   # median of pixel intensities
    lower = int(max(0, (1.0 - sigma) * median_intensity))
    upper = int(min(255, (1.0 + sigma) * median_intensity))
    edges = cv2.Canny(blurred, threshold1=lower, threshold2=upper)
    cv2.imshow("Edges", edges)
    cv2.imwrite("./Files/figures2/edges_8.png", edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Hough transform
    # And write lines in homogeneous representations
    # And divide the lines into vertical group
    homolines = []
    hlines = []
    vlines = []
    lines = cv2.HoughLines(edges, 1, np.pi/180, 52).squeeze()
    if lines is not None:
        for rho, theta in lines:
            x0 = rho * np.cos(theta)
            y0 = rho * np.sin(theta)
            xpt = (int(x0 + 1000 * (-np.sin(theta))), int(y0 + 1000 * (np.cos(theta))))
            ypt = (int(x0 - 1000 * (-np.sin(theta))), int(y0 - 1000 * (np.cos(theta))))
            # cv2.line(img, xpt, ypt, (255, 0, 0), 2)
            homox = (int(x0 + 1000 * (-np.sin(theta))), int(y0 + 1000 * (np.cos(theta))), 1)
            homoy = (int(x0 - 1000 * (-np.sin(theta))), int(y0 - 1000 * (np.cos(theta))), 1)
            homoline = np.cross(homox, homoy) / np.cross(homox, homoy)[2]
            homolines.append(homoline)
            if -np.pi/4 <= theta < np.pi/4 or 3*np.pi/4 <= theta < np.pi or -np.pi <= theta < -3*np.pi/4:
                vlines.append(homoline)
                # cv2.line(img, xpt, ypt, (0, 0, 255), 2)      # red
            else:
                hlines.append(homoline)
                # cv2.line(img, xpt, ypt, (0, 255, 0), 2)        # green
    # cv2.imshow("Lines", img)
    # cv2.imwrite("./Files/figures2/lines_8.png", img)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()

    # sort the lines
    hlines = sorted(hlines, key=lambda x: x[0] * np.sin(x[1]), reverse=True)
    vlines = sorted(vlines, key=lambda y: y[0] * np.cos(y[1]), reverse=False)
    # Find intersection for each pair between sorted horizontal lines and vertical lines
    corner_list = []
    idx = 0
    for hline in hlines:
        for vline in vlines:
            pt = np.cross(hline, vline) / np.cross(hline, vline)[2]
            condition_mat = [np.linalg.norm((pt - corner_list[i])) >= 15 for i in range(len(corner_list))]
            if np.asarray(condition_mat).all():
                corner_list.append(pt)
                # cv2.circle(img, (int(pt[0]), int(pt[1])), radius=2, color=(255, 255, 0), thickness=-1)
                # cv2.putText(img, str(idx), (int(pt[0]-2), int(pt[1]-2)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                #             fontScale=.3, color=(255, 255, 0), thickness=1)
                idx += 1
    # cv2.imshow("Corners", img)
    # # cv2.imwrite("./Files/figures2/corners_8.png", img)
    # cv2.waitKey(0)
    # # cv2.destroyAllWindows()

    return edges, [homolines, hlines, vlines], corner_list


# img = cv2.imread('Files/Dataset1/Pic_2.jpg')
# edges, lines, corners = corner_detection(img)


# Reproject corners
def calculate_vij(H, i, j):
    """
    Calculate Vij from H
    """
    h = H.T
    output = np.zeros((6, 1), dtype=float)
    output[0] = h[i, 0] * h[j, 0]
    output[1] = h[i, 0] * h[j, 1] + h[i, 1] * h[j, 0]
    output[2] = h[i, 1] * h[j, 1]
    output[3] = h[i, 2] * h[j, 0] + h[i, 0] * h[j, 2]
    output[4] = h[i, 2] * h[j, 1] + h[i, 1] * h[j, 2]
    output[5] = h[i, 2] * h[j, 2]

    return output


def calculate_w(H_mats):
    """
    Calculate the image of the absolute conic W (symmetric 3x3) from b (6x1) calculated from
    homography matrices
    """
    # obtain the vector v (2x6)
    V = np.zeros((2 * len(H_mats), 6), dtype=float)
    for k, H in enumerate(H_mats):
        V[2 * k, :] = calculate_vij(H, 0, 1)[:, 0]
        V[2 * k + 1, :] = calculate_vij(H, 0, 0)[:, 0] - calculate_vij(H, 1, 1)[:, 0]
    u, s, vh = np.linalg.svd(np.dot(V.T, V))
    b = vh[-1, :]
    output = np.zeros((3, 3), dtype=float)
    output[0, :] = [b[0], b[1], b[3]]
    output[1, :] = [b[1], b[2], b[4]]
    output[2, :] = [b[3], b[4], b[5]]
    print('W=', output)

    return output


def calculate_k(W):
    """
    Obtain the intrinsic camera matrix K from W
    """
    output = np.zeros((3, 3), dtype=float)
    x0 = (W[0, 1] * W[0, 2] - W[0, 0] * W[1, 2]) / (W[0, 0] * W[1, 1] - W[0, 1] * W[0, 1])
    la = W[2, 2] - (W[0, 2] * W[0, 2] + x0 * (W[0, 1] * W[0, 2] - W[0, 0] * W[1, 2])) / W[0, 0]
    ax = np.sqrt(la / W[0, 0])
    ay = np.sqrt(la * W[0, 0] / (W[0, 0] * W[1, 1] - W[0, 1] * W[0, 1]))
    s = - W[0, 1] * ax * ax * ay / la
    y0 = s * x0 / ay - W[0, 2] * ax * ax / la
    output[0, 0] = ax
    output[0, 1] = s
    output[0, 2] = x0
    output[1, 1] = ay
    output[1, 2] = y0
    output[2, 2] = 1
    print('Camera intrinsic Parameter K :', output)

    return output


def calculate_world_coord(numh, numv, d):
    """ Get the coordinates of corners in world frame """
    corner_ls = []
    for j in range(numv):
        for i in range(numh):
            corner_ls.append(np.array([i * d, j * d, 1]))

    return corner_ls


def calculate_extr_mat(K, H):
    """ Using Zhang's algorithm to calculate the Extrinsic camera parameter [R|t]
    from K and H"""
    h1 = H[:, 0]
    h2 = H[:, 1]
    h3 = H[:, 2]
    K_inv = np.linalg.inv(K)
    epsilon = 1 / np.linalg.norm(np.dot(K_inv, h1))
    r1 = epsilon * np.dot(K_inv, h1)
    r2 = epsilon * np.dot(K_inv, h2)
    r3 = np.cross(r1, r2)
    t = epsilon * np.dot(K_inv, h3)
    # condition the rotation matrix
    Q = np.array([r1, r2, r3]).T
    u, s, vh = np.linalg.svd(Q)
    R = np.dot(u, vh)
    print('the Extrinsic camera parameter is R:{} and t:{}'.format(R, t))

    return R, t


def corner_reprojection(corners, P):

    output = []
    H_mat = P[:, [0, 1, 3]]
    for c in corners:
        output.append(np.dot(H_mat, c) / np.dot(H_mat, c)[-1])

    return output


def calculate_hmat(pts1, pts2):
    A = np.zeros((2 * len(pts1), 8))
    b = np.zeros((2 * len(pts1), 1))
    for i in range(len(pts1)):
        A[2 * i] = [pts1[i][0], pts1[i][1], 1, 0, 0, 0, -pts1[i][0]*pts2[i][0], -pts1[i][1]*pts2[i][0]]
        b[2 * i] = pts2[i][0]
        A[2 * i + 1] = [0, 0, 0, pts1[i][0], pts1[i][1], 1, -pts1[i][0]*pts2[i][1], -pts1[i][1]*pts2[i][1]]
        b[2 * i + 1] = pts2[i][1]
    h = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), b)
    H = np.append(h, 1)

    return H.reshape((3, 3))


def camera_calibration(img_path, N, **kwargs):
    d = kwargs.pop("d", 25)

    img_ls = []
    H_mats = []
    img_arr = np.zeros((3, 80 * N), dtype=float)
    repro_arr = np.zeros((3, 80 * N), dtype=float)
    world_corners = calculate_world_coord(8, 10, d=d)

    for i in range(1, N+1, 1):
        file_name = "Pic_{}.jpg".format(i)
        img = cv2.imread(os.path.join(img_path, file_name))
        # img = cv2.resize(img, None, fx=0.25, fy=0.25)
        img_ls.append(img)
        _, _, img_corners = corner_detection(img)
        img_arr[:, (i-1) * 80 : i * 80] = np.array(img_corners).T
        H = calculate_hmat(world_corners, img_corners)
        H_mats.append(H)
    # intrisic parameter
    W = calculate_w(H_mats)
    K = calculate_k(W)
    # extrinsic parameter
    for idx, h in enumerate(H_mats):
        R, t = calculate_extr_mat(K, h)
        P = np.dot(K, np.hstack((R, t.reshape((3, 1)))))
        repro_arr[:, idx * 80 : (idx+1) * 80] = np.array(corner_reprojection(world_corners, P)).T
        num = 0
        for i in range(80):
            pt = repro_arr[:, idx * 80 + i].reshape((3, 1))
            orig_pt = img_arr[:, idx * 80 + i].reshape((3, 1))
            cv2.circle(img_ls[idx], (int(pt[0]), int(pt[1])), radius=2, color=(0, 0, 255), thickness=-1)
            cv2.circle(img_ls[idx], (int(orig_pt[0]), int(orig_pt[1])), radius=2, color=(255, 255, 0), thickness=-1)
            cv2.putText(img_ls[idx], str(num), (int(pt[0]-2), int(pt[1]-2)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        fontScale=.3, color=(255, 255, 0), thickness=1)
            num += 1
        cv2.imshow("Reprojected Corners", img_ls[idx])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("K=", K)

    return K


path = "Files/Dataset1"
K = camera_calibration(path, N=40)

# path = "Files/Dataset2"
# K = camera_calibration(path, N=30)