import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io


def compute_hc(src, dst):
    """
    The module computes the homography matrix by AH = b.
    :param src: source points.
    :param dst: mapped points.
    :return: homography matrix H.
    """

    A = np.zeros((8, 8))
    b = np.zeros((8, 1))

    for i in range(len(src)):
        A[2 * i] = [src[i][0], src[i][1], 1, 0, 0, 0, -src[i][0]*dst[i][0], -src[i][1]*dst[i][0]]
        A[2 * i + 1] = [0, 0, 0, src[i][0], src[i][1], 1, -src[i][0]*dst[i][1], -src[i][1]*dst[i][1]]
        b[2 * i] = dst[i][0]
        b[2 * i + 1] = dst[i][1]

    h = np.dot(np.linalg.pinv(A), b)
    homo_mat = np.append(h, 1)

    return homo_mat.reshape((3, 3))


def compute_pixel_val(img, loc):
    """
    This module returns the pixel value given by weighting average of neighbor pixels.
    :param img: the mapping img.
    :param loc: the coordinates of mapped points.
    :return: the pixel value.
    """

    loc0_f = np.int(np.floor(loc[0]))
    loc1_f = np.int(np.floor(loc[1]))
    loc0_c = np.int(np.ceil(loc[0]))
    loc1_c = np.int(np.ceil(loc[1]))
    a = img[loc0_f][loc1_f]
    b = img[loc0_f][loc1_c]
    c = img[loc0_c][loc1_f]
    d = img[loc0_c][loc1_c]
    dx = float(loc[0] - loc0_f)
    dy = float(loc[1] - loc1_f)
    Wa = 1 / np.linalg.norm([dx, dy])
    Wb = 1 / np.linalg.norm([1 - dx, dy])
    Wc = 1 / np.linalg.norm([dx, 1 - dy])
    Wd = 1 / np.linalg.norm([1 - dx, 1 - dy])
    output = (a * Wa + b * Wb + c * Wc + d * Wd) / (Wa + Wb + Wc + Wd)

    return output


def ptp_mapping(img, h_mat):
    # Determine the coordinates of domain plane
    m, n, d = np.shape(img)
    P = np.array([0, 0, 1])
    Q = np.array([0, m-1, 1])
    S = np.array([n-1, 0, 1])
    R = np.array([n-1, m-1, 1])
    # Compute the projection of corners
    locP = np.dot(h_mat, P) / np.dot(h_mat, P)[2]
    locQ = np.dot(h_mat, Q) / np.dot(h_mat, Q)[2]
    locS = np.dot(h_mat, S) / np.dot(h_mat, S)[2]
    locR = np.dot(h_mat, R) / np.dot(h_mat, R)[2]
    # Determin the size of mapped image
    x_min = np.int(np.floor(np.min([locP[1], locQ[1], locR[1], locS[1]])))
    x_max = np.int(np.ceil(np.max([locP[1], locQ[1], locR[1], locS[1]])))
    x_scale = x_max - x_min
    y_min = np.int(np.floor(np.min([locP[0], locQ[0], locR[0], locS[0]])))
    y_max = np.int(np.ceil(np.max([locP[0], locQ[0], locR[0], locS[0]])))
    y_scale = y_max - y_min

    # Compute scaling factor
    # if we fix width, then the scaling factor s_w = w_o / w_i
    scaling_w = y_scale / n
    # if we fix length, then the scaling factor s_h = h_o / h_i
    scaling_h = x_scale / m
    # s = max{s_w, s_h}
    s = np.maximum(scaling_w, scaling_h)
    if s < 1:
        s = 1
    # Determine the ourput size
    output = np.zeros((int(np.ceil(x_scale/s)), int(np.ceil(y_scale/s)), 3))
    # Compute the projection
    h_inv = np.linalg.pinv(h_mat) / np.linalg.pinv(h_mat)[2][2]
    # Create array
    y_pt, x_pt = np.meshgrid(np.arange(0, y_scale*s, 1*s), np.arange(0, x_scale*s, 1*s))
    # flattened array along axis=0
    pts = np.vstack((y_pt.ravel(), x_pt.ravel())).T + np.array([[y_min, x_min]])
    # Add third coordinates
    pts = np.append(pts, np.ones([len(pts), 1]), 1)
    # Transformation
    translated_pts = (np.dot(h_inv, pts.T)).T
    # Normalization
    translated_pts = translated_pts[:, :2] / translated_pts[:, [-1]]
    for i in range(0, x_scale):
        for j in range(0, y_scale):
            loc0 = translated_pts[i*y_scale + j, 1]
            loc1 = translated_pts[i*y_scale + j, 0]
            if (loc0 > 0) and (loc1 > 0) and (loc0 < img.shape[0] - 1) and (loc1 < img.shape[1] - 1):
                output[i][j] = compute_pixel_val(img, [loc0, loc1])

    return output.astype(np.uint8)


point = ['P', 'Q', 'S', 'R']
# Image projection
img1 = io.imread('hw3_Task1_Images/Images/Img1.JPG')
PQSR1 = np.array([[642, 498], [642, 532], [666, 503], [666, 537]])
PQSR = np.array([[0, 0], [0, 85], [75, 0], [75, 85]])
# Plot
plt.imshow(img1)
plt.scatter(PQSR1[0][0], PQSR1[0][1])
plt.annotate(point[0] + '({},{})'.format(PQSR1[0][0], PQSR1[0][1]), (PQSR1[0][0]-50, PQSR1[0][1]-50), c='r')
plt.scatter(PQSR1[1][0], PQSR1[1][1])
plt.annotate(point[1] + '({},{})'.format(PQSR1[1][0], PQSR1[1][1]), (PQSR1[1][0]-100, PQSR1[1][1]+100), c='r')
plt.scatter(PQSR1[2][0], PQSR1[2][1])
plt.annotate(point[2] + '({},{})'.format(PQSR1[2][0], PQSR1[2][1]), (PQSR1[2][0]+100, PQSR1[2][1]-100), c='r')
plt.scatter(PQSR1[3][0], PQSR1[3][1])
plt.annotate(point[3] + '({},{})'.format(PQSR1[3][0], PQSR1[3][1]), (PQSR1[3][0]+50, PQSR1[3][1]+50), c='r')
plt.axis('off')
plt.savefig('img1_labelledpt.jpeg')
plt.show()
plt.clf()
# Computation
H = compute_hc(PQSR1, PQSR)
output = ptp_mapping(img1, H)
plt.imshow(output)
plt.axis('off')
plt.savefig("img1_results.jpeg")
plt.show()
plt.clf()

img2 = io.imread('hw3_Task1_Images/Images/Img2.jpeg')
PQSR2 = np.array([[480, 722], [481, 874], [600, 739], [606, 923]])
PQSR = np.array([[0, 0], [0, 74], [84, 0], [84, 74]])
# Plot
plt.imshow(img2)
for i in range(4):
    plt.scatter(PQSR2[i][0], PQSR2[i][1])
    plt.annotate(point[i] + '({},{})'.format(PQSR2[i][0], PQSR2[i][1]), (PQSR2[i][0], PQSR2[i][1]), c='r')
plt.axis('off')
plt.savefig('img2_labelledpt.jpeg')
plt.show()
plt.clf()
# Computation
H2 = compute_hc(PQSR2, PQSR)
output = ptp_mapping(img2, H2)
plt.imshow(output)
plt.axis('off')
plt.savefig("img2_results.jpeg")
plt.show()
plt.clf()


img3 = io.imread('hw3_Task1_Images/Images/Img3.JPG')
PQSR3 = np.array([[2060, 700], [2092, 1483], [2666, 720], [2695, 1333]])
PQSR = np.array([[0, 0], [0, 36], [55, 0], [55, 36]])
# Plot
plt.imshow(img3)
for i in range(4):
    plt.scatter(PQSR3[i][0], PQSR3[i][1])
    plt.annotate(point[i] + '({},{})'.format(PQSR3[i][0], PQSR3[i][1]), (PQSR3[i][0], PQSR3[i][1]), c='r')
plt.title('painting3.jpeg')
plt.axis('off')
plt.savefig('img3_labelledpt.jpeg')
plt.show()
plt.clf()
# Computation
H3 = compute_hc(PQSR3, PQSR)
output = ptp_mapping(img3, H3)
plt.imshow(output)
plt.axis('off')
plt.savefig("img3_results.jpeg")
plt.show()
plt.clf()


img4 = io.imread('hw3_Task1_Images/Images/Img4.jpeg')
PQSR4 = np.array([[625, 622], [609, 1044], [778, 521], [764, 970]])
PQSR = np.array([[0, 0], [0, 120], [40, 0], [40, 120]])
# Plot
plt.imshow(img4)
for i in range(4):
    plt.scatter(PQSR4[i][0], PQSR4[i][1])
    plt.annotate(point[i] + '({},{})'.format(PQSR4[i][0], PQSR4[i][1]), (PQSR4[i][0], PQSR4[i][1]), c='r')
plt.axis('off')
plt.savefig('img4_method1_labels.jpeg')
plt.show()
plt.clf()
# Computation
H4 = compute_hc(PQSR4, PQSR)
output = ptp_mapping(img4, H4)
plt.imshow(output)
plt.axis('off')
plt.savefig("img4_results.jpeg")
plt.show()
plt.clf()


img5 = io.imread('hw3_Task1_Images/Images/img5.jpeg')
PQSR5 = np.array([[573, 302], [572, 386], [692, 296], [692, 384]])
PQSR = np.array([[0, 0], [0, 100], [200, 0], [200, 100]])
# Plot
plt.imshow(img5)
for i in range(4):
    plt.scatter(PQSR5[i][0], PQSR5[i][1])
    plt.annotate(point[i] + '({},{})'.format(PQSR5[i][0], PQSR5[i][1]), (PQSR5[i][0], PQSR5[i][1]), c='r')
plt.axis('off')
plt.savefig('img5_labelledpt.jpeg')
plt.show()
plt.clf()
# Computation
H5 = compute_hc(PQSR5, PQSR)
output = ptp_mapping(img5, H5)
plt.imshow(output)
plt.axis('off')
plt.savefig("img5_results.jpeg")
plt.show()
plt.clf()


img6 = io.imread('hw3_Task1_Images/Images/Img6.jpeg')
PQSR6 = np.array([[232, 63], [232, 1466], [693, 133], [692, 1453]])
PQSR = np.array([[0, 0], [0, 50], [20, 0], [20, 50]])
# Plot
plt.imshow(img6)
for i in range(4):
    plt.scatter(PQSR6[i][0], PQSR6[i][1])
    plt.annotate(point[i] + '({},{})'.format(PQSR6[i][0], PQSR6[i][1]), (PQSR6[i][0], PQSR6[i][1]), c='r')
plt.axis('off')
plt.savefig('img6_method1_labels.jpeg')
plt.show()
plt.clf()
# Computation
H6 = compute_hc(PQSR6, PQSR)
output = ptp_mapping(img6, H6)
plt.imshow(output)
plt.axis('off')
plt.savefig("img6_results.jpeg")
plt.show()
plt.clf()