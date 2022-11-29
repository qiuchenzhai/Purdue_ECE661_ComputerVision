import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io

# read image
img1 = io.imread('painting1.jpeg')
img2 = io.imread('painting2.jpeg')
img3 = io.imread('painting3.jpeg')
img = io.imread('kittens.jpeg')

# record the coordinates of points
PQSR1 = np.array([[298, 510], [238, 1610], [1780, 352], [1686, 1830]])
PQSR2 = np.array([[344, 700], [334, 2334], [1890, 756], [1886, 2006]])
PQSR3 = np.array([[106, 441], [121, 1370], [1221, 302], [1102, 1866]])
PQSR = np.array([[0, 0], [0, 1125], [1920, 0], [1920, 1125]])


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
    output = np.divide((a * Wa + b * Wb + c * Wc + d * Wd), (Wa + Wb + Wc + Wd), where=(Wa + Wb + Wc + Wd)!=0)
    return output


def mapping(src_img, dst_img, points_src, h_mat):
    """
    The mapping function maps the source image to the destination.
    :param src_img: source image.
    :param dst_img: the projection image.
    :param points_src: the P', Q', S', R' points.
    :param h_mat: the HC matrix H.
    :return: the projected image.
    """
    temp = np.zeros(dst_img.shape[0:2])
    pts = np.array([[points_src[0][0], points_src[0][1]], [points_src[1][0], points_src[1][1]], [points_src[3][0],
                    points_src[3][1]], [points_src[2][0], points_src[2][1]]])
    cv2.fillPoly(temp, [pts], 255)
    # plt.imshow(temp)
    # plt.show()
    # plt.clf()
    loc_hc = np.array([[np.dot(h_mat, np.array([j, i, 1])) for j in range(temp.shape[1])] for i in range(temp.shape[0])])
    loc = np.array([loc_hc[:, :, 0] / loc_hc[:, :, 2], loc_hc[:, :, 1] / loc_hc[:, :, 2]])
    for i in range(temp.shape[0]):
        for j in range(temp.shape[1]):
            loc0 = loc[1, i, j]
            loc1 = loc[0, i, j]
            if (loc0 > 0) and (loc1 > 0) and (loc0 < src_img.shape[0]-1) and (loc1 < src_img.shape[1]-1):
                dst_img[i, j][temp[i, j] == 255] = compute_pixel_val(src_img, [loc0, loc1])
            else:
                dst_img[i, j] = 0

    return dst_img


# Image projection
h_ac = compute_hc(PQSR2, PQSR1)
h_bc = compute_hc(PQSR3, PQSR2)
pts = np.array([[0, 0], [0, img3.shape[0]], [img3.shape[1], 0], [img3.shape[1], img3.shape[0]]])
dst_img = mapping(img1, img3, pts, np.dot(h_ac, h_bc))
plt.imshow(dst_img)
plt.axis('off')
plt.savefig('atoc.jpeg')
plt.clf()