import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io
from scipy.optimize import least_squares


# Read images
img1 = io.imread('1.jpeg')
img2 = io.imread('2.jpeg')
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)
img3 = io.imread('3.jpeg')
img3 = cv2.resize(img3, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)
img4 = io.imread('4.jpeg')
img4 = cv2.resize(img4, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)
img5 = io.imread('5.jpeg')
img5 = cv2.resize(img5, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)
# Converts image to gray images
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
gray5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)


# =========================================================================================
# Extract SIFT feature
num_features = 2000
sift = cv2.xfeatures2d.SIFT_create(nfeatures=num_features)
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
kp3, des3 = sift.detectAndCompute(gray3, None)
kp4, des4 = sift.detectAndCompute(gray4, None)
kp5, des5 = sift.detectAndCompute(gray5, None)


def ncc_metric(img1, kp1, img2, kpts, k):

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialization
    h, w = img1.shape
    n = int(k / 2)
    # Neighborhood of kp1
    img1_padded = np.zeros([h + 2 * n, w + 2 * n])
    img1_padded[n: n + h, n: n + w] = img1
    img2_padded = np.zeros([h + 2 * n, w + 2 * n])
    img2_padded[n: n + h, n: n + w] = img2
    # find the ncc between kp1 and kp2's in corner list of img2 and return the correspondence
    neighbor1 = img1_padded[int(kp1[1]): int(kp1[1]) + 2 * n, int(kp1[0]): int(kp1[0]) + 2 * n]
    max_ncc = -1e6
    # index = int(len(corner2)-1)
    for i in range(len(kpts)):
        kp2 = kpts[i].pt
        neighbor2 = img2_padded[int(kp2[1]): int(kp2[1]) + 2 * n, int(kp2[0]): int(kp2[0]) + 2 * n]
        sum1 = np.sum(neighbor1 - np.mean(neighbor1) ** 2)
        sum2 = np.sum(neighbor2 - np.mean(neighbor2) ** 2)
        ncc = np.sum((neighbor1 - np.mean(neighbor1)) * (neighbor2 - np.mean(neighbor2))) / np.sqrt(sum1 * sum2)
        if ncc > max_ncc:
            max_ncc = ncc
            index = i

    return index, max_ncc


def match_pt(img1, kp1, img2, kp2, num_best):
    """
    Use SSD to build points correspondences.
    """
    pts = []
    match_pts = []
    match_dist = []
    for i in range(len(kp1)):
        pt1 = kp1[i].pt
        pts.append(pt1)
        index, max_ncc = ncc_metric(img1, pt1, img2, kp2, k=21)
        match_pts.append(kp2[index].pt)
        match_dist.append(max_ncc)

    # Sort
    pts = [x for _, x in sorted(zip(match_dist, pts))]
    pts = np.array(pts)[0:num_best, :]
    match_pts = [y for _, y in sorted(zip(match_dist, match_pts))]
    match_pts = np.array(match_pts)[0:num_best, :]
    match_dist = sorted(match_dist)
    match_dist = np.array(match_dist)[0:num_best]

    # # Plot
    # img = np.hstack((img1, img2))
    # w, h, d = img1.shape
    # for i in range(num_best):
    #     pt1 = [int(pts[i][0]), int(pts[i][1])]
    #     pt2 = [int(match_pts[i][0] + h), int(match_pts[i][1])]
    #     cv2.line(img, tuple(pt1), tuple(pt2), (0, 255, 255), 1)
    #     cv2.circle(img, tuple(pt1), 4, (0, 255, 255), 2)
    #     cv2.circle(img, tuple(pt2), 4, (0, 255, 255), 2)
    # plt.imshow(img, cmap='gray')
    # plt.axis('off')
    # plt.show()
    # plt.clf()

    return pts, match_pts, match_dist


# Build points correspondences
pts1, match_pts12, match_dist12 = match_pt(img1, kp1, img2, kp2, num_best=200)
pts2, match_pts23, match_dist23 = match_pt(img2, kp2, img3, kp3, num_best=200)
pts3, match_pts34, match_dist34 = match_pt(img3, kp3, img4, kp4, num_best=200)
pts4, match_pts45, match_dist45 = match_pt(img4, kp4, img5, kp5, num_best=200)


# =========================================================================================
# RANSAC


def random_miniset(pts1, pts2, num_pts):

    random_selection = np.random.permutation(len(pts1))
    selection1 = []
    selection2 = []
    for i in range(num_pts):
        selection1.append(pts1[random_selection[i]])
        selection2.append(pts2[random_selection[i]])

    return selection1, selection2


def compute_homography(pts1, pts2):

    A = np.zeros((2 * len(pts1), 8))
    b = np.zeros((2 * len(pts1), 1))

    for i in range(len(pts1)):
        pt1 = np.array([pts1[i][0], pts1[i][1], 1], dtype=float)
        pt2 = np.array([pts2[i][0], pts2[i][1], 1], dtype=float)
        A[2 * i] = [pt1[0], pt1[1], 1, 0, 0, 0, -pt1[0]*pt2[0], -pt1[1]*pt2[0]]
        A[2 * i + 1] = [0, 0, 0, pt1[0], pt1[1], 1, -pt1[0]*pt2[1], -pt1[1]*pt2[1]]
        b[2 * i] = pt2[0]
        b[2 * i + 1] = pt2[1]

    h = np.dot(np.linalg.pinv(A), b)
    homo_mat = np.append(h, 1)
    H = homo_mat.reshape((3, 3))

    return H


def get_liers(pts1, pts2, map_pts1, delta):

    inlier_pts1 = []
    inlier_pts2 = []
    outlier_pts1 = []
    outlier_pts2 = []

    for i in range(len(pts2)):
        if np.sqrt((pts2[i][0] - map_pts1[i][0])**2 + (pts2[i][1] - map_pts1[i][1])**2) < delta:
            inlier_pts1.append(pts1[i])
            inlier_pts2.append(pts2[i])
        else:
            outlier_pts1.append(pts1[i])
            outlier_pts2.append(pts2[i])

    return inlier_pts1, inlier_pts2, outlier_pts1, outlier_pts2


def apply_H(H, pts):

    pts = np.append(pts, np.ones([len(pts), 1]), 1)
    mapped_pts = np.zeros(pts.shape, dtype=float)
    for i in range(len(pts)):
        mapped_pts[i] = np.dot(H, pts[i]) / np.dot(H, pts[i])[2]

    return mapped_pts


def ransac_rejection(pts1, pts2, n, epsilon, p, delta):

    N = int( np.log(1-p) / np.log( 1 - (1 - epsilon)**n ) )
    n_total = len(pts1)
    M = int( (1 - epsilon) * n_total)
    for i in range(N):
        selected_pts1, selected_pts2 = random_miniset(pts1, pts2, num_pts=6)
        H = compute_homography(selected_pts1, selected_pts2)
        map_pts = apply_H(H, pts1)
        inlier_pt1, inlier_pt2, outlier_pt1, outlier_pt2 = get_liers(pts1, pts2, map_pts, delta)

    return inlier_pt1, inlier_pt2, outlier_pt1, outlier_pt2


def show_liers(img1, img2, inlier_pt1, inlier_pt2, outlier_pt1, outlier_pt2):

    img = np.hstack((img1, img2))
    w, h, d = img1.shape

    for i in range(len(inlier_pt1)):
        pt1 = [int(inlier_pt1[i][0]), int(inlier_pt1[i][1])]
        pt2 = [int(inlier_pt2[i][0] + h), int(inlier_pt2[i][1])]
        cv2.line(img, tuple(pt1), tuple(pt2), (0, 255, 255), 1)
        cv2.circle(img, tuple(pt2), 4, (0, 255, 255), 2)
        cv2.circle(img, tuple(pt2), 4, (0, 255, 255), 2)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title('Inlier')
    plt.show()
    plt.clf()

    img = np.hstack((img1, img2))
    for i in range(len(outlier_pt1)):
        pt1 = [int(outlier_pt1[i][0]), int(outlier_pt1[i][1])]
        pt2 = [int(outlier_pt2[i][0] + h), int(outlier_pt2[i][1])]
        cv2.line(img, tuple(pt1), tuple(pt2), (0, 255, 255), 1)
        cv2.circle(img, tuple(pt2), 4, (0, 255, 255), 2)
        cv2.circle(img, tuple(pt2), 4, (0, 255, 255), 2)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title('Outlier')
    plt.show()
    plt.clf()


# RANSAC paramters
epsilon = 0.1
delta = 8
p = 0.99
n = 6


# the first pair
in1, in12, out1, out12 = ransac_rejection(pts1, match_pts12, n, epsilon, p, delta)
show_liers(img1, img2, in1, in12, out1, out12)
# the second pair
in2, in23, out2, out23 = ransac_rejection(pts2, match_pts23, n, epsilon, p,  delta)
show_liers(img2, img3, in2, in23, out2, out23)
# the third pair
in3, in34, out3, out34 = ransac_rejection(pts3, match_pts34, n, epsilon, p,  delta)
show_liers(img3, img4, in3, in34, out3, out34)
# the fourth pair
in4, in45, out4, out45 = ransac_rejection(pts4, match_pts45, n, epsilon, p,  delta)
show_liers(img4, img5, in4, in45, out4, out45)


# =========================================================================================
# Linear Least Square


def homography_estimate(pts1, pts2):

    if len(pts1) % 2 == 1:
        pts1.append([0, 0])
        pts2.append([0, 0])
    pts1 = np.array(pts1, dtype=float)
    pts2 = np.array(pts2, dtype=float)
    A = np.zeros((2 * len(pts1), 8))
    b = np.zeros((2 * len(pts1), 1))

    for i in range(len(pts1)):
        pt1 = np.array([pts1[i][0], pts1[i][1], 1], dtype=float)
        pt2 = np.array([pts2[i][0], pts2[i][1], 1], dtype=float)
        A[2 * i] = [0, 0, 0, -pt2[2] * pt1[0], -pt2[2] * pt1[1], -pt2[2] * pt1[2], pt2[1] * pt1[0], pt2[1] * pt1[1]]
        A[2 * i + 1] = [pt2[2] * pt1[0], pt2[2] * pt1[1], pt2[2] * pt1[2], 0, 0, 0, -pt2[0] * pt1[0], -pt2[0] * pt1[1]]
        b[2 * i] = -pt2[1] * pt1[2]
        b[2 * i + 1] = pt2[0] * pt1[2]

    h = np.dot(np.linalg.pinv(A), b)
    homo_mat = np.append(h, 1)
    H = homo_mat.reshape((3, 3))

    return H


H_12 = homography_estimate(in1, in12)
H_23 = homography_estimate(in2, in23)
H_33 = np.identity(3)
H_34 = homography_estimate(in3, in34)
H_45 = homography_estimate(in4, in45)

H_43 = homography_estimate(in34, in3)
H_54 = homography_estimate(in45, in4)
H_13 = np.dot(H_12, H_23) / np.dot(H_12, H_23)[2][2]
H_53 = np.dot(H_54, H_43) / np.dot(H_54, H_43)[2][2]


# =========================================================================================
# Create panoramic Image


def compute_boundary(img, H_mat):

    b = np.zeros((4, 3), dtype=float)
    b[0] = np.dot(H_mat, np.array([0, 0, 1])) / np.dot(H_mat, np.array([0, 0, 1]))[2]
    b[1] = np.dot(H_mat, np.array([img.shape[0], 0, 1])) / np.dot(H_mat, np.array([img.shape[0], 0, 1]))[2]
    b[2] = np.dot(H_mat, np.array([0, img.shape[1], 1])) / np.dot(H_mat, np.array([0, img.shape[1], 1]))[2]
    b[3] = np.dot(H_mat, np.array([img.shape[0], img.shape[1], 1])) / np.dot(H_mat, np.array([img.shape[0], img.shape[1], 1]))[2]

    return b


def init_pano_img(imgs, H_matrices):

    b1 = compute_boundary(imgs[0], H_matrices[0])[:, 0:2]
    b2 = compute_boundary(imgs[1], H_matrices[1])[:, 0:2]
    b3 = compute_boundary(imgs[2], H_matrices[2])[:, 0:2]
    b4 = compute_boundary(imgs[3], H_matrices[3])[:, 0:2]
    b5 = compute_boundary(imgs[4], H_matrices[4])[:, 0:2]
    x_min, y_min = np.amin(np.amin([b1, b2, b3, b4, b5], 0), 0)
    x_max, y_max = np.amax(np.amax([b1, b2, b3, b4, b5], 0), 0)
    x_scale = np.int(np.ceil(x_max)) - np.int(np.floor(x_min))
    y_scale = np.int(np.ceil(y_max)) - np.int(np.floor(y_min))
    output = np.zeros((x_scale, y_scale, 3))

    return output, x_min, y_min


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


def img_map(panoramic_img, img, h_mat, x_min, y_min):

    x_scale, y_scale, d_scale = np.shape(panoramic_img)
    # # Determine the coordinates of domain plane
    # m, n, d = np.shape(img)
    # # Compute scaling factor
    # # if we fix width, then the scaling factor s_w = w_o / w_i
    # scaling_w = y_scale / n
    # # if we fix length, then the scaling factor s_h = h_o / h_i
    # scaling_h = x_scale / m
    # # s = max{s_w, s_h}
    # s = np.maximum(scaling_w, scaling_h)
    # if s < 1:
    #     s = 1
    s = 1
    panoramic_img = cv2.resize(panoramic_img, None, fx=1/s, fy=1/s)
    h_inv = np.linalg.pinv(h_mat) / np.linalg.pinv(h_mat)[2][2]
    # Create array
    y_pt, x_pt = np.meshgrid(np.arange(0, y_scale * s, 1 * s), np.arange(0, x_scale * s, 1 * s))
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
            loc0 = translated_pts[i * y_scale + j, 1]
            loc1 = translated_pts[i * y_scale + j, 0]
            if (loc0 > 0) and (loc1 > 0) and (loc0 < img.shape[0] - 1) and (loc1 < img.shape[1] - 1):
                panoramic_img[i][j] = compute_pixel_val(img, [loc0, loc1])

    return panoramic_img.astype(np.uint8)


imgs = [img1, img2, img3, img4, img5]
H_matrices = [H_13, H_23, H_33, H_43, H_53]
pano_img, x_min, y_min = init_pano_img(imgs, H_matrices)
pano_img = img_map(pano_img, img1, H_13, x_min, y_min)
pano_img = img_map(pano_img, img5, H_53, x_min, y_min)
pano_img = img_map(pano_img, img2, H_23, x_min, y_min)
pano_img = img_map(pano_img, img4, H_43, x_min, y_min)
pano_img = img_map(pano_img, img3, H_33, x_min, y_min)



# =========================================================================================
# Levenberg Marquardt Method


def LM(pts1, pts2, H_linear):

    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)

    def func(h):
        cost = []
        for i in range(len(pts1)):
            x = pts1[i][0] * h[0] + pts1[i][1] * h[1] + h[2] / (pts1[i][0] * h[6] + pts1[i][7] * h[4] + 1)
            y = pts1[i][0] * h[3] + pts1[i][4] * h[1] + h[5] / (pts1[i][0] * h[6] + pts1[i][7] * h[4] + 1)
            cost.append(pts2[i][0] - x)
            cost.append(pts2[i][1] - y)
        return np.asarray(cost)

    sol = least_squares(func, H_linear.squeeze(), method='lm')
    H_nonlinear = sol.x
    H_nonlinear.append(H_nonlinear, 1)

    return H_nonlinear.reshape((3, 3))


H_12_nonliear = LM(in1, in12, H_12)
H_23_nonliear = LM(in2, in23, H_23)
H_33_nonliear = np.identity(3)
H_34_nonliear = LM(in3, in34, H_34)
H_45_nonliear = LM(in4, in45, H_45)

H_43_nonliear = np.linalg.inv(H_34_nonliear)
H_54_nonliear = np.linalg.inv(H_45_nonliear)
H_13_nonliear = np.dot(H_12_nonliear, H_23_nonliear) / np.dot(H_12_nonliear, H_23_nonliear)[2][2]
H_53_nonliear = np.dot(H_54_nonliear, H_43_nonliear) / np.dot(H_54_nonliear, H_43_nonliear)[2][2]

# =========================================================================================
# Create panoramic Image

imgs = [img1, img2, img3, img4, img5]
H_matrices = [H_13_nonliear, H_23_nonliear, H_33_nonliear, H_43_nonliear, H_53_nonliear]
pano_img, x_min, y_min = init_pano_img(imgs, H_matrices)
pano_img = img_map(pano_img, img1, H_13_nonliear, x_min, y_min)
pano_img = img_map(pano_img, img5, H_53_nonliear, x_min, y_min)
pano_img = img_map(pano_img, img2, H_23_nonliear, x_min, y_min)
pano_img = img_map(pano_img, img4, H_43_nonliear, x_min, y_min)
pano_img = img_map(pano_img, img3, H_33_nonliear, x_min, y_min)