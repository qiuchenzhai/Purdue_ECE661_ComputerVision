import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io
from scipy import signal


def Haar(sigma):
    # Haar filter
    kernel_size = int(np.ceil(4 * sigma))
    if kernel_size % 2 > 0:
        kernel_size += 1
    dx = np.ones((kernel_size, kernel_size))
    dx[:, :int(kernel_size/2)] = -1
    dy = np.ones((kernel_size, kernel_size))
    dy[int(kernel_size/2):, :] = -1

    return dx, dy


def Harris_corners(img, sigma, k, **kwargs):
    # Initialization
    win_size = kwargs.pop("win_size", 15)
    height, width = img.shape
    dx_filter, dy_filter = Haar(sigma=sigma)
    dx = signal.convolve2d(img, dx_filter, mode='same')
    dy = signal.convolve2d(img, dy_filter, mode='same')
    # determine neighboring window size
    kernel_size = int(np.ceil(5 * sigma))
    if kernel_size % 2 > 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size))
    # compute entries of matrix C
    sum11 = signal.convolve2d(dx ** 2, kernel, mode='same')
    sum22 = signal.convolve2d(dy ** 2, kernel, mode='same')
    sum12 = signal.convolve2d(dx * dy, kernel, mode='same')
    # trace and determinant of matrix C
    tr = sum11 + sum22
    det = sum11 * sum22 - sum12 ** 2
    # Response of harris corner detector
    r = det - k * tr * tr
    mask = np.ones(r.shape)
    # to reject negative corner detector responses
    mask[r < 0] = 0
    corners = []
    n = int(win_size/2)
    # Suppress non-maximum values
    for i in range(n, width-n):
        for j in range(n, height-n):
            if mask[j, i] > 0:
              max_val = np.amax(r[j-n: j+n+1, i-n: i+n+1])
              if r[j, i] == max_val:
                  corners.append([i, j])

    return corners


def ssd_metric(img1, kp1, img2, corners, k):
    # Initialization
    h, w = img1.shape
    n = int(k/2)
    # Neighborhood of kp1
    img1_padded = np.zeros([h + 2 * n, w + 2 * n])
    img1_padded[n: n+h, n: n+w] = img1
    img2_padded = np.zeros([h + 2 * n, w + 2 * n])
    img2_padded[n: n+h, n: n+w] = img2
    # find the ssd between kp1 and kp2's in corner list of img2 and return the correspondence
    neighbor1 = img1_padded[kp1[1]: kp1[1] + 2 * n, kp1[0]: kp1[0] + 2 * n]
    min_ssd = 1e6
    index = int(len(corner2)-1)
    for idx, kp2 in enumerate(corners):
        neighbor2 = img2_padded[kp2[1]: kp2[1] + 2 * n, kp2[0]: kp2[0] + 2 * n]
        ssd = np.sum((neighbor1 - neighbor2) ** 2)
        if min_ssd > ssd:
            min_ssd = ssd
            index = idx

    return index, min_ssd


def ncc_metric(img1, kp1, img2, corners, k):
    # Initialization
    h, w = img1.shape
    n = int(k/2)
    # Neighborhood of kp1
    img1_padded = np.zeros([h + 2 * n, w + 2 * n])
    img1_padded[n: n+h, n: n+w] = img1
    img2_padded = np.zeros([h + 2 * n, w + 2 * n])
    img2_padded[n: n+h, n: n+w] = img2
    # find the ncc between kp1 and kp2's in corner list of img2 and return the correspondence
    neighbor1 = img1_padded[kp1[1]: kp1[1] + 2 * n, kp1[0]: kp1[0] + 2 * n]
    max_ncc = -1e6
    index = int(len(corner2)-1)
    for idx, kp2 in enumerate(corners):
        neighbor2 = img2_padded[kp2[1]: kp2[1] + 2 * n, kp2[0]: kp2[0] + 2 * n]
        sum1 = np.sum(neighbor1 - np.mean(neighbor1) ** 2)
        sum2 = np.sum(neighbor2 - np.mean(neighbor2) ** 2)
        ncc = np.sum((neighbor1 - np.mean(neighbor1)) * (neighbor2 - np.mean(neighbor2))) / np.sqrt(sum1 * sum2)
        if ncc > max_ncc:
            max_ncc = ncc
            index = idx

    return index, max_ncc


def build_corr(img1, corner1, img2, corner2, metric):
    # Assuming the corner list of img1 is shorter than corner list of img2
    match_corners = corner2
    metric_value = np.zeros((len(corner1), 1))
    # Determine the correspondence for each point in the corner list of img1
    if metric == 'SSD':
        for idx, kp in enumerate(corner1):
            index, min_ssd = ssd_metric(img1, kp, img2, corner2, k=25)
            match_corners[idx] = corner2[int(index)]
            metric_value[idx] = min_ssd
    if metric == 'NCC':
        for idx, kp in enumerate(corner1):
            index, max_ncc = ncc_metric(img1, kp, img2, corner2, k=25)
            match_corners[idx] = corner2[int(index)]
            metric_value[idx] = max_ncc
    # Return the matching pairs between two corner lists
    match_pairs = [(corner1[i], match_corners[i]) for i in range(len(corner1))]

    return match_corners, match_pairs, metric_value


def filter_corr(match_pairs, metric_value, mode, thres):
    # Find the good correspondences among all the correspondences by comparing the metric value
    pairs = []
    if mode == 'SSD':
        for i in range(len(metric_value)):
            if metric_value[i] < thres:
                pairs.append(match_pairs[i])
    if mode == 'NCC':
        for i in range(len(metric_value)):
            if metric_value[i] > thres:
                pairs.append(match_pairs[i])

    return pairs


# read images
img1 = io.imread('hw4_Task1_Images/pair3/1.JPG')
img2 = io.imread('hw4_Task1_Images/pair3/2.JPG')
# Initialization
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)
w, h, d = img1.shape
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
sigma = 2.2
k = 0.05
win_size = 15         # 15 for pair 1, 3 ; 13 for pair 2; 25 for pair 4; 31 for pair 5
corner1 = Harris_corners(gray1, sigma=sigma, k=k, win_size=win_size)
corner2 = Harris_corners(gray2, sigma=sigma, k=k, win_size=win_size)
# Plot the detected corners
img = np.hstack((img1, img2))
for i in range(len(corner1)):
    pt1 = corner1[i]
    cv2.circle(img, tuple(pt1), 4, (255, 0, 0), 2)
    cv2.circle(img, tuple(pt1), 4, (255, 0, 0), 2)
for j in range(len(corner2)):
    pt2 = [corner2[j][0] + h, corner2[j][1]]
    cv2.circle(img, tuple(pt2), 4, (255, 0, 0), 2)
    cv2.circle(img, tuple(pt2), 4, (255, 0, 0), 2)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.savefig('Corners_sigma_{}.jpeg'.format(sigma))
plt.show()
plt.clf()

# Create corner correspondence
method = 'SSD'         # 'SSD' or 'NCC'
threshold = 10000      # 10000 for pair 1, 3, 5; 20000 for pair 2; 12000 for pair 4;
name = 'SSD_sigma_{}.jpeg'.format(sigma)
# # Hyper-parameters for using NCC metric
# method = 'NCC'
# threshold = 0.042     # 0.042 for pair 1; 0.58 for pair 2; 0.22 for pair 3, 4; 0.041 for pair 5;
# name = 'NCC_sigma_{}.jpeg'.format(sigma)
if len(corner1) > len(corner2):
    match_corners, match_pairs, metric_value = build_corr(gray2, corner2, gray1, corner1, metric=method)
    good_pairs = filter_corr(match_pairs, metric_value, mode=method, thres=threshold)
    img = np.hstack((img1, img2))
    for idx, i in enumerate(good_pairs):
        pt1 = i[1]
        pt2 = [i[0][0] + h, i[0][1]]
        cv2.line(img, tuple(pt1), tuple(pt2), (0, 255, 255), 1)
        cv2.circle(img, tuple(pt1), 2, (0, 255, 255), 1)
        cv2.circle(img, tuple(pt2), 2, (0, 255, 255), 1)
else:
    match_corners, match_pairs, metric_value = build_corr(gray1, corner1, gray2, corner2, metric=method)
    good_pairs = filter_corr(match_pairs, metric_value, mode=method, thres=threshold)
    img = np.hstack((img1, img2))
    for idx, i in enumerate(good_pairs):
        pt1 = i[0]
        pt2 = [i[1][0] + h, i[1][1]]
        cv2.line(img, tuple(pt1), tuple(pt2), (0, 255, 255), 1)
        cv2.circle(img, tuple(pt1), 2, (0, 255, 255), 1)
        cv2.circle(img, tuple(pt2), 2, (0, 255, 255), 1)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.savefig(name)
plt.show()
plt.clf()

# SIFT
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good_matches = []
for a, b in matches:
    if a.distance < 0.6 * b.distance:
        good_matches.append([a])
img = np.zeros((1, 1))
img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, img, flags=2)
plt.imshow(img, cmap='gray')
plt.savefig('SIFT_img.jpeg')
plt.show()
plt.clf()


