import numpy as np
import cv2
from scipy.linalg import null_space
import matplotlib.pyplot as plt
from scipy import optimize
from skimage import io
from pylab import *
from math import *
from scipy import signal
from scipy.optimize import least_squares


# ======================================================================================================================
# read images
point = ['1', '2', '3', '4', '5', '6', '7', '8']
img1 = io.imread('/content/1.jpg')
interest_pt1 = np.array([[780, 1078], [2453, 463], [3268, 844], [1507, 1792], [950, 1763], [3084, 1474], [1588, 2486], [2392, 960]])
# plt.imshow(img1)
# for i in range(len(point)):
#     plt.scatter(interest_pt1[i][0], interest_pt1[i][1])
#     plt.annotate(point[i] + '({},{})'.format(interest_pt1[i][0], interest_pt1[i][1]), (interest_pt1[i][0], interest_pt1[i][1]), c='r')
# plt.axis('off')
# plt.show()
# plt.clf()

img2 = io.imread('/content/2.jpg')
interest_pt2 = np.array([[906, 1148], [2522, 444], [3416, 809], [1822, 1887], [1063, 1850], [3201, 1453], [1843, 2588], [2568, 964]])
# plt.imshow(img2)
# for i in range(len(point)):
#     plt.scatter(interest_pt2[i][0], interest_pt2[i][1])
#     plt.annotate(point[i] + '({},{})'.format(interest_pt2[i][0], interest_pt2[i][1]), (interest_pt2[i][0], interest_pt2[i][1]), c='r')
# plt.axis('off')
# plt.show()
# plt.clf()

w, h, d = img1.shape
pts1 = interest_pt1
pts2 = interest_pt2
img = np.hstack((img1, img2))
for i in range(len(point)):
    pt1 = np.array(pts1[i])
    pt2 = np.array([pts2[i][0] + h, pts2[i][1]])
    cv2.line(img, tuple(pt1), tuple(pt2), (255, 0, 255), 5)
    cv2.circle(img, tuple(pt1), 20, (0, 255, 255), 20)
    cv2.circle(img, tuple(pt2), 20, (0, 255, 255), 20)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()
plt.clf()


def normalize_pt(pts):
    mu_x = np.mean(pts[:, 0])
    mu_y = np.mean(pts[:, 1])
    dist_x = (pts[:, 0] - mu_x) ** 2
    dist_y = (pts[:, 1] - mu_y) ** 2
    dist = np.sqrt(dist_x + dist_y)
    mu = np.sum(dist) / len(pts)
    scale = np.sqrt(2) / mu
    T = np.array([[scale, 0, -scale * mu_x], [0, scale, -scale * mu_y], [0, 0, 1]])
    new_pts = []
    for i in range(len(pts)):
        new_pts.append(np.matmul(T, [pts[i][0], pts[i][1], 1]))

    return T, np.array(new_pts)


def calculate_fund_mat(pts1, pts2, T1, T2):
    A = []
    for i in range(len(pts1)):
        A.append(np.array([pts2[i][0]*pts1[i][0], pts2[i][0]*pts1[i][1], pts2[i][0], pts2[i][1]*pts1[i][0],
                           pts2[i][1]*pts1[i][1], pts2[i][1], pts1[i][0], pts1[i][1], 1.0]))
    u, s, vh = np.linalg.svd(A)
    F = vh[-1, :].reshape(3, 3)
    # print(F)
    # rank constraint
    u, s, vh = np.linalg.svd(F)
    s = [s[0], s[1], 0]
    F = np.dot(u * s, vh)
    # de-normalize
    F = np.matmul(np.matmul(T2.T, F), T1)

    return F


# ===================== normalize the point =====================
T1, norm_pt1 = normalize_pt(pts1)
T2, norm_pt2 = normalize_pt(pts2)
# ===================== calculate fund mat =====================
F = calculate_fund_mat(norm_pt1, norm_pt2, T1, T2)
print('Fundamental Matrix:', F)
print('Rank of Fundamental Matrix:', np.linalg.matrix_rank(F))
# ===================== calculate the null vectors =====================
e = null_space(F) / null_space(F)[2]
print('The left epipoles is:', e.T)
e_prime = null_space(F.T) / null_space(F.T)[2]
print('The right epipoles is:', e_prime.T)
# ===================== calculate the canonical configuration =====================
P = np.hstack((np.eye(3), np.zeros((3, 1))))
print(P)
e_prime_x = [[0, -e_prime[2], e_prime[1]], [e_prime[2], 0, -e_prime[0]], [-e_prime[1], e_prime[0], 0]]
P_prime = np.hstack((np.dot(e_prime_x, F), e_prime))
print(P_prime)


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


def img_map(img, h_mat):
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


def rectify_img(null_vec, img, pts, img_sz):
    w, h = img_sz[0], img_sz[1]
    theta = arctan(-(null_vec[1] - h / 2) / (null_vec[0] - w / 2))
    print('theta =', theta)
    R = np.array([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]])
    print('R =', R)
    # f = cos(theta) * (null_vec[0] - w/2) - sin(theta) * (null_vec[1] - h/2)
    # print(f)
    G = np.eye(3)
    G[2, 0] = 1 / (cos(theta) * (null_vec[0] - w / 2) - sin(theta) * (null_vec[1] - h / 2))
    print('G =', G)
    T = np.eye(3)
    T[0, 2], T[1, 2] = -w / 2, -h / 2
    print('T =', T)
    H = np.matmul(np.matmul(G, R), T)
    # print('H =', H)
    center = np.array([[w / 2, h / 2, 1]])
    center_new = np.matmul(H, center.T) / np.matmul(H, center.T)[2]
    print('new center coordinate:', center_new)
    T = np.eye(3)
    T[0, 2], T[1, 2] = w / 2 - center_new[0], h / 2 - center_new[1]
    H = np.matmul(T, H)
    print('H =', H)
    rectified_img = img_map(img, H)
    plt.imshow(rectified_img, cmap='gray')
    plt.axis('off')
    plt.show()
    plt.clf()
    new_pts = []
    for i in range(len(pts)):
        new_pts.append(np.matmul(H, [pts[i][0], pts[i][1], 1]) / np.matmul(H, [pts[i][0], pts[i][1], 1])[2])
    return H, rectified_img, new_pts


# ===================== calculate H1 ==========================
print('================ calculate H1 ==========================')
H1, rect_img1, rect_pts1 = rectify_img(e, img1, pts1, img1.shape)
print(H1)
print(rect_pts1)
# ===================== calculate H2 ===========================
print('================ calculate H2 ==========================')
H2, rect_img2, rect_pts2 = rectify_img(e_prime, img2, pts2, img1.shape)
print(H2)
print(rect_pts2)


def corner_detection(img, **kwargs):
    sigma = kwargs.pop("sigma", 0.33)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Canny edge detector
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    median_intensity = np.median(blurred)   # median of pixel intensities
    lower = int(max(0, (1.0 - sigma) * median_intensity))
    upper = int(min(255, (1.0 + sigma) * median_intensity))
    edges = cv2.Canny(blurred, threshold1=lower, threshold2=upper)
    return edges

def extract_corner(img, win, win_sz):
    corners=[]
    cnt = 0
    xmin, xmax, ymin, ymax = win[0], win[1], win[2], win[3]
    # Suppress non-maximum values
    for i in range(xmin, xmax):
        for j in range(ymin, ymax):
            if img[j, i] == 255:
                cnt += 1
                if cnt % win_sz == 0:
                    corners.append([i, j])
    return corners


edges1 = corner_detection(rect_img1)
plt.imshow(edges1, cmap='gray')
plt.axis('off')
plt.show()
plt.clf()
print(edges1.shape)

corners1 = extract_corner(edges1, win = [530, 2300, 950, 2560], win_sz=51)
print(len(corners1))
print(corners1)
img = edges1
for j in range(len(corners1)):
    pt1 = corners1[j]
    cv2.circle(img, tuple(pt1), 20, (255, 0, 0), 10)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()
plt.clf()

edges2 = corner_detection(rect_img2)
plt.imshow(edges2, cmap='gray')
plt.axis('off')
plt.show()
plt.clf()

print(edges2.shape)
corners2 = extract_corner(edges2, win = [600, 2600, 850, 2600], win_sz=51)
print(len(corners2))
print(corners2)
img = edges2
for j in range(len(corners2)):
    pt2 = corners2[j]
    cv2.circle(img, tuple(pt2), 20, (255, 0, 0), 10)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()
plt.clf()


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
    index = int(len(corners)-1)
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
    index = int(len(corners)-1)
    for idx, kp2 in enumerate(corners):
        neighbor2 = img2_padded[kp2[1]: kp2[1] + 2 * n, kp2[0]: kp2[0] + 2 * n]
        sum1 = np.sum(neighbor1 - np.mean(neighbor1) ** 2)
        sum2 = np.sum(neighbor2 - np.mean(neighbor2) ** 2)
        # print(neighbor1.shape)
        # print(neighbor2.shape)
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
            index, min_ssd = ssd_metric(img1, kp, img2, corner2, k=21)
            match_corners[idx] = corner2[int(index)]
            metric_value[idx] = min_ssd
    if metric == 'NCC':
        for idx, kp in enumerate(corner1):
            index, max_ncc = ncc_metric(img1, kp, img2, corner2, k=21)
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


rect_img2 = cv2.resize(rect_img2, (rect_img1.shape[1], rect_img1.shape[0]), interpolation=cv2.INTER_AREA)
method = 'NCC'
w, h, d = rect_img1.shape
threshold = 1
gray1 = cv2.cvtColor(rect_img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(rect_img2, cv2.COLOR_BGR2GRAY)
if len(corners1) > len(corners2):
    match_corners, match_pairs, metric_value = build_corr(gray2, corners2, gray1, corners1, metric=method)
    good_pairs = filter_corr(match_pairs, metric_value, mode=method, thres=threshold)
    img = np.hstack((rect_img1, rect_img2))
    for idx, i in enumerate(good_pairs):
        pt1 = i[1]
        pt2 = [i[0][0] + h, i[0][1]]
        cv2.line(img, tuple(pt1), tuple(pt2), (0, 255, 255), 11)
        cv2.circle(img, tuple(pt1), 12, (0, 255, 255), 11)
        cv2.circle(img, tuple(pt2), 12, (0, 255, 255), 11)
else:
    match_corners, match_pairs, metric_value = build_corr(gray1, corners1, gray2, corners2, metric=method)
    good_pairs = filter_corr(match_pairs, metric_value, mode=method, thres=threshold)
    img = np.hstack((rect_img1, rect_img2))
    for idx, i in enumerate(good_pairs):
        pt1 = i[0]
        pt2 = [i[1][0] + h, i[1][1]]
        cv2.line(img, tuple(pt1), tuple(pt2), (0, 255, 255), 11)
        cv2.circle(img, tuple(pt1), 12, (0, 255, 255), 11)
        cv2.circle(img, tuple(pt2), 12, (0, 255, 255), 11)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()
plt.clf()
cv2.imwrite("./cor_corr.png", img)


def triangulate_pts(corner_pairs, P, P_prime):
    pts = []
    for pt1, pt2 in corner_pairs:
        A = np.zeros((4, 4))
        A[0,:] = pt1[0] * P[2,:] - P[0,:]
        A[1, :] = pt1[1] * P[2,:] - P[1,:]
        A[2, :] = pt2[0] * P_prime[2,:] - P_prime[0,:]
        A[3, :] = pt2[1] * P_prime[2,:] - P_prime[1,:]
        u, s, vh = np.linalg.svd(np.dot(A.T, A))
        pt = vh[-1, :]
        pts.append(pt/pt[2])

    return np.asarray(pts)


world_pts = triangulate_pts(zip(interest_pt1, interest_pt2), P, P_prime)
x, y, z = world_pts[:, 0], world_pts[:, 1], world_pts[:, 2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:], y[:], z[:], color='b', zdir='y', depthshade=False)
pts = world_pts
for i in range(8):
    ax.text(x[i], y[i], z[i], str(i + 1), zdir='y')
for p, q in [(0, 1), (1, 2), (2, 3), (0, 3), (4, 6), (5, 6), (0, 4), (3, 6), (2, 4)]:
    ax.plot([pts[p][0], pts[q][0]], [pts[p][1], pts[q][1]], zs=[pts[p][2], pts[q][2]])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


def LM(pts1, pts2, P, P_prime):
    def func(p_mat):
        reproj_pairs = []
        world_pts = triangulate_pts(zip(pts1, pts2), P, p_mat.reshape(3, 4))
        for i in range(len(pts1)):
            pt1_reproj = np.dot(P, world_pts[i]) / np.dot(P, world_pts[i])[-1]
            pt2_reproj = np.dot(p_mat.reshape(3, 4), world_pts[i]) / np.dot(p_mat.reshape(3, 4), world_pts[i])[-1]
            reproj_pairs.append([pt1_reproj, pt2_reproj])
        cost = np.hstack((pts1, pts2)) - np.hstack((pt1_reproj, pt2_reproj))
        return cost.flatten()
    sol = least_squares(func, P_prime.flatten(), method='lm')
    P_prime = sol.x
    P_prime = P_prime.reshape(3, 4)
    return P_prime / P_prime[-1, -1]


corr1 = []
corr2 = []
for i in range(len(good_pairs)):
    corr1.append([good_pairs[i][0][0], good_pairs[i][0][1], 1])
    corr2.append([good_pairs[i][1][0], good_pairs[i][1][1], 1])
pts1 = np.asarray(corr1)
pts2 = np.asarray(corr2)
P_prime_refined = LM(pts2, pts1, P, P_prime)
print(P_prime_refined)
e_prime_refined = P_prime_refined[:, 3]
e_prime_refined_x = [[0, -e_prime_refined[2], e_prime_refined[1]], [e_prime_refined[2], 0, -e_prime_refined[0]], [-e_prime_refined[1], e_prime_refined[0], 0]]
F_refined = np.dot(e_prime_refined_x, np.dot(P_prime_refined, np.dot(P.T, np.linalg.inv(np.dot(P, P.T)))))
print('The refined Fundamental Matrix:', F)
print('Rank of Fundamental Matrix:', np.linalg.matrix_rank(F))

world_pts = triangulate_pts(zip(interest_pt1, interest_pt2), P, P_prime_refined)
x, y, z = world_pts[:, 0], world_pts[:, 1], world_pts[:, 2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:], y[:], z[:], color='b', zdir='y', depthshade=False)
pts = world_pts
for i in range(8):
    ax.text(x[i], y[i], z[i], str(i + 1), zdir='y')
for p, q in [(0, 1), (1, 2), (2, 3), (0, 3), (4, 6), (5, 6), (0, 4), (3, 6), (2, 4)]:
    ax.plot([pts[p][0], pts[q][0]], [pts[p][1], pts[q][1]], zs=[pts[p][2], pts[q][2]])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
