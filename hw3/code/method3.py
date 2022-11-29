import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io


import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io


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


def compute_coe(array1, array2):
    output = np.array([array1[0] * array2[0], (array1[0]*array2[1]+array1[1]*array2[0])/2,
                       array1[1]*array2[2], (array1[0]*array2[2]+array1[2]*array2[0])/2,
                       (array1[1]*array2[2]+array1[2]*array2[1])/2])

    return output


def H_matrix(pts):
    # PQ \ PS
    l1 = np.cross(pts[0], pts[1]) / np.max(np.cross(pts[0], pts[1]))
    m1 = np.cross(pts[0], pts[2]) / np.max(np.cross(pts[0], pts[2]))
    # PS \ SR
    l2 = np.cross(pts[0], pts[2]) / np.max(np.cross(pts[0], pts[2]))
    m2 = np.cross(pts[2], pts[3]) / np.max(np.cross(pts[2], pts[3]))
    # SR \ QR
    l3 = np.cross(pts[2], pts[3]) / np.max(np.cross(pts[2], pts[3]))
    m3 = np.cross(pts[1], pts[3]) / np.max(np.cross(pts[1], pts[3]))
    # PQ \ QR
    l4 = np.cross(pts[0], pts[1]) / np.max(np.cross(pts[0], pts[1]))
    m4 = np.cross(pts[1], pts[3]) / np.max(np.cross(pts[1], pts[3]))
    # AB \ BC
    l5 = np.cross(pts[4], pts[5]) / np.max(np.cross(pts[4], pts[5]))
    m5 = np.cross(pts[5], pts[6]) / np.max(np.cross(pts[5], pts[6]))
    # Compute S_mat
    A = []
    A.append(compute_coe(l1, m1))
    A.append(compute_coe(l2, m2))
    A.append(compute_coe(l3, m3))
    A.append(compute_coe(l4, m4))
    A.append(compute_coe(l5, m5))
    A = np.asarray(A)
    b = np.array([[-l1[2] * m1[2]], [-l2[2] * m2[2]], [-l3[2] * m3[2]], [-l4[2] * m4[2]], [-l5[2] * m5[2]]])
    sol = np.dot(np.linalg.pinv(A), b) / np.max(np.abs(np.dot(np.linalg.pinv(A), b)))
    S = np.zeros((2, 2), dtype=float)
    S[0][0] = sol[0]
    S[0][1] = sol[1] / 2
    S[1][0] = sol[1] / 2
    S[1][1] = sol[2]
    # Compute SVD of S
    S_mat = np.array(S, dtype=float)
    U, D, V = np.linalg.svd(S_mat, full_matrices=True)
    temp = np.dot(np.dot(U, np.sqrt(np.diag(D))), U.transpose())
    v_vec = np.dot(np.linalg.pinv(temp), np.array([sol[3] / 2, sol[4] / 2]))
    H = np.array([[temp[0][0], temp[0][1], 0], [temp[1][0], temp[1][1], 0], [v_vec[0], v_vec[1], 1]], dtype=float)

    return H


# # Method 3 for image1
# img1 = io.imread('hw3_Task1_Images/Images/Img1.JPG')
# PQSR1 = np.array([[462, 165], [431, 730], [1415, 482], [1462, 810], [165, 69], [98, 876], [425, 883]])
# point1 = ["P", "Q", "S", "R", "A", "B", "C"]
# # Plot
# plt.imshow(img1)
# for i in range(7):
#     plt.scatter(PQSR1[i][0], PQSR1[i][1])
#     plt.annotate(point1[i] + '({},{})'.format(PQSR1[i][0], PQSR1[i][1]), (PQSR1[i][0], PQSR1[i][1]), c='r')
# plt.axis('off')
# plt.savefig('img1_method3_labels.jpeg')
# plt.show()
# plt.clf()
# pts = np.array([[462, 165, 1], [431, 730, 1], [1415, 482, 1], [1462, 810, 1], [165, 69, 1], [98, 876, 1], [425, 883, 1]])
# # Computation
# H1 = H_matrix(pts)
# output1 = img_map(img1, np.linalg.inv(H1))
# plt.imshow(output1, cmap='gray')
# plt.axis('off')
# plt.savefig("img1_method3_result.jpeg")
# plt.show()
# plt.clf()

# Method 3 for image2
img2 = io.imread('hw3_Task1_Images/Images/Img2.jpeg')
PQSR2 = np.array([[368, 552], [362, 853], [621, 510], [642, 974], [309, 342], [291, 896], [496, 1023]])
point2 = ["P", "Q", "S", "R", "A", "B", "C"]
# Plot
plt.imshow(img2)
for i in range(7):
    plt.scatter(PQSR2[i][0], PQSR2[i][1])
    plt.annotate(point2[i] + '({},{})'.format(PQSR2[i][0], PQSR2[i][1]), (PQSR2[i][0], PQSR2[i][1]), c='r')
plt.axis('off')
plt.savefig('img2_method3_labels.jpeg')
plt.show()
plt.clf()
pts = np.array([[368, 552, 1], [362, 853, 1], [621, 510, 1], [642, 974, 1], [309, 342, 1], [291, 896, 1], [496, 1023, 1]])
# Computation
H2 = H_matrix(pts)
output2 = img_map(img2, np.linalg.inv(H2))
plt.imshow(output2, cmap='gray')
plt.axis('off')
plt.savefig("img2_method3_result.jpeg")
plt.show()
plt.clf()

# Method 3 for image3
point = ['P', 'Q', 'S', 'R', 'A', 'B', 'C']
img3 = io.imread('hw3_Task1_Images/Images/Img3.JPG')
PQSR3 = np.array([[2060, 700], [2092, 1483], [2666, 720], [2695, 1333], [689, 750], [741, 2088], [1774, 1621]])
plt.imshow(img3, cmap='gray')
for i in range(7):
    plt.scatter(PQSR3[i][0], PQSR3[i][1])
    plt.annotate(point[i] + '({},{})'.format(PQSR3[i][0], PQSR3[i][1]), (PQSR3[i][0], PQSR3[i][1]), c='r')
plt.axis('off')
plt.savefig('img3_method3_labels.jpeg')
plt.show()
plt.clf()
pts3 = np.array([[2060, 700, 1], [2092, 1483, 1], [2666, 720, 1], [2695, 1333, 1], [689, 750, 1], [741, 2088, 1], [1774, 1621, 1]])
H3 = H_matrix(pts3)
output3 = img_map(img3, np.linalg.inv(H3))
plt.imshow(output3, cmap='gray')
plt.axis('off')
plt.savefig("img3_method3_result.jpeg")
plt.show()
plt.clf()


# Method 3 for image4
img4 = io.imread('hw3_Task1_Images/Images/Img4.jpeg')
PQSR4 = np.array([[625, 622], [609, 1044], [778, 521], [764, 970], [396, 776], [378, 1153], [497, 1094]])
point4 = ["P", "Q", "S", "R", "A", "B", "C"]
# Plot
plt.imshow(img4)
for i in range(7):
   plt.scatter(PQSR4[i][0], PQSR4[i][1])
   plt.annotate(point4[i] + '({},{})'.format(PQSR4[i][0], PQSR4[i][1]), (PQSR4[i][0], PQSR4[i][1]), c='r')
plt.axis('off')
plt.savefig('img4_method3_labels.jpeg')
plt.show()
plt.clf()
pts4 = np.array([[625, 622, 1], [609, 1044, 1], [778, 521, 1], [764, 970, 1], [396, 776, 1], [378, 1153, 1], [497, 1094, 1]])
H4 = H_matrix(pts4)
output4 = img_map(img4, np.linalg.pinv(H4))
plt.imshow(output4, cmap='gray')
plt.axis('off')
plt.savefig("img4_method3_result.jpeg")
plt.show()
plt.clf()


# Method 3 for image5
img5 = io.imread('hw3_Task1_Images/Images/img5.jpeg')
PQSR5 = np.array([[573, 302], [572, 386], [692, 296], [692, 384], [110, 230], [106, 316], [184, 227], [181, 313]])
point5 = ["P", "Q", "S", "R", "A", "B", "C", "D"]
plt.imshow(img5)
for i in range(len(point5)):
   plt.scatter(PQSR5[i][0], PQSR5[i][1])
   plt.annotate(point5[i] + '({},{})'.format(PQSR5[i][0], PQSR5[i][1]), (PQSR5[i][0], PQSR5[i][1]), c='r')
plt.axis('off')
plt.savefig('img5_method3_labels.jpeg')
plt.show()
plt.clf()
pts = np.array([[573, 302, 1], [572, 386, 1], [692, 296, 1], [692, 384, 1], [110, 230, 1], [106, 316, 1], [184, 227, 1], [181, 313, 1]])
H = H_matrix(pts)
output5 = img_map(img5, np.linalg.inv(H))
plt.imshow(output5, cmap='gray')
plt.axis('off')
plt.savefig("img5_method3_result.jpeg")
plt.show()
plt.clf()







