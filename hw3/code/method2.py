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


def H_projective(pts):
    l1 = np.cross(pts[0], pts[1])
    l2 = np.cross(pts[2], pts[3])
    int_pt12 = np.cross(l1, l2) / np.cross(l1, l2)[2]

    l3 = np.cross(pts[0], pts[2])
    l4 = np.cross(pts[1], pts[3])
    int_pt34 = np.cross(l3, l4) / np.cross(l3, l4)[2]

    l_VL = np.cross(int_pt12, int_pt34) / np.cross(int_pt12, int_pt34)[2]
    H = np.array([[1, 0, 0], [0, 1, 0], [l_VL[0], l_VL[1], l_VL[2]]])

    return H


def H_affine(pts):
    l1 = np.cross(pts[0], pts[1]) / np.cross(pts[0], pts[1])[2]
    m1 = np.cross(pts[0], pts[2]) / np.cross(pts[0], pts[2])[2]
    l2 = np.cross(pts[1], pts[3]) / np.cross(pts[1], pts[3])[2]
    m2 = np.cross(pts[2], pts[3]) / np.cross(pts[2], pts[3])[2]

    A = np.array([[l1[0] * m1[0], l1[0] * m1[1] + l1[1] * m1[0]], [l2[0] * m2[0], l2[0] * m2[1] + l2[1] * m2[0]]])
    b = np.array([[-l1[1] * m1[1]], [-l2[1] * m2[1]]])
    S = np.zeros((2, 2), dtype=float)
    S[0][0] = np.dot(np.linalg.pinv(A), b)[0]
    S[0][1] = np.dot(np.linalg.pinv(A), b)[1]
    S[1][0] = np.dot(np.linalg.pinv(A), b)[1]
    S[1][1] = 1

    U, D, V = np.linalg.svd(S, full_matrices=True)
    sol = np.dot(np.dot(U, np.sqrt(np.diag(D))), U.transpose())
    H = np.array([[sol[0][0], sol[0][1], 0], [sol[1][0], sol[1][1], 0], [0, 0, 1]])

    return H


point = ["P", "Q", "S", "R"]
# Remove distortion of img1
img1 = io.imread('hw3_Task1_Images/Images/Img1.JPG')
PQSR1 = np.array([[462, 165], [431, 730], [1415, 482], [1462, 810]])
# Plot
plt.imshow(img1)
for i in range(4):
    plt.scatter(PQSR1[i][0], PQSR1[i][1])
    plt.annotate(point[i] + '({},{})'.format(PQSR1[i][0], PQSR1[i][1]), (PQSR1[i][0], PQSR1[i][1]), c='r')
plt.axis('off')
plt.savefig('img1_method2_labelled1.jpeg')
plt.show()
plt.clf()
# Computation
pts = np.array([[462, 165, 1], [431, 730, 1], [1415, 482, 1], [1462, 810, 1]])
H1 = H_projective(pts)
output1 = img_map(img1, H1)
plt.imshow(output1, cmap='gray')
plt.axis('off')
plt.savefig("img1_method2_projective.jpeg")
plt.show()
plt.clf()
H2 = H_affine(pts)
output2 = img_map(img1, np.dot(np.linalg.pinv(H2), H1))
plt.imshow(output2, cmap='gray')
plt.axis('off')
plt.savefig("img1_method2_affine.jpeg")
plt.show()
plt.clf()



# Remove distortion of img2
img2 = io.imread('hw3_Task1_Images/Images/Img2.jpeg')
PQSR2 = np.array([[368, 552], [362, 853], [621, 510], [642, 974]])
# Plot
plt.imshow(img2)
for i in range(4):
    plt.scatter(PQSR2[i][0], PQSR2[i][1])
    plt.annotate(point[i] + '({},{})'.format(PQSR2[i][0], PQSR2[i][1]), (PQSR2[i][0], PQSR2[i][1]), c='r')
plt.axis('off')
plt.savefig('img2_method2_labels.jpeg')
plt.show()
plt.clf()
# Computation
pts = np.array([[368, 552, 1], [362, 853, 1], [621, 510, 1], [642, 974, 1]])
H1 = H_projective(pts)
output1 = img_map(img2, H1)
plt.imshow(output1, cmap='gray')
plt.axis('off')
plt.savefig("img2_method2_projective.jpeg")
plt.show()
plt.clf()
H2 = H_affine(pts)
output2 = img_map(img2, np.dot(np.linalg.pinv(H2), H1))
plt.imshow(output2, cmap='gray')
plt.axis('off')
plt.savefig("img2_method2_affine.jpeg")
plt.show()
plt.clf()


# Remove distortion of img3
img3 = io.imread('hw3_Task1_Images/Images/Img3.JPG')
PQSR3 = np.array([[2060, 700], [2092, 1483], [2666, 720], [2695, 1333]])
# Plot
plt.imshow(img3, cmap='gray')
for i in range(4):
    plt.scatter(PQSR3[i][0], PQSR3[i][1])
    plt.annotate(point[i] + '({},{})'.format(PQSR3[i][0], PQSR3[i][1]), (PQSR3[i][0], PQSR3[i][1]), c='r')
plt.axis('off')
plt.savefig('img3_method2_labels.jpeg')
plt.show()
plt.clf()
# Computation
pts = np.array([[2060, 700, 1], [2092, 1483, 1], [2666, 720, 1], [2695, 1333, 1]])
H1 = H_projective(pts)
output1 = img_map(img3, H1)
plt.imshow(output1, cmap='gray')
plt.axis('off')
plt.savefig("img3_method2_projective.jpeg")
plt.show()
plt.clf()
H2 = H_affine(pts)
output2 = img_map(img3, np.dot(np.linalg.pinv(H2), H1))
plt.imshow(output2, cmap='gray')
plt.axis('off')
plt.savefig("img3_method2_affine.jpeg")
plt.show()
plt.clf()


# Remove distortion of img4
img4 = io.imread('hw3_Task1_Images/Images/Img4.jpeg')
PQSR4 = np.array([[625, 622], [609, 1044], [778, 521], [764, 970]])
# Plot
plt.imshow(img4)
for i in range(4):
    plt.scatter(PQSR4[i][0], PQSR4[i][1])
    plt.annotate(point[i] + '({},{})'.format(PQSR4[i][0], PQSR4[i][1]), (PQSR4[i][0], PQSR4[i][1]), c='r')
plt.axis('off')
plt.savefig('img4_method2_labels.jpeg')
plt.show()
plt.clf()
# Computation
pts = np.array([[625, 622, 1], [609, 1044, 1], [778, 521, 1], [764, 970, 1]])
H1 = H_projective(pts)
output1 = img_map(img4, H1)
plt.imshow(output1, cmap='gray')
plt.axis('off')
plt.savefig("img4_method2_projective.jpeg")
plt.show()
plt.clf()
H2 = H_affine(pts)
output2 = img_map(img4, np.dot(np.linalg.pinv(H2), H1))
plt.imshow(output2, cmap='gray')
plt.axis('off')
plt.savefig("img4_method2_affine.jpeg")
plt.show()
plt.clf()

# Remove distortion of img 5
img5 = io.imread('hw3_Task1_Images/Images/img5.jpeg')
PQSR5 = np.array([[573, 302], [572, 386], [692, 296], [692, 384]])
# Plot
plt.imshow(img5)
for i in range(4):
    plt.scatter(PQSR5[i][0], PQSR5[i][1])
    plt.annotate(point[i] + '({},{})'.format(PQSR5[i][0], PQSR5[i][1]), (PQSR5[i][0], PQSR5[i][1]), c='r')
plt.axis('off')
plt.savefig('img5_method2_labels.jpeg')
plt.show()
plt.clf()
# Computation
pts = np.array([[573, 302, 1], [572, 386, 1], [692, 296, 1], [692, 384, 1]])
H1 = H_projective(pts)
output1 = img_map(img5, H1)
plt.imshow(output1, cmap='gray')
plt.axis('off')
plt.savefig("img5_method2_projective.jpeg")
plt.show()
plt.clf()
H2 = H_affine(pts)
output2 = img_map(img5, np.dot(np.linalg.pinv(H2), H1))
plt.imshow(output2, cmap='gray')
plt.axis('off')
plt.savefig("img5_method2_affine.jpeg")
plt.show()
plt.clf()

# Remove distortion of img6
img6 = io.imread('hw3_Task1_Images/Images/Img6.jpeg')
PQSR6 = np.array([[232, 63], [232, 1466], [693, 133], [692, 1453]])
# Plot
plt.imshow(img6)
for i in range(4):
    plt.scatter(PQSR6[i][0], PQSR6[i][1])
    plt.annotate(point[i] + '({},{})'.format(PQSR6[i][0], PQSR6[i][1]), (PQSR6[i][0], PQSR6[i][1]), c='r')
plt.axis('off')
plt.savefig('img6_method2_labels.jpeg')
plt.show()
plt.clf()
# Computation
pts = np.array([[232, 63, 1], [232, 1466, 1], [693, 133, 1], [692, 1453, 1]])
H1 = H_projective(pts)
output1 = img_map(img6, H1)
plt.imshow(output1, cmap='gray')
plt.axis('off')
plt.savefig("img6_method2_projective.jpeg")
plt.show()
plt.clf()
H2 = H_affine(pts)
output2 = img_map(img6, np.dot(np.linalg.pinv(H2), H1))
plt.imshow(output2, cmap='gray')
plt.axis('off')
plt.savefig("img6_method2_affine.jpeg")
plt.show()
plt.clf()

