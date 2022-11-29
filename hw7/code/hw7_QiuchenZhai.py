import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


# Load images
def load_img(img_class, img_path):
    # initialization
    train_dataset = []
    test_dataset = []
    # load training data
    for facade in img_class:
        folder = img_path + "/training/{}".format(facade)
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                train_dataset.append(img)
    # load testing data
    folder = img_path + "/testing"
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            test_dataset.append(img)
    # create training labels for five facades
    a = np.concatenate(([0] * 20, [1] * 20), axis=None)
    b = np.concatenate(([2] * 20, [3] * 20), axis=None)
    c = np.concatenate((a, b), axis=None)
    training_labels = np.concatenate((c, [4] * 20), axis=None)
    # create test label
    a = np.concatenate(([0] * 5, [1] * 5), axis=None)
    b = np.concatenate(([2] * 5, [3] * 5), axis=None)
    c = np.concatenate((a, b), axis=None)
    test_labels = np.concatenate((c, [4] * 5), axis=None)

    return train_dataset, test_dataset, training_labels, test_labelss


img_class = ["beach", "building", "car", "mountain", "tree"]
img_path = "/imagesDatabasedHW7"
train_data, test_data, training_labels, test_labels = load_img(img_class, img_path)


# LBP features extraction
def bilinear_interp(A, B, C, D):

    du = np.cos(2 * np.pi / 8)
    dv = np.sin(2 * np.pi / 8)
    output = (1-du) * (1-dv) * A + (1-du) * dv * B + du * (1-dv) * C + du * dv * D

    return output


def get_binary_pattern(img, x, y):
    # num_neighbor = 8
    # radius = 1
    # binary_vector = np.zeros((1, num_neighbor))
    # for idx in num_neighbor:
    #     du = radius * np.cos(2 * np.pi * idx / num_neighbor)
    #     dv = radius * np.sin(2 * np.pi * idx / num_neighbor)
    #     du = 0 if np.abs(du) < 1e-6 else du = du
    #     dv = 0 if np.abs(dv) < 1e-6 else dv = dv
    pattern_vec = img[x-1:x+2, y-1:y+2].flatten()
    p0 = pattern_vec[7]
    p1 = bilinear_interp(pattern_vec[4], pattern_vec[5], pattern_vec[7], pattern_vec[8])
    p2 = pattern_vec[5]
    p3 = bilinear_interp(pattern_vec[4], pattern_vec[5], pattern_vec[1], pattern_vec[2])
    p4 = pattern_vec[1]
    p5 = bilinear_interp(pattern_vec[4], pattern_vec[3], pattern_vec[1], pattern_vec[0])
    p6 = pattern_vec[3]
    p7 = bilinear_interp(pattern_vec[4], pattern_vec[3], pattern_vec[7], pattern_vec[6])
    binary_vector = np.array([p0, p1, p2, p3, p4, p5, p6, p7]) >= pattern_vec[4]

    return binary_vector.astype(int)


def encode_feature(binary_vector):
    P = len(binary_vector)
    final_vector = binary_vector
    int_value = 1e6
    for idx in range(P+1):
        binary_vector = np.roll(binary_vector, 1)
        if binary_vector.dot(1 << np.arange(binary_vector.shape[-1] - 1, -1, -1)) <= int_value:
            int_value = binary_vector.dot(1 << np.arange(binary_vector.shape[-1] - 1, -1, -1))
            final_vector = binary_vector
    if final_vector[0] == 0:
        if final_vector.dot(1 << np.arange(final_vector.shape[-1] - 1, -1, -1)) == 1:
            feature = 1
        elif final_vector.dot(1 << np.arange(final_vector.shape[-1] - 1, -1, -1)) == 3:
            feature = 2
        elif final_vector.dot(1 << np.arange(final_vector.shape[-1] - 1, -1, -1)) == 7:
            feature = 3
        elif final_vector.dot(1 << np.arange(final_vector.shape[-1] - 1, -1, -1)) == 15:
            feature = 4
        elif final_vector.dot(1 << np.arange(final_vector.shape[-1] - 1, -1, -1)) == 31:
            feature = 5
        elif final_vector.dot(1 << np.arange(final_vector.shape[-1] - 1, -1, -1)) == 63:
            feature = 6
        elif final_vector.dot(1 << np.arange(final_vector.shape[-1] - 1, -1, -1)) == 127:
            feature = 7
        elif final_vector.dot(1 << np.arange(final_vector.shape[-1] - 1, -1, -1)) == 0:
            feature = 0
        else:
            feature = P + 1
    elif final_vector.dot(1 << np.arange(final_vector.shape[-1] - 1, -1, -1)) == 255:
        feature = P
    else:
        feature = P + 1

    return feature


def lbp_feature_extraction(dataset, num_neighbor):

    img_hists = np.zeros((len(dataset), num_neighbor+2))
    for idx, img in enumerate(dataset):
        histogram = []
        for x in range(1, img.shape[0]-1):
            for y in range(1, img.shape[1]-1):
                binary_vec = get_binary_pattern(img, x, y)
                feature_integer = encode_feature(binary_vec)
                histogram.append(feature_integer)
        hist, _ = np.histogram(histogram, bins=num_neighbor+2)
        img_hists[idx, :] = hist / np.sum(hist)

    return img_hists


train_lbps = lbp_feature_extraction(train_data, num_neighbor=8)
test_lbps = lbp_feature_extraction(test_data, num_neighbor=8)


# kNN and confusion matrix
confusion_mat = np.zeros((5, 5))
for idx in range(len(test_data)):
    ed = [np.linalg.norm(train_lbps[iter] - test_lbps[idx]) for iter in range(len(train_data))]
    best_prediction = [x for _, x in sorted(zip(ed, test_labels))]
    # print(best_prediction)
    prediction_list = np.array(best_prediction)[0:5]
    prediction_list = prediction_list.tolist()
    prediction = max(set(prediction_list), key=prediction_list.count)
    confusion_mat[test_labels[idx], prediction] += 1
    if test_labels[idx] != prediction:
        print(idx, 'gt=', test_labels[idx], 'prediction=', prediction)
print('confusion matrix =', confusion_mat)
