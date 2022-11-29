import numpy as np
import os
import cv2
import math


# ================================ read dataset ========================================
def load_img(img_path):
    pos_dataset = []
    pos_folder = img_path + "positive"
    pos_fseq = os.listdir(pos_folder)
    pos_fseq.sort()
    for fname in pos_fseq:
        img = cv2.imread(os.path.join(pos_folder, fname))
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            pos_dataset.append(gray)
    pos_dataset = np.asarray(pos_dataset)                          # (710, 20, 40)
    pos_label = np.asarray([[1]*len(pos_fseq)])                    # (1, 710)

    neg_dataset = []
    neg_folder = img_path + "negative"
    neg_fseq = os.listdir(pos_folder)
    neg_fseq.sort()
    for fname in neg_fseq:
        img = cv2.imread(os.path.join(neg_folder, fname))
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            neg_dataset.append(gray)
    neg_dataset = np.asarray(neg_dataset)                          # (710, 20, 40)
    neg_label = np.asarray([[0] * len(neg_fseq)])                  # (1, 710)

    return pos_dataset, pos_label, neg_dataset, neg_label


train_path = "ECE661_2020_hw11_DB2/train/"
train_pos_data, train_pos_label, train_neg_data, train_neg_label = load_img(train_path)
# print(train_pos_data.shape, train_pos_label.shape, train_neg_data.shape, train_neg_label.shape)


# ================================ compute feature =================================
def uint(num):
    return np.uint8(num)


def sum_pixels(img, A, B, C, D):
    output = img[uint(D[0]), uint(D[1])] - img[uint(B[0]), uint(B[1])] - img[uint(C[0]), uint(C[1])] + img[uint(A[0]), uint(A[1])]
    return output.astype(np.float64)


def Haar_feature_extraction(data):
    feature_ls = []
    for idx, img in enumerate(data):
        # Compute the Intergral image
        for i in range(img.ndim):
            img = img.cumsum(axis=i)
        # calculate kernels
        horizontal_kernel = [np.hstack((np.zeros((1, n)), np.ones((1, n)))) for n in range(1, int(img.shape[1]/2)-1)]
        vertical_kernel = [np.hstack((np.ones((n, 2)), np.zeros((n, 2)))) for n in range(1, int(img.shape[0]/2)-1)]
        # compute feature using kernels
        feature = []
        for kernel in horizontal_kernel:
            h, w = kernel.shape
            for j in range(img.shape[0] - 1):
                for i in range(img.shape[1] - w):
                    sum1 = sum_pixels(img, [j, i], [j, i + w/2], [j + 1, i], [j + 1, i + w/2])
                    sum2 = sum_pixels(img, [j, i + w/2], [j, i + w], [j + 1, i + w/2], [j + 1, i + w])
                    feature.append(sum2-sum1)
        for kernel in vertical_kernel:
            h, w = kernel.shape
            for j in range(img.shape[0] - h):
                for i in range(img.shape[1] - 2):
                    sum1 = sum_pixels(img, [j, i], [j, i + 2], [j + h/2, i], [j + h/2, i + 2])
                    sum2 = sum_pixels(img, [j + h/2, i], [j + h/2, i + 2], [j + h, i], [j + h, i + 2])
                    feature.append(sum1-sum2)
        feature = np.asarray(feature).flatten()
        feature_ls.append(feature)

    return np.asarray(feature_ls)


# train_pos_feature = Haar_feature_extraction(train_pos_data)
# train_neg_feature = Haar_feature_extraction(train_neg_data)
train_data = np.concatenate((train_pos_data, train_neg_data), axis=0)         # (1420, 20, 40)
# print(train_data.shape)
train_label = np.concatenate((train_pos_label, train_neg_label), axis=1)      # (1, 1420)
# print(train_label.shape)
train_feature = Haar_feature_extraction(train_data)                           # (num_samples, num_features)
# print(train_feature.shape)

# ================================== training =====================================
def build_weak_classifier(feature, label):
    num_samples, num_feature = feature.shape
    num_pos, num_neg = np.sum(label), num_samples-np.sum(label)
    # Initialization
    wgt = np.concatenate((np.ones((num_pos, )) / (2 * num_pos), np.ones((num_neg, )) / (2 * num_neg)), axis=0)
    best_classifier_ls = []
    confidence_factor = []
    for t in range(20):
        # wgt normalization
        wgt = wgt / np.sum(wgt)
        sorted_wgt = [x for _, x in sorted(zip(feature, wgt))]
        sorted_label = [x for _, x in sorted(zip(feature, label))]
        # error estimation
        T_pos, T_neg = np.sum(wgt[:num_pos]), np.sum(wgt[num_pos:])
        S_pos, S_neg = np.cumsum(sorted_wgt * sorted_label), np.cumsum(sorted_wgt * np.abs(1 - sorted_label))
        err1, err2 = (S_pos + (T_neg - S_neg)), (S_neg + (T_pos - S_pos))
        min_err = np.minimum(err1, err2)
        min_err_idx = np.argmin(min_err)
        theta = feature[min_err_idx]
        polarity = ((err1[min_err_idx] <= err2[min_err_idx]) - 0.5) * 2
        if polarity == 1:
            classification = np.asarray((feature[t] >= theta) * 1, dtype=np.uint8)
        elif polarity == -1:
            classification = np.asarray((feature[t] < theta) * 1, dtype=np.uint8)
        else:
            print('Error detected')
        wrong_num_pred = np.sum(np.abs(classification - sorted_label))
        classifier_param = [feature[min_err_idx], polarity, wrong_num_pred]
        best_classifier_ls.append(classifier_param)
        # compute confidence factor and update weights
        beta_t = min_err / (1 - min_err)
        alpha_t = np.log(1 / beta_t)
        confidence_factor.append(alpha_t)
        wgt = wgt * (beta_t ** (np.abs(classification - sorted_label)))

    return best_classifier_ls, confidence_factor


best_weak_classifier, confidence_factor = build_weak_classifier(train_feature, train_label)
strong_classifier = np.asarray(np.matmul(best_weak_classifier, confidence_factor) > np.sum(0.5 * confidence_factor))


# ================================== testing =====================================
test_path = "ECE661_2020_hw11_DB2/test/"
test_pos_data, test_pos_label, test_neg_data, test_neg_label = load_img(test_path)
test_data = np.concatenate((test_pos_data, test_neg_data), axis=0)
train_label = np.concatenate((test_pos_label, test_neg_label), axis=1)
test_feature = Haar_feature_extraction(test_data)                           # (num_samples, num_features)
