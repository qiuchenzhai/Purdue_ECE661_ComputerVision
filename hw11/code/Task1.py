import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


# read images
def load_img(img_path):
    train_dataset = []
    test_dataset = []
    # read training data
    train_folder = img_path + "train"
    train_fseq = os.listdir(train_folder)
    train_fseq.sort()
    for fname in train_fseq:
        img = cv2.imread(os.path.join(train_folder, fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        vec = gray.flatten()                                            # 16348x1
        train_dataset.append(vec)
    train_dataset = np.asarray(train_dataset).T                         # (16384, 630)
    train_dataset = train_dataset / np.linalg.norm(train_dataset, axis=0)
    mean_vec = np.expand_dims(np.mean(train_dataset, axis=1), axis=1)
    train_dataset = train_dataset - mean_vec
    # read testing data
    test_folder = img_path + "test"
    test_fseq = os.listdir(test_folder)
    test_fseq.sort()
    for fname in test_fseq:
        img = cv2.imread(os.path.join(test_folder, fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        vec = gray.flatten()
        test_dataset.append(vec)
    test_dataset = np.asarray(test_dataset).T                            # (16384, 630)
    test_dataset = test_dataset / np.linalg.norm(test_dataset, axis=0)
    test_dataset = test_dataset - mean_vec
    # create labels
    train_label = []
    for i in range(30):
        train_label.append([i + 1] * 21)
    train_label = np.asarray(train_label).flatten()
    test_label = train_label

    return train_dataset, test_dataset, train_label, test_label, mean_vec


img_path = "ECE661_2020_hw11_DB1/"
train_data, test_data, train_label, test_label, global_mean = load_img(img_path)
# print(train_data.shape, test_data.shape, global_mean.shape)


def PCA_feature_extraction(data, num_eig):
    u, s, vh = np.linalg.svd(np.matmul(data.T, data))     # 630 x  630
    w = np.matmul(data, vh.T)
    w = w / np.linalg.norm(w, axis=0)
    w = - w[:, 0:num_eig]                            # 16348 x K
    PCA_val = np.matmul(w.T, data)

    return w, PCA_val


def NN_classifier(train_feature, test_feature, train_label, test_label):
    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(train_feature.T, train_label)
    predict_label = classifier.predict(test_feature.T)
    accuracy_mat = np.zeros(test_label.shape)
    accuracy_mat[predict_label == test_label] = 1
    accuracy = np.sum(accuracy_mat) / 630

    return accuracy


def calculated_Z(M):
    # compute LDA eigen vecs
    u, s, vh = np.linalg.svd(np.matmul(M.T, M))
    # print(s.shape)                                     # (30,)
    evecs = np.matmul(M, vh.T)                           # (16384, 30)
    # construct z matrix
    Z = np.matmul(evecs, np.diag(s ** (-0.5)))

    return Z


class_mean = np.zeros((16384, 30))
mean_mat = []
for i in range(30):
    class_mean[:, i] = np.mean(train_data[:, i*21:(i+1)*21] + global_mean, axis=1)
mean_mat = class_mean - global_mean
Z = calculated_Z(mean_mat)


# diagonalize Z
X_w = np.zeros(train_data.shape)
for i in range(30):
    X_w[:, i*21:(i+1)*21] = train_data[:, i*21:(i+1)*21] + global_mean - np.expand_dims(mean_mat[:, i], axis=1)


def LDA_feature_extraction(Z, X_w, num_eig):
    temp = np.matmul(Z.T, X_w)
    u, s, vh = np.linalg.svd(np.matmul(temp, temp.T))
    w = np.matmul(Z, vh.T)
    w = w / np.linalg.norm(w, axis=0)
    w = w[:, 0:num_eig]
    return w                                # LDA feature


PCA_acc_ls = []
PCA_err_ls = []
LDA_acc_ls = []
LDA_err_ls = []
for k in range(20):
    K = k + 1
    # PCA
    train_PCAfeature, train_PCAval = PCA_feature_extraction(train_data, num_eig=K)
    test_PCAval = np.matmul(train_PCAfeature.T, test_data)
    PCA_accuracy = NN_classifier(train_PCAval, test_PCAval, train_label, test_label)
    PCA_acc_ls.append(PCA_accuracy)
    PCA_err_ls.append(1 - PCA_accuracy)
    # LDA
    w = LDA_feature_extraction(Z, X_w, num_eig=K)
    train_LDAval = np.matmul(w.T, train_data)
    test_LDAval = np.matmul(w.T, test_data)
    LDA_accuracy = NN_classifier(train_LDAval, test_LDAval, train_label, test_label)
    LDA_acc_ls.append(LDA_accuracy)
    LDA_err_ls.append(1 - LDA_accuracy)


# plot
M = [x+1 for x in range(20)]
plt.subplot(121)
plt.scatter(M, PCA_acc_ls)
plt.plot(M, PCA_acc_ls)
plt.scatter(M, LDA_acc_ls)
plt.plot(M, LDA_acc_ls)
plt.legend(['PCA', 'LDA'])
plt.ylabel('accuracy')
plt.xlabel('subspace dimensionality')
plt.title('classification accuracy')
plt.subplot(122)
plt.scatter(M, PCA_err_ls)
plt.plot(M, PCA_err_ls)
plt.scatter(M, LDA_err_ls)
plt.plot(M, LDA_err_ls)
plt.legend(['PCA', 'LDA'])
plt.ylabel('error rate')
plt.xlabel('subspace dimensionality')
plt.title('classification error rate')
plt.show()
print(M)
print(PCA_acc_ls)
print(LDA_err_ls)