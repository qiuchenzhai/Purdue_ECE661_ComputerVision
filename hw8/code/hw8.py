import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from sklearn.metrics import confusion_matrix
import pickle


# Load images
def load_img(img_class, img_path):
    # initialization
    train_dataset = []
    test_dataset = []
    testing_labels = []
    # load training data
    for facade in img_class:
        folder = img_path + "/training/{}".format(facade)
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename))
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            if img is not None:
                train_dataset.append(img)
    # load testing data
    folder = img_path + "/testing"
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        if img is not None:
            test_dataset.append(img)
        if 'cloudy' in filename:
            testing_labels.append(0)
        if 'rain' in filename:
            testing_labels.append(1)
        if 'shine' in filename:
            testing_labels.append(2)
        if 'sunrise' in filename:
            testing_labels.append(3)
    a = np.concatenate(([0] * 290, [1] * 204), axis=None)
    b = np.concatenate(([2] * 242, [3] * 347), axis=None)
    training_labels = np.concatenate((a, b), axis=None)

    return train_dataset, test_dataset, training_labels, testing_labels


img_class = ["cloudy", "rain", "shine", "sunrise"]
img_path = "/Users/qiuchen/PycharmProjects/trial folders/hw8/imagesHW8"
train_data, test_data, training_labels, test_labels = load_img(img_class, img_path)

# split training set into training and validation set
training_set, validation_set, training_label, validation_label = train_test_split(train_data, training_labels,
                                                                                  test_size=0.30, random_state=0)


def compute_conv_kernel(M=3):
    # Generate random uniformly distributed convolutional operators
    kernel = np.random.uniform(low=-1, high=1, size=(M, M))

    return np.asarray(kernel - np.sum(kernel) / (M * M))


def compute_convd_vec(dataset, conv_operator, **kwargs):
    # convolve the image with convolutional operator and downsample the output
    dsample_sz = kwargs.pop("downsample_size", 16)
    output = np.zeros((len(dataset), dsample_sz * dsample_sz), dtype=float)
    for idx, img in enumerate(dataset):
        feature = cv2.resize(cv2.filter2D(img, -1, conv_operator), (dsample_sz, dsample_sz), interpolation=cv2.INTER_AREA).flatten()
        output[idx] = feature

    return output


def get_gram_mat(vector):
    """
    Generate Gram-matrix based C^2 / 2 dimensional feature vectors
    :param vector: vector with size: (len(dataset), 256)
    :return: C^2 / 2 dimensional feature vectors
    """

    output = np.zeros((len(vector), Nchannels, Nchannels))
    gram_mat = []
    for i in range(len(vector)):
        output[i] = np.dot(vector[i], vector[i].transpose())
        gram_mat.append(output[i][np.triu_indices(Nchannels, k=0)])

    return np.asarray(gram_mat)


# perform the classification task using Gram matrix
Ntrials = 10
Nchannels = 65
accuracy = 0
for j in range(Ntrials):
    # Compute C different KxK feature maps
    dsample_sz = 16
    train_temp = np.zeros((len(training_set), Nchannels, dsample_sz * dsample_sz), dtype=float)
    validation_temp = np.zeros((len(validation_set), Nchannels, dsample_sz * dsample_sz), dtype=float)
    kernel_list = []
    for c in range(Nchannels):
        kernel = compute_conv_kernel(M=3)
        kernel_list.append(kernel.flatten())
        train_temp[:, c, :] = compute_convd_vec(training_set, conv_operator=kernel, downsample_size=dsample_sz)
        validation_temp[:, c, :] = compute_convd_vec(validation_set, conv_operator=kernel, downsample_size=dsample_sz)
    # Generate Gram-matrx-based C^2/2 dimensional feature vectors for both training and validation images
    train_gram_mat = get_gram_mat(train_temp)
    validation_gram_mat = get_gram_mat(validation_temp)
    # Train a SVM classifier using Opencv or scikit-learn
    clf = svm.SVC(kernel='linear')
    clf.fit(train_gram_mat, training_label)
    train_predictions = clf.predict(train_gram_mat)
    print('train_prediction :', metrics.accuracy_score(training_label, train_predictions))
    # Evaluate the classification accuray on validation set, check if new features improve the accuracy
    clf_predictions = clf.predict(validation_gram_mat)
    # print(clf_predictions)
    validation_accuracy = metrics.accuracy_score(validation_label, clf_predictions)
    print('validation_prediction :', validation_accuracy)
    # Save best convolutional operators and SVM model in .xml file format for reproduciability
    if validation_accuracy > accuracy:
        best_kernel = kernel_list
        accuracy = metrics.accuracy_score(validation_label, clf_predictions)
        pkl_filename = "svm_model.pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(clf, file)


# Test image
test_temp = np.zeros((len(test_data), Nchannels, dsample_sz * dsample_sz), dtype=float)
for c in range(Nchannels):
    kernel = np.asarray(best_kernel[c]).reshape((3, 3))
    test_temp[:, c, :] = compute_convd_vec(test_data, conv_operator=kernel, downsample_size=dsample_sz)
test_gram_mat = get_gram_mat(test_temp)
# Compute the confusion matrix for test images using best model parameters
filename = "svm_model.pkl"
model = pickle.load(open(filename, 'rb'))
test_predictions = model.predict(test_gram_mat)
test_accuracy = metrics.accuracy_score(test_labels, test_predictions)
print('Test Accuracy =', test_accuracy)
print(confusion_matrix(test_labels, test_predictions))

