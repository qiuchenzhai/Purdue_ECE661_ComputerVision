import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from skimage import io


def compute_thres(img_mono):
    """
    Compute the threshold that maximizes the between class variance
    """
    img = img_mono.flatten()
    hist, bin_edges = np.histogram(img, bins=256, range=(0, 256), density=True)
    plt.hist(hist, bins=256)
    plt.show()
    plt.clf()
    pr_hist = hist * np.arange(256)
    mean_total = np.sum(pr_hist)
    optimum = -1e3
    for i in range(256):
        omega = np.sum(hist[:i+1])
        mu = np.sum(pr_hist[:i+1])
        if ((mean_total*omega - mu) **2 / (omega * (1-omega)) > optimum) and (omega!=0) and (omega!=1):
            optimum = (mean_total*omega - mu) **2 / (omega * (1-omega))
            thres = i

    return thres


def otsu_mask(img_mono, num_iter):
    """
    Otsu Algorithm
    """
    mask = np.ones(img_mono.shape, dtype=np.uint8)
    for i in range(num_iter):
        thres = compute_thres(img_mono[np.nonzero(mask)])
        # print(thres)
        ret, mask = cv2.threshold(img_mono, thres, 255, cv2.THRESH_BINARY)

    return mask


def otsu_rgb(img, num_iter):
    """
    Get mask using Otsu algorithm using RGB values
    """
    b, g, r = cv2.split(img)
    b_iter, g_iter, r_iter = num_iter[0], num_iter[1], num_iter[2]
    mask_b = otsu_mask(b, b_iter)
    mask_g = otsu_mask(g, g_iter)
    mask_r = otsu_mask(r, r_iter)
    mask = cv2.bitwise_and(cv2.bitwise_and(mask_b, mask_g), mask_r)
    # Plot
    figure(num=None, figsize=(10, 1.5), dpi=160, facecolor='w', edgecolor='k')
    plt.subplot(141)
    plt.imshow(mask_b, cmap='gray')
    plt.colorbar()
    plt.axis('off')
    plt.title('mask of B channel')
    plt.subplot(142)
    plt.imshow(mask_g, cmap='gray')
    plt.colorbar()
    plt.axis('off')
    plt.title('mask of G channel')
    plt.subplot(143)
    plt.imshow(mask_r, cmap='gray')
    plt.colorbar()
    plt.axis('off')
    plt.title(r'mask of R channel')
    plt.subplot(144)
    plt.imshow(mask, cmap='gray')
    plt.colorbar()
    plt.axis('off')
    plt.title(r'the combined mask')
    plt.show()
    plt.clf()
    return mask_b, mask_g, mask_r, mask


def plot_rgb(img):
    """
    Plot the original image and the RGB channels
    """
    b, g, r = cv2.split(img)
    figure(num=None, figsize=(10, 1.5), dpi=160, facecolor='w', edgecolor='k')
    plt.subplot(141)
    plt.imshow(img, cmap='gray')
    plt.colorbar()
    plt.axis('off')
    plt.title('original image')
    plt.subplot(142)
    plt.imshow(b, cmap='gray')
    plt.colorbar()
    plt.axis('off')
    plt.title('B channel')
    plt.subplot(143)
    plt.imshow(g, cmap='gray')
    plt.colorbar()
    plt.axis('off')
    plt.title(r'G channel')
    plt.subplot(144)
    plt.imshow(r, cmap='gray')
    plt.colorbar()
    plt.axis('off')
    plt.title(r'R channel')
    plt.show()
    plt.clf()


def refine_mask(mask, kernel_size, num_iter, method):
    """
    Denoise mask
    """
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    if method == 'dilation' or 'dilate':
        output = cv2.dilate(mask, kernel, iterations=num_iter)
    elif method == 'erosion' or 'erose':
        output = cv2.erode(mask, kernel, iterations=num_iter)
    # plt.subplot(121)
    # plt.imshow(mask, cmap='gray')
    # plt.axis('off')
    # plt.title('before')
    # plt.subplot(122)
    # plt.imshow(output, cmap='gray')
    # plt.axis('off')
    # plt.title('after')
    # plt.show()
    # plt.clf()
    return output


def get_contour(mask, window_size):
    """
    Obtain the contour
    """
    output = np.zeros(mask.shape, dtype=np.uint8)
    for i in range(window_size, mask.shape[0]-window_size):
        for j in range(window_size, mask.shape[1]-window_size):
            if np.min(mask[i - window_size:i+window_size+1, j-window_size:j+window_size+1])==0:
                output[i, j] = 255

    output[mask == 0] = 0
    plt.imshow(output, cmap='gray')
    plt.axis('off')
    plt.show()
    plt.clf()
    return output


def get_texture(img, win_size = [3, 5, 7], Niter):
    """
    Get mask from Otsu algorithm based on texture features
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    texture_img = np.zeros(img.shape, dtype=np.uint8)
    masks = np.zeros(img.shape, dtype=np.uint8)
    for n in range(len(win_size)):
        N = win_size[n]
        temp = np.zeros((mask.shape[0] + 2 * int((N - 1) / 2), mask.shape[1] + 2 * int((N - 1) / 2)), dtype=np.uint8)
        temp[int((N - 1) / 2):temp.shape[0] - int((N - 1) / 2), int((N - 1) / 2):temp.shape[1] - int((N - 1) / 2)] = gray
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                x = i + int((N - 1) / 2)
                y = j + int((N - 1) / 2)
                window = temp[x - int((N - 1) / 2): x + int((N - 1) / 2), y - int((N - 1) / 2): y + int((N - 1) / 2)]
                mask[i, j] = np.var(window)
        texture_img[:, :, n] = mask
        masks[:, :, n] = otsu_mask(mask, Niter[n])
    mask = cv2.bitwise_and(cv2.bitwise_and(masks[:, :, 0], masks[:, :, 1]), masks[:, :, 2])
    figure(num=None, figsize=(10, 1.5), dpi=160, facecolor='w', edgecolor='k')
    plt.subplot(141)
    plt.imshow(masks[:, :, 0], cmap='gray')
    plt.colorbar()
    plt.axis('off')
    plt.title('window size 3x3')
    plt.subplot(142)
    plt.imshow(masks[:, :, 1], cmap='gray')
    plt.colorbar()
    plt.axis('off')
    plt.title('window size 5x5')
    plt.subplot(143)
    plt.imshow(masks[:, :, 2], cmap='gray')
    plt.colorbar()
    plt.axis('off')
    plt.title(r'window size 7x7')
    plt.subplot(144)
    plt.imshow(mask, cmap='gray')
    plt.colorbar()
    plt.axis('off')
    plt.title(r'the combined mask')
    plt.show()
    plt.clf()

    return mask, texture_img




