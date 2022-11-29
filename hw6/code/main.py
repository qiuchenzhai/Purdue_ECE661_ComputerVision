import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from skimage import io
from utils import *

# Read images
img1 = io.imread('/hw6_images/cat.jpg')
img2 = io.imread('/hw6_images/Red-Fox_.jpg')
img3 = io.imread('/hw6_images/pigeon.jpeg')
# ========================================================================================
# img1
img = img1
plot_rgb(img)
iters = [1, 1, 1]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# b, g, r = cv2.split(img)
# img = cv2.merge((b,g,r))
# RGB_based
mask_b, mask_g, mask_r, final_mask = otsu_rgb(img, num_iter=iters)
final_mask = refine_mask(final_mask, kernel_size=3, num_iter=1, method='dilation')
final_mask = refine_mask(final_mask, kernel_size=3, num_iter=1, method='erosion')

foreground_img = cv2.bitwise_and(img, cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR))
background_img = cv2.bitwise_xor(img, cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR))
# plt.subplot(121)
# plt.imshow(foreground_img, cmap='gray')
# plt.axis('off')
# plt.subplot(122)
# plt.imshow(background_img, cmap='gray')
# plt.axis('off')
# plt.show()
# plt.clf()

contour_img = get_contour(final_mask, window_size=1)
plt.imshow(contour_img, cmap='gray')
plt.axis('off')
plt.show()
plt.clf()
# Texture-based
Niter = [1, 1, 1]
mask, texture_img = get_texture(img, win_size=[3, 5, 7], Niter=Niter)
plt.imshow(texture_img, cmap='gray')
plt.show()
plt.clf()

mask = refine_mask(mask, kernel_size=7, num_iter=1, method='dilation')
mask = refine_mask(mask, kernel_size=9, num_iter=1, method='erosion')

foreground_img = cv2.bitwise_and(img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
background_img = cv2.bitwise_xor(img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
# plt.subplot(121)
# plt.imshow(foreground_img, cmap='gray')
# plt.axis('off')
# plt.subplot(122)
# plt.imshow(background_img, cmap='gray')
# plt.axis('off')
# plt.show()
# plt.clf()

contour_img = get_contour(mask, window_size=1)
plt.imshow(contour_img, cmap='gray')
plt.axis('off')
plt.show()
plt.clf()


# ========================================================================================
# img2
img = img2
plot_rgb(img)
iters = [2, 2, 1]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# b, g, r = cv2.split(img)
# img = cv2.merge((b,g,r))
# RGB_based
mask_b, mask_g, mask_r, final_mask = otsu_rgb(img, num_iter=iters)
final_mask = refine_mask(final_mask, kernel_size=5, num_iter=1, method='dilation')
final_mask = refine_mask(final_mask, kernel_size=3, num_iter=1, method='erosion')

foreground_img = cv2.bitwise_and(img, cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR))
background_img = cv2.bitwise_xor(img, cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR))
# plt.subplot(121)
# plt.imshow(foreground_img, cmap='gray')
# plt.axis('off')
# plt.subplot(122)
# plt.imshow(background_img, cmap='gray')
# plt.axis('off')
# plt.show()
# plt.clf()

contour_img = get_contour(final_mask, window_size=1)
plt.imshow(contour_img, cmap='gray')
plt.axis('off')
plt.show()
plt.clf()
# Texture-based
Niter = [1, 1, 1]
mask, texture_img = get_texture(img, win_size=[3, 5, 7], Niter=Niter)
plt.imshow(texture_img, cmap='gray')
plt.show()
plt.clf()

mask = refine_mask(mask, kernel_size=7, num_iter=1, method='dilation')
mask = refine_mask(mask, kernel_size=3, num_iter=1, method='erosion')

foreground_img = cv2.bitwise_and(img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
background_img = cv2.bitwise_xor(img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
# plt.subplot(121)
# plt.imshow(foreground_img, cmap='gray')
# plt.axis('off')
# plt.subplot(122)
# plt.imshow(background_img, cmap='gray')
# plt.axis('off')
# plt.show()
# plt.clf()

contour_img = get_contour(mask, window_size=1)
plt.imshow(contour_img, cmap='gray')
plt.axis('off')
plt.show()
plt.clf()


# ========================================================================================
# img3
img = img3
plot_rgb(img)
iters = [2, 2, 2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# b, g, r = cv2.split(img)
# img = cv2.merge((b,g,r))
# RGB_based
mask_b, mask_g, mask_r, final_mask = otsu_rgb(img, num_iter=iters)
final_mask = refine_mask(final_mask, kernel_size=3, num_iter=2, method='dilation')
final_mask = refine_mask(final_mask, kernel_size=3, num_iter=1, method='erosion')

foreground_img = cv2.bitwise_and(img, cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR))
background_img = cv2.bitwise_xor(img, cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR))
# plt.subplot(121)
# plt.imshow(foreground_img, cmap='gray')
# plt.axis('off')
# plt.subplot(122)
# plt.imshow(background_img, cmap='gray')
# plt.axis('off')
# plt.show()
# plt.clf()

contour_img = get_contour(final_mask, window_size=1)
plt.imshow(contour_img, cmap='gray')
plt.axis('off')
plt.show()
plt.clf()
# Texture-based
Niter = [1, 2, 2]
mask, texture_img = get_texture(img, win_size=[3, 5, 7], Niter=Niter)
plt.imshow(texture_img, cmap='gray')
plt.show()
plt.clf()

mask = refine_mask(mask, kernel_size=9, num_iter=1, method='dilation')
mask = refine_mask(mask, kernel_size=3, num_iter=2, method='erosion')

foreground_img = cv2.bitwise_and(img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
background_img = cv2.bitwise_xor(img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
# plt.subplot(121)
# plt.imshow(foreground_img, cmap='gray')
# plt.axis('off')
# plt.subplot(122)
# plt.imshow(background_img, cmap='gray')
# plt.axis('off')
# plt.show()
# plt.clf()

contour_img = get_contour(mask, window_size=1)
plt.imshow(contour_img, cmap='gray')
plt.axis('off')
plt.show()
plt.clf()