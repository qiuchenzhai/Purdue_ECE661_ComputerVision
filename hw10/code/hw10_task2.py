import numpy as np
import cv2
import matplotlib.pyplot as plt
from pylab import *
from math import *


path = 'Task2_Images/'
left_img = cv2.imread(path + 'Left.ppm')
right_img = cv2.imread(path + 'Right.ppm')
Dgt = cv2.imread(path + 'left_truedisp.pgm') # groundtruth_disparity map
mask = cv2.imread(path + 'mask0nocc.png')
# preprocee
left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
Dgt = cv2.cvtColor(Dgt, cv2.COLOR_BGR2GRAY)
Dgt = (Dgt.astype(np.float32) / 16.0).astype(np.int16)
dmax = np.amax(Dgt)

# Estimate Disparity map D
M = 3           # 7, 9, 5
k = int(M / 2)
D = np.zeros(left_img.shape)
for i in range(k, D.shape[0]-k-1):          # row
    for j in range(k, D.shape[1]-k-1):
        left_neigh = [left_img[m, n] for m in range(i-k, i+k+1) for n in range(j-k, j+k+1)]
        left_bit = left_neigh > left_neigh[int(M**2/2)]
        cost = []
        for idx in range(np.maximum(k, j-dmax), j+1):
            right_neigh = [right_img[m, n] for m in range(i - k, i + k + 1) for n in range(idx - k, idx + k + 1)]
            right_bit = right_neigh > right_neigh[int(M ** 2 / 2)]
            cost.append(np.sum(np.bitwise_xor(left_bit * 1, right_bit * 1)))
        cost = np.asarray(cost)
        d = j - np.maximum(k, j-dmax) - np.argmin(cost)
        D[i, j] = d
figure(num=None, figsize=(10, 1.5), dpi=160, facecolor='w', edgecolor='k')
plt.subplot(131)
plt.imshow(D, cmap='gray')
plt.axis('off')
plt.title('Estimated Disparsity map (M = {})'.format(M))

# Compute the percentage accuracy
delta1, delta2 = 1, 2
N = np.count_nonzero(mask)
diff = np.abs(Dgt - D)
diff[mask == 0] = 1e6
accuracy_mat1 = (diff < delta1) * 1
accuracy1 = np.sum(accuracy_mat1) / N
plt.subplot(132)
plt.imshow(accuracy_mat1*255, cmap='gray')
plt.axis('off')
plt.title(r'Disparity error ($\delta$ = {})'.format(delta1))

accuracy_mat2 = (diff < delta2) * 1
accuracy2 = np.sum(accuracy_mat2) / N
plt.subplot(133)
plt.imshow(accuracy_mat2*255, cmap='gray')
plt.axis('off')
plt.title(r'Disparity error ($\delta$ = {})'.format(delta2))
plt.show()
plt.savefig('DM_{}.png'.format(M))
plt.clf()
print('The number of valid pixels is {}'.format(N))
print('accuracy percentage (\delta = 1) is {}'.format(accuracy1))
print('accuracy percentage (\delta = 2) is {}'.format(accuracy2))



