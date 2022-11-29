import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import math

point = ['P', 'Q', 'S', 'R']
# Image projection
img1 = io.imread('hw3_Task1_Images/Images/Img1.JPG')
PQSR1 = np.array([[642, 500], [643, 530], [667, 501], [666, 537]])
PQSR = np.array([[0, 0], [0, 85], [75, 0], [75, 85]])
# Plot
plt.imshow(img1)
plt.scatter(PQSR1[0][0], PQSR1[0][1])
plt.annotate(point[0] + '({},{})'.format(PQSR1[0][0], PQSR1[0][1]), (PQSR1[0][0]-50, PQSR1[0][1]-50), c='r')
plt.scatter(PQSR1[1][0], PQSR1[1][1])
plt.annotate(point[1] + '({},{})'.format(PQSR1[1][0], PQSR1[1][1]), (PQSR1[1][0]-100, PQSR1[1][1]+100), c='r')
plt.scatter(PQSR1[2][0], PQSR1[2][1])
plt.annotate(point[2] + '({},{})'.format(PQSR1[2][0], PQSR1[2][1]), (PQSR1[2][0]+100, PQSR1[2][1]-100), c='r')
plt.scatter(PQSR1[3][0], PQSR1[3][1])
plt.annotate(point[3] + '({},{})'.format(PQSR1[3][0], PQSR1[3][1]), (PQSR1[3][0]+50, PQSR1[3][1]+50), c='r')
plt.axis('off')
plt.show()
plt.clf()

# Compute known angles and length-ratio
P = np.array([642, 500, 1])
Q = np.array([643, 530, 1])
S = np.array([667, 501, 1])
R = np.array([666, 537, 1])
length_ratio = 75 / 85
alpha = math.acos(85 / np.sqrt(75**2 + 85**2))
beta = math.acos(75 / np.sqrt(75**2 + 85**2))
print('alpha =', alpha * 180 / math.pi)
print('beta =', beta * 180 / math.pi)
print('length_ratio =', length_ratio)
l_prime = np.cross(P, Q) / np.cross(P, Q)[2]
m_prime = np.cross(Q, S) / np.cross(Q, S)[2]
n_prime = np.cross(P, S) / np.cross(P, S)[2]


# Compute corresponding length-ratio in the distorted image
C = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
cos_alpha = np.dot(np.dot(l_prime, C), m_prime) / np.sqrt(np.dot(np.dot(l_prime, C), l_prime) * np.dot(np.dot(m_prime, C), m_prime))
alpha_prime = math.acos(cos_alpha)
print('alpha_prime =', alpha_prime * 180 / math.pi)
cos_beta = np.dot(np.dot(n_prime, C), m_prime) / np.sqrt(np.dot(np.dot(n_prime, C), n_prime) * np.dot(np.dot(m_prime, C), m_prime))
beta_prime = math.acos(cos_beta)
print('beta_prime = ', beta_prime * 180 / math.pi)
print('length-ratio computed from distorted image =', math.sin(alpha_prime)/math.sin(beta_prime))