import math
import numpy as np
import numpy.linalg as lin
import scipy.linalg as linalg
from PIL import Image
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from helper import *

'''
Some constants
'''

l = 0.0001
iters = 10000

'''
Below is the proximal operator of the problems
'''

def shrink(x, l):
    # x should be a 1xn row vector
    # Is lambda just a number?
    dim = len(x)
    for i in range(dim):
        if (x[i] > 0):
            x[i] = max(abs(x[i]) - l, 0)
            assert(x[i] >= 0)
        else:
            x[i] = 0 - max(abs(x[i]) - l, 0)
            assert(x[i] <= 0)
    return x

'''
Below is the calculation of Lipschitz constant for the problem
'''

def step_size(A):
    # A should be an nxn matrix
    max_eigan = np.real(max(lin.eigh(A.T.dot(A))[0]))
    return 1 / (2 * max_eigan)

'''
The main ISTA/FISTA routine
'''

def fista(A, b):
    s = step_size(A)
    x_est = np.zeros(A.shape[1])
    y = x_est
    t = 1.0
    for i in range(iters):
        shrink_val = y - 2 * s * A.T.dot((A.dot(y)  - b))
        x_new = shrink(shrink_val, l * s)
        t_new = (1 + math.sqrt(1 + 4 * t * t)) / 2
        y = x_new + ((t-1)/(t_new))*(x_new - x_est)
        t = t_new
        x_est = x_new
    return x_est

def ista(A, b):
    t = step_size(A)
    x_est = np.zeros(A.shape[1])
    for i in range(iters):
        x_est = shrink(x_est - 2 * t * A.T.dot(A.dot(x_est) - b), l * s)
    return x_est

'''
Difference measurements
'''

def diff(x, x_est):
    dim = len(x)
    total_diff = 0
    for i in range(dim):
        total_diff += abs(x[i] - x_est[i])
    return total_diff

'''
Actual Testing
'''

# Blurring operator testing
array = Image.open('./image/64x64.tif')
array = np.array(array)
vec = vectorize(array, array.shape)
psf = gauss_map(64, 64, 3)
conv_mat = build_conv(psf, 64, 64)
b = conv_mat.dot(vec)
b = devectorize(b, array.shape)
plot_figure(b, 'Blurred')


# rand_x = np.random.rand(81)
# rand_x = rand_x * 10
# print(rand_x)
# rand_A = np.random.rand(81, 81)
# # rand_A = sparse.rand(250, 200, density=0.01).A
# rand_b = rand_A.dot(rand_x)
# # noise = np.random.normal(scale=0.01, size=rand_b.shape)
# noise = np.zeros(rand_b.shape)
# noise = noise + 0.1
# rand_b = rand_b
# x_est = fista(rand_A, rand_b)
# # x_est = ista(rand_A, rand_b)
# print(x_est)
# print(diff(rand_x, x_est))
# avg_diff = 0
# for j in range(25):
#     rand_x = np.random.rand(200)
#     rand_x = rand_x * 10
#     rand_A = np.random.rand(250, 200)
#     rand_b = rand_A.dot(rand_x)
#     x_est = fista(rand_A, rand_b)
#     # x_est = ista(rand_A, rand_b)
#     # print(diff(rand_x, x_est))
#     avg_diff += diff(rand_x, x_est)
# print(avg_diff / 100)
