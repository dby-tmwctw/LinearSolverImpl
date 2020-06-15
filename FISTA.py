import math
import numpy as np
import numpy.linalg as lin
import scipy.linalg as linalg
from PIL import Image
import matplotlib.pyplot as plt
import scipy.sparse as sparse

l = 0.0001
iters = 10000

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

def step_size(A):
    # A should be an nxn matrix
    max_eigan = np.real(max(lin.eigh(A.T.dot(A))[0]))
    return 1 / (2 * max_eigan)

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

def diff(x, x_est):
    dim = len(x)
    total_diff = 0
    for i in range(dim):
        total_diff += abs(x[i] - x_est[i])
    return total_diff

def gauss_map(size_x, size_y=None, sigma_x=5, sigma_y=None):
    if size_y == None:
        size_y = size_x
    if sigma_y == None:
        sigma_y = sigma_x

    assert isinstance(size_x, int)
    assert isinstance(size_y, int)

    x0 = size_x // 2
    y0 = size_y // 2

    x = np.arange(0, size_x, dtype=float)
    y = np.arange(0, size_y, dtype=float)[:,np.newaxis]

    x -= x0
    y -= y0

    exp_part = x**2/(2*sigma_x**2)+ y**2/(2*sigma_y**2)
    return 1/(2*np.pi*sigma_x*sigma_y) * np.exp(-exp_part)

def vectorize(image, shape):
    arr = np.array(image)
    flat_arr = arr.ravel()
    vector = np.array(flat_arr)
    return vector

def devectorize(vector, shape):
    arr2 = np.asarray(vector).reshape(shape)
    return arr2

def convmtx(h,n):
    return linalg.toeplitz(np.hstack([h, np.zeros(n-1)]), np.hstack([h[0], np.zeros(n-1)]))

def plot_figure(image, name):
    fig1 = plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title(name)
    plt.show()

def build_conv(psf, n):
    conv_matrix = np.zeros(n*n, n*n)
    center = n / 2
    for i in range(0, n*n):
        curr_row = i / n
        curr_col = i % n
        row_start = max(0, center - curr_row)
        col_start = max(0, center - curr_col)
        row_end = min()

# n = 256
# psf = gauss_map(n, n, 3)
# convmat = convmtx(psf, 256)
# rand_x = Image.open('cameraman.tif')
# rand_x = np.array(rand_V, dtype='float32')
# shape = rand_x.shape
# vec_x = vectorize(rand_x, shape)
# vec_b = convmat.dot(vec_x)
# rand_b = devectorize(vec_b, shape)
# plot_figure(rand_b, 'Blurred')


rand_x = np.random.rand(81)
rand_x = rand_x * 10
print(rand_x)
rand_A = np.random.rand(81, 81)
# rand_A = sparse.rand(250, 200, density=0.01).A
rand_b = rand_A.dot(rand_x)
# noise = np.random.normal(scale=0.01, size=rand_b.shape)
noise = np.zeros(rand_b.shape)
noise = noise + 0.1
rand_b = rand_b
x_est = fista(rand_A, rand_b)
# x_est = ista(rand_A, rand_b)
print(x_est)
print(diff(rand_x, x_est))
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

# import numpy as np
# from PIL import Image
#
# img = Image.open('orig.png').convert('RGBA')
# arr = np.array(img)
#
# # record the original shape
# shape = arr.shape
#
# # make a 1-dimensional view of arr
# flat_arr = arr.ravel()
#
# # convert it to a matrix
# vector = np.matrix(flat_arr)
#
# # do something to the vector
# vector[:,::10] = 128
#
# # reform a numpy array of the original shape
# arr2 = np.asarray(vector).reshape(shape)
#
# # make a PIL image
# img2 = Image.fromarray(arr2, 'RGBA')
# img2.show()

'''
Make the psf the full blurring matrix, and then test image deblurring on this
'''
