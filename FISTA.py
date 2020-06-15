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
    # 2-D image matrix to vector
    # Note that this one stacks rows instead of column
    arr = np.array(image)
    flat_arr = arr.ravel()
    vector = np.array(flat_arr)
    return vector

def devectorize(vector, shape):
    # Vector to 2-D image
    # Note that the vector should stack rows instead of column
    arr2 = np.asarray(vector).reshape(shape)
    return arr2

def plot_figure(image, name):
    # Plot the given figure with given name
    fig1 = plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title(name)
    plt.show()

def circshift(array, m, n):
    # Does circular shift to array
    # Array is an mxn array
    result = np.zeros((m, n))
    for i1 in range(m/2):
        for j1 in range(n/2):
            result[i1+(m-m/2), j1+(n-n/2)] = array[i1, j1]
    for i4 in range(m/2, m):
        for j4 in range(n/2, n):
            result[i4-m/2, j4-n/2] = array[i4, j4]
    for i2 in range(m/2):
        for j2 in range(n/2, n):
            result[i2+(m-m/2), j2-n/2] = array[i2, j2]
    for i3 in range(m/2, m):
        for j3 in range(n/2):
            result[i3-m/2, j3+(n-n/2)] = array[i3, j3]
    return result

def build_conv(psf, n, m):
    # The shape of image is n*n, and the shape of psf is m*m
    conv_matrix = np.zeros((n*n, n*n))
    psf = np.rot90(psf, 2)
    center = m / 2
    for i in range(0, n*n):
        curr_row = i / n
        curr_col = i % n
        row_offset_start = 0 - min(curr_row, m / 2)
        col_offset_start = 0 - min(curr_col, m / 2)
        row_offset_end = min(n - 1 - curr_row, m / 2)
        col_offset_end = min(n - 1 - curr_col, m / 2)
        for j in range(row_offset_start, row_offset_end + 1):
            for k in range(col_offset_start, col_offset_end + 1):
                col_index = (curr_row + j) * n + (curr_col + k)
                psf_row = center + j
                psf_col = center + k
                conv_matrix[i, col_index] = psf[psf_row, psf_col]
    return conv_matrix

array = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
array = np.array(array)
vec = vectorize(array, array.shape)
print(array)
print(vec)
print(devectorize(vec, array.shape))
print(build_conv(array, 3, 3))


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

'''
Make the psf the full blurring matrix, and then test image deblurring on this
'''
