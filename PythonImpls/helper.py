import math
import numpy as np
import numpy.linalg as lin
import scipy.linalg as linalg
from PIL import Image
import matplotlib.pyplot as plt
import scipy.sparse as sparse

def circshift(array, m, n):
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

def resize(img, factor):
    num = int(-np.log2(factor))
    for i in range(num):
        img = 0.25*(img[::2,::2,...]+img[1::2,::2,...]+img[::2,1::2,...]+img[1::2,1::2,...])
    return img

def show_result(psf, rand_B, rand_V):
    fig2 = plt.figure()
    plt.imshow(rand_B, cmap='gray')
    plt.title('Blurred Image')
    plt.show()
    sensor_size = np.array(psf.shape)
    full_size = 2*sensor_size
    calculated_V = ADMM_linear(psf, rand_B)
    np.savetxt('V_EST.txt', calculated_V, fmt='%f')
    fig3 = plt.figure()
    plt.imshow(rand_V, cmap='gray')
    plt.title('Original')
    fig4 = plt.figure()
    plt.imshow(calculated_V, cmap='gray')
    plt.title('Processed')
    plt.show()

def load_psf(n, m, psfName):
    # psf = np.random.rand(m, n)
    # psf = psf / 10
    # psf = Image.open(psfName)
    # psf = np.array(psf, dtype='float32')
    # psf = psf[472:728, 672:928]
    # psf = resize(psf, 0.25)
    psf = gauss_map(n, m, 3)
    fig1 = plt.figure()
    plt.imshow(psf, cmap='gray')
    plt.title('PSF')
    plt.show()
    return psf

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
        row_offset_end = min(n - 1 - curr_row, (m / 2) - 1)
        col_offset_end = min(n - 1 - curr_col, (m / 2) - 1)
        for j in range(row_offset_start, row_offset_end + 1):
            for k in range(col_offset_start, col_offset_end + 1):
                col_index = (curr_row + j) * n + (curr_col + k)
                psf_row = center + j
                psf_col = center + k
                conv_matrix[i, col_index] = psf[psf_row, psf_col]
    return conv_matrix

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
