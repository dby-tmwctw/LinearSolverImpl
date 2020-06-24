import math
import numpy as np
import numpy.linalg as lin
import numpy.fft as fft
import scipy.linalg as linalg
import scipy.signal as signal
from PIL import Image
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from helper import *

'''
Some constants
'''

l = 100
iters = 20
mu = 4
ld = 10

def r_calc(atb, z, v):
    return atb + (mu * z) - (v / mu)

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

def shrink2D(x, l):
    shape = x.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if (x[i,j] > 0):
                x[i,j] = max(abs(x[i,j]) - l, 0)
            else:
                x[i,j] = 0 - max(abs(x[i,j]) - l, 0)
    return x

def shrink2norm(x, l):
    shape = x.shape
    norm = lin.norm(x)
    constant = (1 - (l / norm))
    for i in range(shape[0]):
        for j in range(shape[1]):
            x[i, j] = constant * x[i, j]
    return x

'''
Below is the calculation of Lipschitz constant for the problem
'''

def step_size(A):
    # A should be an nxn matrix
    max_eigan = np.real(max(lin.eigh(A.T.dot(A))[0]))
    return 1 / (2 * max_eigan)

def step_size_fft(psf):
    Ieigs = fft.fft2(circshift(psf, psf.shape[0], psf.shape[1]))
    Ieigs2 = Ieigs ** 2
    step = 1 / (2 * np.amax(Ieigs2))
    return step

'''
Some Fast Fourier-Transform functions
'''

def fourier(A, x):
    return np.real(fft.ifft2(fft.fft2(A) * fft.fft2(x)))

def fourier_adjoint(A, x):
    return np.real(fft.ifft2(fft.fft2(A) * fft.fft2(x[::-1, ::-1])))

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

def ADMM(A, b):
    ata = A.T.dot(A)
    inv = lin.inv(ata + mu * np.identity(ata.shape[0]))
    z = np.zeros(len(b))
    x = np.zeros(len(b))
    v = np.zeros(len(b))
    for i in range(iters):
        z = shrink(x + v / mu, ld / mu)
        x = inv.dot(A.T.dot(b) + mu * z - v / mu)
        v = v + mu * (x - z)
    return x

def ista2(A, b):
    t = step_size_fft(A)
    x_est = np.zeros(A.shape)
    A = circshift(A, A.shape[0], A.shape[1])
    for i in range(iters):
        intermediate = fourier_adjoint(fourier(A, x_est) - b, A)
        x_est = shrink2D(x_est - 2 * t * intermediate, l * t)
    return np.real(x_est)

def fista2(A, b):
    s = step_size_fft(A)
    x_est = np.zeros(A.shape)
    y = x_est
    t = 1.0
    A = circshift(A, A.shape[0], A.shape[1])
    for i in range(iters):
        print(i)
        intermediate = y - 2 * s * (fourier_adjoint(fourier(A, y) - b, A) + 0.01 * y)
        x_new = shrink2D(intermediate, 20 * s)
        # x_new = shrink2norm(x_new, l * s)
        # Just testing on fourth root
        t_new = (1 + math.sqrt(math.sqrt(1 + 4 * t * t))) / 2
        y = x_new + ((t-1)/(t_new))*(x_new - x_est)
        t = t_new
        x_est = x_new
        if (i % 100 == 0):
            print(x_est)
    return np.real(x_est)

def ADMM2(A, b):
    # A_fft = fft.fft2(fft.ifftshift(A))
    # inv = 1. / (ata + mu)
    # atb = np.real(fft.fftshift(fft.ifft2(fft.ifftshift(b) * np.conj(A_fft))))
    A = circshift(A, A.shape[0], A.shape[1])
    A_fft = fft.fft2(A)
    ata = np.abs(np.conj(A_fft) * A_fft)
    inv = 1. / (ata + mu)
    atb = fourier_adjoint(b, A)
    z = np.zeros(b.shape)
    x = np.zeros(b.shape)
    v = np.zeros(b.shape)
    for i in range(iters):
        z = shrink2D(x + v / mu, ld / mu)
        # fftspace = inv * fft.fft2(fft.ifftshift(r_calc(atb, z, v)))
        # x = np.real(fft.fftshift(fft.ifft2(fftspace)))
        x = np.real(fft.ifft2(inv * fft.fft2(atb + mu * z - v / mu)))
        v = v + mu * (x - z)
    return np.real(x)

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
array = Image.open('./image/cameraman.tif')
array = np.array(array)
psf = gauss_map(256, 256, 3)
b = fourier(circshift(psf, 256, 256), array)
plot_figure(b, 'Blurred')
x_est = ADMM2(psf, b)
print(x_est)
print(x_est.shape)
plot_figure(x_est, 'Recovered')

# array = Image.open('./image/64x64.tif')
# array = np.array(array)
# psf = gauss_map(64, 64, 3)
# A = build_conv(psf, 64, 64)
# b = A.dot(vectorize(array, array.shape))
# plot_figure(devectorize(b, array.shape), 'Blurred')
# x_est = fista(A, b)
# print(x_est)
# print(x_est.shape)
# plot_figure(devectorize(x_est, array.shape), 'Recovered')


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
# x_est = ADMM(rand_A, rand_b)
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
