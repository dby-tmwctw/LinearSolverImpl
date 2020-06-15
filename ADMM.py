import numpy as np
import numpy.fft as fft
import scipy.sparse as sparse
from PIL import Image
import matplotlib.pyplot as plt

f = 0.25

mu1 = 1e-6
mu2 = 1e-5
mu3 = 4e-5
tau = 0.0001

iters = 200

'''
Code block below are for helper functions
'''

def preprocess(psf, data, V):
    show_im = True
    psf = np.array(psf, dtype='float32')
    data = np.array(data, dtype='float32')
    V = np.array(V, dtype='float32')

    def resize(img, factor):
        num = int(-np.log2(factor))
        for i in range(num):
            img = 0.25*(img[::2,::2,...]+img[1::2,::2,...]+img[::2,1::2,...]+img[1::2,1::2,...])
        return img


    # psf = resize(psf, f)
    # data = resize(data, f)
    # V = resize(V, f)

    """Now we normalize the images so they have the same total power. Technically not a
    necessary step, but the optimal hyperparameters are a function of the total power in
    the PSF (among other things), so it makes sense to standardize it"""

    psf /= np.linalg.norm(psf.ravel())
    data /= np.linalg.norm(data.ravel())
    V /= np.linalg.norm(V.ravel())

    return psf, data, V

def C(M, full_size, sensor_size):
    # Crops from full_size -> sensor_size
    top = (full_size[0] - sensor_size[0]) // 2
    bottom = (full_size[0] + sensor_size[0]) // 2
    left = (full_size[1] - sensor_size[1]) // 2
    right = (full_size[1] + sensor_size[1]) // 2
    result = M[top:bottom,left:right]
    return result

def CT(b, full_size, sensor_size):
    # Zero-pads from sensor_size -> full_size
    v_pad = (full_size[0] - sensor_size[0]) // 2
    h_pad = (full_size[1] - sensor_size[1]) // 2
    return np.pad(b, ((v_pad, v_pad), (h_pad, h_pad)), 'constant', constant_values=(0,0))

def conv_operator(vk, H_fft):
    # Convolute H with vk : h * vk == F-1(F(h) . F(v))
    vk_zeroed = fft.ifftshift(vk)
    return np.real(fft.fftshift(fft.ifft2(fft.fft2(vk_zeroed) * H_fft)))

def conv_operator_adjoint(x, H_fft):
    # Adjoint (transpose?) operator for convolution: hT *T pad(x) == F-1(F(h)T . F(pad(x)))
    x_zeroed = fft.ifftshift(x)
    return np.real(fft.fftshift(fft.ifft2(fft.fft2(x_zeroed) * np.conj(H_fft))))

def psi(v):
    # psi(v)_i,j = [v_i+1,j - v_i,j, v_i,j+1 - v_i,j]
    # Gradient of the matrix estimate. Approximated by finite-difference,
    # which is specifically 2D forward-difference with a circular boundary
    # condition. np.roll shifts the image circularly, hence can be used to
    # compute these differences.
    row_wise = np.roll(v, 1, axis = 0) - v
    column_wise = np.roll(v, 1, axis = 1) - v
    # Note that we are stacking two images on top of each other
    return np.stack((row_wise, column_wise), axis = 2)

def psi_adjoint(U):
    # psiT(u)_i,j = (u^x_i-1,j - u^x_i,j) + (u^y_i,j-1 - u^y_i,j)
    # Basically the inverse of above
    diff1 = np.roll(U[...,0], -1, axis = 0) - U[...,0]
    diff2 = np.roll(U[...,1], -1, axis = 1) - U[...,1]
    # Note that now we merge two images into one
    return diff1 + diff2

def compute_fft(H, full_size, sensor_size):
    return fft.fft2(fft.ifftshift(CT(H, full_size, sensor_size)))

def compute_X_divmat(full_size, sensor_size):
    return 1. / (CT(np.ones(sensor_size), full_size, sensor_size) + mu1)

def compute_psiTpsi(full_size):
    # V's operator - forward part
    psiTpsi = np.zeros(full_size)
    psiTpsi[0, 0] = 4
    psiTpsi[0, 1] = psiTpsi[1, 0] = psiTpsi[0, -1] = psiTpsi[-1, 0] = -1
    psiTpsi = fft.fft2(psiTpsi)
    return psiTpsi

def compute_V_divmat(H_fft, psiTpsi):
    # V's operator - Inverse of psiTpsi
    MTM = mu1 * (np.abs(np.conj(H_fft) * H_fft))
    psiTpsi_component = mu2 * np.abs(psiTpsi)
    id = mu3
    return 1. / (MTM + psiTpsi_component + id)

def soft_thresh(x, tau):
    # This soft threshold zeros out all 0<x<tau/mu2, and all other compoment of
    # the vector are decreased in magnitude by tau/mu2
    return np.sign(x) * np.maximum(0, np.abs(x) - tau)

def r_calc(W, rho, U, eta, X, xi, H_fft):
    return (mu3 * W - rho) + psi_adjoint(mu2 * U - eta) + conv_operator_adjoint(mu1 * X - xi, H_fft)

'''
Code below are the main routine
'''

def init_matrices(H_fft, full_size):
    X = np.zeros(full_size)
    U = np.zeros((full_size[0], full_size[1], 2))
    V = np.zeros(full_size)
    W = np.zeros(full_size)
    xi = np.zeros_like(conv_operator(V, H_fft))
    eta = np.zeros_like(psi(V))
    rho = np.zeros_like(W)
    return X, U, V, W, xi, eta, rho

def ADMM_linear(H, b):
    # In this algorithm, we do extension and croppings to perform
    # linear convolution
    H = np.array(H, dtype = 'float32')
    b = np.array(b, dtype = 'float32')
    sensor_size = np.array(H.shape)
    full_size = 2 * sensor_size
    # Convert H to fft form
    H_fft = compute_fft(H, full_size, sensor_size)
    #Initialize all the variables
    X, U, V, W, xi, eta, rho = init_matrices(H_fft, full_size)
    # Division matrix is the inverse of C^HC+u_1I. This is for x update
    X_divmat = compute_X_divmat(full_size, sensor_size)
    # V update's operator part. First, we calculate the operator, then we
    # get its inverse
    psiTpsi = compute_psiTpsi(full_size)
    V_divmat = compute_V_divmat(H_fft, psiTpsi)
    for i in range(iters):
        # X, U, V, W, xi, eta, rho = ADMM_iter(X, U, V, W, xi, eta, rho, [H_fft, b, X_divmat, V_divmat])
        U = soft_thresh(psi(V) + eta/mu2, tau/mu2)
        X = X_divmat * (xi + mu1 * conv_operator(V, H_fft) + CT(b, full_size, sensor_size))
        # Precursor to simplify calculation
        fftspace = V_divmat * fft.fft2(fft.ifftshift(r_calc(W, rho, U, eta, X, xi, H_fft)))
        V = np.real(fft.fftshift(fft.ifft2(fftspace)))
        W = np.maximum(rho / mu3 + V, 0)
        xi = xi + mu1 * (conv_operator(V, H_fft) - X)
        eta = eta + mu2 * (psi(V) - U)
        rho = rho + mu3 * (V - W)
        # print("This is matrix after " + str(i) + " iteration:")
        # print(C(V, full_size, sensor_size))
    return C(V, full_size, sensor_size)

'''
Below are the actual test for the program
'''

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

# Define m and n
m = 256
n = 256
# Generate psf
psf = load_psf(n, m, '/psf/psf_gaussian_256.tif')
shifted = circshift(psf, m, n)
S = fft.fft2(shifted)
S_original = fft.fft2(psf)
# Load image
rand_V = Image.open('./image/cameraman.tif')
rand_V = np.array(rand_V, dtype='float32')
# Calculate noise image
# rand_B = np.real(fft.ifft2(S * fft.fft2(rand_V)))
rand_B = rand_V
fig1 = plt.figure()
plt.imshow(rand_B, cmap='gray')
plt.title('With Gaussian')
plt.show()
noise = np.random.normal(scale=10, size=(m, n))
fig2 = plt.figure()
plt.imshow(noise, cmap='gray')
plt.title('Gaussian')
plt.show()
print(rand_B)
print(noise)
rand_B = rand_B + noise
fig3 = plt.figure()
plt.imshow(rand_B, cmap='gray')
plt.title('With noise')
plt.show()
# Preprocess
psf, rand_B, rand_V = preprocess(psf, rand_B, rand_V)
# Show image
show_result(psf, rand_B, rand_V)

'''
Need to test if we replace the psf with a function
'''
