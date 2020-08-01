#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <png.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <chrono>
#include <iomanip>

// Convention: Matrix dimension on the front, then input , output on the back
// Convention: Scalar arguments always at the very end
// Convention: All arrays are of size mxn

// When psf is created, it is normalized so a png image can be constructed. Hence this is to adjust back the psf
const int adjust_coefficient = 3;
// Number of iterations
const int num_iters = 12;
// Some constants for the algorithm
const float mu1 = 1e-6;
const float mu2 = 1e-5;
const float mu3 = 4e-5;
const float tau = 0.0001;

// Complex addition
static __device__ __host__ inline float2 ComplexAdd(float2 a, float2 b) {
    float2 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

// Complex addition
static __device__ __host__ inline float2 ComplexSub(float2 a, float2 b) {
    float2 c;
    c.x = a.x - b.x;
    c.y = a.y - b.y;
    return c;
}

// Complex scale
static __device__ __host__ inline float2 ComplexScale(float2 a, float s) {
    float2 c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}
  
// Complex multiplication
static __device__ __host__ inline float2 ComplexMul(float2 a, float2 b) {
    float2 c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

// Complex maximum
static __device__ __host__ inline float2 ComplexSquareMax(float2 a, float2 b) {
    float2 num1;
    float2 num2;
    num1 = ComplexMul(a, a);
    num2 = ComplexMul(b, b);
    if (num1.x < num2.x)
    {
        return b;
    } else
    {
        return a;
    }
}

// Complex square root
static __device__ __host__ inline float ComplexAbs(float2 a) {
    float c;
    c = a.x * a.x + a.y * a.y;
    return sqrtf(c);
}

// Complex conjugate of a number
static __device__ __host__ inline float2 ComplexConj(float2 a) {
    float2 c;
    c.x = a.x;
    c.y = a.y * (-1.0f);
    return c;
}

// Complex conjugate of a number
static __device__ __host__ inline float thresholding(float a, float tau) {
    if (a > 0)
    {
        return fmaxf(a - tau, 0);
    } else
    {
        return 0.0f - fmaxf(0.0f - a - tau, 0);
    }
}

// Complex conjugate of a number
static __device__ __host__ inline float2 ComplexThres(float2 a, float tau) {
    float2 c;
    c.x = thresholding(a.x, tau);
    c.y = thresholding(a.y, tau);
    return c;
}

__global__
void copy_array(int m, int n, float* x, float* y)
{
    // y = x
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < m; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < n; j += (gridDim.y * blockDim.y))
        {
            y[i*n+j] = x[i*n+j];
        }
    }
}

__global__
void copy_array_complex(int m, int n, float2 *x, float2 *y)
{
    // y = x
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < m; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < n; j += (gridDim.y * blockDim.y))
        {
            y[i*n+j].x = x[i*n+j].x;
            y[i*n+j].y = x[i*n+j].y;
        }
    }
}

__global__
void copy_array_padded(int m, int n, float* x, float* y)
{
    // y is padded, while x is the original input
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < m; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < n; j += (gridDim.y * blockDim.y))
        {
            int index_x = i + (m / 2);
            y[(i+(m/2))*2*n+(j+(n/2))] = x[i*n+j];
        }
    }
}

__global__
void multiply_kernel(int m, int n, float2* x, float2* y, float scale)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < m; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < n; j += (gridDim.y * blockDim.y))
        {
            x[i*n+j] = ComplexScale(ComplexMul(x[i*n+j], y[i*n+j]), scale);
        }
    }
}

__global__
void matrix_minus(int m, int n, float2* x, float2* y)
{
    // x = x - y
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < m; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < n; j += (gridDim.y * blockDim.y))
        {
            x[i*n+j] = ComplexSub(x[i*n+j], y[i*n+j]);
        }
    }
}

__global__
void matrix_scale(int m, int n, float2 *x, float k)
{
    // x = kx
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < m; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < n; j += (gridDim.y * blockDim.y))
        {
            x[i*n+j] = ComplexScale(x[i*n+j], k);
        }
    }
}

__global__
void matrix_pad(int m, int n, float2 *image, float2 *signal)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < m; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < n; j += (gridDim.y * blockDim.y))
        {
            signal[(i+(m/2))*2*n+(j+(n/2))].x = image[i*n+j].x;
            signal[(i+(m/2))*2*n+(j+(n/2))].y = image[i*n+j].y;
        }
    }
}

__global__
void circshift(int m, int n, float2 *x, float2 *y)
{
    // Put circshifted x into y
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < m; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < n; j += (gridDim.y * blockDim.y))
        {
            if (i < (m/2))
            {
                if (j < (n/2))
                {
                    y[(i+(m-m/2))*n + (j+(n-n/2))] = x[i*n+j];
                } else
                {
                    y[(i+(m-m/2))*n + (j-(n/2))] = x[i*n+j];
                }
            } else
            {
                if (j < (n/2))
                {
                    y[(i-(m/2))*n + (j+(n-n/2))] = x[i*n+j];
                } else
                {
                    y[(i-(m/2))*n + (j-(n/2))] = x[i*n+j];
                }
            }
        }
    }
}

__global__
void circshift_reverse(int m, int n, float2 *x, float2 *y)
{
    // Put reverse circshifted x into y
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < m; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < n; j += (gridDim.y * blockDim.y))
        {
            if (i < (m/2))
            {
                if (j < (n/2))
                {
                    y[i*n+j] = x[(i+(m-m/2))*n + (j+(n-n/2))];
                } else
                {
                    y[i*n+j] = x[(i+(m-m/2))*n + (j-(n/2))];
                }
            } else
            {
                if (j < (n/2))
                {
                    y[i*n+j] = x[(i-(m/2))*n + (j+(n-n/2))];
                } else
                {
                    y[i*n+j] = x[(i-(m/2))*n + (j-(n/2))];
                }
            }
        }
    }
}

__global__
void circshift_reverse_and_real(int m, int n, float2 *x, float2 *y)
{
    // Put reverse circshifted x into y
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < m; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < n; j += (gridDim.y * blockDim.y))
        {
            if (i < (m/2))
            {
                if (j < (n/2))
                {
                    y[i*n+j].x = x[(i+(m-m/2))*n + (j+(n-n/2))].x;
                    y[i*n+j].y = 0;
                } else
                {
                    y[i*n+j].x = x[(i+(m-m/2))*n + (j-(n/2))].x;
                    y[i*n+j].y = 0;
                }
            } else
            {
                if (j < (n/2))
                {
                    y[i*n+j].x = x[(i-(m/2))*n + (j+(n-n/2))].x;
                    y[i*n+j].y = 0;
                } else
                {
                    y[i*n+j].x = x[(i-(m/2))*n + (j-(n/2))].x;
                    y[i*n+j].y = 0;
                }
            }
        }
    }
}

__global__
void reverse(int m, int n, float2* x, float2* y)
{
    // x = y reversed
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < m; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < n; j += (gridDim.y * blockDim.y))
        {
            // x[i*n+j] = y[(m*n-1)-(i*n+j)];
            x[i*n+j] = ComplexConj(y[i*n+j]);
        }
    }
}

__global__
void normalize(int m, int n, float2* x, float factor)
{
    // x = factor * x
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < m; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < n; j += (gridDim.y * blockDim.y))
        {
            // x[i*n+j] = y[(m*n-1)-(i*n+j)];
            x[i*n+j] = ComplexScale(x[i*n+j], factor);
        }
    }
}

__global__
void clear_padding_and_restore(int m, int n, float2 *x)
{
    // x = y reversed
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < m; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < n; j += (gridDim.y * blockDim.y))
        {
            if ((i < m / 4) || (i >= ((m + 1) * 3 / 4)) || (j < n / 4) || (j >= ((n + 1) * 3 / 4)))
            {
                x[i*n+j].x = 0.0f;
            } else
            {
                x[i*n+j].x = 4.0f * x[i*n+j].x;
            }
            x[i*n+j].y = 0.0f;
        }
    }
}

__global__
void compute_X_divmat(int m, int n, float *x)
{
    // x = y reversed
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < m; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < n; j += (gridDim.y * blockDim.y))
        {
            float denominator = mu1;
            if ((i >= m / 4) && (i < ((m + 1) * 3 / 4)) && (j >= n / 4) && (j < ((n + 1) * 3 / 4)))
            {
                denominator += 1.0f;
            }
            x[i*n+j] = 1.0f / denominator;
        }
    }
}

__global__
void compute_V_divmat(int m, int n, float2 *filter_shifted, float2 *psiTpsi, float2 *V_divmat)
{
    // x = y reversed
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < m; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < n; j += (gridDim.y * blockDim.y))
        {
            int index = i * n + j;
            float MTM = mu1 * (ComplexAbs(ComplexMul(ComplexConj(filter_shifted[index]), filter_shifted[index])));
            float pTp = mu2 * ComplexAbs(psiTpsi[index]);
            V_divmat[index].x = 1.0f / (MTM + pTp + mu3);
            V_divmat[index].y = 0.0f;
        }
    }
}

__global__
void soft_thresh_specialized(int m, int n, float2 *x)
{
    // Here, we are soft-thresholding x.
    // X is actually an mxnx2 matrices. Each entry is two floating-point numbers
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < m; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < n; j += (gridDim.y * blockDim.y))
        {
            int index = i * n + j;
            x[index].x = thresholding(x[index].x, tau);
            x[index].y = thresholding(x[index].y, tau);
        }
    }
}

__global__
void U_update(int m, int n, float2 *psi_V, float2 *eta, float2 *U)
{
    // Here, we are soft-thresholding x.
    // X is actually an mxnx2 matrices. Each entry is two floating-point numbers
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < m; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < n; j += (gridDim.y * blockDim.y))
        {
            int index = i * n + j;
            U[index].x = psi_V[index].x + (eta[index].x / mu2);
            U[index].y = psi_V[index].y + (eta[index].y / mu2);
            U[index] = ComplexThres(U[index], tau / mu2);
        }
    }
}

__global__
void X_update(int m, int n, float *X_divmat, float2 *xi, float2 *V, float2 *b, float2 *X)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < m; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < n; j += (gridDim.y * blockDim.y))
        {
            int index = i * n + j;
            X[index].x = X_divmat[index] * (xi[index].x + mu1 * V[index].x + b[index].x);
            X[index].y = 0.0f;
        }
    }
}

void V_update(cufftHandle plan, int m, int n, float2 *V_divmat, float2 *r_calc, float2 *draft1, float2 *draft2, float2 *V, dim3 numBlocks, dim3 threads)
{
    circshift<<<numBlocks, threads>>>(m, n, r_calc, draft1);
    cudaDeviceSynchronize();
    cufftExecC2C(plan, draft1, draft1, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    // Possible Error: float too small
    multiply_kernel<<<numBlocks, threads>>>(m, n, draft1, V_divmat, 1.0f);
    cudaDeviceSynchronize();
    cufftExecC2C(plan, draft1, draft1, CUFFT_INVERSE);
    cudaDeviceSynchronize();
    normalize<<<numBlocks, threads>>>(m, n, draft1, (1.0f / (m * n)));
    cudaDeviceSynchronize();
    circshift_reverse_and_real<<<numBlocks, threads>>>(m, n, draft1, V);
    cudaDeviceSynchronize();
}

__global__
void W_update(int m, int n, float2 *rho, float2 *V, float2 *W)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < m; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < n; j += (gridDim.y * blockDim.y))
        {
            int index = i * n + j;
            float curr = rho[index].x / mu3 + V[index].x;
            W[index].x = fmaxf(curr, 0.0f);
            W[index].y = 0.0f;
        }
    }
}

__global__
void Xi_update(int m, int n, float2 *draft1, float2 *X, float2 *xi)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < m; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < n; j += (gridDim.y * blockDim.y))
        {
            int index = i * n + j;
            // draft1[index] = ComplexSub(draft1[index], X[index]);
            // draft1[index] = ComplexScale(draft1[index], mu1);
            draft1[index].x = mu1 * (draft1[index].x - X[index].x);
            xi[index].x = xi[index].x + draft1[index].x;
            xi[index].y = 0;
        }
    }
}

__global__
void Eta_update(int m, int n, float2 *psi_V, float2 *U, float2 *eta)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < m; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < n; j += (gridDim.y * blockDim.y))
        {
            int index = i * n + j;
            eta[index].x = eta[index].x + mu2 * (psi_V[index].x - U[index].x);
            eta[index].y = eta[index].y + mu2 * (psi_V[index].y - U[index].y);
        }
    }
}

__global__
void Rho_update(int m, int n, float2 *V, float2 *W, float2 *rho)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < m; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < n; j += (gridDim.y * blockDim.y))
        {
            int index = i * n + j;
            rho[index].x = rho[index].x + mu3 * (V[index].x - W[index].x);
            rho[index].y = 0.0f;
        }
    }
}

__global__
void r_calculation(int m, int n, float2 *r_calc, float2 *draft1, float2 *draft2)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < m; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < n; j += (gridDim.y * blockDim.y))
        {
            int index = i * n + j;
            r_calc[index] = ComplexAdd(r_calc[index], ComplexAdd(draft1[index], draft2[index]));
        }
    }
}

__global__
void calc_psi_V(int m, int n, float2 *V, float2 *psi_V)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < m; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < n; j += (gridDim.y * blockDim.y))
        {
            float rowNext = i == 0 ? V[(m-1)*n+j].x : V[(i-1)*n+j].x;
            float colNext = j == 0 ? V[i*n+(n-1)].x : V[i*n+(j-1)].x;
            float curr = V[i*n+j].x;
            psi_V[i*n+j].x = rowNext - curr;
            psi_V[i*n+j].y = colNext - curr;
        }
    }
}

__global__
void psi_adjoint(int m, int n, float2 *draft2, float2 *draft1)
{
    // Store psi-adjoint-translated draft2 into draft1
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < m; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < n; j += (gridDim.y * blockDim.y))
        {
            int index = i * n + j;
            float rowNext = i == (m-1) ? draft2[j].x : draft2[(i+1)*n+j].x;
            float colNext = j == (n-1) ? draft2[i*n].y : draft2[i*n+(j+1)].y;
            draft1[index].x = (rowNext - draft2[index].x) + (colNext - draft2[index].y);
            draft1[index].y = 0;
        }
    }
}

__global__
void matrix_dot_product(int m, int n, float2* x, float2* y)
{
    // y = x dot y
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < m; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < n; j += (gridDim.y * blockDim.y))
        {
            int index = i * n + j;
            y[index] = ComplexMul(x[index], y[index]);
        }
    }
}

__global__
void matrix_scale_and_minus(int m, int n, float2 *x, float2 *y, float k, float2 *draft)
{
    // x = kx - y
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < m; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < n; j += (gridDim.y * blockDim.y))
        {
            int index = i * n + j;
            draft[index].x = x[index].x;
            draft[index].y = x[index].y;
            draft[index] = ComplexSub(ComplexScale(draft[index], k), y[index]);
        }
    }
}

float** read_image(const char* name, int* info, png_bytep other)
{
    char header[8];
    png_structp png_ptr;
    png_infop info_ptr;
    int width, height;
    png_byte color_type;
    png_byte bit_depth;
    png_bytep *row_pointers;
    png_bytep *psf_row;

    FILE *fp = fopen(name, "rb");
    fread(header, 1, 8, fp);
    if (png_sig_cmp((png_const_bytep)header, 0, 8))
    {
        printf("That's not an png file\n");
        exit(0);
    }

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    info_ptr = png_create_info_struct(png_ptr);
    setjmp(png_jmpbuf(png_ptr));
    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    width = png_get_image_width(png_ptr, info_ptr);
    height = png_get_image_height(png_ptr, info_ptr);
    info[0] = width;
    info[1] = height;
    color_type = png_get_color_type(png_ptr, info_ptr);
    bit_depth = png_get_bit_depth(png_ptr, info_ptr);
    other[0] = color_type;
    other[1] = bit_depth;
    printf("This image have width %d and height %d\n", width, height);
    row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
    float **ret_arr = (float**) malloc(sizeof(float*) * height);
    for (int i = 0; i < height; i++)
    {
        size_t num = png_get_rowbytes(png_ptr, info_ptr);
        row_pointers[i] = (png_bytep) malloc(num);
        ret_arr[i] = (float*) malloc(sizeof(float) * width);
    }
    png_read_image(png_ptr, row_pointers);
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            ret_arr[i][j] = 0.0 + row_pointers[i][j];
        }
        free(row_pointers[i]);
    }
    free(row_pointers);
    fclose(fp);
    return ret_arr;
}

void write_image(const char *name, png_bytep *image, int width, int height, png_byte color_type, png_byte bit_depth)
{
    printf("Writing file %s...\n", name);
    FILE *fp = fopen(name, "wb");
    if (!fp)
    {
        printf("Can't even open file pointer\n");
        exit(0);
    }
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!png_ptr || !info_ptr)
    {
        printf("Stuck at pointer creation\n");
        exit(0);
    }
    setjmp(png_jmpbuf(png_ptr));
    png_init_io(png_ptr, fp);
    setjmp(png_jmpbuf(png_ptr));
    png_set_IHDR(png_ptr, info_ptr, width, height,
        bit_depth, color_type, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
    png_write_info(png_ptr, info_ptr);
    if (setjmp(png_jmpbuf(png_ptr)))
    {
        printf("Error writing bytes\n");
        exit(0);
    }
    png_write_image(png_ptr, image);
    if (setjmp(png_jmpbuf(png_ptr)))
    {
        printf("Error during end of write\n");
        exit(0);
    }
    png_write_end(png_ptr, NULL);
    fclose(fp);
}

void write_test_result(int numTests, double *record, char *filename)
{
    printf("Creating %s.csv file\n",filename);
    FILE *fp;

    // printf("Creating filename\n");

    // filename = strcat(filename, ".csv");
    // filename = strcat("../test/CUDA_Result/", filename);

    printf("Creating file pointer\n");
    fp = fopen(filename,"w+");

    fprintf(fp,"Num_Iterations, Time(s)\n");

    printf("Starting to write actual file\n");

    for(int i = 0; i < numTests; i++)
    {
        fprintf(fp, "%d,%.10g\n", i+1, record[i]);
    }

    fclose(fp);

    printf("%s.csv file created\n",filename);
}

void copy_image(int m, int n, float **image, float **psf, float2 *signal, float2 *filter)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            signal[i*n+j].x = image[i][j];
            signal[i*n+j].y = 0.0f;
            filter[i*n+j].x = (psf[i][j] * 1.0f) / (1500 * adjust_coefficient * adjust_coefficient);
            filter[i*n+j].y = 0.0f;
        }
    }
}

void write_back_image_padded(int m, int n, png_bytep *new_image, float2 *signal)
{
    // The signal is padded in this case
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            new_image[i][j] = (png_byte)fminf(fabsf(signal[(i+(m/2))*2*n+(j+(n/2))].x), 255.0);
            if ((i == m / 2) && (j < n / 2)) printf("%d ", new_image[i][j]);
        }
    }
}

void fourier(cufftHandle plan, int m, int n, float2 *signal, float2 *filter_shifted, dim3 numBlocks, dim3 threads)
{
    cufftExecC2C(plan, signal, signal, CUFFT_FORWARD);
    // cufftExecC2C(plan, filter_shifted, filter_shifted, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    multiply_kernel<<<numBlocks, threads>>>(m, n, signal, filter_shifted, (1.0f / (m * n)));
    cudaDeviceSynchronize();
    cufftExecC2C(plan, signal, signal, CUFFT_INVERSE);
    // cufftExecC2C(plan, filter_shifted, filter_shifted, CUFFT_INVERSE);
    cudaDeviceSynchronize();
    // normalize<<<numBlocks, threads>>>(m, n, signal, (1.0f / (m * n)));
    // cudaDeviceSynchronize();
    // normalize<<<numBlocks, threads>>>(m, n, filter_shifted);
    // cudaDeviceSynchronize();
}

void printSlice(int m, int n, float2 *mat, int start, const char* title)
{
    printf("%s:\n", title);
    for (int i = start; i < start + 8; i++)
    {
        for (int j = start; j < start + 8; j++)
        {
            float curr = mat[i*n+j].x;
            printf("%6.3f ", curr);
        }
        printf("\n");
    }
}

void printSliceReal(int m, int n, float *mat, int start, const char* title)
{
    printf("%s:\n", title);
    for (int i = start - 4; i < start + 4; i++)
    {
        for (int j = start - 4; j < start + 4; j++)
        {
            printf("%6.3f ", mat[i*n+j]);
        }
        printf("\n");
    }
}

void printSliceScientific(int m, int n, float2 *mat, int start, const char* title)
{
    printf("%s:\n", title);
    for (int i = start; i < start + 8; i++)
    {
        for (int j = start; j < start + 8; j++)
        {
            float curr = mat[i*n+j].x;
            printf("%6.3e ", curr);
        }
        printf("\n");
    }
}

void load_and_pad(int *image_info, int *psf_info, png_bytep other1, png_bytep other2, float2 **pad_signal, float2 **pad_filter)
{
    float **image;
    float **psf;
    png_bytep *new_image;
    image = read_image("../image/cameraman.png", image_info, other1);
    psf = read_image("../psf/psf_gaussian_3.png", psf_info, other2);
    int width = image_info[0];
    int height = image_info[1];
    int pad_height = 2 * height;
    int pad_width = 2 * width;
    int pad_size = pad_height * pad_width * sizeof(float2);
    float2 *signal, *filter, *filter_shifted;
    int size = image_info[0] * image_info[1] * sizeof(float2);
    // The signal (image)
    cudaMallocManaged(reinterpret_cast<void **>(&signal), size);
    // The filter (psf)
    cudaMallocManaged(reinterpret_cast<void **>(&filter), size);
    // Shifted psf
    cudaMallocManaged(reinterpret_cast<void **>(&filter_shifted), size);
    cudaMallocManaged(reinterpret_cast<void **>(pad_signal), pad_size);
    cudaMallocManaged(reinterpret_cast<void **>(pad_filter), pad_size);
    cudaDeviceSynchronize();
    cudaMemset(reinterpret_cast<void **>(pad_signal), 0, pad_size);
    cudaMemset(reinterpret_cast<void **>(pad_filter), 0, pad_size);
    cudaDeviceSynchronize();
    // Copy the image over, do the conversion from real to complex at the same time
    copy_image(image_info[1], image_info[0], image, psf, signal, filter);
    cudaDeviceSynchronize();
    dim3 threads(16, 16);
    dim3 numBlocks(image_info[1] / 16, image_info[0] / 16);
    circshift<<<numBlocks, threads>>>(image_info[1], image_info[0], filter, filter_shifted);
    cudaDeviceSynchronize();
    // Create cufft plan
    cufftHandle plan;
    cufftPlan2d(&plan, image_info[0], image_info[1], CUFFT_C2C);
    // Pre-transform the two filters
    cufftExecC2C(plan, filter_shifted, filter_shifted, CUFFT_FORWARD);
    // Now blur the image
    fourier(plan, image_info[1], image_info[0], signal, filter_shifted, numBlocks, threads);
    matrix_pad<<<numBlocks, threads>>>(height, width, signal, *pad_signal);
    cudaDeviceSynchronize();
    matrix_pad<<<numBlocks, threads>>>(height, width, filter, *pad_filter);
    cudaDeviceSynchronize();
    // printSlice(pad_height, pad_width, *pad_signal, height / 2, "pad_signal_original");
    cudaFree(signal);
    cudaFree(filter);
    cudaFree(filter_shifted);
    cufftDestroy(plan);
}

double* run_admm_image(int numIters)
{
    png_bytep *new_image;
    // Variables used in algorithm
    float2 *X, *U, *V, *W, *xi, *eta, *rho;
    // Intermediate or draft matrices
    float2 *r_calc, *draft1, *draft2;
    // Images and filters
    float2 *pad_signal, *pad_filter, *filter_shifted, *filter_reversed;
    printf("Start to load image\n");
    int *image_info, *psf_info;
    png_bytep other1, other2;
    image_info = (int*) malloc(2 * sizeof(int));
    psf_info = (int*) malloc(2 * sizeof(int));
    other1 = (png_bytep) malloc(2 * sizeof(png_byte));
    other2 = (png_bytep) malloc(2 * sizeof(png_byte));
    load_and_pad(image_info, psf_info, other1, other2, &pad_signal, &pad_filter);
    int size = image_info[0] * image_info[1] * sizeof(float2);
    int width = image_info[0];
    int height = image_info[1];
    int m = height;
    int n = width;
    int pad_height = 2 * height;
    int pad_width = 2 * width;
    int pad_size = pad_height * pad_width * sizeof(float2);
    printf("Finished loading images\n");
    // printSlice(pad_height, pad_width, pad_signal, height / 2, "pad_signal_original");
    printf("Started Allocation\n");
    printf("One array is of size %d\n", pad_size);
    cudaMallocManaged(reinterpret_cast<void **>(&X), pad_size);
    cudaMallocManaged(reinterpret_cast<void **>(&U), pad_size);
    cudaMallocManaged(reinterpret_cast<void **>(&V), pad_size);
    cudaMallocManaged(reinterpret_cast<void **>(&W), pad_size);
    cudaMallocManaged(reinterpret_cast<void **>(&xi), pad_size);
    cudaMallocManaged(reinterpret_cast<void **>(&eta), pad_size);
    cudaMallocManaged(reinterpret_cast<void **>(&rho), pad_size);
    cudaMallocManaged(reinterpret_cast<void **>(&r_calc), pad_size);
    cudaMallocManaged(reinterpret_cast<void **>(&draft1), pad_size);
    cudaMallocManaged(reinterpret_cast<void **>(&draft2), pad_size);
    // Shifted psf
    cudaMallocManaged(reinterpret_cast<void **>(&filter_shifted), pad_size);
    // Used for fourier adjoint
    cudaMallocManaged(reinterpret_cast<void **>(&filter_reversed), pad_size);
    cudaDeviceSynchronize();
    printf("Finished Allocation\n");
    cudaMemset(reinterpret_cast<void **>(&X), 0, pad_size);
    cudaMemset(reinterpret_cast<void **>(&U), 0, pad_size);
    cudaMemset(reinterpret_cast<void **>(&V), 0, pad_size);
    cudaMemset(reinterpret_cast<void **>(&W), 0, pad_size);
    cudaMemset(reinterpret_cast<void **>(&xi), 0, pad_size);
    cudaMemset(reinterpret_cast<void **>(&eta), 0, pad_size);
    cudaMemset(reinterpret_cast<void **>(&rho), 0, pad_size);
    cudaMemset(reinterpret_cast<void **>(&r_calc), 0, pad_size);
    cudaMemset(reinterpret_cast<void **>(&draft1), 0, pad_size);
    cudaMemset(reinterpret_cast<void **>(&draft2), 0, pad_size);
    cudaDeviceSynchronize();
    printf("Finished setting memory\n");
    dim3 threads(16, 16);
    dim3 numBlocks(pad_height / 16, pad_width / 16);
    circshift<<<numBlocks, threads>>>(pad_height, pad_width, pad_filter, filter_shifted);
    cudaDeviceSynchronize();
    reverse<<<numBlocks, threads>>>(pad_height, pad_width, filter_reversed, filter_shifted);
    cudaDeviceSynchronize();
    printf("Transformation matrices ready\n");
    // Create cufft plan
    cufftHandle plan;
    cufftPlan2d(&plan, pad_height, pad_width, CUFFT_C2C);
    // Pre-transform the two filters
    cufftExecC2C(plan, filter_shifted, filter_shifted, CUFFT_FORWARD);
    cufftExecC2C(plan, filter_reversed, filter_reversed, CUFFT_FORWARD);
    // Copy the image back
    new_image = (png_bytep*) malloc(image_info[1] * sizeof(png_bytep));
    for (int i = 0; i < image_info[1]; i++) new_image[i] = (png_bytep) malloc(image_info[0] * sizeof(png_byte));
    write_back_image_padded(image_info[1], image_info[0], new_image, pad_signal);
    cudaDeviceSynchronize();
    write_image("../test/blurred/blurred3.png", new_image, image_info[0], image_info[1], other1[0], other1[1]);
    float *X_divmat;
    float2 *psiTpsi, *psi_V, *V_divmat;
    cudaMallocManaged(reinterpret_cast<void **>(&X_divmat), pad_height * pad_width * sizeof(float));
    cudaMallocManaged(reinterpret_cast<void **>(&V_divmat), pad_height * pad_width * sizeof(float2));
    cudaMallocManaged(reinterpret_cast<void **>(&psiTpsi), pad_height * pad_width * sizeof(float2));
    cudaMallocManaged(reinterpret_cast<void **>(&psi_V), pad_height * pad_width * sizeof(float2));
    cudaDeviceSynchronize();
    // Now, first we calculate the X_divmat
    compute_X_divmat<<<numBlocks, threads>>>(pad_height, pad_width, X_divmat);
    cudaDeviceSynchronize();
    // printSliceReal(pad_height, pad_width, X_divmat, height / 2, "X_divmat");
    cudaMemset(reinterpret_cast<void **>(&psiTpsi), 0, pad_size);
    cudaDeviceSynchronize();
    // Below is the construction of the psiTpsi matrix. Move to a function later
    psiTpsi[0].x = 4.0f;
    psiTpsi[1].x = -1.0f;
    psiTpsi[pad_width - 1].x = -1.0f;
    psiTpsi[pad_width].x = -1.0f;
    psiTpsi[(pad_height - 1) * pad_width].x = -1.0f;
    cufftExecC2C(plan, psiTpsi, psiTpsi, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    // Now we move on to the V_divmat calculation. There is a lot of element wise calculation, hence
    // It can be done in terms of a kernel function. The filter is pre-shifted
    compute_V_divmat<<<numBlocks, threads>>>(pad_height, pad_width, filter_shifted, psiTpsi, V_divmat);
    cudaDeviceSynchronize();
    // printSlice(pad_height, pad_width, V_divmat, height / 2, "V_divmat");
    double *record = (double *) malloc(numIters * sizeof(double));
    auto runStart = std::chrono::system_clock::now();
    // ADMM iterations starts here
    for (int i = 0; i < numIters; i++)
    {
        // printf("Iteration %d\n", i);
        // U-update
        // Here, as we just calculated psi_V, there is no need to calculate that again
        U_update<<<numBlocks, threads>>>(pad_height, pad_width, psi_V, eta, U);
        cudaDeviceSynchronize();
        // printSlice(pad_height, pad_width, U, height / 2, "U");
        // X-update
        copy_array_complex<<<numBlocks, threads>>>(pad_height, pad_width, V, draft1);
        cudaDeviceSynchronize();
        fourier(plan, pad_height, pad_width, draft1, filter_shifted, numBlocks, threads);
        X_update<<<numBlocks, threads>>>(pad_height, pad_width, X_divmat, xi, draft1, pad_signal, X);
        cudaDeviceSynchronize();
        // printSlice(pad_height, pad_width, pad_signal, height / 2, "pad_signal");
        // printSlice(pad_height, pad_width, X, height / 2, "X");
        // R-calculation
        matrix_scale_and_minus<<<numBlocks, threads>>>(pad_height, pad_width, W, rho, mu3, r_calc);
        cudaDeviceSynchronize();
        matrix_scale_and_minus<<<numBlocks, threads>>>(pad_height, pad_width, U, eta, mu2, draft2);
        cudaDeviceSynchronize();
        psi_adjoint<<<numBlocks, threads>>>(pad_height, pad_width, draft2, draft1);
        cudaDeviceSynchronize();
        matrix_scale_and_minus<<<numBlocks, threads>>>(pad_height, pad_width, X, xi, mu1, draft2);
        cudaDeviceSynchronize();
        fourier(plan, pad_height, pad_width, draft2, filter_reversed, numBlocks, threads);
        r_calculation<<<numBlocks, threads>>>(pad_height, pad_width, r_calc, draft1, draft2);
        cudaDeviceSynchronize();
        // printSlice(pad_height, pad_width, r_calc, height / 2, "r_calc");
        // V-update
        V_update(plan, pad_height, pad_width, V_divmat, r_calc, draft1, draft2, V, numBlocks, threads);
        // printSlice(pad_height, pad_width, V, height / 2, "V");
        // W-update
        W_update<<<numBlocks, threads>>>(pad_height, pad_width, rho, V, W);
        cudaDeviceSynchronize();
        // printSlice(pad_height, pad_width, W, height, "W");
        // Xi-update
        copy_array_complex<<<numBlocks, threads>>>(pad_height, pad_width, V, draft1);
        cudaDeviceSynchronize();
        fourier(plan, pad_height, pad_width, draft1, filter_shifted, numBlocks, threads);
        Xi_update<<<numBlocks, threads>>>(pad_height, pad_width, draft1, X, xi);
        cudaDeviceSynchronize();
        // printSliceScientific(pad_height, pad_width, xi, height, "xi");
        // Eta-update
        calc_psi_V<<<numBlocks, threads>>>(pad_height, pad_width, V, psi_V);
        cudaDeviceSynchronize();
        Eta_update<<<numBlocks, threads>>>(pad_height, pad_width, psi_V, U, eta);
        cudaDeviceSynchronize();
        // printSliceScientific(pad_height, pad_width, eta, height, "eta");
        // Rho-update
        Rho_update<<<numBlocks, threads>>>(pad_height, pad_width, V, W, rho);
        cudaDeviceSynchronize();
        // printSliceScientific(pad_height, pad_width, rho, height, "rho");
        auto currEnd = std::chrono::system_clock::now();
        std::chrono::duration<double> currDuration = currEnd - runStart;
        record[i] = currDuration.count();
    }
    write_back_image_padded(image_info[1], image_info[0], new_image, V);
    cudaDeviceSynchronize();
    write_image("../test/recovered/ADMM_recovered3.png", new_image, image_info[0], image_info[1], other1[0], other1[1]);
    cudaFree(X);
    cudaFree(U);
    cudaFree(V);
    cudaFree(W);
    cudaFree(xi);
    cudaFree(eta);
    cudaFree(rho);
    cudaFree(r_calc);
    cudaFree(draft1);
    cudaFree(draft2);
    cudaFree(pad_signal);
    cudaFree(pad_filter);
    cudaFree(X_divmat);
    cudaFree(V_divmat);
    cudaFree(psiTpsi);
    cudaFree(psi_V);
    free(image_info);
    free(psf_info);
    free(other1);
    free(other2);
    free(new_image);
    cufftDestroy(plan);
    return record;
}

// Padding needed: psf, blurred image, image of all ones

int main(void)
{
    int numTests = 200;
    double *record = run_admm_image(numTests);
    // for (int i = 0; i < numTests; i++)
    // {
    //     printf("Test %d\n", i + 1);
    //     record[i] = run_ista_image(i+1);
    // }
    char *filename = "../test/CUDA_Result/ADMM_0_200_single.csv";
    write_test_result(numTests, record, filename);
    free(record);
}