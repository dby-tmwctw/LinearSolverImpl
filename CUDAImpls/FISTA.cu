#include <stdio.h>
#include <curand.h>
#include "cublas_v2.h"
#include <cusolverDn.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <png.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <chrono>
#include <iomanip>
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

const int adjust_coefficient = 3;
const int num_iters = 8000;

__global__
void copy_array(int n, int m, float* x, float* y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < n; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < m; j += (gridDim.y * blockDim.y))
        {
            y[i*m+j] = x[i*m+j];
        }
    }
}

__global__
void copy_vector(int n, int *x, int *y)
{
    // x = y
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        x[i] = y[i];
    }
}

__global__
void vector_minus(int n, float *x, float *y)
{
    // x = x - y
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        x[i] = x[i] - y[i];
    }
}

__global__
void vector_multiply(int n, float k, float *x)
{
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        x[i] = k * x[i];
    }
}

__global__
void shrink(int n, float *x, float l)
{
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        if (x[i] > 0)
        {
            x[i] = fmaxf(fabsf(x[i]) - l, 0);
        } else
        {
            x[i] = 0 - fmaxf(fabsf(x[i]) - l, 0);
        }
    }
}

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

__global__
void multiply_kernel(int n, int m, float2* x, float2* y, float scale)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < n; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < m; j += (gridDim.y * blockDim.y))
        {
            x[i*m+j] = ComplexScale(ComplexMul(x[i*m+j], y[i*m+j]), scale);
        }
    }
}

__global__
void matrix_minus(int m, int n, float2* x, float2* y)
{
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
void matrix_copy(int m, int n, float2* x, float2* y)
{
    // Let x = y
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < m; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < n; j += (gridDim.y * blockDim.y))
        {
            x[i*n+j].x = y[i*n+j].x;
            x[i*n+j].y = y[i*n+j].y;
        }
    }
}

__global__
void modify(int m, int n, float2 *x, float2 *y, float step)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < m; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < n; j += (gridDim.y * blockDim.y))
        {
            x[i*n+j] = ComplexSub(y[i*n+j], ComplexScale(x[i*n+j], 2 * step));
        }
    }
}

__global__
void shrink2D(int m, int n, float2 *x, float l)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < m; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < n; j += (gridDim.y * blockDim.y))
        {
            float real = x[i*n+j].x;
            if (real > 0)
            {
                x[i*n+j].x = fmaxf(fabsf(real) - l, 0);
                x[i*n+j].y = 0.0f;
            } else
            {
                x[i*n+j].x = 0 - fmaxf(fabsf(real) - l, 0);
                x[i*n+j].y = 0.0f;
            }
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
void reverse(int m, int n, float2* x, float2* y)
{
    // x = y reversed
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < m; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < n; j += (gridDim.y * blockDim.y))
        {
            x[i*n+j] = y[(m*n-1)-(i*n+j)];
        }
    }
}

__global__
void multiply_kernel_additional(int m, int n, float2* x, float2 *y, float2 *z, float scale)
{
    // x = y * z
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < m; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < n; j += (gridDim.y * blockDim.y))
        {
            x[i*n+j] = ComplexScale(ComplexMul(y[i*n+j], z[i*n+j]), scale);
        }
    }
}

__global__
void normalize(int m, int n, float2* x)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = row; i < m; i += (gridDim.x * blockDim.x))
    {
        for (int j = col; j < n; j += (gridDim.y * blockDim.y))
        {
            x[i*n+j].x = x[i*n+j].x / (m * n);
            x[i*n+j].y = x[i*n+j].y / (m * n);
        }
    }
}

__global__
void reduce(float2 *input, float2 *output)
{
    extern __shared__ float2 sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = input[index];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] = ComplexSquareMax(sdata[tid], sdata[tid+s]);
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        output[blockIdx.x] = sdata[0];
    }
}

__global__
void collapseAns(float2 *ans)
{
    int tid = threadIdx.x;
    for (unsigned int s = blockDim.x / 2; s > 0; s>>=1)
    {
        if (tid < s)
        {
            ans[tid] = ComplexSquareMax(ans[tid], ans[tid+s]);
        }
        __syncthreads();
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

void free_image(float **image, int *info, png_bytep other)
{
    for (int i = 0; i < info[1]; i++)
    {
        free(image[i]);
    }
    free(image);
    free(info);
    free(other);
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

void write_back_image(int m, int n, png_bytep *new_image, float2 *signal)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            new_image[i][j] = (png_byte)fminf(fabsf(signal[i*n+j].x), 255.0);
            if (i == m / 2) printf("%d ", new_image[i][j]);
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
    // normalize<<<numBlocks, threads>>>(m, n, filter_shifted);
    // cudaDeviceSynchronize();
}

float step_size(cublasHandle_t handle, cusolverDnHandle_t cusolverH, int m, int n, float *A)
{
    // A should be an mxn matrix;
    float *d_work;
    int lwork = 0;
    int *devInfo;
    float *eigen;
    float *AT;
    float *transpose;
    cudaMallocManaged(&AT, m*n*sizeof(float));
    cudaMallocManaged(&transpose, n*n*sizeof(float));
    cudaMallocManaged(&eigen, n*sizeof(float));
    cudaMallocManaged(&devInfo, sizeof(int));
    cudaDeviceSynchronize();
    cudaMemset(&transpose, 0, n*n*sizeof(float));
    cudaDeviceSynchronize();
    copy_array<<<dim3((m+15) / 16, (n+15) / 16), dim3(16, 16)>>>(m, n, A, AT);
    cudaDeviceSynchronize();
    float a = 1.0f;
    float c = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m, &a, AT, m, A, m, &c, transpose, n);
    cudaDeviceSynchronize();
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cusolverDnSsyevd_bufferSize(cusolverH, jobz, uplo, n, transpose, n, eigen, &lwork);
    cudaMalloc((void**)&d_work, sizeof(float)*lwork);
    cusolverDnSsyevd(cusolverH, jobz, uplo, n, transpose, n, eigen, d_work, lwork, devInfo);
    cudaDeviceSynchronize();
    float step = 1 / (2 * eigen[n-1]);
    cudaFree(eigen);
    cudaFree(devInfo);
    cudaFree(d_work);
    return step;
}

void ista(cublasHandle_t handle, cusolverDnHandle_t cusolverH, int m, int n, float *A, float *x, float *b, float *x_est, int iters, float l)
{
    float *result, *temp;
    cudaMallocManaged(&result, m*sizeof(float));
    cudaMallocManaged(&temp, n*sizeof(float));
    cudaDeviceSynchronize();
    float step = step_size(handle, cusolverH, m, n, A);
    float a = 1.0f;
    float c = 0.0f;
    for (int k = 0; k < iters; k++)
    {
        cublasSgemv(handle, CUBLAS_OP_N, m, n, &a, A, m, x_est, 1, &c, result, 1);
        cudaDeviceSynchronize();
        vector_minus<<<1, m>>>(m, result, b);
        cudaDeviceSynchronize();
        cublasSgemv(handle, CUBLAS_OP_T, m, n, &a, A, m, result, 1, &c, temp, 1);
        cudaDeviceSynchronize();
        vector_multiply<<<1, n>>>(n, 2*step, temp);
        cudaDeviceSynchronize();
        vector_minus<<<1, n>>>(n, x_est, temp);
        cudaDeviceSynchronize();
        shrink<<<1, n>>>(n, x_est, l * step);
        cudaDeviceSynchronize();
    }
    cudaFree(result);
    cudaFree(temp);
}

void run_ista(void)
{
    cublasHandle_t handle;
    cublasStatus_t stat;
    curandGenerator_t gen;
    cusolverDnHandle_t cusolverH;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    int i, j;
    unsigned long long seed = 123456;
    // A is mxn matrix
    float *A;
    // x is R^n, b is R^m
    float *x, *b;
    int m = 32;
    int n = 16;
    int info_gpu = 0;
    cudaMallocManaged(&A, m*n*sizeof(float));
    cudaMallocManaged(&x, n*sizeof(float));
    cudaMallocManaged(&b, m*sizeof(float));
    cudaDeviceSynchronize();
    cudaMemset(&b, 0, m*sizeof(float));
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    cudaDeviceSynchronize();
    curandGenerateUniform(gen, A, m*n*sizeof(float));
    curandGenerateUniform(gen, x, n*sizeof(float));
    cudaDeviceSynchronize();
    // cudaMemcpy(&AT, &A, m*n*sizeof(float), cudaMemcpyDefault);
    for (i = 0; i < n; i++)
    {
        printf("%f\n", x[i]);
    }
    printf("-------------------------\n");
    float a = 1.0f;
    float c = 0.0f;
    stat = cublasCreate(&handle);
    stat = cublasSgemv(handle, CUBLAS_OP_N, m, n, &a, A, m, x, 1, &c, b, 1);
    cudaDeviceSynchronize();
    cusolver_status = cusolverDnCreate(&cusolverH);
    int iters = 8000;
    float l = 0.00001;
    float *x_est;
    cudaMallocManaged(&x_est, n*sizeof(float));
    cudaDeviceSynchronize();
    cudaMemset(&x_est, 0, n*sizeof(float));
    ista(handle, cusolverH, m, n, A, x, b, x_est, iters, l);
    for (j = 0; j < n; j++)
    {
        printf("%f\n", x_est[j]);
    }
    cudaFree(A);
    cudaFree(x);
    cudaFree(b);
    cudaFree(x_est);
    cublasDestroy(handle);
    curandDestroyGenerator(gen);
}

void run_ista_image(void)
{
    float **image;
    float **psf;
    png_bytep *new_image;
    int *image_info, *psf_info;
    png_bytep other1, other2;
    image_info = (int*) malloc(2 * sizeof(int));
    psf_info = (int*) malloc(2 * sizeof(int));
    other1 = (png_bytep) malloc(2 * sizeof(png_byte));
    other2 = (png_bytep) malloc(2 * sizeof(png_byte));
    image = read_image("../image/cameraman.png", image_info, other1);
    psf = read_image("../psf/psf_gaussian_3.png", psf_info, other2);
    float2 *signal, *filter, *filter_shifted, *filter_reversed;
    int size = image_info[0] * image_info[1] * sizeof(float2);
    // The signal (image)
    cudaMallocManaged(reinterpret_cast<void **>(&signal), size);
    // The filter (psf)
    cudaMallocManaged(reinterpret_cast<void **>(&filter), size);
    // Shifted psf
    cudaMallocManaged(reinterpret_cast<void **>(&filter_shifted), size);
    // Used for fourier adjoint
    cudaMallocManaged(reinterpret_cast<void **>(&filter_reversed), size);
    // Copy the image over, do the conversion from real to complex at the same time
    copy_image(image_info[1], image_info[0], image, psf, signal, filter);
    cudaDeviceSynchronize();
    dim3 threads(16, 16);
    dim3 numBlocks(image_info[1] / 16, image_info[0] / 16);
    circshift<<<numBlocks, threads>>>(image_info[1], image_info[0], filter, filter_shifted);
    cudaDeviceSynchronize();
    reverse<<<numBlocks, threads>>>(image_info[1], image_info[0], filter_reversed, filter_shifted);
    cudaDeviceSynchronize();
    // Create cufft plan
    cufftHandle plan;
    cufftPlan2d(&plan, image_info[0], image_info[1], CUFFT_C2C);
    // Pre-transform the two filters
    cufftExecC2C(plan, filter_shifted, filter_shifted, CUFFT_FORWARD);
    cufftExecC2C(plan, filter_reversed, filter_reversed, CUFFT_FORWARD);
    // Now blur the image
    fourier(plan, image_info[1], image_info[0], signal, filter_shifted, numBlocks, threads);
    // Copy the image back
    new_image = (png_bytep*) malloc(image_info[1] * sizeof(png_bytep));
    for (int i = 0; i < image_info[1]; i++) new_image[i] = (png_bytep) malloc(image_info[0] * sizeof(png_byte));
    write_back_image(image_info[1], image_info[0], new_image, signal);
    cudaDeviceSynchronize();
    write_image("../test/blurred/blurred3.png", new_image, image_info[0], image_info[1], other1[0], other1[1]);
    // cufftExecC2C(plan, filter_shifted, filter_shifted, CUFFT_FORWARD);
    float2 *output;
    int numThreads1 = 64;
    int numBlocks1 = (image_info[0] * image_info[1]) / numThreads1;
    cudaMallocManaged(reinterpret_cast<void **>(&output), numBlocks1 * sizeof(float2));
    reduce<<<numBlocks1, numThreads1, numThreads1*sizeof(float2)>>>(filter_shifted, output);
    cudaDeviceSynchronize();
    collapseAns<<<1, numBlocks1>>>(output);
    cudaDeviceSynchronize();
    float2 maxComplex = ComplexMul(output[0], output[0]);
    float max_value = maxComplex.x;
    float step = 1 / (2 * max_value);
    // cufftExecC2C(plan, filter_shifted, filter_shifted, CUFFT_INVERSE);
    // normalize<<<numBlocks, threads>>>(image_info[1], image_info[0], filter_shifted);
    // cudaDeviceSynchronize();
    cudaFree(output);
    printf("%f %f\n", max_value, step);
    float2 *x_est, *temp;
    cudaMallocManaged(reinterpret_cast<void **>(&x_est), size);
    cudaMallocManaged(reinterpret_cast<void **>(&temp), size);
    cudaMemset(reinterpret_cast<void **>(&x_est), 0, size);
    auto runStart = std::chrono::system_clock::now();
    for (int i = 0; i < num_iters; i++)
    {
        printf("Now doing iteration %d\n", i);
        matrix_copy<<<numBlocks, threads>>>(image_info[1], image_info[0], temp, x_est);
        cudaDeviceSynchronize();
        fourier(plan, image_info[1], image_info[0], x_est, filter_shifted, numBlocks, threads);
        matrix_minus<<<numBlocks, threads>>>(image_info[1], image_info[0], x_est, signal);
        cudaDeviceSynchronize();
        fourier(plan, image_info[1], image_info[0], x_est, filter_reversed, numBlocks, threads);
        modify<<<numBlocks, threads>>>(image_info[1], image_info[0], x_est, temp, step);
        cudaDeviceSynchronize();
        shrink2D<<<numBlocks, threads>>>(image_info[1], image_info[0], x_est, 10 * step);
        cudaDeviceSynchronize();
    }
    auto runEnd = std::chrono::system_clock::now();
    std::chrono::duration<double> runDuration = runEnd - runStart;
    printf("Program runtime: %.17g second(s)\n", runDuration.count());
    write_back_image(image_info[1], image_info[0], new_image, x_est);
    cudaDeviceSynchronize();
    write_image("../test/recovered/recovered3.png", new_image, image_info[0], image_info[1], other1[0], other1[1]);
    cufftExecC2C(plan, filter_shifted, filter_shifted, CUFFT_INVERSE);
    cufftExecC2C(plan, filter_reversed, filter_reversed, CUFFT_INVERSE);
    free_image(image, image_info, other1);
    free_image(psf, psf_info, other2);
    cudaFree(signal);
    cudaFree(filter);
    cudaFree(filter_shifted);
    cudaFree(filter_reversed);
    cufftDestroy(plan);
}

int main(void)
{
    run_ista_image();
}