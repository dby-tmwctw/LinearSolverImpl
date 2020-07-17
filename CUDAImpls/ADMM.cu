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

// Convention: Input on the front, output on the back
// Convention: All arrays are of size mxn

const int adjust_coefficient = 3;
const int num_iters = 8000;

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

void copy_image_padded(int m, int n, float **image, float **psf, float2 *signal, float2 *filter)
{
    // Here, the signal and the filter should be padded
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            signal[(i+(m/2))*2*n+(j+(n/2))].x = image[i][j];
            signal[(i+(m/2))*2*n+(j+(n/2))].y = 0.0f;
            filter[(i+(m/2))*2*n+(j+(n/2))].x = (psf[i][j] * 1.0f) / (1500 * adjust_coefficient * adjust_coefficient);
            filter[(i+(m/2))*2*n+(j+(n/2))].y = 0.0f;
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
    // normalize<<<numBlocks, threads>>>(m, n, filter_shifted);
    // cudaDeviceSynchronize();
}

// Padding needed: psf, blurred image, image of all ones

int main(void)
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
    float2 *signal, *filter, filter_shifted_temp;
    int size = image_info[0] * image_info[1] * sizeof(float2);
    int width = image_info[0];
    int height = image_info[1];
    // // The signal (image)
    // cudaMallocManaged(reinterpret_cast<void **>(&signal), size);
    // // The filter (psf)
    // cudaMallocManaged(reinterpret_cast<void **>(&filter), size);
    // cudaMallocManaged(reinterpret_cast<void **>(&filter_shifted_temp), size);
    // // // Copy the image over, do the conversion from real to complex at the same time
    // copy_image(image_info[1], image_info[0], image, psf, signal, filter);
    // // Below is just attempts to create a truely blurred image
    // cufftHandle plan_temp;
    // cufftPlan2d(&plan_temp, height, width, CUFFT_C2C);
    // dim3 threads_temp(16, 16);
    // dim3 numBlocks_temp(width / 16, height / 16);
    // circshift<<<numBlocks_temp, threads_temp>>>(height, width, filter, filter_shifted_temp);
    // cudaDeviceSynchronize();
    // cufftExecC2C(plan, filter_shifted_temp, filter_shifted_temp, CUFFT_FORWARD);
    // cudaDeviceSynchronize();
    // fourier(plan_temp, height, width, signal, filter_shifted_temp, numBlocks_temp, threads_temp);
    // cudaFree(filter_shifted_temp);
    // cufftDestroy(plan_temp);
    int m = height;
    int n = width;
    int pad_height = 2 * height;
    int pad_width = 2 * width;
    int pad_size = pad_height * pad_width * sizeof(float2);
    float2 *X, *U, *V, *W, *xi, *eta, *rho;
    float2 *pad_signal, *pad_filter, *filter_shifted, *filter_reversed;
    printf("Started Allocation\n");
    printf("One array is of size %d\n", pad_size);
    cudaMallocManaged(reinterpret_cast<void **>(&X), pad_size);
    cudaMallocManaged(reinterpret_cast<void **>(&U), pad_size);
    cudaMallocManaged(reinterpret_cast<void **>(&V), pad_size);
    cudaMallocManaged(reinterpret_cast<void **>(&W), pad_size);
    cudaMallocManaged(reinterpret_cast<void **>(&xi), pad_size);
    cudaMallocManaged(reinterpret_cast<void **>(&eta), pad_size);
    cudaMallocManaged(reinterpret_cast<void **>(&rho), pad_size);
    cudaMallocManaged(reinterpret_cast<void **>(&pad_signal), pad_size);
    cudaMallocManaged(reinterpret_cast<void **>(&pad_filter), pad_size);
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
    cudaMemset(reinterpret_cast<void **>(&pad_signal), 0, pad_size);
    cudaMemset(reinterpret_cast<void **>(&pad_filter), 0, pad_size);
    cudaDeviceSynchronize();
    printf("Finished setting memory\n");
    copy_image_padded(height, width, image, psf, pad_signal, pad_filter);
    printf("Finished copying image over\n");
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
    // Now blur the image
    fourier(plan, pad_height, pad_width, pad_signal, filter_shifted, numBlocks, threads);
    // Copy the image back
    new_image = (png_bytep*) malloc(image_info[1] * sizeof(png_bytep));
    for (int i = 0; i < image_info[1]; i++) new_image[i] = (png_bytep) malloc(image_info[0] * sizeof(png_byte));
    write_back_image_padded(image_info[1], image_info[0], new_image, pad_signal);
    cudaDeviceSynchronize();
    write_image("../test/blurred/blurred3.png", new_image, image_info[0], image_info[1], other1[0], other1[1]);
    // int width = image_info[0];
    // int height = image_info[1];
    // int m = height;
    // int n = width;
    // int pad_height = 2 * height;
    // int pad_width = 2 * width;
    // int pad_size = pad_height * pad_width * sizeof(float2);
    // float2 *X, *U, *V, *W, *xi, *eta, *rho;
    // cudaMallocManaged(reinterpret_cast<void **>(&X), pad_size);
    // cudaMallocManaged(reinterpret_cast<void **>(&U), pad_size);
    // cudaMallocManaged(reinterpret_cast<void **>(&V), pad_size);
    // cudaMallocManaged(reinterpret_cast<void **>(&W), pad_size);
    // cudaMallocManaged(reinterpret_cast<void **>(&xi), pad_size);
    // cudaMallocManaged(reinterpret_cast<void **>(&eta), pad_size);
    // cudaMallocManaged(reinterpret_cast<void **>(&rho), pad_size);
}