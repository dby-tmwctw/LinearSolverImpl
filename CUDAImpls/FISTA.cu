#include <stdio.h>
#include <curand.h>
#include "cublas_v2.h"
#include <cusolverDn.h>
#include <math.h>
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

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

int main(void)
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