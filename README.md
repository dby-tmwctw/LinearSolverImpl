# LinearSolverImpl

CUDA and Python Image Reconstruction Algorithms Implementation

## Introduction

This repository majorly features a CUDA Implementation of several Image Reconstruction Algorithms, including **ISTA, FISTA, SAFISTA** and **ADMM**. All of them support 2-D image reconstruction with the Gaussian blur as point spread function. 1-D (vector) reconstruction with matrix-vector multiplication is supported for ISTA and FISTA. Compared to their original implementations in python (which is also provided in repository), the CUDA parallelized version could be **1600x** faster in speed.

## Content Overview
### CUDAImpls
In this folder there are two files: FISTA.cu and ADMM.cu. 

FISTA.cu contains all algorithms except ADMM. In FISTA.cu, ista() and fista() are 1-D vector reconstruction, and functions with name like **run_xxx_image()** is the 2-D image reconstruction. The parameters used in the algorithm could be set at the top, which are defined as constants.

ADMM.cu contains the ADMM algorithm. The **run_admm_image()** is the reconstruction function for 2-D image reconstrucion.

You can customize the image and psf used by digging into the functions and find the relevant line. This will be moved out as an argument in later versions.

## PythonImpls
In this folder there are python implementations of above algorithms. The structure is similar to above.

## Requirements
### Compiler
CUDA 10.1
### Libraries
CUSolver

CUFFT

CURand

CUBLAS

libpng

chrono

## Compilation Command
nvcc xxx.cu -o xxx.o -lcufft -lcurand -lcusolver -lcublas -lpng

./xxx.o
