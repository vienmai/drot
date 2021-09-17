#ifndef KERNELS_HPP_
#define KERNELS_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 128
#define WORK_SIZE 1

template<typename T>
__inline__ __device__ T warp_reduce_sum(T val) {
  for (int w = 16; w > 0; w /= 2)
      val += __shfl_down_sync(0xffffffff, val, w);
  return val;
}

template<typename T>
__inline__ __device__ T block_reduce_sum(T val) {
    static __shared__ T shared[32];

    val = warp_reduce_sum(val);
    if (threadIdx.y % 32==0)
        shared[threadIdx.y / 32] = val;
    __syncthreads();

    val = (threadIdx.y < blockDim.y / 32) ? shared[threadIdx.y % 32] : 0;
    if (threadIdx.y / 32==0)
        val = warp_reduce_sum(val);

    return val;
}

template<typename T>
__global__ void update_x_even(T *X,
        T *a2,
        T *b2,
        T *gamma2,
        T *loginfo,
        const T * __restrict__ phi1,
        const T * __restrict__ phi2,
        const T * __restrict__ C,
        const T stepsize,
        const int nrows,
        const int ncols) {
    const int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset;

    __shared__ T phi2_shared[BLOCK_SIZE];
    T xval=0;
    T colsum=0;
    const T phi1_val=phi1[tidy];

    for (int idx=0; idx<WORK_SIZE; idx++) {
        offset = (blockIdx.x * WORK_SIZE + idx) * BLOCK_SIZE; // Row offset
        if (threadIdx.y + offset < ncols)
            phi2_shared[threadIdx.y] = phi2[threadIdx.y + offset];
        __syncthreads();

        if (tidy < nrows) {
            if (BLOCK_SIZE + offset <= ncols) {
                #pragma unroll
                for (int e=0; e<BLOCK_SIZE; e++) {
                    xval = X[tidy + (e + offset) * nrows];
                    xval = max(xval + phi1_val + phi2_shared[e], T(0));
                    X[tidy + (e + offset) * nrows] = xval;

                    colsum += xval;
                    xval = warp_reduce_sum(xval);
                    if (threadIdx.y % 32==0)
                        atomicAdd(&b2[e + offset], xval);
                }
            } else { // Edge block
                #pragma unroll
                for (int e=ncols-1; e>=offset; e--) {
                    xval = X[tidy + e * nrows];
                    xval = max(xval + phi1_val + phi2_shared[e-offset], T(0));
                    X[tidy + e * nrows] = xval;

                    colsum += xval;
                    xval = warp_reduce_sum(xval);
                    if (threadIdx.y % 32==0)
                        atomicAdd(&b2[e], xval);
                }
            }
        }
    }
    if (tidy < nrows) {
        atomicAdd(&a2[tidy], colsum);
        colsum = warp_reduce_sum(colsum);
        if (threadIdx.y % 32==0)
            atomicAdd(gamma2, colsum);
    }
    if (tidy>=blockDim.y*gridDim.y-2 && blockIdx.x==0)
        loginfo[blockDim.y*gridDim.y - tidy - 1] = T(0);
}

template<typename T>
__global__ void update_x_odd(T *X,
        T *a2,
        T *b2,
        T *gamma2,
        T *loginfo,
        const T * __restrict__ phi1,
        const T * __restrict__ phi2,
        const T * __restrict__ C,
        const T stepsize,
        const int nrows,
        const int ncols) {
    const int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset;

    __shared__ T phi2_shared[BLOCK_SIZE];
    T xval=0;
    T cval=0;
    T colsum=0;
    const T phi1_val=phi1[tidy];

    for (int idx=0; idx<WORK_SIZE; idx++) {
        offset = (blockIdx.x * WORK_SIZE + idx) * BLOCK_SIZE; // Row offset
        if (threadIdx.y + offset < ncols)
            phi2_shared[threadIdx.y] = phi2[threadIdx.y + offset];
        __syncthreads();

        if (tidy < nrows) {
            if (BLOCK_SIZE + offset <= ncols) {
                #pragma unroll
                for (int e=0; e<BLOCK_SIZE; e++) {
                    xval = X[tidy + (e + offset) * nrows];
                    cval = stepsize * C[tidy + (e + offset) * nrows];
                    xval = max(xval + phi1_val + phi2_shared[e] - cval, T(0));
                    X[tidy + (e + offset) * nrows] = xval - cval;

                    colsum += xval;
                    xval = warp_reduce_sum(xval);
                    if (threadIdx.y % 32==0)
                        atomicAdd(&b2[e + offset], xval);
                }
            } else { // Edge blocks
                #pragma unroll
                for (int e=ncols-1; e>=offset; e--) {
                    xval = X[tidy + e * nrows];
                    cval = stepsize * C[tidy + e * nrows];
                    xval = max(xval + phi1_val + phi2_shared[e-offset]
                            - cval, T(0));
                    X[tidy + e * nrows] = xval - cval;

                    colsum += xval;
                    xval = warp_reduce_sum(xval);
                    if (threadIdx.y % 32==0)
                        atomicAdd(&b2[e], xval);
                }
            }
        }
    }
    if (tidy < nrows) {
        atomicAdd(&a2[tidy], colsum);
        colsum = warp_reduce_sum(colsum);
        if (threadIdx.y % 32==0)
            atomicAdd(gamma2, colsum);
    }
    if (tidy>=blockDim.y*gridDim.y-2 && blockIdx.x==0)
        loginfo[blockDim.y*gridDim.y - tidy - 1] = T(0);
}

template<typename T>
__global__ void update_x_even_(T *X,
        T *a2,
        T *b2,
        T *gamma2,
        T *loginfo,
        const T * __restrict__ phi1,
        const T * __restrict__ phi2,
        const T * __restrict__ C,
        const T stepsize,
        const int nrows,
        const int ncols) {
    const int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset;

    __shared__ T phi2_shared[BLOCK_SIZE];
    T xval=0;
    T colsum=0;
    const T phi1_val=phi1[tidy];

    for (int idx=0; idx<WORK_SIZE; idx++) {
        offset = (blockIdx.x * WORK_SIZE + idx) * BLOCK_SIZE; // Column offset
        if (threadIdx.y + offset < ncols)
            phi2_shared[threadIdx.y] = phi2[threadIdx.y + offset];
        __syncthreads();

        if (tidy < nrows) {
            if (BLOCK_SIZE + offset <= ncols) {
                #pragma unroll
                for (int e=0; e<BLOCK_SIZE; e++) {
                    xval = X[tidy + (e + offset) * nrows];
                    xval = max(xval + phi1_val + phi2_shared[e], T(0));
                    X[tidy + (e + offset) * nrows] = xval;

                    colsum += xval;
                    xval = warp_reduce_sum(xval);
                    if (threadIdx.y % 32==0)
                        atomicAdd(&b2[e + offset], xval);
                }
            } else { // Edge blocks
                #pragma unroll
                for (int e=ncols-1; e>=offset; e--) {
                    xval = X[tidy + e * nrows];
                    xval = max(xval + phi1_val + phi2_shared[e-offset], T(0));
                    X[tidy + e * nrows] = xval;

                    colsum += xval;
                    xval = warp_reduce_sum(xval);
                    if (threadIdx.y % 32==0)
                        atomicAdd(&b2[e], xval);
                }
            }
        }
    }
    if (tidy < nrows) {
        atomicAdd(&a2[tidy], colsum);
        colsum = warp_reduce_sum(colsum);
        if (threadIdx.y % 32==0)
            atomicAdd(gamma2, colsum);
    }
    if (tidy>=blockDim.y*gridDim.y-5 && blockIdx.x==0)
        loginfo[blockDim.y*gridDim.y - tidy - 1] = T(0);
}

template<typename T>
__global__ void update_x_odd_(T *X,
        T *a2,
        T *b2,
        T *gamma2,
        T *loginfo,
        const T * __restrict__ phi1,
        const T * __restrict__ phi2,
        const T * __restrict__ C,
        const T stepsize,
        const int nrows,
        const int ncols) {
    const int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset;

    __shared__ T phi2_shared[BLOCK_SIZE];
    T xval=0;
    T cval=0;
    T fval=0;
    T colsum=0;
    const T phi1_val=phi1[tidy];

    for (int idx=0; idx<WORK_SIZE; idx++) {
        offset = (blockIdx.x * WORK_SIZE + idx) * BLOCK_SIZE; // Column offset
        if (threadIdx.y + offset < ncols)
            phi2_shared[threadIdx.y] = phi2[threadIdx.y + offset];
        __syncthreads();

        if (tidy < nrows) {
            if (BLOCK_SIZE + offset <= ncols) {
                #pragma unroll
                for (int e=0; e<BLOCK_SIZE; e++) {
                    xval = X[tidy + (e + offset) * nrows];
                    cval = C[tidy + (e + offset) * nrows];
                    xval = max(xval + phi1_val + phi2_shared[e]
                            - stepsize * cval, T(0));
                    X[tidy + (e + offset) * nrows] = xval - stepsize * cval;

                    colsum += xval;
                    fval += xval * cval;
                    xval = warp_reduce_sum(xval);
                    if (threadIdx.y % 32==0)
                        atomicAdd(&b2[e + offset], xval);
                }
            } else {
                #pragma unroll
                for (int e=ncols-1; e>=offset; e--) {
                    xval = X[tidy + e * nrows];
                    cval = C[tidy + e * nrows];
                    xval = max(xval + phi1_val + phi2_shared[e-offset]
                            - stepsize * cval, T(0));
                    X[tidy + e * nrows] = xval - stepsize * cval;

                    colsum += xval;
                    fval += xval * cval;
                    xval = warp_reduce_sum(xval);
                    if (threadIdx.y % 32==0)
                        atomicAdd(&b2[e], xval);
                }
            }
        }
    }
    if (tidy < nrows) {
        atomicAdd(&a2[tidy], colsum);
        colsum = warp_reduce_sum(colsum);
        fval = warp_reduce_sum(fval);
        if (threadIdx.y % 32==0) {
            atomicAdd(gamma2, colsum);
            atomicAdd(&loginfo[2], fval);
        }
    }
    if (tidy>=blockDim.y*gridDim.y-2 && blockIdx.x==0) {
        loginfo[blockDim.y*gridDim.y - tidy - 1] = T(0);
    }
}

template<typename T>
__global__ void update_auxs_even(T *a1,
        T *b1,
        T *a2,
        T *b2,
        T *gamma1,
        T *gamma2,
        T *loginfo,
        T *phi1,
        T *phi2,
        const T* __restrict__ p,
        const T* __restrict__ q,
        const int nrows,
        const int ncols) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    T val1=*gamma1; // this way gamma will not be modified when updating
    T val2=*gamma2; // gamma1 and gamma2 by tid=0 && blockIdx.y=0
    T gamma;

    if (tid < nrows && blockIdx.y==0) {
        gamma = (2*(val2 - T(1)) - val1) / (T) (nrows + ncols);
        val1 = a1[tid];
        val2 = a2[tid] - p[tid];
        phi1[tid] = (val1 - 2*val2 + gamma) / (T) ncols;
        a1[tid] = val1 - val2;

        a2[tid] = T(0); // Reset for the next reduction

        val2 *= val2;
        val2 = warp_reduce_sum(val2);
        if (threadIdx.x % 32==0)
            atomicAdd(&loginfo[0], val2);
    }
    if (tid < ncols && blockIdx.y==1) {
        gamma = (2*(val2 - T(1)) - val1) / (T) (nrows + ncols);
        val1 = b1[tid];
        val2 = b2[tid] - q[tid];
        phi2[tid] = (val1 - 2*val2 + gamma) / (T) nrows;
        b1[tid] = val1 - val2;

        b2[tid] = T(0);

        val2 *= val2;
        val2 = warp_reduce_sum(val2);
        if (threadIdx.x % 32==0)
            atomicAdd(&loginfo[1], val2);
    }
    if (tid==31 && blockIdx.y==0) {
        *gamma1 += - *gamma2 + T(1);
        *gamma2 = T(0);
    }
}

template<typename T>
__global__ void update_auxs_odd(T *a1,
        T *b1,
        T *a2,
        T *b2,
        T *gamma1,
        T *gamma2,
        T *loginfo,
        T *phi1,
        T *phi2,
        const T* __restrict__ p,
        const T* __restrict__ q,
        const int nrows,
        const int ncols) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    T val1=*gamma1;
    T val2=*gamma2;
    T dval;
    T gamma;

    if (tid < nrows && blockIdx.y==0) {
        gamma = (2*(val2 - T(1)) - val1) / (T) (nrows + ncols);
        val1 = a1[tid];
        dval = p[tid];
        val2 = a2[tid] - dval;
        a1[tid] = val1 - val2;
        val1 = (val1 - 2*val2 + gamma) / (T) ncols;
        phi1[tid] = val1;

        a2[tid] = T(0); // Reset for the next reduction

        dval *= val1;
        val2 *= val2;
        val2 = warp_reduce_sum(val2);
        dval = warp_reduce_sum(dval);
        if (threadIdx.x % 32==0) {
            atomicAdd(&loginfo[0], val2);
            atomicAdd(&loginfo[3], dval);
        }
    }
    if (tid < ncols && blockIdx.y==1) {
        gamma = (2*(val2 - T(1)) - val1) / (T) (nrows + ncols);
        val1 = b1[tid];
        dval = q[tid];
        val2 = b2[tid] - dval;
        b1[tid] = val1 - val2;
        val1 = (val1 - 2*val2 + gamma) / (T) nrows;
        phi2[tid] = val1;

        b2[tid] = T(0);

        dval *= val1;
        val2 *= val2;
        val2 = warp_reduce_sum(val2);
        dval = warp_reduce_sum(dval);
        if (threadIdx.x % 32==0) {
            atomicAdd(&loginfo[1], val2);
            atomicAdd(&loginfo[4], dval);
        }
    }
    if (tid==31 && blockIdx.y==0) {
        *gamma1 += - *gamma2 + T(1);
        *gamma2 = T(0);
    }
}

template<typename T>
__global__ void zeros(T *X, const size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < n) {
        X[tid] = T(0);
        tid += gridDim.x * blockDim.x;
    }
}

template<typename T>
__global__ void init_auxs(T *a1,
        T *b1,
        T *a2,
        T *b2,
        T *phi1,
        T *phi2,
        T *common,
        const int nrows,
        const int ncols) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < nrows && blockIdx.y==0) {
        a1[tid] = T(0);
        a2[tid] = T(0);
        phi1[tid] = T(0);
    }
    if (tid < ncols && blockIdx.y==1) {
        b1[tid] = T(0);
        b2[tid] = T(0);
        phi2[tid] = T(0);
        if (tid < 16)
            common[tid] = T(0);
    }
}

template<typename T>
__global__ void init_x(T *X,
        const T* __restrict__ C,
        const T* __restrict__ p,
        const T* __restrict__ q,
        const T stepsize,
        const int nrows,
        const int ncols) {
    const int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset;
    __shared__ T q_shared[BLOCK_SIZE];
    const T pval=p[tidy];

    for (int idx=0; idx<WORK_SIZE; idx++) {
        offset = (blockIdx.x * WORK_SIZE + idx) * BLOCK_SIZE;
        if (threadIdx.y + offset < ncols)
            q_shared[threadIdx.y] = q[threadIdx.y + offset];
        __syncthreads();

        if (tidy < nrows) {
            if (BLOCK_SIZE + offset <= ncols) {
                #pragma unroll
                for (int e=0; e<BLOCK_SIZE; e++)
                    X[tidy + (e + offset) * nrows] = pval * q_shared[e]
                        - stepsize * C[tidy + (e + offset) * nrows];
            } else { // Edge blocks
                #pragma unroll
                for (int e=ncols-1; e>=offset; e--)
                    X[tidy + e * nrows] = pval * q_shared[e - offset]
                        - stepsize * C[tidy + e * nrows];
            }
        }
    }
}
#endif
