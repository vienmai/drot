#ifndef DROT_HPP_
#define DROT_HPP_

#include <math.h>
#include <cuda_runtime.h>
#include <vector>
#include "kernels.hpp"
#include "timer.hpp"
#include "reader.hpp"

template<typename T> void step(T *X,
        T *a1,
        T *b1,
        T *a2,
        T *b2,
        T *gamma1,
        T *gamma2,
        T *phi1,
        T *phi2,
        T *normsq,
        const T *C,
        const T *p,
        const T *q,
        const T stepsize,
        const int nrows,
        const int ncols,
        const int iteration);

template<typename T> T drot(const T *C,
        const T *p,
        const T *q,
        const int nrows,
        const int ncols,
        const T stepsize,
        const int maxiters,
        const T eps,
        const bool verbose=true,
        const bool log=false,
        const std::string &filename={}) {
    const size_t matsize = nrows*ncols*sizeof(T);
    const size_t rowsize = nrows*sizeof(T);
    const size_t colsize = ncols*sizeof(T);

    // Initilizing
    T *dC;
    T *dp;
    T *dq;
    T *dX;
    T *da1;
    T *da2;
    T *db1;
    T *db2;
    T *dphi1;
    T *dphi2;
    T *dcommon; // Loginfo and auxiuliary scalars

    cudaMalloc((void**)&dC, matsize);
    cudaMalloc((void**)&dp, rowsize);
    cudaMalloc((void**)&dq, colsize);
    cudaMemcpy(dC, &C[0], matsize, cudaMemcpyHostToDevice);
    cudaMemcpy(dp, &p[0], rowsize, cudaMemcpyHostToDevice);
    cudaMemcpy(dq, &q[0], colsize, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dX, matsize);
    cudaMalloc((void**)&da1, rowsize);
    cudaMalloc((void**)&da2, rowsize);
    cudaMalloc((void**)&db1, colsize);
    cudaMalloc((void**)&db2, colsize);
    cudaMalloc((void**)&dphi1, rowsize);
    cudaMalloc((void**)&dphi2, colsize);
    cudaMalloc((void**)&dcommon, 16*sizeof(T));

    dim3 grid_auxs((max(nrows, ncols) + BLOCK_SIZE - 1) / BLOCK_SIZE, 2);
    dim3 block_auxs(BLOCK_SIZE);

    zeros<T> <<<1024, 256>>>(dX, nrows*ncols); // Much faster than creating
    init_auxs<T> <<<grid_auxs, block_auxs>>>(da1, db1, da2, db2, // and
            dphi1, dphi2, dcommon, nrows, ncols); // copying from CPU

    dim3 grid_x((ncols + BLOCK_SIZE * WORK_SIZE - 1) / (BLOCK_SIZE * WORK_SIZE),
            (nrows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block_x(1, BLOCK_SIZE);

    init_x<T> <<<block_x, grid_x>>>(dX, dC, dp, dq, stepsize, nrows, ncols);
    cudaDeviceSynchronize();

    // Optimizing
    int k = 0;
    bool done = false;
    T loginfo[5];
    T res = 0;
    T fval = 0;

    utility::timer t;
    double tend;
    std::vector<int> iterations;
    std::vector<double> times;
    std::vector<T> residuals;
    std::vector<T> objectives;
    if (log) {
        iterations.reserve(maxiters + 1); // No implicit copies
        times.reserve(maxiters + 1);      // when calling push_back
        residuals.reserve(maxiters + 1);
        objectives.reserve(maxiters + 1);
    }
    if (verbose)
        printf("%8s %10s %10s\n", "Iter", "Time", "Residual");
    while ((!done) && (k < maxiters)) {
        step(dX, da1, db1, da2, db2, &dcommon[14], &dcommon[15], dphi1, dphi2,
               &dcommon[0], dC, dp, dq, stepsize, nrows, ncols, k);

        cudaMemcpy(&loginfo[0], &dcommon[0], 5*sizeof(T), cudaMemcpyDeviceToHost);
        res = sqrt(loginfo[0] + loginfo[1]);
        fval = k % 2 == 0 ? fval: loginfo[2];
        //fval = k % 2 == 0 ? fval: abs(loginfo[2] - (loginfo[3] + loginfo[4]) / stepsize);
        done = res <= eps ? true : false;
        k++;

        if (log) {
            tend = t.elapsed();
            iterations.push_back(k);
            times.push_back(tend);
            residuals.push_back(res);
            objectives.push_back(fval);
        }
        if (verbose)
            printf("%8d %10.5f %10.5f \n", k, tend, res);
    }

    // Saving
    if (log)
        utility::csv<T>(filename, iterations, times, residuals, objectives);

    // Cleaning
    cudaFree(dC);
    cudaFree(dp);
    cudaFree(dq);
    cudaFree(dX);
    cudaFree(da1);
    cudaFree(da2);
    cudaFree(db1);
    cudaFree(db2);
    cudaFree(dphi1);
    cudaFree(dphi2);
    cudaFree(dcommon);

    return fval;
}

template<typename T> void step(T *X,
        T *a1,
        T *b1,
        T *a2,
        T *b2,
        T *gamma1,
        T *gamma2,
        T *phi1,
        T *phi2,
        T *normsq,
        const T *C,
        const T *p,
        const T *q,
        const T stepsize,
        const int nrows,
        const int ncols,
        const int iteration) {

    dim3 grid_x((ncols + BLOCK_SIZE * WORK_SIZE - 1) / (BLOCK_SIZE * WORK_SIZE),
            (nrows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block_x(1, BLOCK_SIZE);
    dim3 grid_auxs((max(nrows, ncols) + BLOCK_SIZE - 1) / BLOCK_SIZE, 2);
    dim3 block_auxs(BLOCK_SIZE);

    if (iteration % 2 == 0) {
        update_x_even_<float> <<<grid_x, block_x>>>(X, a2, b2, gamma2, normsq,
                phi1, phi2, C, stepsize, nrows, ncols);
        update_auxs_even<float> <<<grid_auxs, block_auxs>>>(a1, b1, a2, b2, gamma1, gamma2,
            normsq, phi1, phi2, p, q,  nrows, ncols);
    } else {
        update_x_odd_<float> <<<grid_x, block_x>>>(X, a2, b2, gamma2, normsq,
                phi1, phi2, C, stepsize, nrows, ncols);
        update_auxs_odd<float> <<<grid_auxs, block_auxs>>>(a1, b1, a2, b2, gamma1, gamma2,
            normsq, phi1, phi2, p, q,  nrows, ncols);
    }
}
#endif
