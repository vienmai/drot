#include <vector>
#include <string>
#include "drot.hpp"
#include "reader.hpp"

int main(int argc, char *argv[]) {
    const int nrows = 512;
    const int ncols = 512;

    auto C = utility::load<float>("examples/data/cmatrix", nrows, ncols);
    std::vector<float> p(nrows, 1/(float) nrows);
    std::vector<float> q(ncols, 1/(float) ncols);

    const float stepsize = 2/(float) (nrows + ncols);
    const int maxiters = 500;
    const float eps = 0.0f;

    const std::string filename = "examples/output/drot_"
        + std::to_string(nrows) + "_" + std::to_string(stepsize) + ".csv";
    const bool verbose=true;
    const bool log=true;
    drot(&C[0], &p[0], &q[0], nrows, ncols, stepsize, maxiters, eps, verbose,
            log, filename);

    return 0;
}
