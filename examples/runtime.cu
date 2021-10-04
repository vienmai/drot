#include <vector>
#include <string>
#include "drot.hpp"
#include "reader.hpp"

int main(int argc, char *argv[]) {
    const std::vector<int> dims{100, 200, 500, 1000, 2000, 5000, 10000, 20000};
    const int maxiters = 100;
    const int ntests = 10;
    const float eps = 1E-4;
    std::string filename, filename_;

    for (const auto &nrows: dims) {
        auto ncols = nrows;
        filename = "examples/data/cmatrix_" + std::to_string(nrows);
        auto C = utility::load<float>(filename, nrows, ncols);
        std::vector<float> p(nrows, 1/(float) nrows);
        std::vector<float> q(ncols, 1/(float) ncols);
        const float stepsize = 1/(float) (nrows + ncols);

        for (int idx=0; idx<ntests; idx++) {
            filename_ = "examples/output/drot_runtime_" + std::to_string(nrows)
                + "_test_" + std::to_string(idx) + ".csv";
            drot<float>(&C[0], &p[0], &q[0], nrows, ncols, stepsize,
                maxiters, eps, false, true, filename_);
        }
    }
    return 0;
}
