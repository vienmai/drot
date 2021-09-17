#include <vector>
#include <string>
#include "drot.hpp"
#include "reader.hpp"

template<typename T>
void multi_experiment(const int nrows,
        const int ncols,
        const int maxiters,
        const T eps,
        const T alpha=1.0,
        const int ntests=50) {
    std::vector<T> out;
    out.reserve(ntests);
    std::string filename;

    std::vector<T> p(nrows, 1/(T) nrows);
    std::vector<T> q(ncols, 1/(T) ncols);

    for (int idx=0; idx<ntests; idx++) {
        std::cout << "\n *** Experiment " << idx+1 << "/" << ntests << " ***\n";
        filename = "examples/data/cmatrix_" + std::to_string(nrows) + "_test_" +
            std::to_string(idx);
        auto C = utility::load<T>(filename, nrows, ncols);
        const T stepsize = alpha/(T) (nrows + ncols);
        auto fval = drot<T>(&C[0], &p[0], &q[0], nrows, ncols, stepsize,
                maxiters, eps);
        out.push_back(fval);
    }
    const std::string filename_ = "examples/output/drot_"
        + std::to_string(nrows) + "_ntests_" + std::to_string(ntests) + ".csv";
    utility::csv<T>(filename_, out);
}

int main(int argc, char *argv[]) {
    const int nrows = 500;
    const int ncols = 500;
    const int maxiters = 1000;
    const int ntests = 50;
    const float eps = 1E-4;

    multi_experiment<float>(nrows, ncols, maxiters, eps, 2.0f, ntests);

    return 0;
}
