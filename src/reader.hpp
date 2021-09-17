#ifndef UTILITY_READER_HPP_
#define UTILITY_READER_HPP_

#include <fstream>
#include <vector>

namespace utility {
template <typename T>
std::vector<T> load(const std::string &filename,
        const int nrows, const int ncols) {
    std::ifstream ifs(filename, std::ifstream::binary);
    std::vector<T> values(std::size_t(nrows) * ncols);
    std::vector<T> out(std::size_t(nrows) * ncols);
    ifs.read(reinterpret_cast<char *>(&values[0]),
        std::size_t(nrows) * ncols * sizeof(T));
    ifs.close();

    for(int col=0; col<ncols; col++) {
        for(int row=0; row<nrows; row++) {
            out[col * nrows + row] = values[row * ncols + col];
        }
    }
    return out;
}
template <typename T>
void csv(const std::string &filename, const std::vector<T> &values) {
    std::cout << filename << '\n';
    std::ofstream file(filename);
    if (file) {
            file << "f\n";
            for (const T &val : values)
                    file << std::fixed << val << '\n';
    } else {
        std::cout << "File not found\n";
        exit(1);
    }
}

template <typename T>
void csv(const std::string &filename,
        const std::vector<int> &iterations,
        const std::vector<double> &times,
        const std::vector<T> &residuals,
        const std::vector<T> &objectives) {
    std::cout << filename << '\n';
    std::ofstream file(filename);
    if (file) {
            file << "k,t,r\n";
            for (int i=0; i< iterations.size(); i++)
                file << std::fixed << iterations[i] << ','
                     << times[i] << ',' << residuals[i] << ','
                     << objectives[i] << '\n';
    } else {
        std::cout << "File not found\n";
        exit(1);
    }
}
};
#endif
