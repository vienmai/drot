# DROT
Douglas-Rachford Splitting for Optimal Transport

To run experiments on the T7600's GPU:

1. Make sure the cost matrix `C` has been generated; we can do so by runing the Python notebook in `examples/`. The previous step should generate a `cmatrix` in `examples/data/`. 

2. Check if the dimensions of `C` in `examples/test.cu` are correct. If so, execute the following commands:
```bash
$ mkdir bin
$ make
```
- For a single run:
```bash
  $ ./bin/test
```
- For performance profile:
```bash
$ ./bin/multiexp
```
3. Play with the notebook 


There is also the `pydrot` folder that contains the `CPU` code for sanity check. It has not been carefully optimized for performance.
