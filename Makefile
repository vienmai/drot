CXX=nvcc
CXXFLAGS=-Icuda_helper -O3 --gpu-architecture=compute_35 --gpu-code=sm_35

SRCDIR := examples
SRC := $(wildcard $(SRCDIR)/*.cu)
EXE := $(addprefix bin/, $(notdir $(patsubst %.cu,%,$(filter %.cu,$(SRC)))))
INC := -I src/

.PHONY: all
all: $(EXE)

# Pattern rules
$(EXE) : bin/% : $(SRCDIR)/%.cu
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $<

.PHONY: clean
clean:
	$(RM) $(EXE)
