################################################################################
### Makefile template, based on the following blog:
###  http://hiltmon.com/blog/2013/07/03/a-simple-c-plus-plus-project-structure
################################################################################

CC := g++
CFLAGS := -g -Wall -fpic -O2

NVCC := nvcc
NVCCFLAGS := -ccbin=$(CC) $(foreach FLAG, $(CFLAGS), -Xcompiler $(FLAG))
CUDA_ARCH := -gencode arch=compute_35,code=sm_35

INCDIR := include
SRCDIR := src
TOOLSDIR := tools
BUILDDIR := build

INCLUDE := $(INCDIR) /usr/local/cuda/include
LIBRARY := /usr/local/cuda/lib64
LIBS := dl m z rt glog cudart cublas curand

INCLUDE := $(foreach INC, $(INCLUDE), -I $(INC))
LIBRARY := $(foreach LIB, $(LIBRARY), -L $(LIB))
LIBS := $(foreach LIB, $(LIBS), -l$(LIB))

### all header files
HEADERS := $(shell find $(INCDIR) -type f -name *.h)

### $(SRCDIR)包含所有的库cpp
SRC_SRC := $(shell find $(SRCDIR) -type f -name *.cpp)
OBJ_SRC := $(addprefix $(BUILDDIR)/, ${SRC_SRC:.cpp=.o})

### ${TOOLSDIR}包含所有的有main函数的cpp
SRC_TOOLS := $(shell find $(TOOLSDIR) -type f -name *.cu)
OBJ_TOOLS := $(addprefix $(BUILDDIR)/, ${SRC_TOOLS:.cu=.cuo})
TGT_TOOLS := $(addprefix $(BUILDDIR)/, ${SRC_TOOLS:.cu=.cubin})

### 所有与build相关的目录
ALL_BUILD_DIRS := $(sort $(dir $(OBJ_SRC) $(TGT_TOOLS)))

all: $(TGT_TOOLS)

$(TGT_TOOLS): %.cubin : %.cuo $(OBJ_SRC)
	$(CC) -o $@ $^ $(LIBRARY) $(LIBS)

$(OBJ_SRC): $(BUILDDIR)/%.o : %.cpp $(HEADERS) | $(ALL_BUILD_DIRS)
	$(CC) $(CFLAGS) -c -o $@ $< $(INCLUDE)

$(OBJ_TOOLS): $(BUILDDIR)/%.cuo : %.cu $(HEADERS) | $(ALL_BUILD_DIRS)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -c -o $@ $< $(INCLUDE)

$(ALL_BUILD_DIRS):
	@ mkdir -p $@

clean:
	rm -rf $(BUILDDIR)

.PHONY: clean
