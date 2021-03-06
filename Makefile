################################################################################
### Makefile template, based on the following blog:
###  http://hiltmon.com/blog/2013/07/03/a-simple-c-plus-plus-project-structure
################################################################################

CC := g++
CFLAGS := -g -Wall -fpic -O2 -Wno-unused-function

NVCC := nvcc
NVCCFLAGS := -ccbin=$(CC) $(foreach FLAG, $(CFLAGS), -Xcompiler $(FLAG))
CUDA_ARCH := -gencode arch=compute_35,code=sm_35 \
             -gencode arch=compute_50,code=sm_50 \
             -gencode arch=compute_52,code=sm_52 \
             -gencode arch=compute_61,code=sm_61

INCDIR := include
SRCDIR := src
TOOLSDIR := tools
BUILDDIR := build
PROJECT := checker

INCLUDE := $(INCDIR) /usr/local/cuda/include /usr/local/cuda/samples/common/inc
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
TGT_SRC := $(BUILDDIR)/lib/lib$(PROJECT).so

### $(SRCDIR)包含所有的库cu
SRC_SRC_CU := $(shell find $(SRCDIR) -type f -name *.cu)
OBJ_SRC_CU := $(addprefix $(BUILDDIR)/, ${SRC_SRC_CU:.cu=.cuo})

### ${TOOLSDIR}包含所有的有main函数的cu
SRC_TOOLS := $(shell find $(TOOLSDIR) -type f -name *.cu)
OBJ_TOOLS := $(addprefix $(BUILDDIR)/, ${SRC_TOOLS:.cu=.cuo})
TGT_TOOLS := $(addprefix $(BUILDDIR)/, ${SRC_TOOLS:.cu=.cubin})

### 所有与build相关的目录
ALL_BUILD_DIRS := $(sort $(dir $(OBJ_SRC) $(OBJ_SRC_CU) $(TGT_SRC) $(TGT_TOOLS)))

all: $(TGT_SRC) $(TGT_TOOLS)

$(TGT_SRC): $(OBJ_SRC) $(OBJ_SRC_CU)
	$(CC) -shared -o $@ $^ $(LIBRARY) $(LIBS)

$(TGT_TOOLS): %.cubin : %.cuo $(OBJ_SRC) $(OBJ_SRC_CU)
	$(CC) -o $@ $^ $(LIBRARY) $(LIBS)

$(OBJ_SRC): $(BUILDDIR)/%.o : %.cpp $(HEADERS) | $(ALL_BUILD_DIRS)
	$(CC) $(CFLAGS) -c -o $@ $< $(INCLUDE)

$(OBJ_SRC_CU): $(BUILDDIR)/%.cuo : %.cu $(HEADERS) | $(ALL_BUILD_DIRS)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -c -o $@ $< $(INCLUDE)

$(OBJ_TOOLS): $(BUILDDIR)/%.cuo : %.cu $(HEADERS) | $(ALL_BUILD_DIRS)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -c -o $@ $< $(INCLUDE)

$(ALL_BUILD_DIRS):
	@ mkdir -p $@

clean:
	rm -rf $(BUILDDIR)

.PHONY: clean
