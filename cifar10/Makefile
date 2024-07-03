# Project: MNIST with GA
# Makefile for Linux or Windows with mingw gcc
# -DNO_ASSERT (same as -DNDEBUG): disables assertation checks
# -D__USE_OPENMP__: combined with -fopenmp, enables OpenMP support

ifeq ($(OS),Windows_NT)
	detected_OS := Windows
else
	detected_OS := $(shell uname -s)
endif

GCC           = gcc
CUDA          = False
NVCC          = nvcc

CIFAR10CNN    = cifar10cnn
CIFAR10GABP   = cifar10gabp
CIFAR10KD     = cifar10kd
CIFAR10KDGA   = cifar10kdga
CIFAR10PERMUT = cifar10permut
SAVELOGITS    = savelogits

OBJDIR        = obj
INCDIR        = ../umml
DATADIR       = $(INCDIR)/datasets
CUDALIB       = /usr/local/cuda/lib64

DATA_INCLUDES = $(DATADIR)/tinysets.hpp $(DATADIR)/clfsets.hpp $(DATADIR)/regrsets.hpp $(DATADIR)/timeseries.hpp \
				$(DATADIR)/breastcancer.hpp $(DATADIR)/iris.hpp $(DATADIR)/heartdisease.hpp \
				$(DATADIR)/binfile.hpp $(DATADIR)/csvloader.hpp \
				$(DATADIR)/mnistloader.hpp $(DATADIR)/cifarloader.hpp $(DATADIR)/postersloader.hpp

ML_INCLUDES   = $(INCDIR)/eigen.hpp $(INCDIR)/pca.hpp $(INCDIR)/kdtree.hpp $(INCDIR)/knn.hpp \
				$(INCDIR)/kmeans.hpp $(INCDIR)/clustermap.hpp $(INCDIR)/multiclass.hpp

NN_INCLUDES   = $(INCDIR)/nn/layers/input.hpp $(INCDIR)/nn/layers/dense.hpp $(INCDIR)/nn/layers/softmax.hpp \
				$(INCDIR)/nn/layers/maxpool2d.hpp $(INCDIR)/nn/layers/conv2d.hpp $(INCDIR)/nn/layers/dropout.hpp \
				$(INCDIR)/nn/op/maxpool.hpp $(INCDIR)/nn/op/conv.hpp \
				$(INCDIR)/nn/op/momentum.hpp $(INCDIR)/nn/op/adam.hpp \
				$(INCDIR)/nn/layers.hpp $(INCDIR)/nn/ffnn.hpp \
				$(INCDIR)/nn/gdstep.hpp $(INCDIR)/nn/lossfunc.hpp $(INCDIR)/nn/backprop.hpp

COM_INCLUDES  =	$(INCDIR)/compiler.hpp $(INCDIR)/types.hpp $(INCDIR)/algo.hpp $(INCDIR)/cpuinfo.hpp \
				$(INCDIR)/rand.hpp $(INCDIR)/utils.hpp $(INCDIR)/dev.hpp \
				$(INCDIR)/uvec.hpp $(INCDIR)/umat.hpp $(INCDIR)/ucub.hpp $(INCDIR)/utensor.hpp \
				$(INCDIR)/func.hpp $(INCDIR)/preproc.hpp $(INCDIR)/metrics.hpp $(INCDIR)/stats.hpp \
				$(INCDIR)/dataframe.hpp $(INCDIR)/splitter.hpp $(INCDIR)/glplot.hpp \
				$(ML_INCLUDES) $(NN_INCLUDES)

CPU_INCLUDES  = $(INCDIR)/blas_cpu.hpp $(INCDIR)/kernels_cpu.hpp \
				$(INCDIR)/nn/op/maxpool_cpu.hpp $(INCDIR)/nn/op/conv_cpu.hpp

OCL_INCLUDES  = $(INCDIR)/ocl.hpp $(INCDIR)/blas_ocl.hpp $(INCDIR)/kernels_ocl.hpp \
				$(INCDIR)/nn/op/maxpool_ocl.hpp $(INCDIR)/nn/op/conv_ocl.hpp

CUDA_INCLUDES =	$(COM_INCLUDES) $(INCDIR)/cuda.hpp $(INCDIR)/blas_cuda.hpp $(INCDIR)/kernels_cuda.hpp \
				$(INCDIR)/nn/op/maxpool_cuda.hpp $(INCDIR)/nn/op/conv_cuda.hpp

BIO_INCLUDES  = $(INCDIR)/bio/initializer.hpp $(INCDIR)/bio/crossover.hpp $(INCDIR)/bio/mutation.hpp \
				$(INCDIR)/bio/ga.hpp $(INCDIR)/bio/bitstr.hpp $(INCDIR)/bio/symvol.hpp $(INCDIR)/bio/pso.hpp

INCLUDES      = $(COM_INCLUDES) $(CPU_INCLUDES) $(OCL_INCLUDES) $(BIO_INCLUDES) $(DATA_INCLUDES)
			
#SOURCES       = $(CIFAR10CNN).cpp $(CIFAR10GA).cpp $(SAVELOGITS).cpp
			
#OBJECTS   = $(SOURCES:%.cpp=$(OBJDIR)/%.o)

ifeq ($(detected_OS),Windows)
	EXECUTABLES = $(CIFAR10CNN).exe $(CIFAR10GABP).exe $(CIFAR10KD).exe $(CIFAR10KDGA).exe $(CIFAR10PERMUT).exe $(SAVELOGITS).exe
else
	EXECUTABLES = $(CIFAR10CNN) $(CIFAR10GABP) $(CIFAR10KD) $(CIFAR10KDGA) $(CIFAR10PERMUT) $(SAVELOGITS)
endif			

NVCCFLAGS = -std=c++11 -c -arch=sm4_60
#ARCHFLAGS = -msse3 -mssse3 -msse4.1 -mavx -mavx2
ARCHFLAGS = -msse2
CXXFLAGS += -O3 $(ARCHFLAGS) -std=c++11 -fomit-frame-pointer -Wall -Wno-unknown-pragmas -s -c -I$(INCDIR)
# -DNO_ASSERT
#-Wfatal-errors -fno-exceptions
# -O3 -DNDEBUG -mavx2

ifeq ($(detected_OS),Windows)
	#DINCS = -IC:/msys/mingw64/include 
	#DLIBS = -LC:/msys/mingw64/lib
	DINCS = -IC:/usr/local/include
	DLIBS = -LC:/usr/local/lib
	CXXFLAGS := $(DINCS) $(CXXFLAGS) -D_WIN32
	GLIBS  = $(DLIBS) -s -lopengl32 -lglew32 -lfreeglut -lglu32 -lstdc++ -lm
	LIBS  = -s -lstdc++ -lm
else
	GLIBS  = -s -lglut -lGLEW -lGLU -lGL -lX11 -ldl -lstdc++ -lm
	LIBS  = -s -ldl -lstdc++ -lm
endif

ifeq ($(findstring openmp,$(MAKECMDGOALS)),openmp)
	CXXFLAGS += -fopenmp -D__USE_OPENMP__
	LIBS += -fopenmp
	GLIBS += -fopenmp
endif

ifeq ($(findstring blas,$(MAKECMDGOALS)),blas)
	CXXFLAGS += -D__USE_BLAS__
	LIBS += -lopenblas
	GLIBS += -lopenblas
endif

ifeq ($(findstring opencl,$(MAKECMDGOALS)),opencl)
	CXXFLAGS += -D__USE_OPENCL__
	LIBS += -lOpenCL
endif


.PHONY: all help everything openmp opencl blas clean

all: help

help:
	@echo ""
	@echo "make target [blas] [openmp] [opencl]"
	@echo ""
	@echo "Targets"
	@echo "everything : builds everything"
	@echo "clean      : clean object files and executabe files"
	@echo ""
	@echo "blas   - enables openblas matrix multiplication."
	@echo "openmp - enables openmp parallelism."
	@echo "opencl - enables opencl GPU parallelism."
	@echo ""
	@echo "Individual targets:"
	@echo " cifar10cnn     : CIFAR-10, train a CNN with backpropagation."
	@echo " cifar10gabp    : CIFAR-10, train with a hybrid BP+GA."
	@echo " cifar10kd      : CIFAR-10, knowledge distillation (KD)."
	@echo " cifar10kdga    : CIFAR-10, knowledge distillation (KD) with a GA."
	@echo " cifar10permut  : CIFAR-10, SGD replacement with a GA."
	@echo " savelogits     : CIFAR-10, save teacher logits to a bin file."

blas:
	@echo "Built with openblas matrix multiplication enabled."

openmp:
	@echo "Built with openmp parallelism enabled."

opencl:
	@echo "Built with OpenCL GPU support enabled."


everything: $(CIFAR10CNN) $(CIFAR10GABP) $(CIFAR10KD) $(CIFAR10KDGA) $(CIFAR10PERMUT) $(SAVELOGITS)

$(OBJDIR)/%.o: %.cu $(CUDA_INCLUDES)
	$(NVCC) $(NVCCFLAGS) -D__USE_CUDA__ $< -o $@

$(OBJDIR)/%.o: %.cpp $(INCLUDES)
	$(GCC) $(CXXFLAGS) $< -o $@

$(CIFAR10CNN): $(OBJDIR)/$(CIFAR10CNN).o
	$(GCC) $^ -o $@ $(LIBS)

$(CIFAR10GABP): $(OBJDIR)/$(CIFAR10GABP).o
	$(GCC) $^ -o $@ $(LIBS)

$(CIFAR10KD): $(OBJDIR)/$(CIFAR10KD).o
	$(GCC) $^ -o $@ $(LIBS)

$(CIFAR10KDGA): $(OBJDIR)/$(CIFAR10KDGA).o
	$(GCC) $^ -o $@ $(LIBS)

$(CIFAR10PERMUT): $(OBJDIR)/$(CIFAR10PERMUT).o
	$(GCC) $^ -o $@ $(LIBS)

$(SAVELOGITS): $(OBJDIR)/$(SAVELOGITS).o
	$(GCC) $^ -o $@ $(LIBS)

clean:
	rm -f $(EXECUTABLES)
	rm -f $(OBJDIR)/*.o
