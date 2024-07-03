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

MNISTCNN      = mnistcnn
MNISTCNN-CUDA = mnistcnn-cuda
MNISTKD       = mnistkd
MNISTKNN      = mnistknn
MNISTGA       = mnistga
MNISTSV       = mnistsv
MNISTRARW     = mnistrarw
MNISTPSO      = mnistpso
MNISTBPGA     = mnistbpga
MNISTPSOGA    = mnistpsoga
MNISTKDGA     = mnistkdga
MNISTKDPSO    = mnistkdpso
MNISTKDMERGE  = mnistkdmerge
MNISTPERMUT   = mnistpermut
MNISTCLUSTER  = mnistcluster
MNISTPCA      = mnistpca
MODMNIST      = modmnist

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
				$(INCDIR)/nn/gdstep.hpp  $(INCDIR)/nn/lossfunc.hpp $(INCDIR)/nn/backprop.hpp

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
			
#SOURCES       = $(MNISTCNN).cpp $(MNISTKD).cpp $(MNISTKNN).cpp $(MNISTGA).cpp $(MODMNIST).cpp
#OBJECTS   = $(SOURCES:%.cpp=$(OBJDIR)/%.o)

ifeq ($(detected_OS),Windows)
	EXECUTABLES = $(MNISTCNN).exe $(MNISTCNN-CUDA).exe $(MNISTKD).exe $(MNISTKNN).exe $(MNISTBPGA).exe $(MNISTGAPSO).exe \
				  $(MNISTGA).exe $(MNISTSV).exe $(MNISTRARW).exe $(MNISTPSO).exe \
				  $(MNISTKDGA).exe $(MNISTKDPSO).exe $(MNISTKDMERGE).exe \
				  $(MNISTPERMUT).exe $(MNISTCLUSTER).exe $(MNISTPCA).exe $(MODMNIST).exe
else
	EXECUTABLES = $(MNISTCNN) $(MNISTCNN-CUDA) $(MNISTKD) $(MNISTKNN) $(MNISTBPGA) $(MNISTGAPSO) \
				  $(MNISTGA) $(MNISTSV) $(MNISTRARW) $(MNISTPSO) \
				  $(MNISTKDGA) $(MNISTKDPSO) $(MNISTKDMERGE) \
				  $(MNISTPERMUT) $(MNISTCLUSTER) $(MNISTPCA) $(MODMNIST)
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
	@echo " mnistcnn       : MNIST (digits), train a CNN with backpropagation."
	@echo " mnistcnn-cuda  : MNIST (digits), train a CNN with backpropagation. (CUDA)"
	@echo " mnistkd        : MNIST (digits), knowledge distillation."
	@echo " mnistknn       : MNIST (digits), KNN and KMeans."
	@echo " mnistga        : MNIST (digits), outlier detection with GA."
	@echo " mnistsv        : MNIST (digits), outlier detection with GA (Symvol)."
	@echo " mnistrarw      : MNIST (digits), outlier detection with GA (RARw)."
	@echo " mnistpso       : MNIST (digits), outlier detection with PSO (RARw)."
	@echo " mnistbpga      : MNIST (digits), BP+GA hybrid training."
	@echo " mnistpsoga     : MNIST (digits), PSO+GA hybrid training."
	@echo " mnistkdga      : MNIST (digits), knowledge distillation with a GA (choose samples)."
	@echo " mnistkdpso     : MNIST (digits), knowledge distillation with PSO (choose samples)."
	@echo " mnistkdmerge   : MNIST (digits), knowledge distillation with a GA (filter merging)."
	@echo " mnistpermut    : MNIST (digits), SGD replacement with a GA."
	@echo " mnistcluster   : MNIST (digits), in-class clustering."
	@echo " mnistpca       : MNIST (digits), clustering and PCA graph."
	@echo " modmnist       : Modifies MNIST dataset."

blas:
	@echo "Built with openblas matrix multiplication enabled."

openmp:
	@echo "Built with openmp parallelism enabled."

opencl:
	@echo "Built with OpenCL GPU support enabled."


everything: $(EXECUTABLES)

$(OBJDIR)/%.o: %.cu $(CUDA_INCLUDES)
	$(NVCC) $(NVCCFLAGS) -D__USE_CUDA__ $< -o $@

$(OBJDIR)/%.o: %.cpp $(INCLUDES)
	$(GCC) $(CXXFLAGS) $< -o $@

$(MNISTCNN): $(OBJDIR)/$(MNISTCNN).o
	$(GCC) $^ -o $@ $(LIBS)

$(MNISTCNN-CUDA): $(OBJDIR)/$(MNISTCNN-CUDA).o
	$(GCC) $^ -o $@ -L$(CUDALIB) $(LIBS) -lcuda -lcudart

$(MNISTKD): $(OBJDIR)/$(MNISTKD).o
	$(GCC) $^ -o $@ $(LIBS)

$(MNISTKNN): $(OBJDIR)/$(MNISTKNN).o
	$(GCC) $^ -o $@ $(GLIBS)

$(MNISTGA): $(OBJDIR)/$(MNISTGA).o
	$(GCC) $^ -o $@ $(GLIBS)

$(MNISTSV): $(OBJDIR)/$(MNISTSV).o
	$(GCC) $^ -o $@ $(GLIBS)

$(MNISTRARW): $(OBJDIR)/$(MNISTRARW).o
	$(GCC) $^ -o $@ $(GLIBS)

$(MNISTPSO): $(OBJDIR)/$(MNISTPSO).o
	$(GCC) $^ -o $@ $(GLIBS)

$(MNISTBPGA): $(OBJDIR)/$(MNISTBPGA).o
	$(GCC) $^ -o $@ $(LIBS)

$(MNISTPSOGA): $(OBJDIR)/$(MNISTPSOGA).o
	$(GCC) $^ -o $@ $(LIBS)

$(MNISTKDGA): $(OBJDIR)/$(MNISTKDGA).o
	$(GCC) $^ -o $@ $(LIBS)

$(MNISTKDPSO): $(OBJDIR)/$(MNISTKDPSO).o
	$(GCC) $^ -o $@ $(LIBS)

$(MNISTKDMERGE): $(OBJDIR)/$(MNISTKDMERGE).o
	$(GCC) $^ -o $@ $(LIBS)

$(MNISTPERMUT): $(OBJDIR)/$(MNISTPERMUT).o
	$(GCC) $^ -o $@ $(LIBS)

$(MNISTCLUSTER): $(OBJDIR)/$(MNISTCLUSTER).o
	$(GCC) $^ -o $@ $(GLIBS)

$(MNISTPCA): $(OBJDIR)/$(MNISTPCA).o
	$(GCC) $^ -o $@ $(GLIBS)

$(MODMNIST): $(OBJDIR)/$(MODMNIST).o
	$(GCC) $^ -o $@ $(LIBS)


clean:
	rm -f $(EXECUTABLES)
	rm -f $(OBJDIR)/*.o
