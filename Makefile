CXX  = g++
NVCC = nvcc

# ---- CUDA paths (adjust if needed) ----
CUDA_HOME ?= /usr/local/cuda-12.6
CUDA_INC  := $(CUDA_HOME)/include
CUDA_LIB  := $(CUDA_HOME)/lib64   # CUDA 12 on Ubuntu uses lib64

# ---- OpenCV flags ----
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS   := $(shell pkg-config --libs opencv4)

# ---- Detect whether compiler is clang or gcc to pick OpenMP runtime ----
# For GCC -> -lgomp, for Clang -> -lomp
OMP_LIB := $(shell $(CXX) -v 2>&1 | grep -qi clang && echo -lomp || echo -lgomp)

# ---- Base flags ----
CXXFLAGS  = -fopenmp -std=c++17 -O3 -I. -I$(CUDA_INC) $(OPENCV_CFLAGS)
NVCCFLAGS = -std=c++17 -O3 -Xcompiler "-fopenmp -fPIC" -I. -I$(CUDA_INC) \
            -gencode arch=compute_61,code=sm_61 \
            -gencode arch=compute_61,code=compute_61 \
            -DUSE_CUDA

LDFLAGS   =
LDLIBS    = $(OPENCV_LIBS)

# ---- Conditional CUDA ----
ifdef USE_CUDA
  CXXFLAGS  += -DUSE_CUDA
  NVCCFLAGS += -DUSE_CUDA

  CUDA_SOURCES = neural_network/nn_layer_cuda.cu
  cuda_objects = $(CUDA_SOURCES:.cu=.o)

  # Link with nvcc, add CUDA libs and rpath, and force OpenMP runtime
  LD       = $(NVCC)
  LDFLAGS += -L$(CUDA_LIB) -Xlinker -rpath -Xlinker $(CUDA_LIB)
  LDLIBS  += -lcudart -lcublas -lcublasLt $(OMP_LIB)
else
  cuda_objects =
  # CPU-only link with g++, keep OpenMP
  LD = $(CXX)
  LDLIBS += $(OMP_LIB)
endif

# ---- Sources ----
NN_SOURCES = neural_network/nn_layer.cpp \
             neural_network/activation_function.cpp \
             neural_network/softmax.cpp \
             neural_network/neural_network.cpp \
             timer.cpp

TRAINING_SOURCES = main.cpp \
                   dataset_loader.cpp \
                   normalize.cpp \
                   loss_function/cross_entropy_loss.cpp \
                   neural_network/trainer.cpp \
                   $(NN_SOURCES)

INFERENCE_SOURCES = main_inference.cpp \
                    image_preprocessor.cpp \
                    normalize.cpp \
                    $(NN_SOURCES)

TRAINING_EXECUTABLE  = ocr_train
INFERENCE_EXECUTABLE = ocr_inference

# ---- Rules ----
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(TRAINING_EXECUTABLE): $(TRAINING_SOURCES:.cpp=.o) $(cuda_objects)
	$(LD) $^ -o $@ $(LDFLAGS) $(LDLIBS)

$(INFERENCE_EXECUTABLE): $(INFERENCE_SOURCES:.cpp=.o) $(cuda_objects)
	$(LD) $^ -o $@ $(LDFLAGS) $(LDLIBS)

# ---- Targets ----
all: $(TRAINING_EXECUTABLE) $(INFERENCE_EXECUTABLE)

cpu:
	$(MAKE) clean && $(MAKE) all

gpu:
	$(MAKE) clean && $(MAKE) USE_CUDA=1 all

clean:
	rm -f *.o neural_network/*.o loss_function/*.o $(TRAINING_EXECUTABLE) $(INFERENCE_EXECUTABLE)

.PHONY: all cpu gpu clean
