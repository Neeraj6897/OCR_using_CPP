CXX = g++
CXXFLAGS = -pg -fopenmp -std=c++17 -g -O3 -I. `pkg-config --cflags opencv4`
LDFLAGS = -fopenmp `pkg-config --libs opencv4`

NN_SOURCES = neural_network/nn_layer.cpp \
             neural_network/activation_function.cpp \
             neural_network/softmax.cpp \
             neural_network/neural_network.cpp

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

TRAINING_EXECUTABLE = ocr_train
INFERENCE_EXECUTABLE = ocr_inference

all: $(TRAINING_EXECUTABLE) $(INFERENCE_EXECUTABLE)

$(TRAINING_EXECUTABLE): $(TRAINING_SOURCES)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(INFERENCE_EXECUTABLE): $(INFERENCE_SOURCES)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TRAINING_EXECUTABLE) $(INFERENCE_EXECUTABLE) $(EXTRACT_EXECUTABLE)
	#rm -f *.bin *.pgm *.txt *.png gmon.out

.PHONY: all clean
