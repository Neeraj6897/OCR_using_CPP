CXX = g++
CXXFLAGS = -std=c++17 -g -O3 -I.

SOURCES = main.cpp \
          dataset_loader.cpp \
          normalize.cpp \
          loss_function/cross_entropy_loss.cpp \
          neural_network/nn_layer.cpp \
          neural_network/activation_function.cpp \
          neural_network/softmax.cpp \
          neural_network/neural_network.cpp \
          neural_network/trainer.cpp

EXECUTABLE = ocr_app

all: $(EXECUTABLE)

$(EXECUTABLE): $(SOURCES)
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -f $(EXECUTABLE)
