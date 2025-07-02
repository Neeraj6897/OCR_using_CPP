#include <iostream>
#include <vector>
#include <memory>
#include <string>

#include "dataset_loader.h"
#include "normalize.h"
#include "neural_network/neural_network.h"
#include "neural_network/nn_layer.h"
#include "neural_network/activation_function.h"
#include "neural_network/softmax.h"
#include "neural_network/trainer.h"

using namespace std;

int main() {
    string train_images_path = "/home/amd/neeraj/test/work/dataset/train-images-idx3-ubyte";
    string train_labels_path = "/home/amd/neeraj/test/work/dataset/train-labels-idx1-ubyte";
    string test_images_path  = "/home/amd/neeraj/test/work/dataset/t10k-images-idx3-ubyte";
    string test_labels_path  = "/home/amd/neeraj/test/work/dataset/t10k-labels-idx1-ubyte";

    int image_width = 28;
    int image_height = 28;
    int image_size = image_width * image_height;
    int num_classes = 10;

    float learning_rate = 0.01f;
    int epochs = 50;

    cout << " Loading dataset..." <<endl;
    
    // Loading data in its original format (2D vector of bytes)
    auto train_images_2d = loadImages(train_images_path);
    auto train_labels = loadLabels(train_labels_path);
    auto test_images_2d = loadImages(test_images_path);
    auto test_labels = loadLabels(test_labels_path);

    if (train_images_2d.empty() || test_images_2d.empty()) {
        cerr << " Error: Failed to load dataset." << endl;
        return 1;
    }

    cout << " Preparing data for training input (1-D vector)" << endl;

    auto train_images_1d = flatteningImages(train_images_2d);
    auto test_images_1d = flatteningImages(test_images_2d);

    const int num_train_images = train_images_2d.size();
    const int num_test_images = test_images_2d.size();

    for (int i = 0; i < num_train_images; i++) {
        normalizeImage(train_images_1d, i * image_size, image_size);
    }

    for (int i = 0; i < num_test_images; i++) {
        normalizeImage(test_images_1d, i * image_size, image_size);
    }

    cout << " Data loaded: " << num_train_images << " training images, and " 
              << num_test_images << " testing images." << endl;

    
    cout << " Building the neural network..." << endl;
    NeuralNetwork network;
    network.addLayer(make_unique<NN_Layer>(image_size, 128));
    network.addLayer(make_unique<RELU>());
    network.addLayer(make_unique<NN_Layer>(128, num_classes));
    network.addLayer(make_unique<SoftMaxLayer>());

    cout << "\n Starting the training for " << epochs << " epochs..." << endl;

    Trainer trainer(network, learning_rate);
    trainer.train(train_images_1d, train_labels, num_train_images, image_size, epochs);

    cout << " Training complete." << endl;
    cout << "==================================================================" << endl;

    cout << "\n Evaluating network on test data..." << endl;
    float accuracy = trainer.test(test_images_1d, test_labels, num_test_images, image_size);

    cout << "------------------------------------" << endl;
    cout << " Final Test Accuracy: " << accuracy << "%" << endl;
    cout << "------------------------------------" << endl;

    return 0;
}