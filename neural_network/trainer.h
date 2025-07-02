#pragma once

#include "neural_network.h"
#include "../loss_function/cross_entropy_loss.h"
#include <vector>

class Trainer {
public:
    Trainer(NeuralNetwork& network, float learning_rate);

    vector<float> to_one_hot(unsigned char label, int num_classes);

    void train(const vector<float>& images_1d, 
               const vector<unsigned char>& labels, 
               int num_images, 
               int image_size, 
               int epochs);

    // Evaluate for accuracy
    float test(const vector<float>& images_1d, 
                     const vector<unsigned char>& labels,
                     int num_images,
                     int image_size);

private:
    NeuralNetwork& network_;
    CrossEntropyLoss loss_function_;
    float learning_rate_;
};