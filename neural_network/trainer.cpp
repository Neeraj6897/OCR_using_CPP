#include "trainer.h"
#include "nn_layer.h"
#include <iostream>
#include <vector>
#include <algorithm> //For std::max_element

Trainer::Trainer(NeuralNetwork& network, float learning_rate)
    : network_(network), learning_rate_(learning_rate) {}

vector<float> Trainer::to_one_hot(unsigned char label, int num_classes) {
    vector<float> one_hot(num_classes, 0.0f);
    if (label < num_classes) {
        one_hot[label] = 1.0f;
    }
    return one_hot;
}

void Trainer::train(const vector<float>& images_1d, const vector<unsigned char>& labels, int num_images, int image_size, int epochs) {
    
    const int num_classes = 10; //MNIST data --> (0-9)

    for (int epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0.0f;

        for (int i = 0; i < num_images; i++) {
            auto image_start = images_1d.begin() + i * image_size;
            vector<float> current_image(image_start, image_start + image_size);
            
            // 1. Forward Pass
            vector<float> predicted_output = network_.predict(current_image);

            // 2. Loss Calculation
            vector<float> actual_label = to_one_hot(labels[i], num_classes);
            float loss_value = loss_function_.compute(predicted_output, actual_label);
            epoch_loss = epoch_loss + loss_value;

            // 3. Backward Pass
            vector<float> gradient = loss_function_.derivative(predicted_output, actual_label);
            
            // Propagate the gradient back through the layers, in reverse order.
            for (int j = network_.layers_.size() - 1; j >= 0; j--) {
                gradient = network_.layers_[j]->backward(gradient);
            }

            // 4. Weight Update
            for (auto& layer_ptr : network_.layers_) {
                layer_ptr->update(learning_rate_);   
            }
        }
        std::cout << " Epoch " << epoch + 1 << "/" << epochs 
                  << " | Average Loss: " << epoch_loss / num_images << std::endl;
    }
}

float Trainer::test(const std::vector<float>& images_1d, const std::vector<unsigned char>& labels, int num_images, int image_size) {
    int correct_count = 0;
    for (int i = 0; i < num_images; i++) {
        auto image_start = images_1d.begin() + i * image_size;
        vector<float> current_image(image_start, image_start + image_size);

        vector<float> output = network_.predict(current_image);
        
        auto max_it = std::max_element(output.begin(), output.end());
        int predicted_label = std::distance(output.begin(), max_it);
        
        if (predicted_label == labels[i]) {
            correct_count++;
        }
    }
    // Returning accuracy as a percentage
    return ((float)(correct_count)/num_images) * 100.0f;
}
