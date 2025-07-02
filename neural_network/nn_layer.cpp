#include "nn_layer.h"
#include <iostream>
#include <random>
#include <stdexcept>

using namespace std;

NN_Layer::NN_Layer(int input_size, int output_size)
    : input_size_(input_size), output_size_(output_size)
      {
        weights_.resize(output_size_ * input_size_);
        biases_.resize(output_size_);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-0.5f, 0.5f);

        for (float& weight : weights_) {
            weight = dis(gen);
        }

        for (float& bias : biases_) {
            bias = dis(gen);
        }

        gradient_weights_.resize(output_size_ * input_size_);
        gradient_biases_.resize(output_size_);
      }

vector<float> NN_Layer::forward(const vector<float>& input) {
    last_input_ = input;

    if (input.size() != static_cast<size_t>(input_size_)) {
        throw invalid_argument("Input size does not match NN layer's input size");
    }

    vector<float> output(output_size_, 0.0f);

    for(int i=0; i<output_size_; i++) {
        float sum = biases_[i];

        for(int j=0; j<input_size_; j++){
            sum = sum + input[j] * weights_[i*input_size_ + j];
        }
        output[i] = sum;
    }
    return output;
}

vector<float> NN_Layer::backward(const vector<float>& gradient_output) {
    vector<float> gradient_input(input_size_, 0.0f);

    for(int i=0; i<output_size_; i++) {
        gradient_biases_[i] = gradient_output[i];
        for(int j=0; j<input_size_; j++) {
            gradient_weights_[i * input_size_ + j] = gradient_output[i] * last_input_[j];
            //weights_[i * input_size_ + j] -= learning_rate * gradient_output[i] * input[j];
            gradient_input[j] += gradient_output[i] * weights_[i * input_size_ + j];
        }
    }
    return gradient_input;
}

void NN_Layer::update(float learning_rate) {
    for(int i=0; i<output_size_; i++) {
        biases_[i] -= learning_rate * gradient_biases_[i];
        for(int j=0; j<input_size_; j++) {
            weights_[i * input_size_ + j] -= learning_rate * gradient_weights_[i * input_size_ + j];
        }
    }
}