#include "nn_layer.h"
#include <iostream>
#include <random>
#include <stdexcept>
#include <fstream>
#include <omp.h>
#include "../timer.h"

using namespace std;

NN_Layer::NN_Layer(int input_size, int output_size)
    : input_size_(input_size), output_size_(output_size)
      {
        weights_.resize(output_size_ * input_size_);
        biases_.resize(output_size_);

        //std::random_device rd; 
        //std::mt19937 gen(rd());
        std::mt19937 gen(42);
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

    {
    TIMING_SCOPE("Forward " + to_string(input_size_) + "x" + to_string(output_size_));

    if (input_size_ * output_size_ > 20000) {
        #pragma omp parallel for schedule(static)
        for(int i=0; i<output_size_; i++) {
            float sum = biases_[i];

            for(int j=0; j<input_size_; j++){
                sum = sum + input[j] * weights_[i*input_size_ + j];
            }
            output[i] = sum;
        }
    }
    else {
        for(int i=0; i<output_size_; i++) {
            float sum = biases_[i];

            for(int j=0; j<input_size_; j++){
                sum = sum + input[j] * weights_[i*input_size_ + j];
            }
            output[i] = sum;
        }
    }
    
    }
    return output;
}

vector<float> NN_Layer::backward(const vector<float>& gradient_output) {
    vector<float> gradient_input(input_size_, 0.0f);

    // Move timer RIGHT before computation
    {
        TIMING_SCOPE("Backward " + to_string(input_size_) + "x" + to_string(output_size_));
        
        // Use consistent threshold with forward pass
        if (input_size_ * output_size_ > 20000) {  // Change from 64 to consistent threshold
            fill(gradient_input.begin(), gradient_input.end(), 0.0f);
            
            #pragma omp parallel
            {
                vector<float> local_gradient_input(input_size_, 0.0f);
                
                #pragma omp for schedule(static)
                for(int i = 0; i < output_size_; i++) {
                    gradient_biases_[i] = gradient_output[i];
                    for(int j = 0; j < input_size_; j++) {
                        gradient_weights_[i * input_size_ + j] = gradient_output[i] * last_input_[j];
                        local_gradient_input[j] += gradient_output[i] * weights_[i * input_size_ + j];
                    }
                }
                
                #pragma omp critical
                {
                    for(int j = 0; j < input_size_; j++) {
                        gradient_input[j] += local_gradient_input[j];
                    }
                }
            }
        } else {
            // Serial execution for small layers
            for(int i = 0; i < output_size_; i++) {
                gradient_biases_[i] = gradient_output[i];
                for(int j = 0; j < input_size_; j++) {
                    gradient_weights_[i * input_size_ + j] = gradient_output[i] * last_input_[j];
                    gradient_input[j] += gradient_output[i] * weights_[i * input_size_ + j];
                }
            }
        }
    }
    return gradient_input;
}

void NN_Layer::update(float learning_rate) {
    // Move timer RIGHT before computation + add size-based logic
    {
        TIMING_SCOPE("Update " + to_string(input_size_) + "x" + to_string(output_size_));
        
        // Only parallelize large operations
        if (input_size_ * output_size_ > 20000) {
            #pragma omp parallel for schedule(static)
            for(int i = 0; i < output_size_; i++) {
                biases_[i] -= learning_rate * gradient_biases_[i];
                for(int j = 0; j < input_size_; j++) {
                    weights_[i * input_size_ + j] -= learning_rate * gradient_weights_[i * input_size_ + j];
                }
            }
        } else {
            // Serial execution for small layers
            for(int i = 0; i < output_size_; i++) {
                biases_[i] -= learning_rate * gradient_biases_[i];
                for(int j = 0; j < input_size_; j++) {
                    weights_[i * input_size_ + j] -= learning_rate * gradient_weights_[i * input_size_ + j];
                }
            }
        }
    }
}

// Adding below code for serialization purpose
void NN_Layer::saveWeights(const string& filename) const {
    ofstream out(filename, ios::binary);
    if (! out) {
        throw runtime_error("Could not open file for saving weights");
    }

    //out.write(reinterpret_cast<const char*>(&input_size_), sizeof(input_size_));
    //out.write(reinterpret_cast<const char*>(&output_size_), sizeof(output_size_));
    out.write(reinterpret_cast<const char*>(weights_.data()), weights_.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(biases_.data()), biases_.size() * sizeof(float));

    if (!out) {
        throw runtime_error("Error writing weights to file");
    }
    out.close();
}

void NN_Layer::loadWeights(const string& filename) {
    ifstream in(filename, ios::binary | ios::ate);
    if (!in) {
        throw runtime_error("Could not open file for loading weights");
    }

    streamsize file_size = in.tellg();
    in.seekg(0, ios::beg);

    size_t expected_bytes = (weights_.size() + biases_.size()) * sizeof(float);

    if (file_size != expected_bytes) {
        string error_msg = "Weight file size mismatch for " + filename +
                           ". Expected " + to_string(expected_bytes) +
                           " bytes, but got " + to_string(file_size) + " bytes.";
        throw runtime_error(error_msg);
    }

    //in.read(reinterpret_cast<char*>(&input_size_), sizeof(input_size_));
    //in.read(reinterpret_cast<char*>(&output_size_), sizeof(output_size_));
    //weights_.resize(output_size_ * input_size_);
    //biases_.resize(output_size_);
    in.read(reinterpret_cast<char*>(weights_.data()), weights_.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(biases_.data()), biases_.size() * sizeof(float));

    if (!in) {
        throw runtime_error("Error reading weights from file");
    }
    in.close();
}
