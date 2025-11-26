#include "activation_function.h"
using namespace std;

vector<float> RELU::forward(const vector<float>& input) {
    input_ = input;
    
    vector<float> output = input;
    for (float& value : output) {
        if (value < 0) {
            value = 0;
        }
    }
    return output;
}

vector<float> RELU::backward(const vector<float>& grad_output) {
    vector<float> grad_input(input_.size());

    if (input_.size() != grad_output.size()) {
        return {}; 
    }

    for (size_t i = 0; i < input_.size(); i++) {
        grad_input[i] = (input_[i] > 0) ? grad_output[i] : 0.0f;
    }
    return grad_input;
}