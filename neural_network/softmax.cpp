#include "softmax.h"
#include <cmath>
#include <numeric>
#include <algorithm>

using namespace std;

vector<float> SoftMaxLayer::forward(const vector<float>& input) {
    vector<float> output(input.size());
    float max_val = *std::max_element(input.begin(), input.end());
    float sum = 0.0f;

    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }

    for (size_t i = 0; i < input.size(); ++i) {
        output[i] /= sum;
    }
    // Store the result for the backward pass
    last_output_ = output;
    return output;
}

vector<float> SoftMaxLayer::backward(const vector<float>& grad_output) {
   return grad_output;
}