#include "cross_entropy_loss.h"
#include <cmath>    // For std::log
#include <vector>

using namespace std;

float CrossEntropyLoss::compute(const vector<float>& predicted, const vector<float>& actual) {
    float loss = 0.0f;
    for (size_t i = 0; i < predicted.size(); i++) {
        // Adding small epsilon to prevent log(0)
        float epsilon = 1e-9f;
        loss += actual[i] * std::log(predicted[i] + epsilon);
    }
    return -loss;
}

std::vector<float> CrossEntropyLoss::derivative(const vector<float>& predicted, const vector<float>& actual) {
    vector<float> grad(predicted.size());
    for (size_t i = 0; i < predicted.size(); i++) {
        grad[i] = predicted[i] - actual[i];
    }
    return grad;
}