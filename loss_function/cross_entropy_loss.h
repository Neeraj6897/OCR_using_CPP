#ifndef CROSS_ENTROPY_LOSS_H
#define CROSS_ENTROPY_LOSS_H

#include <vector>

class CrossEntropyLoss {
public:
    // Change return type and arguments to float
    float compute(const std::vector<float>& predicted, const std::vector<float>& actual);
    std::vector<float> derivative(const std::vector<float>& predicted, const std::vector<float>& actual);
};

#endif