#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include "layer.h"

class RELU : public Layer {
    public:
        std::vector<float> forward(const std::vector<float>& input);
        std::vector<float> backward(const std::vector<float>& gradientOutput);
        void update(float learning_rate) override {};

    private:
        std::vector<float> input_;
};

#endif