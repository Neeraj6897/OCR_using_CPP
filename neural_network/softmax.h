#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "layer.h"

class SoftMaxLayer : public Layer {
    public:
        std::vector<float> forward(const std::vector<float>& input) override;
        std::vector<float> backward(const std::vector<float>& gradient_output) override;
        void update(float learning_rate) override {};

    private:
        std::vector<float> last_output_;
};

#endif