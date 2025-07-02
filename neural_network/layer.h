#ifndef LAYER_H
#define LAYER_H

#include <vector>
using namespace std;

class Layer {
    public:
        virtual ~Layer() = default;

        virtual vector<float> forward(const vector<float>& input) = 0;
        virtual vector<float> backward(const vector<float>& gradient_output) = 0;
        virtual void update(float learning_rate) {}
};

#endif  