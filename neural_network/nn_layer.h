#ifndef NN_LAYER_H
#define NN_LAYER_H

#include "layer.h"
#include <vector>

class NN_Layer : public Layer {
    public:
        NN_Layer(int input_size, int output_size);

        vector<float> forward(const vector<float>& input) override;
        vector<float> backward(const vector<float>& gradient_output) override;
        void update(float learning_rate) override;

    private:
         int input_size_;
        int output_size_;

        vector<float> weights_;
        vector<float> biases_;
        vector<float> last_input_;     
        vector<float> gradient_weights_;
        vector<float> gradient_biases_;
};

#endif
