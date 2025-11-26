#ifndef NN_LAYER_H
#define NN_LAYER_H

#include "layer.h"
#include <vector>
#include <string>
#include <stdexcept>

class NN_Layer : public Layer {
    public:
        NN_Layer(int input_size, int output_size);

        vector<float> forward(const vector<float>& input) override;
        vector<float> backward(const vector<float>& gradient_output) override;
        void update(float learning_rate) override;

        //Adding for serialization purposes
        void saveWeights(const string& filename) const;
        void loadWeights(const string& filename);

        const std::vector<float>& getWeights() const { return weights_; }
        const std::vector<float>& getBiases() const { return biases_; }
        const std::vector<float>& getGradientWeights() const { return gradient_weights_; }
        const std::vector<float>& getGradientBiases() const { return gradient_biases_; }
        
        // Setters for creating identical layers
        void setWeights(const std::vector<float>& weights) { 
            if (weights.size() != weights_.size()) {
                throw std::invalid_argument("Weight vector size mismatch");
            }
            weights_ = weights; 
        }
        
        void setBiases(const std::vector<float>& biases) { 
            if (biases.size() != biases_.size()) {
                throw std::invalid_argument("Bias vector size mismatch");
            }
            biases_ = biases; 
        }

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
