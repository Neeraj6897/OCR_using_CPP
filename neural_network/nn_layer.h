#ifndef NN_LAYER_H
#define NN_LAYER_H

#include "layer.h"
#include <vector>
#include <string>
#include <stdexcept>

class NN_Layer : public Layer {
    public:
        NN_Layer(int input_size, int output_size);
        ~NN_Layer(); // to free GPU memory

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
        
        //Setters for creating identical layers
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

        void enable_cuda(bool on) { use_cuda_ = on; }

    private:
         int input_size_;
        int output_size_;

        vector<float> weights_;
        vector<float> biases_;
        vector<float> last_input_;     
        vector<float> gradient_weights_;
        vector<float> gradient_biases_;

        #ifdef USE_CUDA
            bool use_cuda_ = true;
            bool host_params_dirty_ = false;
            float *d_W_ = nullptr, *d_b_ = nullptr, *d_x_ = nullptr, *d_y_ = nullptr;
            float *d_dY_=nullptr, *d_dW_=nullptr, *d_db_=nullptr, *d_dX_=nullptr;

            void cudaInit_();
            void cudaFree_();
            void cudaUploadParams_();
            void cudaForward(const float* x_host, float* y_host) const;
            void cudaBackward(const float* gradY_host, float* gradX_host);  

            void cudaUpdate(float learning_rate);
            void syncDeviceToHost_();
        #else
            bool use_cuda_ = false;
        #endif
};

#endif
