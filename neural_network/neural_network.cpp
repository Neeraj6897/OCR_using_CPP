#include "neural_network.h"

void NeuralNetwork::addLayer(unique_ptr<Layer> layer) {
    layers_.push_back(move(layer));
}

    vector<float> NeuralNetwork::predict(const vector<float>& input) const {
        vector<float> activation = input;
        for(const auto& layer : layers_) {
            activation = layer->forward(activation);
        }
        return activation;  
    } 