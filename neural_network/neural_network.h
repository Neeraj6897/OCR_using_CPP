#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"
#include <vector>
#include <memory>

using namespace std;

class Trainer;

class NeuralNetwork {
    friend class Trainer; //Allow Trainer to access private members
    
    public:
        void addLayer(unique_ptr<Layer> layer);

        // Forward propagation using predict function
        vector<float> predict(const vector<float>& input) const;

        // Add this accessor for serialization
        Layer* getLayer(int index) const {
        if (index >= 0 && index < static_cast<int>(layers_.size())) {
            return layers_[index].get();
        }
        return nullptr;
        }
    
        int getLayerCount() const {
            return static_cast<int>(layers_.size());
        }

    private:
        vector<unique_ptr<Layer>> layers_;
};

#endif