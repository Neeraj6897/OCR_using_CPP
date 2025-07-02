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

        //Running forward pass through all layers
        vector<float> predict(const vector<float>& input) const;

    private:
        vector<unique_ptr<Layer>> layers_;
};

#endif