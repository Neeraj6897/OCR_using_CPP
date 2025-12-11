#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <iomanip> // Add this for std::setprecision

#include "normalize.h"
#include "neural_network/neural_network.h"
#include "neural_network/nn_layer.h"
#include "neural_network/activation_function.h"
#include "neural_network/softmax.h"
#include "image_preprocessor.h"

using namespace std;

void showUsage(const string& program_name) {
    cerr << "Usage: " << program_name << " <path_to_image>" << endl;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        showUsage(argv[0]);
        return 1;
    }

    string image_path = argv[1];
    cout << "Loading image: " << image_path << endl;

    // Load the network architecture (same as training)
    NeuralNetwork network;
    network.addLayer(make_unique<NN_Layer>(784, 128));
    network.addLayer(make_unique<RELU>());
    network.addLayer(make_unique<NN_Layer>(128, 10));
    network.addLayer(make_unique<SoftMaxLayer>());

    // 2. Load the trained weights
    cout << "\n--- Loading Weights ---" << endl;
    try {
        auto* layer1 = dynamic_cast<NN_Layer*>(network.getLayer(0));
        auto* layer2 = dynamic_cast<NN_Layer*>(network.getLayer(2));
        if (!layer1 || !layer2) {
            cerr << "Error: Could not get network layers." << endl;
            return 1;
        }
        layer1->loadWeights("layer1_weights.bin");
        layer2->loadWeights("layer2_weights.bin");
        cout << "Weights loaded successfully." << endl;
    } catch (const exception& e) {
        cerr << "FATAL ERROR loading weights: " << e.what() << endl;
        return 1;
    }

    cout << "\n--- Preprocessing Image ---" << endl;
    vector<float> image_vector = ImagePreprocessor::preprocessImageForMNIST(image_path);
    if (image_vector.size() != 784) {
        cerr << "Error: Image preprocessing failed." << endl;
        return 1;
    }
    //normalizeImage(image_vector, 0, 784);

    // Pass through Layer 0 (NN 784->128)
    vector<float> out_layer0 = network.getLayer(0)->forward(image_vector);
    vector<float> out_layer1 = network.getLayer(1)->forward(out_layer0);
    vector<float> out_layer2 = network.getLayer(2)->forward(out_layer1);
    vector<float> final_output = network.getLayer(3)->forward(out_layer2);

    cout << "\n--- Final Prediction ---" << endl;
    cout << "Final Output Vector: [ ";
    for (size_t i = 0; i < final_output.size(); ++i) {
        cout << fixed << setprecision(4) << final_output[i] << (i == final_output.size() - 1 ? "" : ", ");
    }
    cout << " ]" << endl;

    auto max_it = max_element(final_output.begin(), final_output.end());
    int predicted_digit = distance(final_output.begin(), max_it);
    float confidence = *max_it;

    cout << "\nPredicted Digit: " << predicted_digit << endl;
    cout << "Confidence: " << confidence * 100.0f << "%" << endl;

    return 0;
}
