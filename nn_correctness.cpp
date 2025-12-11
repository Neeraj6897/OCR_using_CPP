#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <cassert>
#include <omp.h>
#include <random>

#include "neural_network/nn_layer.h"

bool compareVectors(const std::vector<float>& vec1, const std::vector<float>& vec2, const std::string& test_name, float tolerance = 1e-6f) {
    if (vec1.size() != vec2.size()) {
        std::cerr << "FAILED: " << test_name << " - Vector sizes differ (" << vec1.size() << " vs " << vec2.size() << ")" << std::endl;
        return false;
    }

    double max_diff = 0.0;
    int diff_count = 0;
    
    for (size_t i = 0; i < vec1.size(); ++i) {
        double diff = std::abs(vec1[i] - vec2[i]);
        if (diff > tolerance) {
            diff_count++;
            max_diff = std::max(max_diff, (double)diff);
        }
    }
    
    if (diff_count == 0) {
        std::cout << "✅ PASSED: " << test_name << " - Perfect match" << std::endl;
        return true;
    } else {
        std::cerr << "❌ FAILED: " << test_name << " - " << diff_count << " differences, max diff: " << max_diff << std::endl;
        // Print first few mismatches for debugging
        for (size_t i = 0; i < vec1.size() && diff_count < 5; ++i) {
            if (std::abs(vec1[i] - vec2[i]) > tolerance) {
                std::cerr << "  Index " << i << ": " << vec1[i] << " vs " << vec2[i] << std::endl;
                diff_count--;
            }
        }
        return false;
    }
}

// Helper function to copy weights between layers
void copyLayerWeights(const NN_Layer& source, NN_Layer& destination) {
    auto source_weights = source.getWeights();
    auto source_biases = source.getBiases();
    
    // You'll need to add these methods to nn_layer.h
    destination.setWeights(source_weights);
    destination.setBiases(source_biases);
}

int main() {
    std::cout << "===== Verifying Computational Accuracy =====" << std::endl;
    std::cout << "OpenMP Max Threads: " << omp_get_max_threads() << std::endl;

    const int input_size = 784;
    const int output_size = 128;
    const float learning_rate = 0.01f;

    // Create one layer with fixed seed, then copy to ensure identical starting conditions
    NN_Layer serial_layer(input_size, output_size);
    NN_Layer parallel_layer(input_size, output_size);
    
    // Make them identical by copying weights
    copyLayerWeights(serial_layer, parallel_layer);

    // Create deterministic test data
    std::vector<float> input_vector(input_size);
    std::vector<float> gradient_vector(output_size);
    
    // Use a fixed pattern instead of iota for more realistic data
    std::mt19937 gen(12345);  // Fixed seed for reproducible results
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (auto& val : input_vector) val = dis(gen);
    for (auto& val : gradient_vector) val = dis(gen);

    std::cout << "\n--- Testing Forward Pass ---" << std::endl;
    
    // Force serial execution
    omp_set_num_threads(1);
    std::vector<float> serial_forward_out = serial_layer.forward(input_vector);

    // Force parallel execution
    omp_set_num_threads(8);
    std::vector<float> parallel_forward_out = parallel_layer.forward(input_vector);

    bool forward_pass = compareVectors(serial_forward_out, parallel_forward_out, "Forward Pass Output");

    std::cout << "\n--- Testing Backward Pass ---" << std::endl;
    
    omp_set_num_threads(1);
    std::vector<float> serial_backward_out = serial_layer.backward(gradient_vector);

    omp_set_num_threads(8);
    std::vector<float> parallel_backward_out = parallel_layer.backward(gradient_vector);

    bool backward_pass = compareVectors(serial_backward_out, parallel_backward_out, "Backward Pass Output");
    bool gradient_weights = compareVectors(serial_layer.getGradientWeights(), parallel_layer.getGradientWeights(), "Backward Pass Gradient Weights");
    bool gradient_biases = compareVectors(serial_layer.getGradientBiases(), parallel_layer.getGradientBiases(), "Backward Pass Gradient Biases");

    std::cout << "\n--- Testing Update Pass ---" << std::endl;
    
    omp_set_num_threads(1);
    serial_layer.update(learning_rate);

    omp_set_num_threads(8);
    parallel_layer.update(learning_rate);

    bool update_weights = compareVectors(serial_layer.getWeights(), parallel_layer.getWeights(), "Update Pass Final Weights");
    bool update_biases = compareVectors(serial_layer.getBiases(), parallel_layer.getBiases(), "Update Pass Final Biases");

    std::cout << "\n===== SUMMARY =====" << std::endl;
    if (forward_pass && backward_pass && gradient_weights && gradient_biases && update_weights && update_biases) {
        std::cout << "🎉 ALL TESTS PASSED - Serial and Parallel computations are identical!" << std::endl;
    } else {
        std::cout << "⚠️  SOME TESTS FAILED - Check OpenMP implementation for correctness issues." << std::endl;
    }

    return 0;
}