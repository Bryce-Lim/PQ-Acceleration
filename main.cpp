//==============================================================//
// Example usage of AMXInnerProduct class                       //
// Author: Bryce Lim, 2025                                      //
//==============================================================//

#include "AMXInnerProduct.h"
#include <iostream>
#include <chrono>
#include <vector>

int main() {
    auto init_start = std::chrono::high_resolution_clock::now();
    
    // Create AMX inner product calculator
    AMXInnerProduct amx_calculator;
    
    // Initialize AMX functionality
    if (!amx_calculator.initialize()) {
        std::cerr << "Failed to initialize AMX functionality!" << std::endl;
        return -1;
    }
    
    // Set up test data
    int centroid_count = 10;
    int data_count = 20;
    int dimension = 32;
    
    std::vector<std::vector<float>> centroids(centroid_count);
    std::vector<std::vector<float>> data(data_count);
    
    // Initialize vector sizes
    for (int i = 0; i < centroid_count; ++i)
        centroids[i].resize(dimension);
    for (int i = 0; i < data_count; ++i)
        data[i].resize(dimension);

    // Initialize centroids with simple patterns
    for (int i = 0; i < centroid_count; ++i) {
        for (int j = 0; j < dimension; ++j) {
            centroids[i][j] = (1 + i) * 1.0f;
        }
    }

    // Initialize data vectors with test patterns
    for (int i = 0; i < data_count; i += 2) {
        for (int j = 0; j < dimension; ++j) {
            data[i][j] = 1.0f;
        }
    }
    
    auto init_end = std::chrono::high_resolution_clock::now();
    
    // Perform the computation
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> results = amx_calculator.compute_inner_products(centroids, data);
    auto end = std::chrono::high_resolution_clock::now();
    
    // Print results
    AMXInnerProduct::print_float_vectors(results);
    
    // Print timing information
    auto init_duration = std::chrono::duration_cast<std::chrono::microseconds>(init_end - init_start);
    std::cout << "Initialization took: " << init_duration.count() << " microseconds" << std::endl;

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Calculation function took: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Total runtime took: " << duration.count() + init_duration.count() << " microseconds" << std::endl;

    /*
     * How to use results?
     * results[i][j] gives the inner product between centroid i and data_point j
     */
    
    // Example: Print a specific inner product
    std::cout << "\nExample: Inner product between centroid 0 and data point 0: " 
              << results[0][0] << std::endl;

    return 0;
}

// Compilation Instructions:
// g++ -O2 -march=native -fno-strict-aliasing -mavx512bf16 main.cpp AMXInnerProduct.cpp -o AMXInnerProduct
