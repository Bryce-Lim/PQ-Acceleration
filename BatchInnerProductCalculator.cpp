#include "BatchInnerProductCalculator.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <cmath>
#include <cassert>

BatchInnerProductCalculator::BatchInnerProductCalculator(int dim) : dimension(dim) {
    space = new hnswlib::InnerProductSpace(dimension);
    dist_func = space->get_dist_func();
    dist_func_param = space->get_dist_func_param();
    
    std::cout << "Initialized BatchInnerProductCalculator for " << dimension 
              << "-dimensional vectors" << std::endl;
}

BatchInnerProductCalculator::~BatchInnerProductCalculator() {
    delete space;
}

std::vector<std::vector<float>> BatchInnerProductCalculator::calculateInnerProducts(
    const std::vector<std::vector<float>>& centroids,
    const std::vector<std::vector<float>>& data) {
    
    // Validate inputs
    if (centroids.empty() || data.empty()) {
        throw std::invalid_argument("Centroids and data cannot be empty");
    }
    
    // Check dimensions
    for (const auto& centroid : centroids) {
        if (centroid.size() != dimension) {
            throw std::invalid_argument("Centroid dimension mismatch");
        }
    }
    
    for (const auto& vec : data) {
        if (vec.size() != dimension) {
            throw std::invalid_argument("Data vector dimension mismatch");
        }
    }
    
    size_t num_centroids = centroids.size();
    size_t num_data = data.size();
    
    // Initialize results matrix
    std::vector<std::vector<float>> results(num_centroids, std::vector<float>(num_data));
    
    // Calculate all pairwise inner products
    // Iterate through centroids in outer loop for better cache locality
    for (size_t i = 0; i < num_centroids; ++i) {
        const float* centroid_data = centroids[i].data();
        
        for (size_t j = 0; j < num_data; ++j) {
            const float* data_vec = data[j].data();
            
            // Use hnswlib's optimized distance function
            // Note: hnswlib's inner product space returns (1 - inner_product)
            float distance = dist_func(centroid_data, data_vec, dist_func_param);
            results[i][j] = 1.0f - distance;  // Convert back to actual inner product
        }
    }
    
    return results;
}

std::vector<std::vector<float>> BatchInnerProductCalculator::calculateInnerProductsOptimized(
    const std::vector<std::vector<float>>& centroids,
    const std::vector<std::vector<float>>& data) {

    const size_t num_centroids = centroids.size();
    const size_t num_data = data.size();
    
    // Pre-allocate results
    std::vector<std::vector<float>> results(num_centroids);
    for (size_t i = 0; i < num_centroids; ++i) {
        results[i].resize(num_data);
    }

    // OPTIMIZATION: Block/tile processing for better cache locality
    const size_t centroid_block_size = 8;  // Process 8 centroids at a time
    const size_t data_block_size = 64;     // Process 64 data points at a time

//    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (size_t ci = 0; ci < num_centroids; ci += centroid_block_size) {
        for (size_t dj = 0; dj < num_data; dj += data_block_size) {
            
            size_t ci_end = std::min(ci + centroid_block_size, num_centroids);
            size_t dj_end = std::min(dj + data_block_size, num_data);
            
            // Process block
            for (size_t i = ci; i < ci_end; ++i) {
                const float* centroid_data = centroids[i].data();
                float* result_row = results[i].data();
                
                for (size_t j = dj; j < dj_end; ++j) {
                    const float* data_vec = data[j].data();
                    float distance = dist_func(centroid_data, data_vec, dist_func_param);
                    result_row[j] = 1.0f - distance;
                }
            }
        }
    }

    return results;
}

std::vector<float> BatchInnerProductCalculator::calculateWithSingleCentroid(
    const std::vector<float>& centroid,
    const std::vector<std::vector<float>>& data) {
    
    if (centroid.size() != dimension) {
        throw std::invalid_argument("Centroid dimension mismatch");
    }
    
    std::vector<float> results;
    results.reserve(data.size());
    
    const float* centroid_data = centroid.data();
    
    for (const auto& vec : data) {
        if (vec.size() != dimension) {
            throw std::invalid_argument("Data vector dimension mismatch");
        }
        
        const float* data_vec = vec.data();
        float distance = dist_func(centroid_data, data_vec, dist_func_param);
        results.push_back(1.0f - distance);
    }
    
    return results;
}

std::vector<std::vector<std::pair<float, size_t>>> BatchInnerProductCalculator::findTopKSimilar(
    const std::vector<std::vector<float>>& centroids,
    const std::vector<std::vector<float>>& data,
    int k) {
    
    auto all_products = calculateInnerProducts(centroids, data);
    std::vector<std::vector<std::pair<float, size_t>>> top_k_results;
    
    for (size_t i = 0; i < centroids.size(); ++i) {
        std::vector<std::pair<float, size_t>> similarities;
        
        for (size_t j = 0; j < data.size(); ++j) {
            similarities.emplace_back(all_products[i][j], j);
        }
        
        // Sort by similarity (descending) and keep top-k
        std::partial_sort(similarities.begin(), 
                        similarities.begin() + std::min(k, (int)similarities.size()),
                        similarities.end(),
                        [](const auto& a, const auto& b) { return a.first > b.first; });
        
        similarities.resize(std::min(k, (int)similarities.size()));
        top_k_results.push_back(std::move(similarities));
    }
    
    return top_k_results;
}

void BatchInnerProductCalculator::benchmark(
    const std::vector<std::vector<float>>& centroids,
    const std::vector<std::vector<float>>& data,
    int num_iterations) {
    
    std::cout << "\n=== Benchmarking Inner Product Calculations ===" << std::endl;
    std::cout << "Centroids: " << centroids.size() << " x " << dimension << std::endl;
    std::cout << "Data: " << data.size() << " x " << dimension << std::endl;
    std::cout << "Total calculations: " << centroids.size() * data.size() << std::endl;
    std::cout << "Iterations: " << num_iterations << std::endl;
    
    // Benchmark standard implementation
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> results1;
    
    for (int iter = 0; iter < num_iterations; ++iter) {
        results1 = calculateInnerProducts(centroids, data);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto standard_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Benchmark optimized implementation
    start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> results2;
    
    for (int iter = 0; iter < num_iterations; ++iter) {
        results2 = calculateInnerProductsOptimized(centroids, data);
    }
    
    end = std::chrono::high_resolution_clock::now();
    auto optimized_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Benchmark naive implementation for comparison
    start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> results3;
    
    for (int iter = 0; iter < num_iterations; ++iter) {
        results3 = naiveInnerProducts(centroids, data);
    }
    
    end = std::chrono::high_resolution_clock::now();
    auto naive_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Display results
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Standard hnswlib:   " << standard_duration.count() << " μs" << std::endl;
    std::cout << "Optimized hnswlib:  " << optimized_duration.count() << " μs" << std::endl;
    std::cout << "Naive calculation:  " << naive_duration.count() << " μs" << std::endl;
    
    double speedup_vs_naive = (double)naive_duration.count() / standard_duration.count();
    std::cout << "\nSpeedup vs naive:   " << speedup_vs_naive << "x" << std::endl;
    
    // Verify accuracy
    bool accurate = verifyResults(results1, results3, 1e-5f);
    std::cout << "Accuracy check:     " << (accurate ? "✓ PASSED" : "✗ FAILED") << std::endl;
}

std::vector<std::vector<float>> BatchInnerProductCalculator::naiveInnerProducts(
    const std::vector<std::vector<float>>& centroids,
    const std::vector<std::vector<float>>& data) {
    
    std::vector<std::vector<float>> results(centroids.size(), 
                                          std::vector<float>(data.size()));
    
    for (size_t i = 0; i < centroids.size(); ++i) {
        for (size_t j = 0; j < data.size(); ++j) {
            float sum = 0.0f;
            for (int k = 0; k < dimension; ++k) {
                sum += centroids[i][k] * data[j][k];
            }
            results[i][j] = sum;
        }
    }
    
    return results;
}

bool BatchInnerProductCalculator::verifyResults(
    const std::vector<std::vector<float>>& a,
    const std::vector<std::vector<float>>& b,
    float tolerance) {
    
    if (a.size() != b.size()) return false;
    
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].size() != b[i].size()) return false;
        
        for (size_t j = 0; j < a[i].size(); ++j) {
            if (std::abs(a[i][j] - b[i][j]) > tolerance) {
                return false;
            }
        }
    }
    
    return true;
}
