#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <hnswlib/hnswlib.h>

class InnerProductBenchmark {
private:
    int dimension;
    int num_vectors;
    std::vector<std::vector<float>> vectors;
    hnswlib::InnerProductSpace* space;
    
public:
    InnerProductBenchmark(int dim, int num_vecs) : dimension(dim), num_vectors(num_vecs) {
        space = new hnswlib::InnerProductSpace(dimension);
        generateRandomVectors();
    }
    
    ~InnerProductBenchmark() {
        delete space;
    }
    
    void generateRandomVectors() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        vectors.resize(num_vectors);
        for (int i = 0; i < num_vectors; ++i) {
            vectors[i].resize(dimension);
            for (int j = 0; j < dimension; ++j) {
                vectors[i][j] = dis(gen);
            }
        }
        
        std::cout << "Generated " << num_vectors << " random vectors of dimension " << dimension << std::endl;
    }
    
    // Manual inner product calculation (baseline)
    float manualInnerProduct(const std::vector<float>& a, const std::vector<float>& b) {
        float result = 0.0f;
        for (int i = 0; i < dimension; ++i) {
            result += a[i] * b[i];
        }
        return result;
    }
    
    // hnswlib optimized inner product (uses SIMD/AVX512 internally)
    float hnswlibInnerProduct(const std::vector<float>& a, const std::vector<float>& b) {
        // hnswlib's distance function for inner product space
        // Note: hnswlib returns 1 - inner_product for inner product space
        float distance = space->get_dist_func()(a.data(), b.data(), space->get_dist_func_param());
        return 1.0f - distance;  // Convert back to actual inner product
    }
    
    // Batch inner product calculation using hnswlib
    std::vector<float> batchInnerProduct(int reference_idx) {
        std::vector<float> results;
        results.reserve(num_vectors);
        
        const float* ref_data = vectors[reference_idx].data();
        auto dist_func = space->get_dist_func();
        auto dist_param = space->get_dist_func_param();
        
        for (int i = 0; i < num_vectors; ++i) {
            float distance = dist_func(ref_data, vectors[i].data(), dist_param);
            results.push_back(1.0f - distance);  // Convert to inner product
        }
        
        return results;
    }
    
    void benchmarkComparison() {
        const int reference_idx = 0;
        const int num_iterations = 1000;
        
        std::cout << "\n=== Benchmarking Inner Product Calculations ===" << std::endl;
        std::cout << "Reference vector index: " << reference_idx << std::endl;
        std::cout << "Number of comparisons per iteration: " << num_vectors - 1 << std::endl;
        std::cout << "Number of iterations: " << num_iterations << std::endl;
        
        // Benchmark manual calculation
        auto start = std::chrono::high_resolution_clock::now();
        float manual_sum = 0.0f;
        
        for (int iter = 0; iter < num_iterations; ++iter) {
            for (int i = 1; i < num_vectors; ++i) {
                manual_sum += manualInnerProduct(vectors[reference_idx], vectors[i]);
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto manual_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Benchmark hnswlib calculation
        start = std::chrono::high_resolution_clock::now();
        float hnswlib_sum = 0.0f;
        
        for (int iter = 0; iter < num_iterations; ++iter) {
            for (int i = 1; i < num_vectors; ++i) {
                hnswlib_sum += hnswlibInnerProduct(vectors[reference_idx], vectors[i]);
            }
        }
        
        end = std::chrono::high_resolution_clock::now();
        auto hnswlib_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Benchmark batch calculation
        start = std::chrono::high_resolution_clock::now();
        float batch_sum = 0.0f;
        
        for (int iter = 0; iter < num_iterations; ++iter) {
            auto batch_results = batchInnerProduct(reference_idx);
            for (int i = 1; i < batch_results.size(); ++i) {
                batch_sum += batch_results[i];
            }
        }
        
        end = std::chrono::high_resolution_clock::now();
        auto batch_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Results
        std::cout << "\n=== Results ===" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Manual calculation:    " << manual_duration.count() << " μs (sum: " << manual_sum << ")" << std::endl;
        std::cout << "hnswlib calculation:   " << hnswlib_duration.count() << " μs (sum: " << hnswlib_sum << ")" << std::endl;
        std::cout << "hnswlib batch:         " << batch_duration.count() << " μs (sum: " << batch_sum << ")" << std::endl;
        
        double speedup_individual = (double)manual_duration.count() / hnswlib_duration.count();
        double speedup_batch = (double)manual_duration.count() / batch_duration.count();
        
        std::cout << "\nSpeedup (individual):  " << speedup_individual << "x" << std::endl;
        std::cout << "Speedup (batch):       " << speedup_batch << "x" << std::endl;
        
        // Verify accuracy
        float diff = std::abs(manual_sum - hnswlib_sum);
        std::cout << "\nAccuracy check - difference: " << diff << std::endl;
        if (diff < 1e-3) {
            std::cout << "✓ Results match within tolerance" << std::endl;
        } else {
            std::cout << "⚠ Results differ significantly" << std::endl;
        }
    }
    
    void findTopKSimilar(int reference_idx, int k = 5) {
        std::cout << "\n=== Finding Top-" << k << " Most Similar Vectors ===" << std::endl;
        
        std::vector<std::pair<float, int>> similarities;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_vectors; ++i) {
            if (i != reference_idx) {
                float similarity = hnswlibInnerProduct(vectors[reference_idx], vectors[i]);
                similarities.push_back({similarity, i});
            }
        }
        
        // Sort by similarity (descending)
        std::sort(similarities.begin(), similarities.end(), 
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Top-" << k << " most similar to vector " << reference_idx << ":" << std::endl;
        for (int i = 0; i < std::min(k, (int)similarities.size()); ++i) {
            std::cout << "  Vector " << similarities[i].second 
                      << ": similarity = " << std::fixed << std::setprecision(6) 
                      << similarities[i].first << std::endl;
        }
        
        std::cout << "Time taken: " << duration.count() << " μs" << std::endl;
    }
};

int main() {
    std::cout << "=== hnswlib AVX512 Inner Product Benchmark ===" << std::endl;
    
    // Check if AVX512 is available
    std::cout << "Note: Ensure your CPU supports AVX512 and compile with -mavx512f -mavx512dq flags" << std::endl;
    
    // Test different vector sizes
    std::vector<std::pair<int, int>> test_configs = {
        {128, 1000},   // 128-dim vectors, 1000 vectors
        {256, 1000},   // 256-dim vectors, 1000 vectors
        {512, 500},    // 512-dim vectors, 500 vectors
        {1024, 250}    // 1024-dim vectors, 250 vectors
    };
    
    for (const auto& config : test_configs) {
        int dim = config.first;
        int num_vecs = config.second;
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "Testing with " << dim << "-dimensional vectors, " << num_vecs << " vectors" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        InnerProductBenchmark benchmark(dim, num_vecs);
        benchmark.benchmarkComparison();
        benchmark.findTopKSimilar(0, 5);
    }
    
    std::cout << "\n=== Benchmark Complete ===" << std::endl;
    std::cout << "For maximum performance, ensure you compiled with:" << std::endl;
    std::cout << "g++ -std=c++17 -O3 -march=native -mavx512f -mavx512dq your_code.cpp -o your_program" << std::endl;
    
    return 0;
}
