#ifndef BATCH_INNER_PRODUCT_CALCULATOR_H
#define BATCH_INNER_PRODUCT_CALCULATOR_H

#include <vector>
#include <hnswlib/hnswlib.h>

class BatchInnerProductCalculator {
private:
    int dimension;
    hnswlib::InnerProductSpace* space;
    hnswlib::DISTFUNC<float> dist_func;
    void* dist_func_param;
    
    // Private helper methods
    std::vector<std::vector<float>> naiveInnerProducts(
        const std::vector<std::vector<float>>& centroids,
        const std::vector<std::vector<float>>& data);
    
    bool verifyResults(const std::vector<std::vector<float>>& a,
                      const std::vector<std::vector<float>>& b,
                      float tolerance);
    
public:
    // Constructor and destructor
    BatchInnerProductCalculator(int dim);
    ~BatchInnerProductCalculator();
    
    // Disable copy constructor and assignment operator to avoid double deletion
    BatchInnerProductCalculator(const BatchInnerProductCalculator&) = delete;
    BatchInnerProductCalculator& operator=(const BatchInnerProductCalculator&) = delete;
    
    // Main calculation functions
    std::vector<std::vector<float>> calculateInnerProducts(
        const std::vector<std::vector<float>>& centroids,
        const std::vector<std::vector<float>>& data);
    
    std::vector<std::vector<float>> calculateInnerProductsOptimized(
        const std::vector<std::vector<float>>& centroids,
        const std::vector<std::vector<float>>& data);
    
    // Utility functions
    std::vector<float> calculateWithSingleCentroid(
        const std::vector<float>& centroid,
        const std::vector<std::vector<float>>& data);
    
    std::vector<std::vector<std::pair<float, size_t>>> findTopKSimilar(
        const std::vector<std::vector<float>>& centroids,
        const std::vector<std::vector<float>>& data,
        int k);
    
    // Benchmarking
    void benchmark(const std::vector<std::vector<float>>& centroids,
                   const std::vector<std::vector<float>>& data,
                   int num_iterations = 10);
    
    // Getters
    int getDimension() const { return dimension; }
};

#endif // BATCH_INNER_PRODUCT_CALCULATOR_H
