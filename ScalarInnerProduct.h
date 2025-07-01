// ScalarInnerProduct.h
#ifndef SCALAR_INNER_PRODUCT_H
#define SCALAR_INNER_PRODUCT_H

#include <vector>

class ScalarInnerProduct {
public:
    /**
     * Computes the inner product matrix between centroids and data vectors.
     * 
     * @param centroids: Vector of centroid vectors (rows)
     * @param data: Vector of data vectors (columns)
     * @return Matrix where output[row][col] = inner product of centroids[row] and data[col]
     */
    std::vector<std::vector<float>> compute(
        const std::vector<std::vector<float>>& centroids,
        const std::vector<std::vector<float>>& data
    );
    
    /**
     * Helper function to print a matrix
     */
    void printMatrix(const std::vector<std::vector<float>>& matrix);
};

#endif // SCALAR_INNER_PRODUCT_H
