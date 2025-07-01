#include "AMXInnerProduct.h"
#include "ScalarInnerProduct.h"
#include "BatchInnerProductCalculator.h"
#include "arrow/api.h"
#include "arrow/io/api.h"
#include "arrow/ipc/api.h"
#include "parquet/arrow/reader.h"
#include <algorithm>
#include <random>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

// Define these constants based on your data
const int dim = 512;                                                                               // Adjust to your embedding dimension
const int max_elements = 1000000;                                                                  // Maximum number of vectors to load
const int num_centroids = 100;
const std::string dataroot = "/mnt/ceph/district9/dataset/openai/openai_large_5m/"; // Set your data directory

static void differenceAnalyzer(std::vector<std::vector<float>> scalar_results, std::vector<std::vector<float>> AMX_results) {

    AMXInnerProduct amx_calculator;
    std::vector<std::vector<float>> subtract_results(scalar_results.size(), std::vector<float>(scalar_results[0].size()));
    float average_error = 0.0f;
    float max_error = 0.0f;

    for (int i = 0; i < scalar_results.size(); i++) {
        for (int j = 0; j < scalar_results[0].size(); j++) {
            subtract_results[i][j] = std::fabs(AMX_results[i][j] - scalar_results[i][j]);
            average_error += subtract_results[i][j];
            if (subtract_results[i][j] > max_error) {
                max_error = subtract_results[i][j];
            }
        }
    }

    average_error = average_error / (scalar_results.size() * scalar_results[0].size());
    std::cout << "Average difference between Scalar and AMX -- " << average_error << std::endl;
    std::cout << "Largest difference between Scalar and AMX -- " << max_error << std::endl;
}

int main()
{
    // Start Timer for Initialization
    auto init_start = std::chrono::high_resolution_clock::now();

    // Reading parquet files (0, 1 - 1m size)
    std::vector<std::vector<float>> data;
    data.reserve(max_elements);

    int cnt = 0;
    size_t partition_size = 500000;

    for (int file_idx = 0; file_idx < 2; file_idx++)
    {
        auto pool = arrow::default_memory_pool();
        std::shared_ptr<arrow::io::RandomAccessFile> input;

        std::string path = dataroot + "train-0";
        path += std::to_string(file_idx);
        path += "-of-10.parquet";

        auto maybe_input = arrow::io::ReadableFile::Open(path);
        if (!maybe_input.ok())
        {
            std::cerr << "Error opening file: " << maybe_input.status().ToString() << std::endl;
            return -1;
        }
        input = maybe_input.ValueUnsafe();

        std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
        auto status = parquet::arrow::OpenFile(input, pool, &arrow_reader);
        if (!status.ok())
        {
            std::cerr << "Error opening parquet file: " << status.ToString() << std::endl;
            return -2;
        }

        std::shared_ptr<arrow::Table> table;
        status = arrow_reader->ReadTable(&table);
        if (!status.ok())
        {
            std::cerr << "Error reading table: " << status.ToString() << std::endl;
            return -3;
        }

        auto emb_col = table->column(1);
        if (emb_col->chunks().size() != 1)
        {
            std::cout << "Multiple chunks found: " << emb_col->chunks().size() << std::endl;
        }

        for (auto &arr : emb_col->chunks())
        {
            auto val = std::static_pointer_cast<arrow::DoubleArray>(
                std::static_pointer_cast<arrow::ListArray>(arr)->values());

            for (int i = 0; i < partition_size && data.size() < max_elements; i++)
            {
                std::vector<float> vec(dim);
                for (int j = 0; j < dim; j++)
                {
                    vec[j] = (float)val->Value(i * dim + j);
                }
                data.push_back(vec);
            }
        }
        cnt++;
    }

    // Normalize vectors
    for (auto &emb : data)
    {
        float mag = 0;
        for (int d = 0; d < dim; d++)
        {
            mag += emb[d] * emb[d];
        }
        mag = sqrt(mag);

        if (mag > 0)
        { // Avoid division by zero
            for (int d = 0; d < dim; d++)
            {
                emb[d] /= mag;
            }
        }
    }

    // Sample random centroids
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<std::vector<float>> random_centroids;
    std::sample(data.begin(), data.end(), std::back_inserter(random_centroids), num_centroids, gen);

    std::cout << "Successfully loaded " << data.size() << " vectors of dimension " << data[0].size() << std::endl;
    std::cout << "Sampled " << random_centroids.size() << " random centroids" << std::endl;

    auto init_end = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<float>> centroids_copy = random_centroids;
    std::vector<std::vector<float>> data_copy = data;

    auto start = std::chrono::high_resolution_clock::now();
    AMXInnerProduct amx_calculator;
    std::vector<std::vector<float>> AMX_results;
    if (amx_calculator.initialize())
    {
        amx_calculator.reset_timers();
        AMX_results = amx_calculator.compute_inner_products(centroids_copy, data_copy);
        std::cout << "DEBUG: AMX calculation successful!" << std::endl;
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto hnsw_start = std::chrono::high_resolution_clock::now();
    BatchInnerProductCalculator calculator(dim);
    auto hnsw_results = calculator.calculateInnerProductsOptimized(random_centroids, data);
    auto hnsw_end = std::chrono::high_resolution_clock::now();

    auto scalar_start = std::chrono::high_resolution_clock::now();
    ScalarInnerProduct scalar_calculator;
    std::vector<std::vector<float>> scalar_results = scalar_calculator.compute(random_centroids, data);
    auto scalar_end = std::chrono::high_resolution_clock::now();

    // Print timing information
    auto init_duration = std::chrono::duration_cast<std::chrono::microseconds>(init_end - init_start);
    std::cout << "Preprocessing / Initialization took: " << init_duration.count() << " microseconds\n" << std::endl;

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "AMX Calculation function took: " << duration.count() << " microseconds" << std::endl;
    std::cout << "AMX runtime + Preprocessing took: " << duration.count() + init_duration.count() << " microseconds\n" << std::endl;

    auto scalar_duration = std::chrono::duration_cast<std::chrono::microseconds>(scalar_end - scalar_start);
    std::cout << "Scalar Calculation function took: " << scalar_duration.count() << " microseconds" << std::endl;
    std::cout << "Scalar runtime + Preprocessing took: " << scalar_duration.count() + init_duration.count() << " microseconds\n" << std::endl;

    auto hnsw_duration = std::chrono::duration_cast<std::chrono::microseconds>(hnsw_end - hnsw_start);
    std::cout << "HNSWLIB Calculation function took: " << hnsw_duration.count() << " microseconds" << std::endl;
    std::cout << "HNSWLIB runtime + Preprocessing took: " << hnsw_duration.count() + init_duration.count() << " microseconds\n" << std::endl;

    amx_calculator.print_timing_stats();
    differenceAnalyzer(scalar_results, AMX_results);

//    amx_calculator.print_float_vectors(AMX_results);
//    scalar_calculator.printMatrix(scalar_results);

    return 0;
}
