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
#include <thread>
#include <future>
#include <mutex>
#include <iomanip>

// Define these constants based on your data
const int dim = 1024;
const int max_elements = 985600;
const int num_centroids = 1600;
const int rounds = 10;
const std::vector<int> thread_tests = {60, 80, 100, 112, 120, 160, 200, 224};
const std::string dataroot = "/mnt/ceph/district9/dataset/openai/openai_large_5m/";

// Thread-safe AMX wrapper for data partitioning
class ThreadedAMXInnerProduct
{
private:
    struct ThreadResult
    {
        std::vector<std::vector<float>> results;
        std::chrono::duration<double> compute_time;
        bool success;
        std::string error_message;
    };

    int calculate_optimal_threads(size_t data_size)
    {
        // Conservative approach: limit threads based on data size and memory
        int max_threads = std::thread::hardware_concurrency();

        // For very large datasets, use more threads
        if (data_size > 500000)
        {
            return std::min(224, max_threads);
        }
        else if (data_size > 100000)
        {
            return std::min(4, max_threads);
        }
        else if (data_size > 50000)
        {
            return std::min(2, max_threads);
        }
        else
        {
            return 1; // Single threaded for small datasets
        }
    }

    void compute_partition(
        const std::vector<std::vector<float>> &centroids,
        const std::vector<std::vector<float>> &data_partition,
        ThreadResult &result,
        int thread_id)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        try
        {
            // Each thread gets its own AMX instance
            AMXInnerProduct thread_amx;
            if (!thread_amx.initialize())
            {
                result.success = false;
                result.error_message = "Failed to initialize AMX for thread " + std::to_string(thread_id);
                return;
            }

            // Make local copies to avoid shared memory issues
            std::vector<std::vector<float>> local_centroids = centroids;
            std::vector<std::vector<float>> local_data = data_partition;

            // Compute inner products for this partition
            result.results = thread_amx.compute_inner_products(local_centroids, local_data);
            result.success = true;
        }
        catch (const std::exception &e)
        {
            result.success = false;
            result.error_message = "Thread " + std::to_string(thread_id) + " error: " + e.what();
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        result.compute_time = end_time - start_time;
    }

public:
    std::vector<std::vector<float>> compute_inner_products_threaded(
        std::vector<std::vector<float>> &centroids,
        std::vector<std::vector<float>> &data,
        int num_threads = 0)
    {
        if (num_threads == 0)
        {
            num_threads = calculate_optimal_threads(data.size());
        }

        std::cout << "Using " << num_threads << " threads for " << data.size() << " data vectors" << std::endl;

        // If only one thread is optimal, use single-threaded approach
        if (num_threads == 1)
        {
            AMXInnerProduct single_amx;
            if (!single_amx.initialize())
            {
                throw std::runtime_error("Failed to initialize AMX");
            }
            return single_amx.compute_inner_products(centroids, data);
        }

        // Calculate partition sizes
        const size_t data_per_thread = (data.size() + num_threads - 1) / num_threads;
        std::vector<std::future<void>> futures;
        std::vector<ThreadResult> thread_results(num_threads);

        // Launch threads
        for (int t = 0; t < num_threads; ++t)
        {
            // Calculate data range for this thread
            size_t start_idx = t * data_per_thread;
            size_t end_idx = std::min(start_idx + data_per_thread, data.size());

            if (start_idx >= end_idx)
            {
                // No data for this thread
                thread_results[t].success = true;
                thread_results[t].results = std::vector<std::vector<float>>(centroids.size());
                continue;
            }

            // Create data partition for this thread
            std::vector<std::vector<float>> data_partition(
                data.begin() + start_idx,
                data.begin() + end_idx);

            // Launch thread
            futures.emplace_back(
                std::async(std::launch::async, [this, &centroids, data_partition, &thread_results, t]()
                           { compute_partition(centroids, data_partition, thread_results[t], t); }));
        }

        // Wait for all threads to complete
        for (auto &future : futures)
        {
            future.wait();
        }

        // Check for errors
        for (int t = 0; t < num_threads; ++t)
        {
            if (!thread_results[t].success)
            {
                throw std::runtime_error(thread_results[t].error_message);
            }
        }

        // Merge results - combine column-wise since we partitioned data
        std::vector<std::vector<float>> final_result(centroids.size());

        // Pre-allocate space for efficiency
        for (size_t i = 0; i < centroids.size(); ++i)
        {
            final_result[i].reserve(data.size());
        }

        // Merge results from each thread
        for (int t = 0; t < num_threads; ++t)
        {
            if (!thread_results[t].results.empty())
            {
                for (size_t i = 0; i < centroids.size(); ++i)
                {
                    if (i < thread_results[t].results.size())
                    {
                        final_result[i].insert(
                            final_result[i].end(),
                            thread_results[t].results[i].begin(),
                            thread_results[t].results[i].end());
                    }
                }
            }
        }

        return final_result;
    }

    // Method to benchmark different thread counts
    void benchmark_thread_scaling(
        std::vector<std::vector<float>> &centroids,
        std::vector<std::vector<float>> &data)
    {
        std::cout << "\n=== Thread Scaling Benchmark ===" << std::endl;

        std::vector<int> thread_counts = thread_tests;

        for (int num_threads : thread_counts)
        {
            if (num_threads > std::thread::hardware_concurrency())
            {
                continue;
            }

            std::cout << "\nTesting with " << num_threads << " threads..." << std::endl;

            // Make copies for thread safety
            std::vector<std::vector<float>> centroids_copy = centroids;
            std::vector<std::vector<float>> data_copy = data;

            auto start = std::chrono::high_resolution_clock::now();

            try
            {
                auto results = compute_inner_products_threaded(centroids_copy, data_copy, num_threads);

                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

                std::cout << "  " << num_threads << " threads: " << duration.count() << " microseconds" << std::endl;

                // Verify result dimensions
                if (results.size() != centroids.size() ||
                    (results.size() > 0 && results[0].size() != data.size()))
                {
                    std::cout << "  WARNING: Result dimensions incorrect!" << std::endl;
                }
            }
            catch (const std::exception &e)
            {
                std::cout << "  " << num_threads << " threads: FAILED - " << e.what() << std::endl;
            }
        }
    }
};

static void differenceAnalyzer(const std::vector<std::vector<float>> &scalar_results,
                               const std::vector<std::vector<float>> &AMX_results)
{
    if (scalar_results.empty() || AMX_results.empty() ||
        scalar_results.size() != AMX_results.size() ||
        scalar_results[0].size() != AMX_results[0].size())
    {
        std::cout << "Cannot compare results: dimension mismatch" << std::endl;
        return;
    }

    float average_error = 0.0f;
    float max_error = 0.0f;
    size_t total_elements = 0;

    for (size_t i = 0; i < scalar_results.size(); i++)
    {
        for (size_t j = 0; j < scalar_results[i].size(); j++)
        {
            float error = std::fabs(AMX_results[i][j] - scalar_results[i][j]);
            average_error += error;
            max_error = std::max(max_error, error);
            total_elements++;
        }
    }

    average_error /= total_elements;
    std::cout << "Average difference between Scalar and AMX: " << average_error << std::endl;
    std::cout << "Largest difference between Scalar and AMX: " << max_error << std::endl;
}

int main()
{
    // Start Timer for Initialization
    auto init_start = std::chrono::high_resolution_clock::now();

    std::cout << "Loading data from parquet files..." << std::endl;

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

        std::cout << "Loading file: " << path << std::endl;

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

    std::cout << "Normalizing vectors..." << std::endl;

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
        {
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

    // Print initialization timing
    auto init_duration = std::chrono::duration_cast<std::chrono::microseconds>(init_end - init_start);
    std::cout << "Preprocessing / Initialization took: " << init_duration.count() << " microseconds\n"
              << std::endl;

    // =========================
    // BENCHMARK COMPARISONS
    // =========================

    std::cout << "=== Performance Benchmarks ===" << std::endl;

    // 1. Single-threaded AMX (baseline)
    std::cout << "\n1. Single-threaded AMX:" << std::endl;
    std::vector<std::vector<float>> centroids_copy = random_centroids;
    std::vector<std::vector<float>> data_copy = data;

    auto start = std::chrono::high_resolution_clock::now();
    AMXInnerProduct single_amx;
    std::vector<std::vector<float>> single_AMX_results;

    if (single_amx.initialize())
    {
        single_amx.reset_timers();
        single_AMX_results = single_amx.compute_inner_products(centroids_copy, data_copy);
        std::cout << "Single-threaded AMX calculation successful!" << std::endl;
    }
    else
    {
        std::cout << "Failed to initialize single-threaded AMX" << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto single_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Single-threaded AMX took: " << single_duration.count() << " microseconds" << std::endl;

    // Print detailed timing from single-threaded AMX
    single_amx.print_timing_stats();

    // 2. Multi-threaded AMX with data partitioning
    std::cout << "\n2. Multi-threaded AMX (Data Partitioning):" << std::endl;

    ThreadedAMXInnerProduct threaded_amx;
    centroids_copy = random_centroids;
    data_copy = data;

    std::vector<std::vector<float>> threaded_AMX_results;
    long total_time = 0;
    for (int i = 0; i < rounds; i++)
    {
        start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<float>> single_run_AMX_results;
        try
        {
            threaded_AMX_results = threaded_amx.compute_inner_products_threaded(centroids_copy, data_copy);
            std::cout << "Multi-threaded AMX calculation successful!" << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cout << "Multi-threaded AMX failed: " << e.what() << std::endl;
        }

        end = std::chrono::high_resolution_clock::now();
        auto threaded_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Multi-threaded AMX took: " << threaded_duration.count() << " microseconds" << std::endl;
        total_time += threaded_duration.count();
        threaded_AMX_results = single_run_AMX_results;
    }
    std::cout << "AVERAGE Multi-threaded AMX took: " << total_time / rounds << " microseconds" << std::endl;

    // 3. HNSWLIB comparison with thread control
    std::cout << "\n3. HNSWLIB with Thread Control:" << std::endl;
    BatchInnerProductCalculator calculator(dim);

    // Test HNSWLIB with different thread counts
    std::vector<int> hnswlib_thread_counts = {224};
    std::vector<long> hnswlib_times;

    total_time = 0;
    for (int i = 0; i < rounds; i++) {
        for (int threads : hnswlib_thread_counts)
        {
            if (threads > std::thread::hardware_concurrency())
                continue;

            std::cout << "\nTesting HNSWLIB with " << threads << " threads:" << std::endl;

            auto hnsw_start = std::chrono::high_resolution_clock::now();
            auto hnsw_results = calculator.calculateInnerProductsOptimizedThreaded(
                random_centroids, data, threads);
            auto hnsw_end = std::chrono::high_resolution_clock::now();

            auto hnsw_duration = std::chrono::duration_cast<std::chrono::microseconds>(hnsw_end - hnsw_start);
            hnswlib_times.push_back(hnsw_duration.count());
            std::cout << "HNSWLIB (" << threads << " threads): " << hnsw_duration.count() << " μs" << std::endl;
            total_time += hnsw_duration.count();
        }
    }
    std::cout << "AVERAGE HNSWLIB: " << total_time / rounds << " μs" << std::endl;

    // =========================
    // ANALYSIS
    // =========================

    // std::cout << "\n=== Performance Analysis ===" << std::endl;
    // 
    // // Calculate speedups
    // if (single_duration.count() > 0 && threaded_duration.count() > 0)
    // {
    //     double speedup = (double)single_duration.count() / threaded_duration.count();
    //     std::cout << "Multi-threaded AMX speedup vs Single-threaded: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    // }

    // // Compare with best HNSWLIB time
    // if (!hnswlib_times.empty() && threaded_duration.count() > 0)
    // {
    //     long best_hnswlib_time = *std::min_element(hnswlib_times.begin(), hnswlib_times.end());
    //     double speedup = (double)best_hnswlib_time / threaded_duration.count();
    //     std::cout << "Multi-threaded AMX speedup vs Best HNSWLIB: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    // }

    // Accuracy comparison
    if (!single_AMX_results.empty() && !threaded_AMX_results.empty())
    {
        std::cout << "\n=== Accuracy Verification ===" << std::endl;
        differenceAnalyzer(single_AMX_results, threaded_AMX_results);
    }

    // Thread scaling benchmarks
    bool benchmark = false;
    if (data.size() > 50000 && benchmark)
    {
        std::cout << "\n=== Thread Scaling Analysis ===" << std::endl;

        std::cout << "\nAMX Thread Scaling:" << std::endl;
        threaded_amx.benchmark_thread_scaling(random_centroids, data);

        std::cout << "\nHNSWLIB Thread Scaling:" << std::endl;
        calculator.benchmarkThreadScaling(random_centroids, data);
    }

    return 0;
}
