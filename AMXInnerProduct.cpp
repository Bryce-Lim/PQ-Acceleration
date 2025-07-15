//==============================================================//
// Author: Bryce Lim, 2025                                      //
// SPDX-License-Identifier: MIT                                 //
//                                                              //
// This code is based in part on Intel's 2022 starter code:     //
// https://github.com/intel/AMX-TMUL-Code-Samples               //
// Licensed under the MIT License.                              //
//                                                              //
// Modifications and extensions by Bryce Lim, 2025              //
//==============================================================//

#include "AMXInnerProduct.h"
#include <iostream>
#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdlib.h>
#include <cstring>
#include <iomanip>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <vector>
#include <immintrin.h>
#include <cstdint>
#include <thread>
#include <future>
#include <atomic>

// Constructor
AMXInnerProduct::AMXInnerProduct() : amx_initialized(false)
{
    reset_timers();
}

// Destructor
AMXInnerProduct::~AMXInnerProduct()
{
    if (amx_initialized)
    {
        _tile_release();
    }
}

// Initialize AMX functionality
bool AMXInnerProduct::initialize()
{
    if (!set_tiledata_use())
    {
        amx_initialized = false;
        return false;
    }
    amx_initialized = true;
    return true;
}

// Reset all timing counters
void AMXInnerProduct::reset_timers()
{
    total_compute_time = std::chrono::duration<double>::zero();
    padding_time = std::chrono::duration<double>::zero();
    conversion_time = std::chrono::duration<double>::zero();
    chunking_time = std::chrono::duration<double>::zero();
    multiplication_time = std::chrono::duration<double>::zero();
    tile_setup_time = std::chrono::duration<double>::zero();
    actual_amx_time = std::chrono::duration<double>::zero();
    tile_load_time = std::chrono::duration<double>::zero();

    compute_calls = 0;
    chunking_calls = 0;
    multiplication_calls = 0;
}

// Print comprehensive timing statistics
void AMXInnerProduct::print_timing_stats() const
{
    std::cout << "\n=== AMX Inner Product Timing Statistics ===\n";
    std::cout << std::fixed << std::setprecision(3);

    std::cout << " Total compute time:        " << std::setw(8) << get_total_compute_time_ms() << " ms\n";
    std::cout << " - Padding time:            " << std::setw(8) << get_padding_time_ms() << " ms\n";
    std::cout << " - Chunking time:           " << std::setw(8) << get_chunking_time_ms() << " ms\n";
    std::cout << "   - Conversion time:       " << std::setw(8) << get_conversion_time_ms() << " ms\n";
    std::cout << "   - Multiplication time:   " << std::setw(8) << get_multiplication_time_ms() << " ms\n";
    std::cout << "     - Result merging time: " << std::setw(8) << get_tile_setup_time_ms() << " ms\n";
    std::cout << "     - Actual AMX time:     " << std::setw(8) << get_actual_amx_time_ms() << " ms\n";
    std::cout << "     - Tile load time:      " << std::setw(8) << get_tile_load_time_ms() << " ms\n";

    std::cout << "===========================================\n\n";
}

// Timing getter methods
double AMXInnerProduct::get_total_compute_time_ms() const
{
    return total_compute_time.count() * 1000.0;
}

double AMXInnerProduct::get_padding_time_ms() const
{
    return padding_time.count() * 1000.0;
}

double AMXInnerProduct::get_conversion_time_ms() const
{
    return conversion_time.count() * 1000.0;
}

double AMXInnerProduct::get_chunking_time_ms() const
{
    return chunking_time.count() * 1000.0;
}

double AMXInnerProduct::get_multiplication_time_ms() const
{
    return multiplication_time.count() * 1000.0;
}

double AMXInnerProduct::get_tile_setup_time_ms() const
{
    return tile_setup_time.count() * 1000.0;
}

double AMXInnerProduct::get_actual_amx_time_ms() const
{
    return actual_amx_time.count() * 1000.0;
}

double AMXInnerProduct::get_tile_load_time_ms() const
{
    return tile_load_time.count() * 1000.0;
}

double AMXInnerProduct::get_average_multiplication_time_ms() const
{
    if (multiplication_calls == 0)
        return 0.0;
    return get_multiplication_time_ms() / multiplication_calls;
}

size_t AMXInnerProduct::get_multiplication_call_count() const
{
    return multiplication_calls;
}

// Main public interface for computing inner products
std::vector<std::vector<float>> AMXInnerProduct::compute_inner_products(
    std::vector<std::vector<float>> &centroids,
    std::vector<std::vector<float>> &data)
{

    auto start_total = std::chrono::high_resolution_clock::now();

    if (!amx_initialized)
    {
        throw std::runtime_error("AMX not initialized. Call initialize() first.");
    }

    // Time padding
    auto start_padding = std::chrono::high_resolution_clock::now();
    padVectors(centroids);
    padVectors(data);
    auto end_padding = std::chrono::high_resolution_clock::now();
    padding_time += end_padding - start_padding;

    // Prepare result matrix
    std::vector<std::vector<float>> results_agg(centroids.size(), std::vector<float>(data.size()));

    // Perform the computation (timing handled inside chunking)
    chunking(results_agg, centroids, data);

    auto end_total = std::chrono::high_resolution_clock::now();
    total_compute_time += end_total - start_total;
    compute_calls++;

    return results_agg;
}

// Helper function to convert float to bfloat16
bfloat16_t AMXInnerProduct::float_to_bfloat16(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));

    // Round to nearest even and truncate to bfloat16
    uint32_t rounding_bias = 0x00007FFF + ((bits >> 16) & 1);
    return static_cast<bfloat16_t>((bits + rounding_bias) >> 16);
}

// Helper function to convert bfloat16 to float
float AMXInnerProduct::bfloat16_to_float(bfloat16_t bf16)
{
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

// Initialize tile config
void AMXInnerProduct::init_tile_config(__tilecfg *tileinfo)
{
    int i;
    tileinfo->palette_id = 1;
    tileinfo->start_row = 0;

    // Tile 1: accumulator (float32)
    tileinfo->colsb[0] = MAX_SIZE * sizeof(float);
    tileinfo->rows[0] = MAX_SIZE;

    // Tiles 2,3: bfloat16 operands
    for (i = 1; i < 4; ++i)
    {
        tileinfo->colsb[i] = MAX_COLS * sizeof(bfloat16_t);
        tileinfo->rows[i] = MAX_SIZE;
    }

    _tile_loadconfig(tileinfo);
}

// Initialize float buffer for accumulator
void AMXInnerProduct::init_buffer_float(float *buf, float value)
{
    int rows = MAX_SIZE;
    int cols = MAX_SIZE;

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            buf[i * cols + j] = value;
        }
    }
}

// Set_tiledata_use() - Invoke syscall to set ARCH_SET_STATE_USE
bool AMXInnerProduct::set_tiledata_use()
{
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA))
    {
        // printf("\n Fail to do XFEATURE_XTILEDATA \n\n");
        return false;
    }
    else
    {
        // printf("\n TILE DATA USE SET - OK \n\n");
        return true;
    }
}

// Padding Vectors
void AMXInnerProduct::padVectors(std::vector<std::vector<float>> &vectors)
{
    // 1. Reserve outer vector capacity upfront
    int padded_size = (vectors.size() + 15) & ~15;
    vectors.reserve(padded_size);
    vectors.resize(padded_size);

    // 2. Optimize inner vector padding
    for (auto &vec : vectors)
    {
        if (vec.empty())
        {
            vec.resize(MAX_COLS);
        }
        else
        {
            size_t current_size = vec.size();
            size_t padded_inner_size = ((current_size + MAX_COLS - 1) / MAX_COLS) * MAX_COLS;
            vec.resize(padded_inner_size);
        }
    }
}

void AMXInnerProduct::chunking(std::vector<std::vector<float>> &results_agg,
                               std::vector<std::vector<float>> &centroids,
                               std::vector<std::vector<float>> &data)
{
    int centroid_height = centroids.size() / MAX_SIZE;
    int data_height = data[0].size() / MAX_COLS;

    auto start_chunking = std::chrono::high_resolution_clock::now();

    float results_chunk[MAX_SIZE * MAX_SIZE];
    std::vector<std::vector<bfloat16_t>> centroid_chunk(centroids.size() * centroids[0].size() / (MAX_COLS * MAX_SIZE), std::vector<bfloat16_t>(MAX_COLS * MAX_SIZE));

    // Chunk and format centroids
    auto start_conversion = std::chrono::high_resolution_clock::now();
    int index = 0;
    for (int offset = 0; offset < centroids[0].size(); offset += MAX_COLS)
    {
        for (int i = 0; i < centroids.size(); ++i)
        {
            for (int j = 0; j < MAX_COLS; ++j)
            {
                centroid_chunk[index / (MAX_COLS * MAX_SIZE)][index % (MAX_COLS * MAX_SIZE)] = float_to_bfloat16(centroids[i][offset + j]);
                index++;
            }
        }
    }
    auto end_conversion = std::chrono::high_resolution_clock::now();
    conversion_time += end_conversion - start_conversion;

    // PRE-CONVERT ALL DATA CHUNKS AND TIME THEM SEPARATELY
    std::vector<std::vector<bfloat16_t>> all_data_chunks;
    static int conversion_count = 0;

    for (int offset = 0; offset < data.size(); offset += MAX_SIZE)
    {
        for (int d_offset = 0; d_offset < data[0].size(); d_offset += MAX_COLS)
        {
            std::vector<bfloat16_t> data_chunk(MAX_COLS * MAX_SIZE);

            auto start_conversion = std::chrono::high_resolution_clock::now();
            
	    int k = 0;
	    for (int i = 0; i < MAX_COLS; i += 2) {
    		for (int j = 0; j < MAX_SIZE; ++j) {
        	    data_chunk[k++] = float_to_bfloat16(data[offset + j][d_offset + i]);
        	    data_chunk[k++] = float_to_bfloat16(data[offset + j][d_offset + i + 1]);
    		}
	    }	

	    conversion_count++;
            auto end_conversion = std::chrono::high_resolution_clock::now();
            conversion_time += end_conversion - start_conversion;

            all_data_chunks.push_back(std::move(data_chunk));
        }
    }

    // Tile init!
    __tilecfg tile_data = {0};
    init_tile_config(&tile_data);

    int id = 0;
    int centroid_id = 0;
    int chunk_index = 0;

    auto start_multiply = std::chrono::high_resolution_clock::now();

    // NOW DO ALL THE COMPUTATION USING PRE-CONVERTED CHUNKS
    for (int offset = 0; offset < data.size(); offset += MAX_SIZE)
    {
        for (int d_offset = 0; d_offset < data[0].size(); d_offset += MAX_COLS)
        {
            for (int i = 0; i < centroid_height; ++i)
            {
                _tile_zero(1);
                _tile_loadd(2, centroid_chunk[centroid_id].data(), STRIDE);
                _tile_loadd(3, all_data_chunks[chunk_index].data(), STRIDE);

                _tile_dpbf16ps(1, 2, 3);
                _tile_stored(1, results_chunk, STRIDE);

                // Merge results (same as before)
                int col_offset = (id / data_height) * MAX_SIZE;
                for (int row = 0; row < MAX_SIZE; ++row)
                {
                    if (row + 2 < MAX_SIZE)
                    {
                        _mm_prefetch(&results_chunk[(row + 2) * MAX_SIZE], _MM_HINT_T0);
                        _mm_prefetch(&results_agg[i * MAX_SIZE + (row + 2)][col_offset], _MM_HINT_T0);
                    }
                    float *chunk_row = &results_chunk[row * MAX_SIZE];
                    float *agg_row = &results_agg[i * MAX_SIZE + row][col_offset];

                    int col = 0;
                    for (; col <= MAX_SIZE - 16; col += 16)
                    {
                        __m512 chunk_vec1 = _mm512_loadu_ps(&chunk_row[col]);
                        __m512 agg_vec1 = _mm512_loadu_ps(&agg_row[col]);
                        __m512 result1 = _mm512_add_ps(agg_vec1, chunk_vec1);
                        _mm512_storeu_ps(&agg_row[col], result1);
                    }
                }

                centroid_id = (centroid_id + 1) % centroid_chunk.size();
            }
            chunk_index++;
            id++;
        }
    }

    auto end_multiply = std::chrono::high_resolution_clock::now();
    multiplication_time += end_multiply - start_multiply;

//    std::cout << "Conversion loop executed " << conversion_count << " times" << std::endl;
    auto end_chunking = std::chrono::high_resolution_clock::now();
    chunking_time += end_chunking - start_chunking;
    chunking_calls++;
}

// Print bfloat16 buffer
void AMXInnerProduct::print_buffer_bf16(std::vector<bfloat16_t> buf, int32_t rows, int32_t cols)
{
    printf("BFloat16 Buffer:\n");
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            printf("%6.2f", bfloat16_to_float(buf[i * cols + j]));
        }
        printf("\n");
    }
    printf("\n");
}

// Print float buffer
void AMXInnerProduct::print_buffer_float(float *buf, int32_t rows, int32_t cols)
{
    printf("Float32 Result Buffer:\n");
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            printf("%8.2f", buf[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Print vector of bfloat16 vectors
void AMXInnerProduct::print_bfloat16_vectors(const std::vector<std::vector<bfloat16_t>> &vecs)
{
    std::cout << "Vector of vectors (bfloat16):\n";

    for (size_t i = 0; i < vecs.size(); ++i)
    {
        std::cout << "Vector " << i << ": [";
        for (size_t j = 0; j < vecs[i].size(); ++j)
        {
            std::cout << bfloat16_to_float(vecs[i][j]);
            if (j < vecs[i].size() - 1)
            {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";
    }
    std::cout << "]\n";
}

// Print vector of float vectors
void AMXInnerProduct::print_float_vectors(const std::vector<std::vector<float>> &vecs)
{
    std::cout << "INNER PRODUCT CALCULATION - Vector of vectors (float32):\n";

    for (size_t i = 0; i < vecs.size(); ++i)
    {
        std::cout << "Vector " << i << ": [";
        for (size_t j = 0; j < vecs[i].size(); ++j)
        {
            std::cout << vecs[i][j];
            if (j < vecs[i].size() - 1)
            {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";
    }
    std::cout << "\n";
}

std::vector<std::vector<float>> AMXInnerProduct::compute_inner_products_threaded(
    std::vector<std::vector<float>> &centroids,
    std::vector<std::vector<float>> &data,
    int num_threads)
{
    auto start_total = std::chrono::high_resolution_clock::now();
    std::cout << "Using " << num_threads << " threads for computation" << std::endl;

    // Calculate partition sizes
    const size_t data_per_thread = (data.size() + num_threads - 1) / num_threads;
    std::vector<std::future<ThreadResult>> futures;

    // Launch threads
    for (int t = 0; t < num_threads; ++t)
    {
        size_t start_idx = t * data_per_thread;
        size_t end_idx = std::min(start_idx + data_per_thread, data.size());

        if (start_idx >= end_idx)
            continue;

        // Create data partition
        std::vector<std::vector<float>> data_partition(data.begin() + start_idx, data.begin() + end_idx);

        // Launch async computation
        futures.emplace_back(
            std::async(std::launch::async, [centroids, data_partition, t]() -> ThreadResult
                       {
                ThreadResult result;
                auto thread_start = std::chrono::high_resolution_clock::now();
                
                try {
                    result.results = compute_data_partition(centroids, data_partition, t);
                    result.success = true;
                } catch (const std::exception& e) {
                    result.success = false;
                    result.error_message = "Thread " + std::to_string(t) + " error: " + e.what();
                }
                
                auto thread_end = std::chrono::high_resolution_clock::now();
                result.compute_time = thread_end - thread_start;
                return result; })
            );
    }

    // Collect results
    std::vector<ThreadResult> thread_results;
    for (auto &future : futures)
    {
        thread_results.push_back(future.get());
    }

    // Check for errors
    for (const auto &result : thread_results)
    {
        if (!result.success)
        {
            throw std::runtime_error(result.error_message);
        }
    }

    // Merge results
    std::vector<std::vector<float>> final_result(centroids.size(), std::vector<float>(data.size()));

    // Calculate partition sizes upfront
    std::vector<size_t> partition_start_indices(num_threads + 1);
    for (int t = 0; t <= num_threads; ++t)
    {
        partition_start_indices[t] = std::min((size_t)t * data_per_thread, data.size());
    }

    for (int t = 0; t < num_threads; ++t)
    {
        if (!thread_results[t].success)
            continue;

        size_t start_idx = partition_start_indices[t];
        size_t partition_size = partition_start_indices[t + 1] - start_idx;

        for (size_t i = 0; i < centroids.size(); ++i)
        {
            if (i < thread_results[t].results.size() && thread_results[t].results[i].size() == partition_size)
            {
                std::memcpy(&final_result[i][start_idx], thread_results[t].results[i].data(), partition_size * sizeof(float));
            }
        }
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    total_compute_time += end_total - start_total;
    compute_calls++;

    return final_result;
}

// Static method for thread-safe computation
std::vector<std::vector<float>> AMXInnerProduct::compute_data_partition(
    const std::vector<std::vector<float>> &centroids,
    const std::vector<std::vector<float>> &data_partition,
    int thread_id)
{
    // Each thread creates its own AMX instance
    AMXInnerProduct thread_amx;

    if (!thread_amx.initialize())
    {
        throw std::runtime_error("Failed to initialize AMX for thread " + std::to_string(thread_id));
    }

    // Make local copies to avoid threading issues
    std::vector<std::vector<float>> local_centroids = centroids;
    std::vector<std::vector<float>> local_data = data_partition;

    return thread_amx.compute_inner_products(local_centroids, local_data);
}

int AMXInnerProduct::calculate_optimal_threads(size_t data_size, size_t available_memory)
{
    int max_threads = std::thread::hardware_concurrency();

    // Conservative threading based on data size
    if (data_size < 50000)
    {
        return 1;
    }
    else if (data_size < 200000)
    {
        return std::min(2, max_threads);
    }
    else if (data_size < 500000)
    {
        return std::min(4, max_threads);
    }
    else
    {
        return std::min(224, max_threads);
    }
}

void AMXInnerProduct::benchmark_thread_scaling(
    std::vector<std::vector<float>> &centroids,
    std::vector<std::vector<float>> &data)
{
    std::cout << "\n=== AMX Thread Scaling Benchmark ===" << std::endl;

    std::vector<int> thread_counts = {1, 2, 4, 6, 8, 12, 16, 32};
    std::cout << "Hardware concurrency: " << std::thread::hardware_concurrency() << std::endl;

    for (int num_threads : thread_counts)
    {
        if (num_threads > std::thread::hardware_concurrency())
        {
            continue;
        }

        std::cout << "\nTesting with " << num_threads << " threads..." << std::endl;

        // Make copies for each test
        std::vector<std::vector<float>> centroids_copy = centroids;
        std::vector<std::vector<float>> data_copy = data;

        auto start = std::chrono::high_resolution_clock::now();

        try
        {
            auto results = compute_inner_products_threaded(centroids_copy, data_copy, num_threads);

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            std::cout << "  " << num_threads << " threads: " << duration.count() << " microseconds";

            // Calculate speedup vs single-threaded
            if (num_threads > 1)
            {
                // You'd store the single-threaded baseline for comparison
                // For now, just show the timing
                std::cout << " (result dimensions: " << results.size() << " x " << results[0].size() << ")";
            }
            std::cout << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cout << "  " << num_threads << " threads: FAILED - " << e.what() << std::endl;
        }
    }
}
