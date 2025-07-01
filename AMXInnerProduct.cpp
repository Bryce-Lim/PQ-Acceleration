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

    std::cout << "Total compute time:        " << std::setw(8) << get_total_compute_time_ms() << " ms\n";
    std::cout << "- Padding time:          " << std::setw(8) << get_padding_time_ms() << " ms\n";
    std::cout << "- Conversion time:       " << std::setw(8) << get_conversion_time_ms() << " ms\n";
    std::cout << "- Chunking time:         " << std::setw(8) << get_chunking_time_ms() << " ms\n";
    std::cout << "- Multiplication time:   " << std::setw(8) << get_multiplication_time_ms() << " ms\n";
    std::cout << "  - Result merging time: " << std::setw(8) << get_tile_setup_time_ms() << " ms\n";
    std::cout << "  - Actual AMX time:     " << std::setw(8) << get_actual_amx_time_ms() << " ms\n";
    std::cout << "  - Tile load time:      " << std::setw(8) << get_tile_load_time_ms() << " ms\n";

    std::cout << "\nCall counts:\n";
    std::cout << "  - multiplication calls:   " << multiplication_calls << " calls\n";

    if (multiplication_calls > 0)
    {
        std::cout << "  - Avg multiplication:     " << std::setw(8) << get_average_multiplication_time_ms() << " ms/call\n";
    }
    std::cout << "==========================================\n\n";
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

    if (!amx_initialized)
    {
        throw std::runtime_error("AMX not initialized. Call initialize() first.");
    }

    auto start_total = std::chrono::high_resolution_clock::now();

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
        printf("\n Fail to do XFEATURE_XTILEDATA \n\n");
        return false;
    }
    else
    {
        printf("\n TILE DATA USE SET - OK \n\n");
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
    for (auto& vec : vectors) {
        if (vec.empty()) {
            vec.resize(MAX_COLS);
        } else {
            size_t current_size = vec.size();
            size_t padded_inner_size = ((current_size + MAX_COLS - 1) / MAX_COLS) * MAX_COLS;
            vec.resize(padded_inner_size);
        }
    }
}

// Chunking method (now non-static to access timing variables)
void AMXInnerProduct::chunking(std::vector<std::vector<float>> &results_agg,
                               std::vector<std::vector<float>> &centroids,
                               std::vector<std::vector<float>> &data)
{
    int centroid_height = centroids.size() / MAX_SIZE;
    int data_height = data[0].size() / MAX_COLS;

    auto start_chunking = std::chrono::high_resolution_clock::now();

    float results_chunk[MAX_SIZE * MAX_SIZE];
    std::vector<std::vector<bfloat16_t>> centroid_chunk(centroids.size() * centroids[0].size() / (MAX_COLS * MAX_SIZE), std::vector<bfloat16_t>(MAX_COLS * MAX_SIZE));
    std::vector<bfloat16_t> data_chunk(MAX_COLS * MAX_SIZE);

    // Chunk and format centroids
    auto start_conversion = std::chrono::high_resolution_clock::now();
    const int chunk_size = MAX_COLS * MAX_SIZE;
    int chunk_idx = 0;
    int elem_idx = 0;

    for (int offset = 0; offset < centroids[0].size(); offset += MAX_COLS)
    {
        for (int i = 0; i < centroids.size(); ++i)
        {
            // Direct pointer access for better cache performance
            const float *src = &centroids[i][offset];
            auto *dest = &centroid_chunk[chunk_idx][elem_idx];

            // Process in chunks of 8 (or 4/16 depending on your SIMD capabilities)
            int j = 0;
            for (; j <= MAX_COLS - 8; j += 8)
            {
                // Load 8 floats
                __m256 vals = _mm256_loadu_ps(&src[j]);

                // Convert to bfloat16 (you'd need a SIMD version of your conversion)
                // This is pseudocode - actual SIMD bfloat16 conversion depends on your CPU
                for (int k = 0; k < 8; ++k)
                {
                    dest[j + k] = float_to_bfloat16(src[j + k]);
                }
            }

            // Handle remaining elements
            for (; j < MAX_COLS; ++j)
            {
                dest[j] = float_to_bfloat16(src[j]);
            }

            elem_idx += MAX_COLS;
            if (elem_idx >= chunk_size)
            {
                elem_idx = 0;
                ++chunk_idx;
            }
        }
    }

    auto end_conversion = std::chrono::high_resolution_clock::now();
    conversion_time += end_conversion - start_conversion;

    // Tile init!
    __tilecfg tile_data = {0};
    init_tile_config(&tile_data);

    int id = 0;
    int centroid_id = 0;

    for (int offset = 0; offset < data.size(); offset += MAX_SIZE)
    {
        for (int d_offset = 0; d_offset < data[0].size(); d_offset += MAX_COLS)
        {
            start_conversion = std::chrono::high_resolution_clock::now();
            int k = 0;
            for (int i = 0; i < MAX_COLS; i += 2)
            {
                for (int j = 0; j < MAX_SIZE; ++j)
                {
                    data_chunk[k++] = float_to_bfloat16(data[offset + j][d_offset + i]);
                    data_chunk[k++] = float_to_bfloat16(data[offset + j][d_offset + i + 1]);
                }
            }
            end_conversion = std::chrono::high_resolution_clock::now();
            conversion_time += end_conversion - start_conversion;

            for (int i = 0; i < centroid_height; ++i)
            {
		auto start_multiply = std::chrono::high_resolution_clock::now();

                // Multiplying tiles!
                auto start_load = std::chrono::high_resolution_clock::now();
                _tile_zero(1);
                _tile_loadd(2, centroid_chunk[centroid_id].data(), STRIDE);
                _tile_loadd(3, data_chunk.data(), STRIDE);
                auto end_load = std::chrono::high_resolution_clock::now();
                tile_load_time += end_load - start_load;

                auto start_AMX = std::chrono::high_resolution_clock::now();
                _tile_dpbf16ps(1, 2, 3);
                _tile_stored(1, results_chunk, STRIDE);
                auto end_AMX = std::chrono::high_resolution_clock::now();
                actual_amx_time += end_AMX - start_AMX;
                multiplication_calls++;

		// Merging results_chunk into results_agg
                auto start_merge = std::chrono::high_resolution_clock::now();
                for (int row = 0; row < MAX_SIZE; ++row)
                {
                    // Prefetch next row
                    if (row + 1 < MAX_SIZE)
                    {
                        _mm_prefetch(&results_chunk[(row + 1) * MAX_SIZE], _MM_HINT_T0);
                        _mm_prefetch(&results_agg[i * MAX_SIZE + (row + 1)][(id / data_height) * MAX_SIZE], _MM_HINT_T0);
                    }

                    for (int col = 0; col < MAX_SIZE; col += 8)
                    {
                        __m256 chunk_vec = _mm256_loadu_ps(&results_chunk[row * MAX_SIZE + col]);
                        __m256 agg_vec = _mm256_loadu_ps(&results_agg[i * MAX_SIZE + row][(id / data_height) * MAX_SIZE + col]);
                        __m256 result = _mm256_add_ps(agg_vec, chunk_vec);
                        _mm256_storeu_ps(&results_agg[i * MAX_SIZE + row][(id / data_height) * MAX_SIZE + col], result);
                    }
                }
                auto end_merge = std::chrono::high_resolution_clock::now();

		tile_setup_time += end_merge - start_merge;

                centroid_id = (centroid_id + 1) % centroid_chunk.size();
            
		auto end_multiply = std::chrono::high_resolution_clock::now();
                multiplication_time += end_multiply - start_multiply;

	    }

            id++;
        }
    }

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
