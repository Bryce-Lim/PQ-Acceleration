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

#ifndef AMX_INNER_PRODUCT_H
#define AMX_INNER_PRODUCT_H

#include <string>
#include <vector>
#include <immintrin.h>
#include <stdint.h>
#include <stdbool.h>
#include <chrono>

#define MAX_SIZE 16
#define MAX_COLS 32
#define STRIDE 64
#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18

// Define bfloat16 type
typedef uint16_t bfloat16_t;

// Define tile config data structure
typedef struct __tile_config
{
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved_0[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} __tilecfg;

class AMXInnerProduct
{
private:
    bool amx_initialized;

    // Timing instance variables
    std::chrono::duration<double> total_compute_time;
    std::chrono::duration<double> padding_time;
    std::chrono::duration<double> conversion_time;
    std::chrono::duration<double> chunking_time;
    std::chrono::duration<double> multiplication_time;
    std::chrono::duration<double> tile_setup_time;
    std::chrono::duration<double> actual_amx_time;
    std::chrono::duration<double> tile_load_time;

    // Timing counters
    size_t compute_calls;
    size_t chunking_calls;
    size_t multiplication_calls;

    // Helper functions for type conversion
    static bfloat16_t float_to_bfloat16(float f);
    static float bfloat16_to_float(bfloat16_t bf16);

    // Tile configuration and buffer initialization
    static void init_tile_config(__tilecfg *tileinfo);
    static void init_buffer_float(float *buf, float value);

    // Core computation methods
    static void tile_1_init(std::vector<bfloat16_t> buf, std::vector<std::vector<bfloat16_t>> &vectors);
    static void tile_2_init(std::vector<bfloat16_t> buf, std::vector<std::vector<bfloat16_t>> &vectors);
    void main_multiply(float *result, std::vector<std::vector<bfloat16_t>> &centroids,
                      std::vector<std::vector<bfloat16_t>> &data);
    void chunking(std::vector<std::vector<float>> &results_agg,
                 std::vector<std::vector<float>> &centroids,
                 std::vector<std::vector<float>> &data);

    // Utility methods
    static void padVectors(std::vector<std::vector<float>> &vectors);
    bool set_tiledata_use();

    // Add these helper methods
    int calculate_optimal_threads(size_t data_size, size_t available_memory = 0);
    
    // Thread result structure
    struct ThreadResult {
        std::vector<std::vector<float>> results;
        std::chrono::duration<double> compute_time;
        bool success;
        std::string error_message;
        
        ThreadResult() : success(false) {}
    };

public:
    // Constructor and destructor
    AMXInnerProduct();
    ~AMXInnerProduct();

    // Main public interface
    bool initialize();
    std::vector<std::vector<float>> compute_inner_products(
        std::vector<std::vector<float>> &centroids,
        std::vector<std::vector<float>> &data
    );

    // Timing methods
    void reset_timers();
    void print_timing_stats() const;

    // Individual timing getters (in milliseconds)
    double get_total_compute_time_ms() const;
    double get_padding_time_ms() const;
    double get_conversion_time_ms() const;
    double get_chunking_time_ms() const;
    double get_multiplication_time_ms() const;
    double get_tile_setup_time_ms() const;
    double get_actual_amx_time_ms() const;
    double get_tile_load_time_ms() const;

    // Get timing statistics
    double get_average_multiplication_time_ms() const;
    size_t get_multiplication_call_count() const;

    // Utility methods for printing/debugging
    static void print_buffer_bf16(std::vector<bfloat16_t> buf, int32_t rows, int32_t cols);
    static void print_buffer_float(float *buf, int32_t rows, int32_t cols);
    static void print_bfloat16_vectors(const std::vector<std::vector<bfloat16_t>> &vecs);
    static void print_float_vectors(const std::vector<std::vector<float>> &vecs);

    // Check if AMX is properly initialized
    bool is_initialized() const { return amx_initialized; }

    // Multi-threaded interface using data partitioning
    std::vector<std::vector<float>> compute_inner_products_threaded(
        std::vector<std::vector<float>>& centroids,
        std::vector<std::vector<float>>& data,
        int num_threads = 0
    );
    
    // Static method for thread-safe computation
    static std::vector<std::vector<float>> compute_data_partition(
        const std::vector<std::vector<float>>& centroids,
        const std::vector<std::vector<float>>& data_partition,
        int thread_id
    );
    
    // Benchmark different thread counts
    void benchmark_thread_scaling(
        std::vector<std::vector<float>>& centroids,
        std::vector<std::vector<float>>& data
    );

};

#endif
