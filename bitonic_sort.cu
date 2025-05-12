/*
 * Parallel bitonic sort using CUDA.
 * Compile with
 * nvcc -arch=sm_11 bitonic_sort.cu
 * Based on http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
 * License: BSD 3
 * credits: https://gist.github.com/mre/1392067 
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>

// Helper function to check if an array is sorted
bool isSorted(const int* array, int size) {
    for (int i = 1; i < size; i++) {
        if (array[i] < array[i-1]) {
            return false;
        }
    }
    return true;
}

__global__ void bitonic_sort_step(int *dev_values, int j, int k)
{
    /* Sorting partners: i and ixj */
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j;

    /* The threads with the lowest ids sort the array. */
    if ((ixj) > i)
    {
        if ((i & k) == 0)
        {
            /* Sort ascending */
            if (dev_values[i] > dev_values[ixj])
            {
                /* exchange(i,ixj); */
                int temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
        if ((i & k) != 0)
        {
            /* Sort descending */
            if (dev_values[i] < dev_values[ixj])
            {
                /* exchange(i,ixj); */
                int temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
    }
}

// Helper function to get the next power of 2
int nextPowerOf2(int n)
{
    return pow(2, ceil(log2(n)));
}

/**
 * Inplace bitonic sort using CUDA.
 */
void bitonic_sort(int *values, int num_vals)
{
    // Check if num_vals is a power of 2, if not, pad the array
    int orig_num_vals = num_vals;
    int padded_size = nextPowerOf2(num_vals);
    
    // Create a padded array if needed
    int *padded_values = values;
    if (padded_size > num_vals) {
        padded_values = new int[padded_size];
        
        // Copy original values
        for (int i = 0; i < num_vals; i++) {
            padded_values[i] = values[i];
        }
        
        // Fill the rest with INT_MAX (will sort to the end)
        for (int i = num_vals; i < padded_size; i++) {
            padded_values[i] = INT_MAX;
        }
        
        num_vals = padded_size;
    }
    
    int *dev_values;
    size_t size = num_vals * sizeof(int);

    cudaMalloc((void **)&dev_values, size);
    cudaMemcpy(dev_values, padded_values, size, cudaMemcpyHostToDevice);

    const int threads_per_block = 1024; // adjust to whatever
    int blocks = (num_vals + threads_per_block - 1) / threads_per_block;

    int j, k;
    /* Major step */
    for (k = 2; k <= num_vals; k <<= 1)
    {
        /* Minor step */
        for (j = k >> 1; j > 0; j = j >> 1)
        {
            dim3 gridDim(blocks);
            dim3 blockDim(threads_per_block);
            void *args[] = {&dev_values, &j, &k};
            cudaLaunchKernel((void*)bitonic_sort_step, gridDim, blockDim, args, 0, NULL);
            cudaDeviceSynchronize(); // Make sure each kernel completes before next launch
        }
    }
    
    // Copy back the sorted array
    if (padded_size > orig_num_vals) {
        // Copy only the original number of elements back to values
        cudaMemcpy(padded_values, dev_values, size, cudaMemcpyDeviceToHost);
        for (int i = 0; i < orig_num_vals; i++) {
            values[i] = padded_values[i];
        }
        delete[] padded_values;
    } else {
        cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
    }
    
    cudaFree(dev_values);
}

int main()
{
    // Constants for benchmarking
    const int WARMUP_RUNS = 5;
    const int BENCHMARK_RUNS = 10000;
    
    // Read integers from file
    std::vector<int> original_values;
    std::ifstream file("random_array_of_10^4_integers.txt");
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open array.txt" << std::endl;
        return 1;
    }
    
    int num;
    while (file >> num) {
        original_values.push_back(num);
    }
    file.close();
    
    int num_vals = original_values.size();
    std::cout << "Array size: " << num_vals << " elements" << std::endl;
    
    // Create array from vector for original values
    int* original_array = new int[num_vals];
    for (int i = 0; i < num_vals; i++) {
        original_array[i] = original_values[i];
    }
    
    // Vector to store all benchmark run times
    std::vector<int> run_times;
    
    // Warm-up runs
    std::cout << "Performing " << WARMUP_RUNS << " warm-up runs..." << std::endl;
    for (int i = 0; i < WARMUP_RUNS; i++) {
        // Create a copy of the original array for each run
        int* array_copy = new int[num_vals];
        for (int j = 0; j < num_vals; j++) {
            array_copy[j] = original_array[j];
        }
        
        // Sort and ignore timing
        bitonic_sort(array_copy, num_vals);
        
        // Check if array is sorted after first warm-up run only
        if (i == 0) {
            bool sorted = isSorted(array_copy, num_vals);
            if (!sorted) {
                std::cerr << "ERROR: Array not correctly sorted after first warm-up run!" << std::endl;
                delete[] array_copy;
                delete[] original_array;
                return 1;  // Exit with error code
            }
            std::cout << "Array correctly sorted, continuing with benchmark." << std::endl;
        }
        
        delete[] array_copy;
    }
    
    // Benchmark runs
    std::cout << "\nPerforming " << BENCHMARK_RUNS << " benchmark runs..." << std::endl;
    for (int i = 0; i < BENCHMARK_RUNS; i++) {
        // Create a copy of the original array for each run
        int* array_copy = new int[num_vals];
        for (int j = 0; j < num_vals; j++) {
            array_copy[j] = original_array[j];
        }
        
        // Create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // Start timing
        cudaEventRecord(start);
        
        // Sort
        bitonic_sort(array_copy, num_vals);
        
        // Stop timing
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        // Calculate elapsed time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        // Convert to integer microseconds
        int microseconds = static_cast<int>(milliseconds * 1000.0f);
        
        // Store the duration
        run_times.push_back(microseconds);
        
        // Clean up
        delete[] array_copy;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    // Calculate statistics
    // 1. Mean
    double mean = std::accumulate(run_times.begin(), run_times.end(), 0.0) / run_times.size();
    
    // 2. Median (sort the vector first)
    std::vector<int> sorted_times = run_times;
    std::sort(sorted_times.begin(), sorted_times.end());
    double median = sorted_times.size() % 2 == 0 
                  ? (sorted_times[sorted_times.size()/2 - 1] + sorted_times[sorted_times.size()/2]) / 2.0
                  : sorted_times[sorted_times.size()/2];
    
    // 3. Standard Deviation
    double variance = 0.0;
    for (const auto& time : run_times) {
        variance += std::pow(time - mean, 2);
    }
    variance /= run_times.size();
    double std_dev = std::sqrt(variance);
    
    // Output statistics
    std::cout << "\nStatistics for " << BENCHMARK_RUNS << " runs (in microseconds):" << std::endl;
    std::cout << "Mean: " << mean << std::endl;
    std::cout << "Median: " << median << std::endl;
    std::cout << "Standard Deviation: " << std_dev << std::endl;
    
    // Clean up
    delete[] original_array;
    
    return 0;
}

