/*
 * Quicksort using C++
 * credits: https://gist.github.com/christophewang/ad056af4b3ab0ceebacf 
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>  // For std::sort
#include <numeric>    // For std::accumulate
#include <cmath>      // For std::sqrt

void quickSort(std::vector<int>& array, int low, int high)
{
    int i = low;
    int j = high;
    int pivot = array[(i + j) / 2];
    int temp;

    while (i <= j)
    {
        while (array[i] < pivot)
            i++;
        while (array[j] > pivot)
            j--;
        if (i <= j)
        {
            temp = array[i];
            array[i] = array[j];
            array[j] = temp;
            i++;
            j--;
        }
    }
    if (j > low)
        quickSort(array, low, j);
    if (i < high)
        quickSort(array, i, high);
}

bool isSorted(const std::vector<int>& array) {
    for (size_t i = 1; i < array.size(); i++) {
        if (array[i] < array[i-1]) {
            return false;
        }
    }
    return true;
}

int main()
{
    // Constants for benchmarking
    const int WARMUP_RUNS = 5;
    const int BENCHMARK_RUNS = 10000;
    
    // Read the input file once
    std::vector<int> original_array;
    std::ifstream file("random_array_of_10^4_integers.txt");
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open array.txt" << std::endl;
        return 1;
    }
    
    // Read integers from file
    int num;
    while (file >> num) {
        original_array.push_back(num);
    }
    file.close();
    
    int n = original_array.size();
    std::cout << "Array size: " << n << " elements" << std::endl;
    
    // Vector to store all benchmark run times
    std::vector<long long> run_times;
    
    // Warm-up runs
    std::cout << "Performing " << WARMUP_RUNS << " warm-up runs..." << std::endl;
    for (int i = 0; i < WARMUP_RUNS; i++) {
        // Create a copy of the original array for each run
        std::vector<int> array_copy = original_array;
        
        // Sort and ignore timing
        quickSort(array_copy, 0, n - 1);
        
        // Check if array is sorted after first warm-up run only
        if (i == 0) {
            bool sorted = isSorted(array_copy);
            if (!sorted) {
                std::cerr << "ERROR: Array not correctly sorted after first warm-up run!" << std::endl;
                return 1;  // Exit with error code
            }
            std::cout << "Array correctly sorted, continuing with benchmark." << std::endl;
        }
    }
    
    // Benchmark runs
    std::cout << "\nPerforming " << BENCHMARK_RUNS << " benchmark runs..." << std::endl;
    for (int i = 0; i < BENCHMARK_RUNS; i++) {
        // Create a copy of the original array for each run
        std::vector<int> array_copy = original_array;
        
        // Start timer
        auto start = std::chrono::high_resolution_clock::now();
        
        // Sort
        quickSort(array_copy, 0, n - 1);
        
        // End timer
        auto end = std::chrono::high_resolution_clock::now();
        
        // Calculate elapsed time
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Store the duration
        run_times.push_back(duration.count());
    }
    
    // Calculate statistics
    // 1. Mean
    double mean = std::accumulate(run_times.begin(), run_times.end(), 0.0) / run_times.size();
    
    // 2. Median (sort the vector first)
    std::vector<long long> sorted_times = run_times;
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
    
    return 0;
}