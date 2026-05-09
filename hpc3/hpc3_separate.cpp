#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <climits>
#include <algorithm>

using namespace std;
using namespace std::chrono;

auto measure = [](string name, auto func) {
    auto start = high_resolution_clock::now();
    func();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << name << " Time: " << duration.count() << " ms\n";
    return duration.count();
};

int main() {
    // Number of elements in the array
    int n = 10000000; // 10 million
    vector<int> arr(n);
    
    // Generate random data
    generate(arr.begin(), arr.end(), []() { return rand() % 100000; });

    cout << "=== Array Size: " << n << " elements ===\n\n";

    int seq_min_val = arr[0], seq_max_val = arr[0];
    long long seq_sum = 0;
    
    int par_min_val = arr[0], par_max_val = arr[0];
    long long par_sum = 0;

    cout << "--- Sequential Execution ---\n";
    auto seq_min_time = measure("Min", [&]() {
        for (int i = 0; i < n; ++i) {
            if (arr[i] < seq_min_val) seq_min_val = arr[i];
        }
    });
    auto seq_max_time = measure("Max", [&]() {
        for (int i = 0; i < n; ++i) {
            if (arr[i] > seq_max_val) seq_max_val = arr[i];
        }
    });
    auto seq_sum_time = measure("Sum", [&]() {
        for (int i = 0; i < n; ++i) {
            seq_sum += arr[i];
        }
    });
    double seq_average = static_cast<double>(seq_sum) / n;
    
    cout << "Results: Min=" << seq_min_val << ", Max=" << seq_max_val << ", Sum=" << seq_sum << ", Avg=" << seq_average << "\n\n";

    cout << "--- Parallel Execution ---\n";
    auto par_min_time = measure("Min", [&]() {
        #pragma omp parallel for reduction(min:par_min_val)
        for (int i = 0; i < n; ++i) {
            if (arr[i] < par_min_val) par_min_val = arr[i];
        }
    });
    auto par_max_time = measure("Max", [&]() {
        #pragma omp parallel for reduction(max:par_max_val)
        for (int i = 0; i < n; ++i) {
            if (arr[i] > par_max_val) par_max_val = arr[i];
        }
    });
    auto par_sum_time = measure("Sum", [&]() {
        #pragma omp parallel for reduction(+:par_sum)
        for (int i = 0; i < n; ++i) {
            par_sum += arr[i];
        }
    });
    double par_average = static_cast<double>(par_sum) / n;

    cout << "Results: Min=" << par_min_val << ", Max=" << par_max_val << ", Sum=" << par_sum << ", Avg=" << par_average << "\n\n";

    cout << "--- Speedup ---\n";
    cout << "Min Speedup: " << (par_min_time > 0 ? static_cast<double>(seq_min_time) / par_min_time : 0) << "x\n";
    cout << "Max Speedup: " << (par_max_time > 0 ? static_cast<double>(seq_max_time) / par_max_time : 0) << "x\n";
    cout << "Sum Speedup: " << (par_sum_time > 0 ? static_cast<double>(seq_sum_time) / par_sum_time : 0) << "x\n";

    return 0;
}
