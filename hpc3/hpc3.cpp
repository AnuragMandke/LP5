#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <climits>
#include <algorithm>

using namespace std;
using namespace std::chrono;

int main() {
    // Number of elements in the array
    int n = 10000000; // 10 million
    vector<int> arr(n);
    
    // Generate random data
    generate(arr.begin(), arr.end(), []() { return rand() % 100000; });

    int min_val = arr[0];
    int max_val = arr[0];
    long long sum = 0;

    auto start = high_resolution_clock::now();

    // Parallel Reduction for Min, Max, and Sum in a single pass
    #pragma omp parallel for reduction(min:min_val) reduction(max:max_val) reduction(+:sum)
    for (int i = 0; i < n; ++i) {
        if (arr[i] < min_val) min_val = arr[i];
        if (arr[i] > max_val) max_val = arr[i];
        sum += arr[i];
    }

    // Average is simply sum divided by number of elements
    double average = static_cast<double>(sum) / n;

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    cout << "=== Parallel Reduction Results ===\n";
    cout << "Array Size: " << n << " elements\n";
    cout << "Min:        " << min_val << "\n";
    cout << "Max:        " << max_val << "\n";
    cout << "Sum:        " << sum << "\n";
    cout << "Average:    " << average << "\n";
    cout << "Time Taken: " << duration.count() << " ms\n";

    return 0;
}
