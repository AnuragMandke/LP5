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

    // === Sequential Execution ===
    int seq_min_val = arr[0];
    int seq_max_val = arr[0];
    long long seq_sum = 0;

    auto start_seq = high_resolution_clock::now();

    for (int i = 0; i < n; ++i) {
        if (arr[i] < seq_min_val) seq_min_val = arr[i];
        if (arr[i] > seq_max_val) seq_max_val = arr[i];
        seq_sum += arr[i];
    }
    double seq_average = static_cast<double>(seq_sum) / n;

    auto stop_seq = high_resolution_clock::now();
    auto duration_seq = duration_cast<milliseconds>(stop_seq - start_seq);

    // === Parallel Execution ===
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

    cout << "=== Array Size: " << n << " elements ===\n\n";

    cout << "=== Sequential Reduction Results ===\n";
    cout << "Min:        " << seq_min_val << "\n";
    cout << "Max:        " << seq_max_val << "\n";
    cout << "Sum:        " << seq_sum << "\n";
    cout << "Average:    " << seq_average << "\n";
    cout << "Time Taken: " << duration_seq.count() << " ms\n\n";

    cout << "=== Parallel Reduction Results ===\n";
    cout << "Min:        " << min_val << "\n";
    cout << "Max:        " << max_val << "\n";
    cout << "Sum:        " << sum << "\n";
    cout << "Average:    " << average << "\n";
    cout << "Time Taken: " << duration.count() << " ms\n\n";

    if (duration.count() > 0) {
        cout << "Speedup:    " << static_cast<double>(duration_seq.count()) / duration.count() << "x\n";
    } else {
        cout << "Speedup:    N/A (Parallel execution took < 1 ms)\n";
    }

    return 0;
}
