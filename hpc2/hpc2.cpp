#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <algorithm>

using namespace std;
using namespace std::chrono;

void bubbleSort(vector<int>& arr, bool parallel) {
    int n = arr.size();
    if (parallel) {
        for (int p = 0; p < n; ++p) {
            #pragma omp parallel for
            for (int i = p % 2; i < n - 1; i += 2)
                if (arr[i] > arr[i + 1]) swap(arr[i], arr[i + 1]);
        }
    } else {
        for (int i = 0; i < n - 1; ++i)
            for (int j = 0; j < n - i - 1; ++j)
                if (arr[j] > arr[j + 1]) swap(arr[j], arr[j + 1]);
    }
}

void mergeSort(vector<int>& arr, int l, int r, bool parallel) {
    if (l < r) {
        int m = l + (r - l) / 2;
        if (parallel && r - l > 1000) {
            #pragma omp task shared(arr)
            mergeSort(arr, l, m, true);
            #pragma omp task shared(arr)
            mergeSort(arr, m + 1, r, true);
            #pragma omp taskwait
        } else {
            mergeSort(arr, l, m, false);
            mergeSort(arr, m + 1, r, false);
        }
        inplace_merge(arr.begin() + l, arr.begin() + m + 1, arr.begin() + r + 1);
    }
}

auto measure = [](string name, vector<int> a, auto sort_func) {
    auto start = high_resolution_clock::now();
    sort_func(a);
    auto end = high_resolution_clock::now();
    
    cout << name << ": " << duration_cast<milliseconds>(end - start).count() << " ms";
    if (is_sorted(a.begin(), a.end())) {
        cout << " (Verified)\n";
    } else {
        cout << " (FAILED)\n";
    }
};

int main() {
    int n_bubble = 100000, n_merge = 1000000;
    vector<int> b_arr(n_bubble), m_arr(n_merge);
    generate(b_arr.begin(), b_arr.end(), rand);
    generate(m_arr.begin(), m_arr.end(), rand);

    cout << "=== Bubble Sort (" << n_bubble << ") ===\n";
    measure("Sequential", b_arr, [&](vector<int>& arr){ bubbleSort(arr, false); });
    measure("Parallel",   b_arr, [&](vector<int>& arr){ bubbleSort(arr, true); });

    cout << "\n=== Merge Sort (" << n_merge << ") ===\n";
    measure("Sequential", m_arr, [&](vector<int>& arr){ mergeSort(arr, 0, arr.size()-1, false); });
    measure("Parallel",   m_arr, [&](vector<int>& arr){ 
        #pragma omp parallel
        #pragma omp single
        mergeSort(arr, 0, arr.size()-1, true); 
    });

    return 0;
}
