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

auto measure = [](string name, auto func) {
    auto start = high_resolution_clock::now();
    func();
    cout << name << ": " << duration_cast<milliseconds>(high_resolution_clock::now() - start).count() << " ms\n";
};

int main() {
    int n_bubble = 100000, n_merge = 1000000;
    vector<int> b_arr(n_bubble), m_arr(n_merge);
    generate(b_arr.begin(), b_arr.end(), rand);
    generate(m_arr.begin(), m_arr.end(), rand);

    cout << "=== Bubble Sort (" << n_bubble << ") ===\n";
    measure("Sequential", [&]{ vector<int> a=b_arr; bubbleSort(a, false); });
    measure("Parallel",   [&]{ vector<int> a=b_arr; bubbleSort(a, true); });

    cout << "\n=== Merge Sort (" << n_merge << ") ===\n";
    measure("Sequential", [&]{ vector<int> a=m_arr; mergeSort(a, 0, a.size()-1, false); });
    measure("Parallel",   [&]{ 
        vector<int> a=m_arr; 
        #pragma omp parallel
        #pragma omp single
        mergeSort(a, 0, a.size()-1, true); 
    });

    return 0;
}
