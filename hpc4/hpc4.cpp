#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

// ---------------------------------------------------------
// CUDA Kernel for Vector Addition
// ---------------------------------------------------------
__global__ void vectorAdd(int* a, int* b, int* c, int n) {
    // Calculate global thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// ---------------------------------------------------------
// CUDA Kernel for Matrix Multiplication
// ---------------------------------------------------------
__global__ void matrixMul(int* a, int* b, int* c, int N) {
    // Calculate global row and column
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (row < N && col < N) {
        int sum = 0;
        // Dot product of row of 'a' and column of 'b'
        for (int i = 0; i < N; ++i) {
            sum += a[row * N + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}

void performVectorAddition() {
    int n = 1 << 24; // ~16.7 Million elements
    size_t size = n * sizeof(int);

    // Host memory allocation
    int *h_a = new int[n];
    int *h_b = new int[n];
    int *h_c = new int[n];

    // Initialize host vectors
    for (int i = 0; i < n; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Device memory allocation
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy data from Host to Device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define Grid and Block dimensions (1D)
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch Vector Add Kernel
    auto start = chrono::high_resolution_clock::now();
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize(); // Wait for GPU to finish
    auto stop = chrono::high_resolution_clock::now();

    // Copy result back to Host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    cout << "=== Vector Addition ===" << endl;
    cout << "Elements: " << n << endl;
    cout << "Time taken: " << chrono::duration_cast<chrono::milliseconds>(stop - start).count() << " ms" << endl;
    
    // Verify result
    bool correct = true;
    for (int i = 0; i < 10; ++i) { // Check first 10
        if (h_c[i] != h_a[i] + h_b[i]) correct = false;
    }
    cout << "Status: " << (correct ? "SUCCESS" : "FAILED") << endl << endl;

    // Free memory
    delete[] h_a; delete[] h_b; delete[] h_c;
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

void performMatrixMultiplication() {
    int N = 1024; // 1024x1024 Matrix
    size_t size = N * N * sizeof(int);

    // Host memory allocation
    int *h_a = new int[N * N];
    int *h_b = new int[N * N];
    int *h_c = new int[N * N];

    // Initialize host matrices
    for (int i = 0; i < N * N; ++i) {
        h_a[i] = 1;
        h_b[i] = 2;
    }

    // Device memory allocation
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy data from Host to Device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define Grid and Block dimensions (2D)
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // Launch Matrix Mul Kernel
    auto start = chrono::high_resolution_clock::now();
    matrixMul<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    auto stop = chrono::high_resolution_clock::now();

    // Copy result back to Host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    cout << "=== Matrix Multiplication ===" << endl;
    cout << "Matrix Size: " << N << "x" << N << endl;
    cout << "Time taken: " << chrono::duration_cast<chrono::milliseconds>(stop - start).count() << " ms" << endl;

    // Verify result (Since a is 1 and b is 2, the dot product of a row and col is 1*2*N = 2N)
    bool correct = true;
    if (h_c[0] != 2 * N) correct = false;
    cout << "Status: " << (correct ? "SUCCESS" : "FAILED") << endl;

    // Free memory
    delete[] h_a; delete[] h_b; delete[] h_c;
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

int main() {
    // Make sure to compile this with `nvcc` instead of `g++`
    // Example: nvcc hpc4.cpp -o hpc4.exe
    
    performVectorAddition();
    performMatrixMultiplication();
    
    return 0;
}
