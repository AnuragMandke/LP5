#include <iostream>
#include <chrono>
using namespace std;

// ── Kernels ────────────────────────────────────────────────

__global__ void vectorAdd(int* a, int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void matrixMul(int* a, int* b, int* c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        int sum = 0;
        for (int i = 0; i < N; ++i)
            sum += a[row * N + i] * b[i * N + col];
        c[row * N + col] = sum;
    }
}

// ── Helper: allocate + copy a host array to the GPU ────────

int* toGPU(int* h, size_t size) {
    int* d;
    cudaMalloc(&d, size);
    cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);
    return d;
}

// ── Tasks ──────────────────────────────────────────────────

void runVectorAdd() {
    const int n = 1 << 24;
    int *h_a = new int[n], *h_b = new int[n], *h_c = new int[n];
    for (int i = 0; i < n; ++i) { h_a[i] = i; h_b[i] = i * 2; }

    int *d_a = toGPU(h_a, n * sizeof(int));
    int *d_b = toGPU(h_b, n * sizeof(int));
    int *d_c; cudaMalloc(&d_c, n * sizeof(int));

    auto t0 = chrono::high_resolution_clock::now();
    vectorAdd<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    auto ms = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - t0).count();

    cudaMemcpy(h_c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int i = 0; i < 10; ++i) ok &= (h_c[i] == h_a[i] + h_b[i]);

    cout << "=== Vector Addition ===\n"
         << "Elements: " << n << "\nTime: " << ms << " ms\n"
         << "Status: " << (ok ? "SUCCESS" : "FAILED") << "\n\n";

    delete[] h_a; delete[] h_b; delete[] h_c;
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

void runMatMul() {
    const int N = 1024;
    int *h_a = new int[N*N], *h_b = new int[N*N], *h_c = new int[N*N];
    for (int i = 0; i < N*N; ++i) { h_a[i] = 1; h_b[i] = 2; }

    int *d_a = toGPU(h_a, N*N * sizeof(int));
    int *d_b = toGPU(h_b, N*N * sizeof(int));
    int *d_c; cudaMalloc(&d_c, N*N * sizeof(int));

    dim3 block(16, 16), grid((N+15)/16, (N+15)/16);

    auto t0 = chrono::high_resolution_clock::now();
    matrixMul<<<grid, block>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    auto ms = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - t0).count();

    cudaMemcpy(h_c, d_c, N*N * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "=== Matrix Multiplication ===\n"
         << "Size: " << N << "x" << N << "\nTime: " << ms << " ms\n"
         << "Status: " << (h_c[0] == 2*N ? "SUCCESS" : "FAILED") << "\n";

    delete[] h_a; delete[] h_b; delete[] h_c;
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

int main() {
    // Compile with: nvcc solution.cu -o solution
    runVectorAdd();
    runMatMul();
}