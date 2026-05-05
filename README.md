# Laboratory Practice 5 (LP5)

This repository contains practical assignments for the **Laboratory Practice 5 (LP5)** course, focusing on **High-Performance Computing (HPC)** and **Deep Learning (DL)**.

## 🚀 High-Performance Computing (HPC)

The HPC practicals demonstrate the use of parallel programming paradigms using **OpenMP** (CPU parallelism) and **CUDA** (GPU parallelism) in C++.

*   **HPC 1: Parallel Graph Traversal (`hpc1/hpc1.cpp`)**
    *   Implementation of Breadth-First Search (BFS) and Depth-First Search (DFS) algorithms for undirected graphs/trees using OpenMP.
*   **HPC 2: Parallel Sorting Algorithms (`hpc2/hpc2.cpp`)**
    *   Implementation and performance comparison of sequential vs. parallel Bubble Sort and Merge Sort using OpenMP tasks and parallel regions.
*   **HPC 3: Parallel Reduction (`hpc3/hpc3.cpp`)**
    *   Implementation of parallel reduction using OpenMP to efficiently find the Minimum, Maximum, Sum, and Average of a large array (10+ million elements).
*   **HPC 4: CUDA Operations (`hpc4/hpc4.cpp`)**
    *   Implementation of Vector Addition and Matrix Multiplication using CUDA for GPU acceleration.

### How to Compile & Run (HPC)
**For OpenMP (HPC 1, 2, 3):**
```bash
g++ -fopenmp filename.cpp -o output
./output
```

**For CUDA (HPC 4):**
```bash
nvcc filename.cpp -o output
./output
```

---

## 🧠 Deep Learning (DL)

The DL practicals demonstrate various neural network architectures using **TensorFlow/Keras** in Python (Jupyter Notebooks).

*   **DL 1: Artificial Neural Networks (ANN)**
    *   `DL1/DL1.ipynb`: Linear Regression using a simple ANN to predict Boston Housing Prices.
*   **DL 2: Deep Neural Networks for Classification**
    *   `DL2/DL2_Binary_IMDB.ipynb`: Binary sentiment classification using the IMDB movie reviews dataset.
    *   `DL2/DL2_Multiclass_OCR.ipynb`: Multiclass classification for OCR letter recognition using the UCI Letter dataset.
*   **DL 3: Convolutional Neural Networks (CNN)**
    *   `DL3/DL3_Fashion_MNIST_CNN.ipynb`: Fashion classification using the Fashion MNIST dataset.
    *   `DL3/DL3_Plant_Disease_CNN.ipynb`: Plant disease detection using the PlantVillage dataset.
*   **DL 4: Recurrent Neural Networks (RNN)**
    *   `DL4/DL4_RNN_Google_Stock.ipynb`: Time-series prediction of Google stock prices using an RNN and historical data fetched via `yfinance`.

### Prerequisites (DL)
To run the Jupyter Notebooks, ensure you have the required Python packages installed:
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow tensorflow-datasets yfinance jupyter
```
To run a notebook:
```bash
jupyter notebook
```