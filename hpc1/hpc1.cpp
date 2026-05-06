#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

// ------------------- SEQUENTIAL TRAVERSALS -------------------

void sequential_bfs(int start, const vector<vector<int>>& graph, bool print_nodes = false) {
    int n = graph.size();
    vector<int> visited(n, 0);
    queue<int> q;

    q.push(start);
    visited[start] = 1;

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        if (print_nodes) cout << u << " ";

        for (int v : graph[u]) {
            if (!visited[v]) {
                visited[v] = 1;
                q.push(v);
            }
        }
    }
    if (print_nodes) cout << endl;
}

void sequential_dfs_util(int u, const vector<vector<int>>& graph, vector<int>& visited, bool print_nodes) {
    visited[u] = 1;
    if (print_nodes) cout << u << " ";

    for (int v : graph[u]) {
        if (!visited[v]) {
            sequential_dfs_util(v, graph, visited, print_nodes);
        }
    }
}

void sequential_dfs(int start, const vector<vector<int>>& graph, bool print_nodes = false) {
    int n = graph.size();
    vector<int> visited(n, 0);
    sequential_dfs_util(start, graph, visited, print_nodes);
    if (print_nodes) cout << endl;
}


// ------------------- PARALLEL TRAVERSALS -------------------

void parallel_bfs(int start, const vector<vector<int>>& graph, bool print_nodes = false) {
    int n = graph.size();
    vector<int> visited(n, 0);
    vector<int> current_level;

    current_level.push_back(start);
    visited[start] = 1;

    while (!current_level.empty()) {
        vector<int> next_level;

        #pragma omp parallel for
        for (int i = 0; i < current_level.size(); ++i) {
            int u = current_level[i];
            
            if (print_nodes) {
                #pragma omp critical
                {
                    cout << u << " ";
                }
            }

            for (int v : graph[u]) {
                if (visited[v] == 0) {
                    // Atomic compare and swap to avoid critical section bottleneck and vector<bool> races
                    if (__sync_bool_compare_and_swap(&visited[v], 0, 1)) {
                        #pragma omp critical
                        {
                            next_level.push_back(v);
                        }
                    }
                }
            }
        }
        current_level = next_level;
    }
    if (print_nodes) cout << endl;
}

void parallel_dfs_util(int u, const vector<vector<int>>& graph, vector<int>& visited, bool print_nodes) {
    if (print_nodes) {
        #pragma omp critical
        {
            cout << u << " ";
        }
    }

    for (int v : graph[u]) {
        if (visited[v] == 0) {
            // Atomic compare and swap replaces the massive critical section
            if (__sync_bool_compare_and_swap(&visited[v], 0, 1)) {
                #pragma omp task shared(graph, visited) firstprivate(v, print_nodes)
                parallel_dfs_util(v, graph, visited, print_nodes);
            }
        }
    }
}

void parallel_dfs(int start, const vector<vector<int>>& graph, bool print_nodes = false) {
    int n = graph.size();
    vector<int> visited(n, 0);
    visited[start] = 1;

    #pragma omp parallel
    {
        #pragma omp single
        {
            parallel_dfs_util(start, graph, visited, print_nodes);
        }
    }
    if (print_nodes) cout << endl;
}

// ------------------- UTILS & MAIN -------------------

// Helper to measure execution time
auto measure = [](string name, auto func) {
    auto start_time = high_resolution_clock::now();
    func();
    auto end_time = high_resolution_clock::now();
    cout << name << ": " << duration_cast<milliseconds>(end_time - start_time).count() << " ms" << endl;
};

int main() {
    // 1. SMALL GRAPH (To verify correctness with printing)
    cout << "=== Verification on Small Graph ===" << endl;
    int n_small = 7;
    vector<vector<int>> small_graph(n_small);
    //          0
    //        /   \
    //       1     2
    //      / \   / \
    //     3   4 5   6
    small_graph[0] = {1, 2};
    small_graph[1] = {0, 3, 4};
    small_graph[2] = {0, 5, 6};
    small_graph[3] = {1};
    small_graph[4] = {1};
    small_graph[5] = {2};
    small_graph[6] = {2};

    cout << "Sequential BFS: "; sequential_bfs(0, small_graph, true);
    cout << "Parallel BFS:   "; parallel_bfs(0, small_graph, true);
    cout << "Sequential DFS: "; sequential_dfs(0, small_graph, true);
    cout << "Parallel DFS:   "; parallel_dfs(0, small_graph, true);
    
    // 2. LARGE GRAPH (To compare performance)
    cout << "\n=== Performance Comparison on Large Graph ===" << endl;
    int n_large = 100000;
    vector<vector<int>> large_graph(n_large);
    
    // Create a large binary tree graph
    for (int i = 0; i < n_large; ++i) {
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        if (left < n_large) {
            large_graph[i].push_back(left);
            large_graph[left].push_back(i);
        }
        if (right < n_large) {
            large_graph[i].push_back(right);
            large_graph[right].push_back(i);
        }
    }
    
    cout << "Graph size: " << n_large << " nodes" << endl;

    measure("Sequential BFS", [&](){ sequential_bfs(0, large_graph, false); });
    measure("Parallel BFS  ", [&](){ parallel_bfs(0, large_graph, false); });
    
    measure("Sequential DFS", [&](){ sequential_dfs(0, large_graph, false); });
    measure("Parallel DFS  ", [&](){ parallel_dfs(0, large_graph, false); });

    return 0;
}
