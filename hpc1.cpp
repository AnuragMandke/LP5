#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

// Parallel BFS using a level-by-level approach
void parallel_bfs(int start, const vector<vector<int>>& graph) {
    int n = graph.size();
    vector<bool> visited(n, false);
    vector<int> current_level;

    current_level.push_back(start);
    visited[start] = true;

    while (!current_level.empty()) {
        vector<int> next_level;

        // Process nodes in the current level in parallel
        #pragma omp parallel for
        for (int i = 0; i < current_level.size(); ++i) {
            int u = current_level[i];
            
            #pragma omp critical
            {
                cout << u << " ";
            }

            // Iterate over all neighbors of node u
            for (int v : graph[u]) {
                if (!visited[v]) {
                    #pragma omp critical
                    {
                        if (!visited[v]) { // Double-check locking
                            visited[v] = true;
                            next_level.push_back(v);
                        }
                    }
                }
            }
        }
        current_level = next_level; // Move to the next depth level
    }
    cout << endl;
}

// Helper function for Parallel DFS using OpenMP tasks
void parallel_dfs_util(int u, const vector<vector<int>>& graph, vector<bool>& visited) {
    #pragma omp critical
    {
        cout << u << " ";
    }

    for (int v : graph[u]) {
        if (!visited[v]) {
            bool do_visit = false;
            #pragma omp critical
            {
                if (!visited[v]) { // Double-check locking
                    visited[v] = true;
                    do_visit = true;
                }
            }
            if (do_visit) {
                // Create a task for the new branch
                #pragma omp task
                parallel_dfs_util(v, graph, visited);
            }
        }
    }
}

// Parallel DFS wrapper
void parallel_dfs(int start, const vector<vector<int>>& graph) {
    int n = graph.size();
    vector<bool> visited(n, false);
    visited[start] = true;

    #pragma omp parallel
    {
        #pragma omp single
        {
            parallel_dfs_util(start, graph, visited);
        }
        // Implicit taskwait at the end of the parallel region ensures all tasks finish
    }
    cout << endl;
}

int main() {
    int n = 7;
    vector<vector<int>> graph(n);

    // Creating a simple undirected tree for traversal
    //          0
    //        /   \
    //       1     2
    //      / \   / \
    //     3   4 5   6
    graph[0] = {1, 2};
    graph[1] = {0, 3, 4};
    graph[2] = {0, 5, 6};
    graph[3] = {1};
    graph[4] = {1};
    graph[5] = {2};
    graph[6] = {2};

    cout << "Parallel BFS Traversal: ";
    parallel_bfs(0, graph);

    cout << "Parallel DFS Traversal: ";
    parallel_dfs(0, graph);

    return 0;
}
