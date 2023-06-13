#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <set>
#include <queue>

namespace {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int> dist(1, 10);
    std::uniform_real_distribution<double> prob(0.0, 1.0);
}

std::vector<std::vector<int>> generateCompleteGraph(int num_vertices) {
    std::vector<std::vector<int>> graph(num_vertices, std::vector<int>(num_vertices, 0));

    for (int i = 0; i < num_vertices; i++) {
        for (int j = i + 1; j < num_vertices; j++) {
            int edge_weight = dist(generator);
            graph[i][j] = edge_weight;
            graph[j][i] = edge_weight;
        }
    }

    return graph;
}

std::vector<std::vector<int>> generateConnectedGraph(int num_vertices, double density) {
    std::vector<std::vector<int>> graph(num_vertices, std::vector<int>(num_vertices, 0));

    // Генерация "скелета" графа
    for (int i = 0; i < num_vertices - 1; i++) {
        int edge_weight = dist(generator);
        graph[i][i + 1] = edge_weight;
        graph[i + 1][i] = edge_weight;
    }

    // Добавление дополнительных ребер
    for (int i = 0; i < num_vertices; i++) {
        for (int j = i + 2; j < num_vertices; j++) {
            if (prob(generator) < density) {
                int edge_weight = dist(generator);
                graph[i][j] = edge_weight;
                graph[j][i] = edge_weight;
            }
        }
    }

    return graph;
}

std::vector<std::vector<int>> generateSparseGraph(int num_vertices) {
    std::vector<std::vector<int>> graph(num_vertices, std::vector<int>(num_vertices, 0));

    for (int i = 0; i < num_vertices - 1; i++) {
        int edge_weight = dist(generator);
        graph[i][i + 1] = edge_weight;
        graph[i + 1][i] = edge_weight;
    }

    return graph;
}


void dijkstraAlgorithm(const std::vector<std::vector<int>> &graph, int start_node, int end_node) {
    int num_vertices = graph.size();
    std::vector<int> shortest_distances(num_vertices, std::numeric_limits<int>::max());
    shortest_distances[start_node] = 0;

    std::set<std::pair<int, int>> active_vertices;
    active_vertices.insert({0, start_node});

    while (!active_vertices.empty()) {
        int cur_node = active_vertices.begin()->second;
        active_vertices.erase(active_vertices.begin());

        for (int i = 0; i < num_vertices; i++) {
            if (graph[cur_node][i] < std::numeric_limits<int>::max()) {
                if (shortest_distances[cur_node] + graph[cur_node][i] < shortest_distances[i]) {
                    active_vertices.erase({shortest_distances[i], i});
                    shortest_distances[i] = shortest_distances[cur_node] + graph[cur_node][i];
                    active_vertices.insert({shortest_distances[i], i});
                }
            }
        }
    }
}

void floydWarshallAlgorithm(const std::vector<std::vector<int>> &graph, int start_node, int end_node) {
    std::vector<std::vector<int>> graph_copy = graph;

    int num_vertices = graph_copy.size();

    for (int i = 0; i < num_vertices; i++) {
        for (int j = 0; j < num_vertices; j++) {
            if (i != j && graph_copy[i][j] == 0) {
                graph_copy[i][j] = std::numeric_limits<int>::max();
            }
        }
    }

    for (int k = 0; k < num_vertices; k++) {
        for (int i = 0; i < num_vertices; i++) {
            for (int j = 0; j < num_vertices; j++) {
                if (graph_copy[i][k] < std::numeric_limits<int>::max() &&
                    graph_copy[k][j] < std::numeric_limits<int>::max() &&
                    graph_copy[i][k] + graph_copy[k][j] < graph_copy[i][j]) {
                    graph_copy[i][j] = graph_copy[i][k] + graph_copy[k][j];
                }
            }
        }
    }
}

void bellmanFordAlgorithm(const std::vector<std::vector<int>> &graph, int start_node, int end_node) {
    int num_vertices = graph.size();
    std::vector<int> distance(num_vertices, std::numeric_limits<int>::max());
    distance[start_node] = 0;

    for (int i = 0; i < num_vertices - 1; i++) {
        for (int u = 0; u < num_vertices; u++) {
            for (int v = 0; v < num_vertices; v++) {
                if (graph[u][v] != std::numeric_limits<int>::max() && distance[u] != std::numeric_limits<int>::max() &&
                    distance[u] + graph[u][v] < distance[v]) {
                    distance[v] = distance[u] + graph[u][v];
                }
            }
        }
    }
}

void aStarAlgorithm(const std::vector<std::vector<int>> &graph, int start_node, int end_node) {
    std::vector<int> distances(graph.size(), std::numeric_limits<int>::max());
    std::vector<int> previous(graph.size(), -1);
    std::vector<bool> visited(graph.size(), false);
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> pq;

    distances[start_node] = 0;
    pq.push({0, start_node});

    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();

        if (visited[u]) {
            continue;
        }

        visited[u] = true;

        for (size_t v = 0; v < graph.size(); v++) {
            if (graph[u][v] != 0 && graph[u][v] != std::numeric_limits<int>::max()) {
                int alt = distances[u] + graph[u][v] +
                          std::abs(static_cast<int64_t>(end_node) - static_cast<int64_t>(v)) /
                          static_cast<double>(graph.size());

                if (alt < distances[v]) {
                    distances[v] = alt;
                    previous[v] = u;
                    pq.push({distances[v], v});
                }
            }
        }
    }

    std::vector<int> shortest_path;
    for (int vertex = end_node; vertex != -1; vertex = previous[vertex]) {
        shortest_path.insert(shortest_path.begin(), vertex);
    }

}


template<typename Func>
uint64_t measureTime(Func function) {
    auto start = std::chrono::high_resolution_clock::now();
    function();
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

int count_edges(const std::vector<std::vector<int>> &graph) {
    int num_edges = 0;
    for (size_t i = 0; i < graph.size(); i++) {
        for (size_t j = 0; j < graph.size(); j++) {
            if (i != j && graph[i][j] != 0 && graph[i][j] != std::numeric_limits<int>::max()) {
                num_edges++;
            }
        }
    }
    return num_edges / 2;
}

int main() {
    std::ofstream out_file;
    out_file.open("results.csv");
    out_file << "NumVertices,NumEdges,GraphType,Algorithm,StartNode,EndNode,TimeTaken(us)\n";


    for (int num_vertices = 10; num_vertices <= 1010; num_vertices += 50) {
        int start_node = 0;
        int end_node = num_vertices - 1;

        auto complete_graph = generateCompleteGraph(num_vertices);
        auto connected_graph = generateConnectedGraph(num_vertices, 0.4);
        auto sparse_graph = generateSparseGraph(num_vertices);

        for (auto &graph_data: {make_pair(complete_graph, "Complete"),
                                make_pair(connected_graph, "Connected"),
                                make_pair(sparse_graph, "Sparse")}) {

            auto &graph = graph_data.first;
            auto &graph_type = graph_data.second;

            int num_edges = count_edges(graph);

            for (int run = 0; run < 10; ++run) {
                auto time_taken = measureTime([&graph, start_node, end_node]() {
                    dijkstraAlgorithm(graph, start_node, end_node);
                });
                out_file << num_vertices << "," << num_edges << "," << graph_type << ",Dijkstra," << start_node << ","
                         << end_node << ","
                         << time_taken << "\n";

                time_taken = measureTime([&graph, start_node, end_node]() {
                    floydWarshallAlgorithm(graph, start_node, end_node);
                });
                out_file << num_vertices << "," << num_edges << "," << graph_type << ",Floyd-Warshall," << start_node
                         << "," << end_node << ","
                         << time_taken << "\n";

                time_taken = measureTime([&graph, start_node, end_node]() {
                    bellmanFordAlgorithm(graph, start_node, end_node);
                });
                out_file << num_vertices << "," << num_edges << "," << graph_type << ",Bellman-Ford," << start_node
                         << "," << end_node << ","
                         << time_taken << "\n";

                time_taken = measureTime([&graph, start_node, end_node]() {
                    aStarAlgorithm(graph, start_node, end_node);
                });
                out_file << num_vertices << "," << num_edges << "," << graph_type << ",A*," << start_node << ","
                         << end_node << ","
                         << time_taken << "\n";
            }
        }
    }

    out_file.close();

    return 0;
}