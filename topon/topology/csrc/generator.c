/**
 * @file generator.c
 * @brief C Program for Polymer Network Generation via Strict Sculpting (Serial Version with Logging).
 *
 * ROLE: this is the standalone searcher. It runs on its own, without
 * Python, and is the tool for long exhaustive searches over many trials.
 * The pure-Python port in topon/topology/generator_python.py is the
 * separate quick path for in-process generation of likely networks.
 * The two are deliberately independent programs, not a library and a
 * wrapper: nothing here is called from Python, and nothing here should
 * grow a Python binding.
 *
 * PROVENANCE: vendored 2026-08-05 from generator_serial_debug11.c
 * (md5 e7631f4bbcb963d50c382721de3b3c18, dated 2025-11-03), the version
 * the shipped generator.exe was built from and the one the Python port
 * mirrors.
 *
 * A later variant exists in the archive (md5 83d7f9d3, 2026-02-27, under
 * experiments/pruning_research/pruning_algorithm_math*) which replaces
 * the per-degree count check in is_move_safe with a cumulative one. It is
 * NOT used here: measured across six standard SC configurations it
 * sculpts 1/6 where this version sculpts 6/6, failing whenever max_func
 * is below the lattice coordination. Treat it as an open experiment.
 *
 * Build:  gcc -O2 -o generator.exe generator.c -lm
 *
 * Lattice construction and the .nodes/.edges format are shared surface
 * with the Python port and must be changed in both; see
 * tests/unit/topology/test_c_generator.py. The sculpting search itself
 * is this program's own business.
 *
 * KNOWN DIVERGENCES from the Python port (pre-existing, not fixed here):
 *   - The degree<=2 guard in the sculpting stages is gated on
 *     is_sc_lattice here, but applied unconditionally in Python.
 *   (Per-axis periodicity, once C-only, has been in Python since V47.)
 *
 * SEARCHES: two, chosen by a named flag that may sit anywhere in argv.
 * --search=strict (the default, and what every existing eight- or
 * nine-argument call gets) is the edge-by-edge sculptor below.
 * --search=exact is the degree-constrained subgraph search ported from
 * topon/topology/degree_matching.py (see the EXACT DEGREE MATCHING
 * section): it needs a count for every degree up to max_func and reaches
 * it exactly, with max_trials bounding the attempts. Both write the same
 * .nodes/.edges files.
 *
 * NEIGHBOUR CUTOFF: an optional ninth argument sets the candidate-edge
 * range for SC/BCC/FCC/Diamond in cell units (MIX carries it inside its
 * own argument). At the default 1.0 each pure lattice keeps its
 * canonical neighbour pattern; any other value rebuilds the edges by
 * the same minimum-image search MIX uses, mirroring
 * generator_python.edges_within_cutoff. Sites, numbering and the
 * .nodes/.edges format are unchanged.
 *
 * @details
 * This program simulates the creation of a polymer network with a specific target
 * degree distribution. It employs a **Strict Sculpting Model** to rigorously avoid
 * unintended "collateral damage," especially the creation of nodes with degrees
 * that have a target count of zero.
 *
 * MODIFIED: Now supports SC, BCC, and FCC initial lattice generation.
 * MODIFIED: Now supports an 'e:N' argument to target a specific *total edge count*.
 *
 * @author Ahmet Burak Yildirim
 * @date November 1, 2025
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <limits.h>
#include <math.h> // For sqrt
#include <stdint.h>
#include <sys/stat.h> // For mkdir
#ifdef _WIN32
#include <direct.h> // For _mkdir on Windows
#include <process.h> // For _getpid
#define mkdir(dir, mode) _mkdir(dir)
#define topon_getpid _getpid
#else
#include <unistd.h> // For getpid
#define topon_getpid getpid
#endif


/* --- Random numbers ---------------------------------------------------
 *
 * xoshiro256** (Blackman and Vigna, 2018), seeded by running the 64-bit
 * seed through splitmix64. Every draw in this file comes from here: the
 * sculptor's shuffles, the MIX site draws and the exact search.
 *
 * This replaces rand()/srand(). MinGW's rand() returns 15 bits from one
 * shared 2^32 cycle, so every seed is only a starting point on the same
 * sequence. Two long searches from different seeds each consume a large
 * arc of it; once a trial of one starts where a trial of the other
 * started, they coincide from then on, and a benchmark of 100 seeds
 * returned a few byte-identical networks. rand() % n is also biased once
 * n is not small next to RAND_MAX, and a 15-bit draw cannot index past
 * 32767 at all.
 *
 * rng_below(n) is Lemire's multiply-shift with rejection, so it is
 * unbiased for any n. */

static uint64_t rng_s[4];

static uint64_t rng_rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t splitmix64(uint64_t* x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void rng_seed(uint64_t seed) {
    uint64_t x = seed;
    for (int i = 0; i < 4; ++i) rng_s[i] = splitmix64(&x);
}

static uint64_t rng_next(void) {
    uint64_t result = rng_rotl(rng_s[1] * 5, 7) * 9;
    uint64_t t = rng_s[1] << 17;
    rng_s[2] ^= rng_s[0];
    rng_s[3] ^= rng_s[1];
    rng_s[1] ^= rng_s[2];
    rng_s[0] ^= rng_s[3];
    rng_s[2] ^= t;
    rng_s[3] = rng_rotl(rng_s[3], 45);
    return result;
}

/* Uniform on [0, n); n must be at least 1. */
static uint64_t rng_below(uint64_t n) {
#if defined(__SIZEOF_INT128__)
    unsigned __int128 m = (unsigned __int128)rng_next() * n;
    uint64_t low = (uint64_t)m;
    if (low < n) {
        uint64_t threshold = (0 - n) % n;
        while (low < threshold) {
            m = (unsigned __int128)rng_next() * n;
            low = (uint64_t)m;
        }
    }
    return (uint64_t)(m >> 64);
#else
    uint64_t threshold = (0 - n) % n;
    uint64_t r;
    do { r = rng_next(); } while (r < threshold);
    return r % n;
#endif
}

/* Uniform on [0, 1) with 53 random bits. */
static double rng_uniform(void) {
    return (double)(rng_next() >> 11) * (1.0 / 9007199254740992.0);
}


// --- Structs and Type Definitions ---

// --- NEW: Coordinate struct ---
typedef struct Coord {
    double x, y, z;
} Coord;

typedef enum NodeStatus {
    ACTIVE,
    IS_DEGREE_0,
    IS_DEGREE_1
} NodeStatus;

typedef enum MoveType {
    SET_D0,
    SET_D1,
    REMOVE_EDGE
} MoveType;

typedef struct MoveLog {
    MoveType type;
    int u;
    int v; // Second node for edges, -1 for node operations
} MoveLog;

typedef struct Edge {
    int u;
    int v;
} Edge;

typedef struct AdjListNode {
    int dest;
    struct AdjListNode* next;
} AdjListNode;

typedef struct AdjList {
    AdjListNode *head;
} AdjList;

typedef struct Graph {
    int V;
    AdjList* array;
    int* degrees;
    Coord* coords; // --- MODIFIED: Added coordinates array ---
} Graph;

typedef struct UnionFind {
    int* parent;
    int n;
} UnionFind;

// --- Forward declarations ---
// --- MODIFIED: Signatures updated for new 'e:N' logic ---
Graph* run_single_trial(Graph* base_graph, int max_func, const int* target_counts, int target_edge_count, long long trial_num, int extensive_logging, const char* dims_str, const char* lattice_type);
/* Per-axis boundaries, stashed for the .nodes writer so it can record
 * them alongside the box. save_graph_to_file is reached through several
 * call sites that do not carry p_dims, and threading it through all of
 * them would touch far more of this file than the header is worth. */
static int g_periodicity[3] = {1, 1, 1};
void addEdge(Graph* graph, int src, int dest);
void print_distribution(const char* stage_name, Graph* g, const int* target_counts, long long trial_num, long long move_num, int max_func, int extensive_logging);
void save_move_log_to_file(const MoveLog* move_log, long long count, const char* dims_str, long long trial);
int is_move_safe(Graph* g, int u, int v, const int* target_counts, int max_func, int current_stage, long long target_degree_sum, long long current_total_degree_sum);


// --- START: HELPER FUNCTION ---
/**
 * @brief Checks if removing an edge between u and v would violate strict count constraints.
 * @param current_stage The stage calling the function (1=SetD0, 2=SetD1, 3=SetMax, 4=Systematic)
 * @param target_degree_sum The target total degree sum (2 * e_target), or -2 if not set.
 * @param current_total_degree_sum The current total degree sum.
 * @return 1 if the move is safe, 0 if it is forbidden.
 */
// --- MODIFIED: Signature updated for new 'e:N' logic ---
int is_move_safe(Graph* g, int u, int v, const int* target_counts, int max_func, int current_stage, long long target_degree_sum, long long current_total_degree_sum) {
    
    // --- NEW: Target Edge Count Check ---
    // This is the primary "stop" signal when using 'e:N'
    if (current_stage == 4 && target_degree_sum != -1) {
        // If we are at or below the target, no more moves are safe.
        // The degree sum *before* this move is current_total_degree_sum.
        // After this move, it will be current_total_degree_sum - 2.
        if (current_total_degree_sum <= target_degree_sum) {
            return 0; // FORBIDDEN: We have hit or gone below our target edge count.
        }
    }
    // --- END NEW CHECK ---
    
    int u_new_degree = g->degrees[u] - 1;
    int v_new_degree = g->degrees[v] - 1;

    // --- Check 1: Check Neighbor 'v' (The "Victim") ---
    // 'v' is always checked for collateral damage, regardless of stage.

    // 1a. Forbidden Degree (target=0)
    if (v_new_degree >= 0 && target_counts[v_new_degree] == 0) {
        return 0; // Forbidden collateral damage on v
    }

    // 1b. Overshooting
    if (v_new_degree >= 0 && target_counts[v_new_degree] > 0) {
        
        // Case A: d0 or d1. These are "sacred" after Stages 1 & 2.
        // Never overshoot them in *any* stage.
        if (v_new_degree <= 1) { 
            int current_count = 0;
            for (int i = 0; i < g->V; ++i) {
                if (g->degrees[i] == v_new_degree) current_count++;
            }
            if (current_count >= target_counts[v_new_degree]) {
                return 0; // Forbid overshooting d0 or d1
            }
        }
        
        // Case B: d2+. Only block overshooting explicit targets in Stage 4.
        // Stages 1, 2, and 3 are *allowed* to overshoot these.
        else if (current_stage == 4) { 
            int current_count = 0;
            for (int i = 0; i < g->V; ++i) {
                if (g->degrees[i] == v_new_degree) current_count++;
            }
            if (current_count >= target_counts[v_new_degree]) {
                return 0; // Forbid overshooting v's target in Stage 4
            }
        }
        // (If stage 1, 2, or 3, we allow overshooting v for d2+)
    }

    // --- Check 2: Check Actor 'u' ---
    // 'u' is *only* checked for damage in Stage 4.
    // In Stages 1, 2, 3, 'u' is the node we are *trying* to change,
    // so its new state isn't "collateral damage."

    if (current_stage == 4) {
        // 2a. Forbidden Degree (target=0)
        if (u_new_degree >= 0 && target_counts[u_new_degree] == 0) {
            return 0; // Forbid u from becoming a forbidden degree
        }

        // 2b. Overshooting (Only for *explicit* targets)
        if (u_new_degree >= 0 && target_counts[u_new_degree] > 0) {
            int current_count = 0;
            for (int i = 0; i < g->V; ++i) {
                if (g->degrees[i] == u_new_degree) current_count++;
            }
            if (current_count >= target_counts[u_new_degree]) {
                return 0; // Forbid overshooting u's target
            }
        }
    }
    
    // If we passed all checks (e.g., in Stage 1, Check 2 is skipped),
    // the move is safe.
    return 1;
}
// --- END: HELPER FUNCTION ---


// --- Union-Find Data Structure Functions ---

UnionFind* createUnionFind(int n) {
    UnionFind* uf = (UnionFind*)malloc(sizeof(UnionFind));
    uf->parent = (int*)malloc(n * sizeof(int));
    uf->n = n;
    for (int i = 0; i < n; i++) uf->parent[i] = i;
    return uf;
}

int find_set(UnionFind* uf, int i) {
    if (uf->parent[i] == i) return i;
    return uf->parent[i] = find_set(uf, uf->parent[i]);
}

void unite_sets(UnionFind* uf, int a, int b) {
    a = find_set(uf, a);
    b = find_set(uf, b);
    if (a != b) uf->parent[b] = a;
}

void freeUnionFind(UnionFind* uf) {
    if (!uf) return;
    free(uf->parent);
    free(uf);
}

// --- Graph Utility Functions ---

AdjListNode* newAdjListNode(int dest) {
    AdjListNode* newNode = (AdjListNode*)malloc(sizeof(AdjListNode));
    newNode->dest = dest;
    newNode->next = NULL;
    return newNode;
}

// --- MODIFIED: createGraph now allocates space for coordinates ---
Graph* createGraph(int V) {
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->V = V;
    graph->array = (AdjList*)malloc(V * sizeof(AdjList));
    graph->degrees = (int*)calloc(V, sizeof(int));
    graph->coords = (Coord*)malloc(V * sizeof(Coord)); // Allocate coords
    for (int i = 0; i < V; ++i) {
        graph->array[i].head = NULL;
        graph->coords[i] = (Coord){0.0, 0.0, 0.0}; // Initialize
    }
    return graph;
}

// --- MODIFIED: freeGraph now frees coordinates ---
void freeGraph(Graph* graph) {
    if (!graph) return;
    for (int i = 0; i < graph->V; ++i) {
        AdjListNode* pCrawl = graph->array[i].head;
        while (pCrawl) {
            AdjListNode* temp = pCrawl;
            pCrawl = pCrawl->next;
            free(temp);
        }
    }
    free(graph->array);
    free(graph->degrees);
    free(graph->coords); // Free coords
    free(graph);
}

// --- MODIFIED: copyGraph now copies coordinates ---
Graph* copyGraph(Graph* src_graph) {
    if (!src_graph) return NULL;
    Graph* new_graph = createGraph(src_graph->V);
    // Copy coordinates
    memcpy(new_graph->coords, src_graph->coords, src_graph->V * sizeof(Coord));
    for (int i = 0; i < src_graph->V; i++) {
        AdjListNode* pCrawl = src_graph->array[i].head;
        while(pCrawl) {
            if (i < pCrawl->dest) addEdge(new_graph, i, pCrawl->dest);
            pCrawl = pCrawl->next;
        }
    }
    return new_graph;
}

void addEdge(Graph* graph, int src, int dest) {
    AdjListNode* newNode = newAdjListNode(dest);
    newNode->next = graph->array[src].head;
    graph->array[src].head = newNode;
    graph->degrees[src]++;
    newNode = newAdjListNode(src);
    newNode->next = graph->array[dest].head;
    graph->array[dest].head = newNode;
    graph->degrees[dest]++;
}

void removeEdge(Graph* graph, int src, int dest) {
    AdjListNode* pCrawl = graph->array[src].head;
    AdjListNode* prev = NULL;
    while (pCrawl && pCrawl->dest != dest) { prev = pCrawl; pCrawl = pCrawl->next; }
    if (pCrawl) {
        if (prev) prev->next = pCrawl->next; else graph->array[src].head = pCrawl->next;
        free(pCrawl);
        graph->degrees[src]--;
    }
    pCrawl = graph->array[dest].head;
    prev = NULL;
    while (pCrawl && pCrawl->dest != src) { prev = pCrawl; pCrawl = pCrawl->next; }
    if (pCrawl) {
        if (prev) prev->next = pCrawl->next; else graph->array[dest].head = pCrawl->next;
        free(pCrawl);
        graph->degrees[dest]--;
    }
}

// --- Connectivity and File I/O ---

int is_subgraph_connected(Graph* g, const NodeStatus* node_status) {
    if (g->V == 0) return 1;
    UnionFind* uf = createUnionFind(g->V);
    int first_active_node = -1;
    for (int i = 0; i < g->V; i++) {
        if (node_status[i] == ACTIVE) {
            if (first_active_node == -1) first_active_node = i;
            AdjListNode* pCrawl = g->array[i].head;
            while (pCrawl) {
                if (node_status[pCrawl->dest] == ACTIVE) unite_sets(uf, i, pCrawl->dest);
                pCrawl = pCrawl->next;
            }
        }
    }
    if (first_active_node == -1) { freeUnionFind(uf); return 1; }
    int root = find_set(uf, first_active_node);
    int connected = 1;
    for (int i = 0; i < g->V; i++) {
        if (node_status[i] == ACTIVE && find_set(uf, i) != root) {
            connected = 0;
            break;
        }
    }
    freeUnionFind(uf);
    return connected;
}

// --- MODIFIED: save_graph_to_file now uses the stored coordinates ---
void save_graph_to_file(Graph* g, const char* dims_str, long long trial) {
    char nodes_filename[256], edges_filename[256];
    // MODIFIED: Use %s for dims_str
    sprintf(nodes_filename, "output/network_N%s_trial%lld.nodes", dims_str, trial);
    sprintf(edges_filename, "output/network_N%s_trial%lld.edges", dims_str, trial);
    FILE* nodes_file = fopen(nodes_filename, "w");
    if (!nodes_file) { perror("Failed to open nodes file"); return; }
    /* Record the true periodic cell. Without it the Python loader has to
     * estimate the box from the coordinate extent as max-min+1, which is
     * exact only for SC: BCC/FCC basis sites sit at +0.5 and never reach
     * the cell edge, so the estimate overshoots by half a cell and sends
     * a large fraction of edges to the wrong periodic replica. Must stay
     * byte-compatible with topon.topology.loader.format_box_header. */
    {
        int bx = 0, by = 0, bz = 0;
        if (sscanf(dims_str, "%dx%dx%d", &bx, &by, &bz) == 3) {
            fprintf(nodes_file, "# BOX %g %g %g\n",
                    (double)bx, (double)by, (double)bz);
        }
    }
    /* Record open axes so the conformation stage knows not to wrap them.
     * Written only when an axis is actually open, so a fully periodic
     * run produces exactly the file format it did before. Must match
     * topon.topology.loader.format_periodicity_header. */
    if (!g_periodicity[0] || !g_periodicity[1] || !g_periodicity[2]) {
        fprintf(nodes_file, "# PERIODICITY %d%d%d\n",
                g_periodicity[0], g_periodicity[1], g_periodicity[2]);
    }
    fprintf(nodes_file, "# NodeID X Y Z Degree\n");
    for (int i = 0; i < g->V; ++i) {
        // Use the stored coordinates directly
        fprintf(nodes_file, "%d %f %f %f %d\n", i, g->coords[i].x, g->coords[i].y, g->coords[i].z, g->degrees[i]);
    }
    fclose(nodes_file);
    FILE* edges_file = fopen(edges_filename, "w");
    if (!edges_file) { perror("Failed to open edges file"); return; }
    fprintf(edges_file, "# Node1 Node2\n");
    for (int i = 0; i < g->V; ++i) {
        AdjListNode* pCrawl = g->array[i].head;
        while(pCrawl) {
            if (i < pCrawl->dest) fprintf(edges_file, "%d %d\n", i, pCrawl->dest);
            pCrawl = pCrawl->next;
        }
    }
    fclose(edges_file);
    printf("Successfully saved network from trial %lld to files.\n", trial);
}

void save_move_log_to_file(const MoveLog* move_log, long long count, const char* dims_str, long long trial) {
    char log_filename[256];
    // MODIFIED: Use %s for dims_str
    sprintf(log_filename, "output/network_N%s_trial%lld.log", dims_str, trial);
    FILE* log_file = fopen(log_filename, "w");
    if (!log_file) {
        perror("Failed to open move log file");
        return;
    }
    fprintf(log_file, "# Successful move log for Trial %lld\n", trial);
    for (long long i = 0; i < count; ++i) {
        const MoveLog* move = &move_log[i];
        switch (move->type) {
            case SET_D0:
                fprintf(log_file, "set node %d to d0\n", move->u);
                break;
            case SET_D1:
                fprintf(log_file, "set node %d to d1\n", move->u);
                break;
            case REMOVE_EDGE:
                fprintf(log_file, "remove edge %d-%d\n", move->u, move->v);
                break;
        }
    }
    fclose(log_file);
    printf("Successfully saved move log for trial %lld.\n", trial);
}


// --- Core Simulation Logic ---

void shuffle_array(int *array, size_t n) {
    if (n > 1) {
        for (size_t i = n - 1; i > 0; i--) {
            size_t j = (size_t)rng_below((uint64_t)i + 1);
            int temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }
}

/* Fisher-Yates over whole Edge structs. Stage 4 used to call
 * shuffle_array((int*)edge_list, n), which shuffles the first n ints of
 * a 2n-int array: endpoints moved between edges, so the loop tried pairs
 * that were not edges, and removeEdge on such a pair is a silent no-op
 * that still counted as a move. On a target the paper run solved at
 * trial 0, one trial made 1.27 M such moves in 15 s while the edge count
 * stayed at 548. The networks it returned were valid (a no-op leaves the
 * graph as it was), but the removal order was not a uniform shuffle: the
 * real edges tried came mostly from the unshuffled second half of the
 * list. The generator benchmark found no descriptor bias against the
 * Python sculptor that it could detect. */
static void shuffle_edges(Edge* edges, long long n) {
    for (long long i = n - 1; i > 0; i--) {
        long long j = (long long)rng_below((uint64_t)i + 1);
        Edge tmp = edges[i];
        edges[i] = edges[j];
        edges[j] = tmp;
    }
}

// --- MODIFIED: Signature updated to parse 'e:N' ---
int parse_degree_distribution(char* str, int* target_counts, int max_degree_val, int* target_edge_count) {
    for(int i = 0; i <= max_degree_val; ++i) target_counts[i] = -2; // -2 means not specified
    char* token = strtok(str, ",");
    while (token != NULL) {
        int degree, count;
        // --- NEW: Check for e:N ---
        if (sscanf(token, "e:%d", &count) == 1) {
            *target_edge_count = count;
        }
        // --- END NEW ---
        else if (sscanf(token, "%d:%d", &degree, &count) == 2) {
            if (degree > max_degree_val) return 0; // Error
            target_counts[degree] = count;
        } else { return 0; } // Error
        token = strtok(NULL, ",");
    }
    return 1;
}

/**
 * @brief Prints the current degree distribution of the graph compared to the target.
 */
void print_distribution(const char* stage_name, Graph* g, const int* target_counts, long long trial_num, long long move_num, int max_func, int extensive_logging) {
    int max_current_degree = 0;
    for(int i=0; i < g->V; ++i) {
        if(g->degrees[i] > max_current_degree) max_current_degree = g->degrees[i];
    }
    
    int max_print_degree = max_current_degree > max_func ? max_current_degree : max_func;
    if (max_print_degree < 6) max_print_degree = 6;

    int buffer_size = max_print_degree + 1;
    int* current_counts = (int*)calloc(buffer_size, sizeof(int));
    long long current_total_degree_sum = 0; // --- NEW ---
    for(int i = 0; i < g->V; ++i) {
        if (g->degrees[i] < buffer_size) {
            current_counts[g->degrees[i]]++;
        }
        current_total_degree_sum += g->degrees[i]; // --- NEW ---
    }
    
    if (move_num > 0) {
         printf("[Trial %lld | Move %-8lld] ", trial_num, move_num);
    } else {
         printf("[Trial %lld | %-24s] ", trial_num, stage_name);
    }

    // --- NEW: Print total edge count ---
    printf("Edges: %-6lld | ", current_total_degree_sum / 2);
    printf("Dist:");
    for(int i = 0; i <= max_print_degree; ++i) {
        if (current_counts[i] > 0 || (i <= max_func && target_counts[i] != -2) ) {
            printf(" d%d:%d", i, current_counts[i]);
            if (i <= max_func && target_counts[i] != -2) {
                if (target_counts[i] == -1) printf("/*");
                else printf("/%d", target_counts[i]);
            }
        }
    }
    printf("\n");
    fflush(stdout);
    free(current_counts);
}
// --- MODIFIED: Added new 'lattice_type' and 'target_edge_count' arguments ---
Graph* run_single_trial(Graph* base_graph, int max_func, const int* target_counts, int target_edge_count, long long trial_num, int extensive_logging, const char* dims_str, const char* lattice_type) {
    Graph* g = copyGraph(base_graph);
    int total_nodes = g->V;
    NodeStatus* node_status = (NodeStatus*)malloc(total_nodes * sizeof(NodeStatus));
    int* node_indices = (int*)malloc(total_nodes * sizeof(int));
    Graph* return_graph = NULL;
    long long move_counter = 0;
    
    // --- NEW: Calculate target total degree sum from target edge count ---
    // -1 (from target_edge_count) * 2 = -2. This is our "not set" flag.
    long long target_degree_sum = (long long)target_edge_count * 2;

    // --- NEW: Check lattice type once at the beginning for efficiency ---
    int is_sc_lattice = (strcmp(lattice_type, "SC") == 0);

    MoveLog* move_log = NULL;
    long long move_log_count = 0;
    long long move_log_capacity = 0;
    if (extensive_logging >= 1) {
        move_log_capacity = g->V * 3; 
        move_log = (MoveLog*)malloc(move_log_capacity * sizeof(MoveLog));
    }

    for(int i=0; i<total_nodes; ++i) {
        node_status[i] = ACTIVE;
        node_indices[i] = i;
    }
    shuffle_array(node_indices, total_nodes);

    int N0_target = (target_counts[0] >= 0) ? target_counts[0] : 0;
    int N1_target = (target_counts[1] >= 0) ? target_counts[1] : 0;
    if (N0_target + N1_target > total_nodes) goto cleanup;
    
    int current_node_offset = 0;

    // --- Stage 1: Set Degree-0 Nodes (Strict) ---
    for(int i=0; i<N0_target; ++i) {
        int node_idx = node_indices[current_node_offset++];
        while(g->degrees[node_idx] > 0) {
            int num_neighbors = g->degrees[node_idx];
            int* neighbors = (int*)malloc(num_neighbors * sizeof(int));
            AdjListNode* pCrawl = g->array[node_idx].head;
            for(int k=0; k<num_neighbors; ++k){ neighbors[k] = pCrawl->dest; pCrawl = pCrawl->next; }
            
            int removed = 0;
            for(int k=0; k<num_neighbors; ++k) {
                int neighbor_idx = neighbors[k];

                // --- MODIFIED: This check is now conditional on being an SC lattice ---
                if (is_sc_lattice && g->degrees[neighbor_idx] <= 2) {
                    continue;
                }
                
                // --- MODIFIED: Pass sums to is_move_safe (not relevant for stage 1, pass -1)
                if (!is_move_safe(g, node_idx, neighbor_idx, target_counts, max_func, 1, target_degree_sum, -1)) {
                    continue;
                }
                
                if (extensive_logging >= 1) { /* log move */ }
                removeEdge(g, node_idx, neighbor_idx);
                
                if (extensive_logging == 1) { /* print per-move */ }
                removed = 1;
                break;
            }
            free(neighbors);
            if(!removed) goto cleanup; 
        }
        node_status[node_idx] = IS_DEGREE_0;
        if (extensive_logging >= 1) { /* log move */ }
    }
    if (extensive_logging >= 1) print_distribution("Stage 1: Set-d0 (Strict)", g, target_counts, trial_num, 0, max_func, extensive_logging);

    // --- Stage 2: Set Degree-1 Nodes (Strict) ---
    for(int i=0; i<N1_target; ++i) {
       int node_idx = node_indices[current_node_offset++];
       if(node_status[node_idx] != ACTIVE) { i--; continue; } 
       
       while(g->degrees[node_idx] > 1) {
            int num_neighbors = g->degrees[node_idx];
            int* neighbors = (int*)malloc(num_neighbors * sizeof(int));
            AdjListNode* pCrawl = g->array[node_idx].head;
            for(int k=0; k<num_neighbors; ++k){ neighbors[k] = pCrawl->dest; pCrawl = pCrawl->next; }
            shuffle_array(neighbors, num_neighbors);
            
            int removed = 0;
            for(int k=0; k<num_neighbors; ++k){
                int neighbor_idx = neighbors[k];
                
                // --- MODIFIED: This check is now conditional on being an SC lattice ---
                if (is_sc_lattice && g->degrees[neighbor_idx] <= 2) {
                    continue;
                }

                // --- MODIFIED: Pass sums to is_move_safe (not relevant for stage 2, pass -1)
                if (!is_move_safe(g, node_idx, neighbor_idx, target_counts, max_func, 2, target_degree_sum, -1)) {
                    continue;
                }
                
                if (extensive_logging >= 1) { /* log move */ }
                removeEdge(g, node_idx, neighbor_idx);

                if(is_subgraph_connected(g, node_status)){
                    if (extensive_logging == 1) { /* print per-move */ }
                    removed = 1;
                    break;
                }
                else { 
                    addEdge(g, node_idx, neighbor_idx); // Backtrack
                    if(extensive_logging >= 1) move_log_count--;
                } 
            }
            free(neighbors);
            if(!removed) goto cleanup;
       }
       node_status[node_idx] = IS_DEGREE_1;
       if (extensive_logging >= 1) { /* log move */ }
    }
    if (extensive_logging >= 1) print_distribution("Stage 2: Set-d1 (Strict)", g, target_counts, trial_num, 0, max_func, extensive_logging);

    // --- Stage 3: Enforce Max Functionality (Strict) ---
    for(int i=0; i<total_nodes; ++i) {
        int node_idx = node_indices[i]; 
        if(node_status[node_idx] != ACTIVE) continue; 
        
        while(g->degrees[node_idx] > max_func) {
            int num_neighbors = g->degrees[node_idx];
            int* neighbors = (int*)malloc(num_neighbors * sizeof(int));
            AdjListNode* pCrawl = g->array[node_idx].head;
            for(int k=0; k<num_neighbors; ++k){ neighbors[k] = pCrawl->dest; pCrawl = pCrawl->next; }
            shuffle_array(neighbors, num_neighbors);

            int removed = 0;
            for(int k=0; k<num_neighbors; ++k) {
                int neighbor_idx = neighbors[k];

                // --- MODIFIED: This check is now conditional on being an SC lattice ---
                if (is_sc_lattice && g->degrees[neighbor_idx] <= 2) {
                    continue;
                }

                // --- MODIFIED: Pass sums to is_move_safe (not relevant for stage 3, pass -1)
                if (!is_move_safe(g, node_idx, neighbor_idx, target_counts, max_func, 3, target_degree_sum, -1)) {
                    continue;
                }

                if (extensive_logging >= 1) {
                    if (move_log_count >= move_log_capacity) {
                        move_log_capacity *= 2;
                        move_log = (MoveLog*)realloc(move_log, move_log_capacity * sizeof(MoveLog));
                    }
                    move_log[move_log_count++] = (MoveLog){REMOVE_EDGE, node_idx, neighbor_idx};
                }
                removeEdge(g, node_idx, neighbor_idx);

                if (is_subgraph_connected(g, node_status)) {
                    if (extensive_logging == 1) {
                        move_counter++;
                        print_distribution("Stage 3: Enforce-Max", g, target_counts, trial_num, move_counter, max_func, extensive_logging);
                    }
                    removed = 1;
                    break;
                } else {
                    addEdge(g, node_idx, neighbor_idx);
                    if(extensive_logging >= 1) move_log_count--;
                }
            }
            free(neighbors);
            if (!removed) goto cleanup;
        }
    }
    if (extensive_logging >= 1) print_distribution("Stage 3: Enforce-Max (Strict)", g, target_counts, trial_num, 0, max_func, extensive_logging);

    // --- Stage 4: Systematic Search Loop ---
    while(1) {
        int* current_counts = (int*)calloc(max_func + 3, sizeof(int));
        int is_done = 1;
        long long current_total_degree_sum = 0; // --- NEW ---
        
        int has_high_degree_nodes = 0;
        for(int i=0; i<total_nodes; ++i) {
             current_total_degree_sum += g->degrees[i]; // --- NEW ---
             if (g->degrees[i] <= max_func + 2) current_counts[g->degrees[i]]++;
             if (node_status[i] == ACTIVE && g->degrees[i] > max_func) {
                 has_high_degree_nodes = 1;
             }
        }

        // ---
        // --- THIS IS THE CORRECTED LOGIC ---
        // ---
        if (has_high_degree_nodes) {
            is_done = 0; // Not done, still have nodes > max_func (e.g., in mf=4 run)
        } else {
            // 1. Check all *explicit* d:N targets first
            for(int i=0; i <= max_func; ++i) {
                if (target_counts[i] >= 0 && target_counts[i] != current_counts[i]) {
                    is_done = 0; // Failed an explicit target
                    break;
                }
            }

            if (is_done) { 
                // 2. If explicit targets are met, check which mode we are in
                if (target_edge_count != -1) {
                    // --- e:N Mode (Bond Percolation) ---
                    // This is for your bond percolation study.
                    if (current_total_degree_sum != target_degree_sum) {
                        is_done = 0; // Edge count is wrong
                    }
                    // We also must be connected (implicitly checked by Stage 4 moves)
                    if (!is_subgraph_connected(g, node_status)) {
                         is_done = 0; // Not connected
                    }
                } else {
                    // --- Legacy Mode (Site Percolation) ---
                    // This is for your site percolation study.
                    // We have already confirmed explicit targets (d0) are met.
                    // We *allow* d1-d6 to be "anything".
                    // The *only* thing to check now is connectivity.
                    
                    if (is_subgraph_connected(g, node_status)) {
                        // SUCCESS! Explicit targets (d0) met AND it's connected.
                        is_done = 1;
                    } else {
                        // FAILED! Explicit targets (d0) met but NOT connected.
                        // This is a "valid" data point, but not a successful trial.
                        free(current_counts);
                        goto cleanup; // Fail the trial
                    }
                }
            }
        }
        // --- END CORRECTED LOGIC ---
        
        if (is_done) {
            free(current_counts);
            print_distribution("Final Distribution", g, target_counts, trial_num, 0, max_func, extensive_logging);
            return_graph = g;
            if (extensive_logging >= 1) {
                save_move_log_to_file(move_log, move_log_count, dims_str, trial_num);
            }
            goto cleanup_success;
        }
        free(current_counts);

        // ---
        // This part of the loop will now only be reached if:
        // 1. We are in 'e:N' mode and haven't hit the target edge count.
        // 2. We are in 'mf=4' mode and haven't finished Stage 3's job.
        // It will *not* be reached by a 'mf=6' site percolation run
        // because that run will either 'goto cleanup' or 'goto cleanup_success'.
        // ---
        if (current_total_degree_sum == 0) goto cleanup;
        
        long long num_edges = current_total_degree_sum / 2;
        Edge* edge_list = (Edge*)malloc(num_edges * sizeof(Edge));
        long long current_edge_idx = 0;
        // --- Later we can add spatial skew?
        for(int j=0; j<total_nodes; ++j) { 
            int i = node_indices[j]; // Use the shuffled index
            if (node_status[i] == ACTIVE) {
                AdjListNode* pCrawl = g->array[i].head;
                while(pCrawl) {
                    // We can still use i < pCrawl->dest to avoid double counting
                    if (i < pCrawl->dest && node_status[pCrawl->dest] == ACTIVE) { 
                        edge_list[current_edge_idx].u = i;
                        edge_list[current_edge_idx].v = pCrawl->dest;
                        current_edge_idx++;
                    }
                    pCrawl = pCrawl->next;
                }
            }
        }
        shuffle_edges(edge_list, current_edge_idx);

        int move_made = 0;
        for(long long i=0; i<current_edge_idx; ++i) {
            int u = edge_list[i].u;
            int v = edge_list[i].v;

            int u_deg = g->degrees[u];
            int v_deg = g->degrees[v];

            if (u_deg <= 1 || v_deg <= 1) continue;
            
            // --- MODIFIED: Legacy check for d2, still useful ---
            if (u_deg == 2 || v_deg == 2) {
                // This logic is for legacy mode, but is also safe for 'e:N' mode.
                // It prevents over-sculpting d1 if d1 is not an explicit target.
                // If d1 target is 0, is_move_safe will catch it anyway.
                if (target_counts[1] != -1) { 
                    int current_d1_count = 0;
                    for (int j = 0; j < total_nodes; j++) {
                        if (node_status[j] == ACTIVE && g->degrees[j] == 1) current_d1_count++;
                    }
                    if (current_d1_count >= target_counts[1]) {
                        continue; 
                    }
                }
            }
            
            // --- MODIFIED: Pass current degree sum to is_move_safe ---
            if (!is_move_safe(g, u, v, target_counts, max_func, 4, target_degree_sum, current_total_degree_sum)) {
                continue;
            }

            if (extensive_logging >= 1) { /* log move */ }
            removeEdge(g, u, v);
            
            if (is_subgraph_connected(g, node_status)) {
                move_made = 1;
                break;
            } else {
                addEdge(g, u, v);
                if(extensive_logging >= 1) move_log_count--;
            }
        }
        free(edge_list);

        if (move_made) {
            move_counter++;
            if (extensive_logging == 1) {
                print_distribution("Stage 4: Systematic Search", g, target_counts, trial_num, move_counter, max_func, extensive_logging);
            }
        } else {
            // --- NEW: If no moves are safe, but we are in e:N mode, check if we are simply stuck
            if (target_edge_count != -1) {
                // We are stuck. We already know 'is_done' is false from the start of the loop.
                // This means we are either stuck *above* the target edge count (failure)
                // or *at* the target edge count but with wrong d:N explicit targets (failure).
                goto cleanup;
            }
            
            // Legacy mode: no moves means failure
            goto cleanup;
        }
    }

cleanup:
    freeGraph(g);
    g = NULL;
cleanup_success:
    if (extensive_logging >= 1) { free(move_log); }
    free(node_status);
    free(node_indices);
    return g;
}

// --- START: LATTICE CREATION LOGIC ---

// Helper for SC lattice (original logic)
Graph* create_sc_lattice(int Nx, int Ny, int Nz, const int* p_dims) {
    int total_nodes = Nx * Ny * Nz; 
    Graph* g = createGraph(total_nodes);
    
    // C++ Lambda REMOVED: auto get_index = [&](int x, int y, int z) { ... };

    for (int z = 0; z < Nz; z++) { 
        for (int y = 0; y < Ny; y++) { 
            for (int x = 0; x < Nx; x++) { 
                int u = x + y * Nx + z * Nx * Ny; 
                g->coords[u] = (Coord){(double)x, (double)y, (double)z};

                if (p_dims[0] || x < Nx - 1) addEdge(g, u, ((x + 1) % Nx) + y * Nx + z * Nx * Ny);
                if (p_dims[1] || y < Ny - 1) addEdge(g, u, x + ((y + 1) % Ny) * Nx + z * Nx * Ny);
                if (p_dims[2] || z < Nz - 1) addEdge(g, u, x + y * Nx + ((z + 1) % Nz) * Nx * Ny);
            }
        }
    }
    return g;
}

// Helper for BCC lattice
Graph* create_bcc_lattice(int Nx, int Ny, int Nz, const int* p_dims) {
    int total_nodes = 2 * Nx * Ny * Nz; // MODIFIED
    Graph* g = createGraph(total_nodes);
    int node_idx = 0;

    int high_res_Nx = 2 * Nx; // MODIFIED
    int high_res_Ny = 2 * Ny; // MODIFIED
    int high_res_Nz = 2 * Nz; // MODIFIED
    // MODIFIED: Use new high_res dims
    long long map_size = (long long)high_res_Nx * high_res_Ny * high_res_Nz;
    int* coord_to_id_map = (int*)malloc(map_size * sizeof(int));
    for(long long i=0; i<map_size; ++i) coord_to_id_map[i] = -1;

    // C++ Lambda REMOVED: auto get_map_idx = [&](int x, int y, int z) { ... };

    // 1. Place nodes
    for (int k = 0; k < Nz; k++) { // MODIFIED
        for (int j = 0; j < Ny; j++) { // MODIFIED
            for (int i = 0; i < Nx; i++) { // MODIFIED
                // Corner node
                int cx = 2*i, cy = 2*j, cz = 2*k;
                g->coords[node_idx] = (Coord){(double)i, (double)j, (double)k};
                // C equivalent: use direct calculation (MODIFIED for new indexing)
                coord_to_id_map[(long long)cx + (long long)cy * high_res_Nx + (long long)cz * high_res_Nx * high_res_Ny] = node_idx++;
                
                // Body-centered node
                int bx = 2*i+1, by = 2*j+1, bz = 2*k+1;
                g->coords[node_idx] = (Coord){(double)i+0.5, (double)j+0.5, (double)k+0.5};
                // C equivalent: use direct calculation (MODIFIED for new indexing)
                coord_to_id_map[(long long)bx + (long long)by * high_res_Nx + (long long)bz * high_res_Nx * high_res_Ny] = node_idx++;
            }
        }
    }

    // 2. Connect nodes (8 nearest neighbors)
    for (long long map_idx = 0; map_idx < map_size; ++map_idx) {
        int id = coord_to_id_map[map_idx];
        if (id == -1) continue;

        // MODIFIED: Use new high_res dims for coordinate extraction
        int z = map_idx / ((long long)high_res_Nx * high_res_Ny);
        int y = (map_idx / high_res_Nx) % high_res_Ny;
        int x = map_idx % high_res_Nx;

        for (int dz = -1; dz <= 1; dz += 2) {
            for (int dy = -1; dy <= 1; dy += 2) {
                for (int dx = -1; dx <= 1; dx += 2) {
                    int nx = x + dx;
                    int ny = y + dy;
                    int nz = z + dz;
                    // Handle periodicity (MODIFIED for new high_res dims)
                    if (p_dims[0]) nx = (nx + high_res_Nx) % high_res_Nx;
                    if (p_dims[1]) ny = (ny + high_res_Ny) % high_res_Ny;
                    if (p_dims[2]) nz = (nz + high_res_Nz) % high_res_Nz;

                    // MODIFIED for new high_res dims
                    if (nx >= 0 && nx < high_res_Nx && ny >= 0 && ny < high_res_Ny && nz >= 0 && nz < high_res_Nz) {
                        // C equivalent: use direct calculation (MODIFIED for new indexing)
                        int neighbor_id = coord_to_id_map[(long long)nx + (long long)ny * high_res_Nx + (long long)nz * high_res_Nx * high_res_Ny];
                        if (neighbor_id != -1 && id < neighbor_id) {
                            addEdge(g, id, neighbor_id);
                        }
                    }
                }
            }
        }
    }
    free(coord_to_id_map);
    return g;
}

// Helper for FCC lattice
Graph* create_fcc_lattice(int Nx, int Ny, int Nz, const int* p_dims) {
    int total_nodes = 4 * Nx * Ny * Nz; // MODIFIED
    Graph* g = createGraph(total_nodes);
    int node_idx = 0;

    int high_res_Nx = 2 * Nx; // MODIFIED
    int high_res_Ny = 2 * Ny; // MODIFIED
    int high_res_Nz = 2 * Nz; // MODIFIED
    // MODIFIED: Use new high_res dims
    long long map_size = (long long)high_res_Nx * high_res_Ny * high_res_Nz;
    int* coord_to_id_map = (int*)malloc(map_size * sizeof(int));
    for(long long i=0; i<map_size; ++i) coord_to_id_map[i] = -1;


    // 1. Place nodes
    for (int k = 0; k < Nz; k++) { // MODIFIED
        for (int j = 0; j < Ny; j++) { // MODIFIED
            for (int i = 0; i < Nx; i++) { // MODIFIED
                // Corner node
                g->coords[node_idx] = (Coord){(double)i, (double)j, (double)k};
                // MODIFIED: New indexing
                coord_to_id_map[(long long)(2*i) + (long long)(2*j) * high_res_Nx + (long long)(2*k) * high_res_Nx * high_res_Ny] = node_idx++;
                // Face nodes
                g->coords[node_idx] = (Coord){(double)i+0.5, (double)j+0.5, (double)k};
                // MODIFIED: New indexing
                coord_to_id_map[(long long)(2*i+1) + (long long)(2*j+1) * high_res_Nx + (long long)(2*k) * high_res_Nx * high_res_Ny] = node_idx++;
                g->coords[node_idx] = (Coord){(double)i+0.5, (double)j, (double)k+0.5};
                // MODIFIED: New indexing
                coord_to_id_map[(long long)(2*i+1) + (long long)(2*j) * high_res_Nx + (long long)(2*k+1) * high_res_Nx * high_res_Ny] = node_idx++;
                g->coords[node_idx] = (Coord){(double)i, (double)j+0.5, (double)k+0.5};
                // MODIFIED: New indexing
                coord_to_id_map[(long long)(2*i) + (long long)(2*j+1) * high_res_Nx + (long long)(2*k+1) * high_res_Nx * high_res_Ny] = node_idx++;
            }
        }
    }

// 2. Connect nodes (12 nearest neighbors)
    for (long long map_idx = 0; map_idx < map_size; ++map_idx) {
        int id = coord_to_id_map[map_idx];
        if (id == -1) continue;

        // MODIFIED: Use new high_res dims for coordinate extraction
        int z = map_idx / ((long long)high_res_Nx * high_res_Ny);
        int y = (map_idx / high_res_Nx) % high_res_Ny;
        int x = map_idx % high_res_Nx;

        // --- MODIFIED: Define all 12 neighbor directions ---
        // (No change to this array)
        int neighbor_offsets[12][3] = {
            {1,1,0}, {1,-1,0}, {-1,1,0}, {-1,-1,0},  // XY plane
            {1,0,1}, {1,0,-1}, {-1,0,1}, {-1,0,-1},  // XZ plane
            {0,1,1}, {0,1,-1}, {0,-1,1}, {0,-1,-1}   // YZ plane
        };

        // --- MODIFIED: Loop over all 12 offsets ---
        for(int i=0; i<12; ++i) {
            int nx = x + neighbor_offsets[i][0];
            int ny = y + neighbor_offsets[i][1];
            int nz = z + neighbor_offsets[i][2];
            
            // Handle periodicity (MODIFIED for new high_res dims)
            if (p_dims[0]) nx = (nx + high_res_Nx) % high_res_Nx;
            if (p_dims[1]) ny = (ny + high_res_Ny) % high_res_Ny;
            if (p_dims[2]) nz = (nz + high_res_Nz) % high_res_Nz;

            // MODIFIED for new high_res dims
            if (nx >= 0 && nx < high_res_Nx && ny >= 0 && ny < high_res_Ny && nz >= 0 && nz < high_res_Nz) {
                // MODIFIED: New indexing
                int neighbor_id = coord_to_id_map[(long long)nx + (long long)ny * high_res_Nx + (long long)nz * high_res_Nx * high_res_Ny];
                
                // The (id < neighbor_id) check correctly prevents double-counting
                if (neighbor_id != -1 && id < neighbor_id) { 
                    addEdge(g, id, neighbor_id);
                }
            }
        }
    }
    free(coord_to_id_map);
    return g;
}

/* Diamond lattice: 8 sites per conventional cubic cell, every site
 * exactly 4-coordinated by construction. Two interpenetrating FCC
 * sublattices offset by (1/4, 1/4, 1/4) along the body diagonal.
 *
 * This is the cleanest backbone for a max_func=4 network: the raw
 * lattice already satisfies that ceiling, so sculpting has nothing to
 * prune unless defects are requested.
 *
 * Mirrors create_diamond_lattice in generator_python_diamond.py,
 * including the site order (cell-major, then the eight basis sites in
 * the order listed below), so the two number their nodes identically.
 *
 * Sites live on a x4 integer grid. A-sublattice sites satisfy
 * (hx+hy+hz) % 4 == 0 and reach their four neighbours through the
 * sign-product +1 offsets; B-sublattice sites satisfy sum % 4 == 3 and
 * use the sign-product -1 offsets. No other residue holds a site.
 */
Graph* create_diamond_lattice(int Nx, int Ny, int Nz, const int* p_dims) {
    static const int basis_hr[8][3] = {
        {0,0,0}, {2,2,0}, {2,0,2}, {0,2,2},     /* A sublattice */
        {1,1,1}, {3,3,1}, {3,1,3}, {1,3,3},     /* B sublattice */
    };
    static const int off_a[4][3] = {            /* sign product +1: A -> B */
        {+1,+1,+1}, {+1,-1,-1}, {-1,+1,-1}, {-1,-1,+1},
    };
    static const int off_b[4][3] = {            /* sign product -1: B -> A */
        {-1,-1,-1}, {-1,+1,+1}, {+1,-1,+1}, {+1,+1,-1},
    };

    int total_nodes = 8 * Nx * Ny * Nz;
    Graph* g = createGraph(total_nodes);
    int node_idx = 0;

    int hr_Nx = 4 * Nx, hr_Ny = 4 * Ny, hr_Nz = 4 * Nz;
    long long map_size = (long long)hr_Nx * hr_Ny * hr_Nz;
    int* coord_to_id_map = (int*)malloc(map_size * sizeof(int));
    if (!coord_to_id_map) {
        fprintf(stderr, "Error: out of memory building diamond lattice.\n");
        freeGraph(g);
        return NULL;
    }
    for (long long i = 0; i < map_size; ++i) coord_to_id_map[i] = -1;

    /* 1. Place sites. */
    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Ny; j++) {
            for (int i = 0; i < Nx; i++) {
                for (int b = 0; b < 8; ++b) {
                    int hx = 4*i + basis_hr[b][0];
                    int hy = 4*j + basis_hr[b][1];
                    int hz = 4*k + basis_hr[b][2];
                    g->coords[node_idx] = (Coord){
                        (double)i + basis_hr[b][0] / 4.0,
                        (double)j + basis_hr[b][1] / 4.0,
                        (double)k + basis_hr[b][2] / 4.0,
                    };
                    coord_to_id_map[(long long)hx
                                    + (long long)hy * hr_Nx
                                    + (long long)hz * hr_Nx * hr_Ny] = node_idx++;
                }
            }
        }
    }

    /* 2. Connect each site to its four tetrahedral neighbours. */
    for (long long map_idx = 0; map_idx < map_size; ++map_idx) {
        int id = coord_to_id_map[map_idx];
        if (id == -1) continue;

        int z = map_idx / ((long long)hr_Nx * hr_Ny);
        int y = (map_idx / hr_Nx) % hr_Ny;
        int x = map_idx % hr_Nx;

        int s = (x + y + z) % 4;
        const int (*offsets)[3] = (s == 0) ? off_a : off_b;

        for (int o = 0; o < 4; ++o) {
            int nx = x + offsets[o][0];
            int ny = y + offsets[o][1];
            int nz = z + offsets[o][2];
            if (p_dims[0]) nx = (nx + hr_Nx) % hr_Nx;
            if (p_dims[1]) ny = (ny + hr_Ny) % hr_Ny;
            if (p_dims[2]) nz = (nz + hr_Nz) % hr_Nz;

            if (nx >= 0 && nx < hr_Nx && ny >= 0 && ny < hr_Ny && nz >= 0 && nz < hr_Nz) {
                int neighbor_id = coord_to_id_map[(long long)nx
                                                  + (long long)ny * hr_Nx
                                                  + (long long)nz * hr_Nx * hr_Ny];
                if (neighbor_id != -1 && id < neighbor_id) {
                    addEdge(g, id, neighbor_id);
                }
            }
        }
    }

    free(coord_to_id_map);
    return g;
}

/* Join every pair of sites within `cutoff` under the minimum image on the
 * periodic axes; an open axis keeps the raw separation, so nothing bonds
 * across a free face. Shared by MIX at every cutoff and by the pure
 * lattices at a non-default one. O(N^2) in the site count: negligible at
 * the cell counts topon uses, seconds past ~20x20x20 (the Python side
 * switches to a cell list there). Mirrors
 * generator_python.edges_within_cutoff, including rint()'s half-to-even
 * agreement with numpy.round and the 1e-12 tolerance. */
static void connect_within_cutoff(Graph* g, const int* p_dims,
                                  const double box[3], double cutoff) {
    double cutoff_sq = cutoff * cutoff;
    for (int a = 0; a < g->V; ++a) {
        for (int b = a + 1; b < g->V; ++b) {
            double d2 = 0.0;
            double dv[3] = {g->coords[a].x - g->coords[b].x,
                            g->coords[a].y - g->coords[b].y,
                            g->coords[a].z - g->coords[b].z};
            for (int ax = 0; ax < 3; ++ax) {
                double d = dv[ax];
                if (p_dims[ax]) d -= box[ax] * rint(d / box[ax]);
                d2 += d * d;
            }
            if (d2 <= cutoff_sq + 1e-12 && d2 > 1e-12) addEdge(g, a, b);
        }
    }
}

/* A copy of `src` with its sites and their numbering but none of its
 * edges: the starting point for rebuilding a pure lattice at a cutoff. */
static Graph* sites_only(const Graph* src) {
    Graph* g = createGraph(src->V);
    memcpy(g->coords, src->coords, (size_t)src->V * sizeof(Coord));
    return g;
}

/* Mixed SC/BCC/FCC lattice.
 *
 * Mirrors PythonTopologyGenerator._create_mixed_lattice. All three
 * lattices share the cubic cell corner and each adds sites on top of it:
 * BCC one body centre, FCC three face centres. So the corner goes into
 * every cell, the body centre with probability f_bcc and each face
 * centre with probability f_fcc. The SC fraction is the remainder and
 * contributes no site of its own, which is what makes the three
 * fractions a partition summing to 1.
 *
 * Edges join every pair within `cutoff` under the minimum image, because
 * a mixed point set has no single neighbour shell to enumerate the way
 * the pure builders do. Non-periodic axes (p_dims[k] == 0) are not
 * wrapped. The search is O(N^2) in the site count, which is negligible
 * at the cell counts topon uses but grows quickly past ~20x20x20.
 *
 * Site order matches the Python builder (z outer, then y, then x, and
 * within a cell corner, body, then the XY/XZ/YZ faces) so that fractions
 * (1, 0, 0) reproduce create_sc_lattice exactly. The draws come from
 * rng_uniform(), so C and Python agree on distributions, not on a given
 * draw.
 */
Graph* create_mixed_lattice(int Nx, int Ny, int Nz, const int* p_dims,
                            double f_bcc, double f_fcc, double cutoff) {
    static const double face_off[3][3] = {
        {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}
    };

    /* Worst case is every optional site drawn: corner + body + 3 faces = 5
     * per cell. Sizing at 4 (the FCC site count) is NOT enough: with
     * f_bcc=0.1, f_fcc=0.9 a 4x4x4 overruns 4*N in ~0.2% of draws. */
    int max_sites = 5 * Nx * Ny * Nz;
    Coord* sites = (Coord*)malloc((size_t)max_sites * sizeof(Coord));
    if (!sites) { fprintf(stderr, "Error: out of memory building mixed lattice.\n"); return NULL; }

    int n = 0;
    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                sites[n++] = (Coord){(double)i, (double)j, (double)k};
                if (f_bcc > 0.0 && rng_uniform() < f_bcc) {
                    sites[n++] = (Coord){i + 0.5, j + 0.5, k + 0.5};
                }
                if (f_fcc > 0.0) {
                    for (int f = 0; f < 3; ++f) {
                        if (rng_uniform() < f_fcc) {
                            sites[n++] = (Coord){i + face_off[f][0],
                                                 j + face_off[f][1],
                                                 k + face_off[f][2]};
                        }
                    }
                }
            }
        }
    }

    Graph* g = createGraph(n);
    for (int i = 0; i < n; ++i) g->coords[i] = sites[i];
    free(sites);

    double box[3] = {(double)Nx, (double)Ny, (double)Nz};
    connect_within_cutoff(g, p_dims, box, cutoff);
    return g;
}

// --- END: LATTICE CREATION LOGIC ---


/* Largest number of distinct partners any site has, ignoring self-loops
 * and repeated pairs. The pure builders emit a repeated pair on a
 * periodic axis two cells long (x+1 and x-1 wrap to the same site), which
 * the Python builders' networkx graph collapses, and a self-loop on an
 * axis one cell long, which networkx keeps and this count (and the exact
 * search) does not. Neither occurs on a cell of three or more per axis. */
static int simple_max_degree(const Graph* g) {
    int best = 0;
    int* seen = (int*)malloc((size_t)(g->V > 0 ? g->V : 1) * sizeof(int));
    for (int i = 0; i < g->V; ++i) seen[i] = -1;
    for (int i = 0; i < g->V; ++i) {
        int d = 0;
        for (AdjListNode* p = g->array[i].head; p; p = p->next) {
            if (p->dest == i || seen[p->dest] == i) continue;
            seen[p->dest] = i;
            d++;
        }
        if (d > best) best = d;
    }
    free(seen);
    return best;
}


// --- START: EXACT DEGREE MATCHING (--search=exact) ---

/* A port of sculpt_exact and build_exact_graph in
 * topon/topology/degree_matching.py, whose module docstring is the
 * specification. The strict sculptor above removes candidate edges one at
 * a time; this search starts from no edges and builds a subgraph of the
 * candidates whose degree sequence is exactly the requested multiset:
 *
 *   1. give every site a target degree by a random permutation of the
 *      requested counts; the sites left over are vacancies (t = 0)
 *   2. greedy random fill, where a degree-1 site may only bond to a site
 *      with t >= 2, so a dangling end always hangs off a junction
 *   3. alternating-path BFS from every site still below target (add an
 *      absent candidate edge, remove a present one, ...) to another
 *      deficient site, then flip the path
 *   4. repair a locally infeasible draw by swapping targets between a
 *      deficient site and a saturated one of lower target (never a
 *      degree-1 site), or by moving a vacancy onto the deficient site,
 *      then augment again
 *   5. accept when every site reached its target and the largest
 *      component holds at least min_giant_fraction of the active sites
 *
 * One call of exact_attempt is one sculpt_exact call. The step bounds
 * and the acceptance floor are the Python constants. The forced double
 * edges of the Python search (double_pairs) have no command-line channel
 * here and are not ported; the pipeline keeps such requests on Python.
 *
 * Besides the random stream, two orders differ from Python, both fixed
 * rather than drawn. A dangling end that repair hands a partner takes the
 * first eligible neighbour in the scaffold's neighbour order, which the C
 * and Python lattice builders lay down differently. And after a vacancy
 * move, the former partners are drained in the order of the site's edge
 * list here, and in set-iteration order there. A comparison of about
 * 16 000 attempts on SC and FCC found no effect of either.
 *
 * Data layout: the scaffold is a CSR array with, for every directed slot,
 * the slot of the reverse direction, so an edge is switched on or off in
 * O(1) from either end. The present edges of a site sit in a small array
 * of CSR slots (a site never carries more than max_f + 1 of them, the +1
 * being the transient while a path is flipped). The BFS reuses one
 * visited-stamp array, one parent array and one queue across every call,
 * so nothing is allocated after exact_create. */

#define EXACT_MAX_AUGMENT_ROUNDS 6
#define EXACT_MAX_REPAIR_STEPS 20000
#define EXACT_DEFAULT_MIN_GIANT_FRACTION 0.99
#define EXACT_PINNED_SHARE 0.5

typedef struct ExactSearch {
    int n;                  /* sites */
    int max_f;
    int cap;                /* present-edge slots per site, max_f + 1 */
    int max_base_deg;       /* most candidate partners of any site */
    long long n_slots;      /* 2 x candidate edges */
    int* nb_off;            /* CSR row offsets, n + 1 */
    int* nb;                /* CSR partner of each slot */
    int* rev;               /* slot of the reverse direction */
    unsigned char* present; /* is the edge of this slot in the subgraph */
    int* t;                 /* target degree, 0 for a vacancy */
    int* deg;               /* achieved degree */
    int* adj;               /* present slots of site i at adj[i * cap] */
    unsigned int* stamp;    /* visited marks, valid when equal to cur */
    unsigned int cur;
    int* par_node;
    int* par_slot;          /* slot in par_node's row that reached the site */
    unsigned char* par_add; /* 1 if reached by an "add" move */
    int* queue;
    int* list;              /* scratch site lists */
    int* list2;
    int* cand;              /* scratch slot list for one expansion */
    int* partners;
    const int* need;        /* requested count per degree 0..max_f */
    double min_giant;
    long long augmentations;
    long long swaps;
} ExactSearch;

typedef struct ExactOutcome {
    int ok;
    long long residual;     /* degree units left unfilled */
    int n_short;            /* sites below their target */
    int giant_known;        /* set once the residual reached 0 */
    double giant_frac;
    int n_components;
    int n_active;
    long long augmentations;
    long long swaps;
    double seconds;
} ExactOutcome;

static ExactSearch* exact_create(const Graph* base, int max_f, const int* need,
                                 double min_giant) {
    ExactSearch* w = (ExactSearch*)calloc(1, sizeof(ExactSearch));
    int n = base->V;
    w->n = n;
    w->max_f = max_f;
    w->cap = max_f + 1;
    w->need = need;
    w->min_giant = min_giant;

    long long total = 0;
    for (int i = 0; i < n; ++i) total += base->degrees[i];
    w->nb_off = (int*)malloc((size_t)(n + 1) * sizeof(int));
    w->nb = (int*)malloc((size_t)(total + 1) * sizeof(int));
    int* seen = (int*)malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
    for (int i = 0; i < n; ++i) seen[i] = -1;
    long long pos = 0;
    for (int i = 0; i < n; ++i) {
        w->nb_off[i] = (int)pos;
        /* The scaffold's own neighbour order, minus self-loops and repeats. */
        for (AdjListNode* p = base->array[i].head; p; p = p->next) {
            if (p->dest == i || seen[p->dest] == i) continue;
            seen[p->dest] = i;
            w->nb[pos++] = p->dest;
        }
        int d = (int)(pos - w->nb_off[i]);
        if (d > w->max_base_deg) w->max_base_deg = d;
    }
    w->nb_off[n] = (int)pos;
    w->n_slots = pos;
    free(seen);

    w->rev = (int*)malloc((size_t)(pos + 1) * sizeof(int));
    for (int i = 0; i < n; ++i) {
        for (int s = w->nb_off[i]; s < w->nb_off[i + 1]; ++s) {
            int j = w->nb[s];
            w->rev[s] = -1;
            for (int r = w->nb_off[j]; r < w->nb_off[j + 1]; ++r) {
                if (w->nb[r] == i) { w->rev[s] = r; break; }
            }
        }
    }

    size_t nn = (size_t)(n > 0 ? n : 1);
    int scratch = (w->max_base_deg > w->cap ? w->max_base_deg : w->cap) + 1;
    w->present = (unsigned char*)calloc((size_t)(pos + 1), 1);
    w->t = (int*)calloc(nn, sizeof(int));
    w->deg = (int*)calloc(nn, sizeof(int));
    w->adj = (int*)malloc(nn * (size_t)w->cap * sizeof(int));
    w->stamp = (unsigned int*)calloc(nn, sizeof(unsigned int));
    w->cur = 0;
    w->par_node = (int*)malloc(nn * sizeof(int));
    w->par_slot = (int*)malloc(nn * sizeof(int));
    w->par_add = (unsigned char*)malloc(nn);
    w->queue = (int*)malloc(nn * sizeof(int));
    w->list = (int*)malloc(nn * sizeof(int));
    w->list2 = (int*)malloc(nn * sizeof(int));
    w->cand = (int*)malloc((size_t)scratch * sizeof(int));
    w->partners = (int*)malloc((size_t)(w->cap + 1) * sizeof(int));
    return w;
}

static void exact_free(ExactSearch* w) {
    if (!w) return;
    free(w->nb_off); free(w->nb); free(w->rev); free(w->present);
    free(w->t); free(w->deg); free(w->adj); free(w->stamp);
    free(w->par_node); free(w->par_slot); free(w->par_add);
    free(w->queue); free(w->list); free(w->list2); free(w->cand);
    free(w->partners);
    free(w);
}

static unsigned int exact_new_stamp(ExactSearch* w) {
    if (++w->cur == 0) {
        memset(w->stamp, 0, (size_t)w->n * sizeof(unsigned int));
        w->cur = 1;
    }
    return w->cur;
}

/* The standing rule: a candidate partner must be an active site, and two
 * degree-1 sites never bond to each other (that would be a free chain,
 * not part of the network). Evaluated at the point of use, because the
 * repair loop moves targets around. */
static int exact_eligible(const ExactSearch* w, int x, int y) {
    return w->t[y] > 0 && !(w->t[x] == 1 && w->t[y] == 1);
}

/* Switch on the edge of slot s, which lies in row x. */
static void exact_add(ExactSearch* w, int x, int s) {
    int y = w->nb[s], r = w->rev[s];
    w->present[s] = 1;
    w->present[r] = 1;
    w->adj[(size_t)x * w->cap + w->deg[x]++] = s;
    w->adj[(size_t)y * w->cap + w->deg[y]++] = r;
}

static void exact_drop_slot(ExactSearch* w, int x, int s) {
    int* a = w->adj + (size_t)x * w->cap;
    int d = w->deg[x];
    for (int k = 0; k < d; ++k) {
        if (a[k] == s) { a[k] = a[d - 1]; break; }
    }
    w->deg[x] = d - 1;
}

/* Switch off the edge of slot s, which lies in row x. */
static void exact_rem(ExactSearch* w, int x, int s) {
    int y = w->nb[s], r = w->rev[s];
    w->present[s] = 0;
    w->present[r] = 0;
    exact_drop_slot(w, x, s);
    exact_drop_slot(w, y, r);
}

/* Alternating BFS from a deficient site s. Moves alternate: add an absent
 * candidate edge, remove a present one, add, ... and the walk ends the
 * moment an "add" reaches another deficient site, which is returned (or
 * -1). Candidates are drawn in random order by a Fisher-Yates shuffle run
 * lazily, so an expansion that finds its end early stops drawing. */
static int exact_augment(ExactSearch* w, int s) {
    unsigned int st = exact_new_stamp(w);
    int head = 0, tail = 0;
    w->stamp[s] = st;
    w->par_node[s] = -1;
    w->par_add[s] = 0;                 /* the first move from s is an add */
    w->queue[tail++] = s;
    while (head < tail) {
        int x = w->queue[head++];
        int nc = 0;
        if (!w->par_add[x]) {
            for (int q = w->nb_off[x]; q < w->nb_off[x + 1]; ++q) {
                int y = w->nb[q];
                if (w->stamp[y] != st && !w->present[q] && exact_eligible(w, x, y))
                    w->cand[nc++] = q;
            }
            for (int j = 0; j < nc; ++j) {
                int r = j + (int)rng_below((uint64_t)(nc - j));
                int q = w->cand[r]; w->cand[r] = w->cand[j]; w->cand[j] = q;
                int y = w->nb[q];
                w->stamp[y] = st;
                w->par_node[y] = x;
                w->par_slot[y] = q;
                w->par_add[y] = 1;
                if (w->deg[y] < w->t[y]) return y;
                w->queue[tail++] = y;
            }
        } else {
            const int* a = w->adj + (size_t)x * w->cap;
            for (int k = 0; k < w->deg[x]; ++k) {
                if (w->stamp[w->nb[a[k]]] != st) w->cand[nc++] = a[k];
            }
            for (int j = 0; j < nc; ++j) {
                int r = j + (int)rng_below((uint64_t)(nc - j));
                int q = w->cand[r]; w->cand[r] = w->cand[j]; w->cand[j] = q;
                int y = w->nb[q];
                w->stamp[y] = st;
                w->par_node[y] = x;
                w->par_slot[y] = q;
                w->par_add[y] = 0;
                w->queue[tail++] = y;
            }
        }
    }
    return -1;
}

/* Augment from s until it reaches its target or no path is left. Flipping
 * a path raises both end degrees by one and leaves every site between
 * them where it was. */
static void exact_drain(ExactSearch* w, int s) {
    while (w->deg[s] < w->t[s]) {
        int y = exact_augment(w, s);
        if (y < 0) return;
        while (w->par_node[y] >= 0) {
            int x = w->par_node[y];
            if (w->par_add[y]) exact_add(w, x, w->par_slot[y]);
            else exact_rem(w, x, w->par_slot[y]);
            y = x;
        }
        w->augmentations++;
    }
}

static int exact_collect_deficient(ExactSearch* w) {
    int m = 0;
    for (int i = 0; i < w->n; ++i) {
        if (w->t[i] > 0 && w->deg[i] < w->t[i]) w->list[m++] = i;
    }
    return m;
}

/* One attempt, the equivalent of one sculpt_exact call. */
static void exact_attempt(ExactSearch* w, ExactOutcome* out) {
    clock_t t0 = clock();
    int n = w->n;
    memset(w->present, 0, (size_t)w->n_slots);
    memset(w->deg, 0, (size_t)n * sizeof(int));
    w->augmentations = 0;
    w->swaps = 0;

    /* 1. Target degrees: the requested multiset, vacancies for the rest,
     *    in random order. */
    int k = 0;
    for (int d = 1; d <= w->max_f; ++d) {
        for (int c = 0; c < w->need[d]; ++c) w->t[k++] = d;
    }
    while (k < n) w->t[k++] = 0;
    shuffle_array(w->t, (size_t)n);

    /* 2. Greedy random fill. */
    int m = 0;
    for (int i = 0; i < n; ++i) if (w->t[i] > 0) w->list[m++] = i;
    shuffle_array(w->list, (size_t)m);
    for (int a = 0; a < m; ++a) {
        int u = w->list[a], nc = 0;
        for (int q = w->nb_off[u]; q < w->nb_off[u + 1]; ++q) {
            int v = w->nb[q];
            if (!w->present[q] && exact_eligible(w, u, v) && w->deg[v] < w->t[v])
                w->cand[nc++] = q;
        }
        for (int j = 0; j < nc && w->deg[u] < w->t[u]; ++j) {
            int r = j + (int)rng_below((uint64_t)(nc - j));
            int q = w->cand[r]; w->cand[r] = w->cand[j]; w->cand[j] = q;
            exact_add(w, u, q);
        }
    }

    /* 3. Augmenting paths. */
    for (int round = 0; round < EXACT_MAX_AUGMENT_ROUNDS; ++round) {
        m = exact_collect_deficient(w);
        if (!m) break;
        shuffle_array(w->list, (size_t)m);
        for (int a = 0; a < m; ++a) exact_drain(w, w->list[a]);
    }

    /* 4. Repair a locally infeasible draw. The random assignment can put
     *    a high target where the scaffold cannot serve it (next to
     *    vacancies, or in a corner the dangling-end rule has emptied). */
    for (int step = 0; step < EXACT_MAX_REPAIR_STEPS; ++step) {
        m = exact_collect_deficient(w);
        if (!m) break;
        int u = w->list[rng_below((uint64_t)m)];
        if (w->t[u] == 1) {
            /* A dangling end with no partner: take any neighbour with room. */
            int done = 0;
            for (int q = w->nb_off[u]; q < w->nb_off[u + 1]; ++q) {
                int v = w->nb[q];
                if (exact_eligible(w, u, v) && w->deg[v] < w->t[v]) {
                    exact_add(w, u, q);
                    done = 1;
                    break;
                }
            }
            if (done) continue;
        }
        /* Swap partners: saturated sites of lower target first, any site
         * of lower target otherwise. Never a degree-1 site. */
        int np = 0;
        for (int i = 0; i < n; ++i) {
            if (w->t[i] >= 2 && w->t[i] < w->t[u] && w->deg[i] == w->t[i]
                && w->deg[u] <= w->t[i]) w->list[np++] = i;
        }
        if (!np) {
            for (int i = 0; i < n; ++i) {
                if (w->t[i] >= 2 && w->t[i] < w->t[u] && w->deg[u] <= w->t[i])
                    w->list[np++] = i;
            }
        }
        if (!np || rng_uniform() < 0.3) {
            /* Move the vacancy: u becomes empty and a vacancy with enough
             * active neighbours takes its target. */
            int nv = 0;
            for (int i = 0; i < n; ++i) {
                if (w->t[i] != 0) continue;
                int c = 0;
                for (int q = w->nb_off[i]; q < w->nb_off[i + 1]; ++q) {
                    int y = w->nb[q];
                    if (w->t[y] > 0 && y != u) c++;
                }
                if (c >= w->t[u]) w->list2[nv++] = i;
            }
            if (nv) {
                int v = w->list2[rng_below((uint64_t)nv)];
                int np2 = w->deg[u];
                const int* a = w->adj + (size_t)u * w->cap;
                for (int j = 0; j < np2; ++j) w->partners[j] = w->nb[a[j]];
                while (w->deg[u] > 0) exact_rem(w, u, a[w->deg[u] - 1]);
                w->t[v] = w->t[u];
                w->t[u] = 0;
                w->swaps++;
                exact_drain(w, v);
                for (int j = 0; j < np2; ++j) exact_drain(w, w->partners[j]);
                continue;
            }
            if (!np) break;
        }
        int v = w->list[rng_below((uint64_t)np)];
        int tmp = w->t[u]; w->t[u] = w->t[v]; w->t[v] = tmp;
        w->swaps++;
        exact_drain(w, v);
        exact_drain(w, u);
    }

    /* 5. Accept or reject. */
    memset(out, 0, sizeof(*out));
    for (int i = 0; i < n; ++i) {
        if (w->t[i] <= 0) continue;
        out->n_active++;
        if (w->deg[i] < w->t[i]) {
            out->residual += w->t[i] - w->deg[i];
            out->n_short++;
        }
    }
    out->augmentations = w->augmentations;
    out->swaps = w->swaps;
    if (out->residual == 0) {
        unsigned int st = exact_new_stamp(w);
        int largest = 0;
        for (int i = 0; i < n; ++i) {
            if (w->t[i] <= 0 || w->stamp[i] == st) continue;
            int head = 0, tail = 0;
            w->stamp[i] = st;
            w->queue[tail++] = i;
            while (head < tail) {
                int x = w->queue[head++];
                const int* a = w->adj + (size_t)x * w->cap;
                for (int j = 0; j < w->deg[x]; ++j) {
                    int y = w->nb[a[j]];
                    if (w->stamp[y] != st) { w->stamp[y] = st; w->queue[tail++] = y; }
                }
            }
            out->n_components++;
            if (tail > largest) largest = tail;
        }
        out->giant_known = 1;
        out->giant_frac = out->n_active ? (double)largest / out->n_active : 1.0;
        out->ok = !(out->giant_frac < w->min_giant);
    }
    out->seconds = (double)(clock() - t0) / CLOCKS_PER_SEC;
}

static void exact_describe_error(const ExactSearch* w, const ExactOutcome* o,
                                 char* buf, size_t len) {
    if (o->residual != 0) {
        snprintf(buf, len, "%lld degree unit(s) unfilled on %d site(s)",
                 o->residual, o->n_short);
    } else {
        snprintf(buf, len, "the giant component holds %.3f of the active sites, "
                 "below min_giant_fraction %g", o->giant_frac, w->min_giant);
    }
}

/* True when most of the request is pinned to the scaffold's ceiling
 * (degree_matching._no_slack). A Diamond lattice at the default cutoff
 * has exactly four candidates per site, so a max_func = 4 request leaves
 * no choice anywhere, and retrying draws the same forced assignment. */
static int exact_no_slack(const ExactSearch* w) {
    int ceiling = w->max_base_deg < w->max_f ? w->max_base_deg : w->max_f;
    long long active = 0, pinned = 0;
    for (int d = 1; d <= w->max_f; ++d) {
        int c = w->need[d];
        if (c <= 0) continue;
        active += c;
        if (d >= ceiling) pinned += c;
    }
    return active > 0 && pinned >= EXACT_PINNED_SHARE * active;
}

/* Why the search stopped, in terms of the scaffold and the request
 * (degree_matching._failure_message). */
static void exact_failure_message(FILE* f, const ExactSearch* w, const char* label,
                                  long long attempts, const ExactOutcome* best) {
    long long n_active = 0, wanted_sum = 0;
    for (int d = 1; d <= w->max_f; ++d) {
        n_active += w->need[d];
        wanted_sum += (long long)d * w->need[d];
    }
    double z_mean = w->n ? (double)w->n_slots / w->n : 0.0;
    char err[256];
    if (best) exact_describe_error(w, best, err, sizeof(err));
    else snprintf(err, sizeof(err), "no attempt completed");
    fprintf(f, "Error: the exact degree search did not reach the requested counts on "
               "%s after %lld attempt(s).\n", label, attempts);
    fprintf(f, "  requested : %lld active sites, degree sum %lld, on %d sites with "
               "%lld candidate edges (mean coordination %.1f, max %d)\n",
            n_active, wanted_sum, w->n, w->n_slots / 2, z_mean, w->max_base_deg);
    fprintf(f, "  best try  : %s", err);
    if (best && best->giant_known) fprintf(f, ", giant component %.3f", best->giant_frac);
    fprintf(f, "\n");
    if (exact_no_slack(w)) {
        int ceiling = w->max_base_deg < w->max_f ? w->max_base_deg : w->max_f;
        long long pinned = 0;
        for (int d = ceiling; d <= w->max_f; ++d) if (d > 0) pinned += w->need[d];
        fprintf(f, "  reason    : the scaffold offers at most %d candidate partners per "
                   "site, and %lld of the %lld active sites are asked for degree %d, so "
                   "most of them must bond to every neighbour they have. There is almost "
                   "no spare edge, and each vacancy%s takes capacity from its neighbours "
                   "that nothing can give back.\n",
                w->max_base_deg, pinned, n_active, ceiling,
                w->need[1] ? " and each dangling-end site" : "");
        fprintf(f, "  fix       : give the scaffold more candidates per site than the "
                   "request needs -- a wider neighbour_cutoff, or a lattice with a higher "
                   "coordination -- or drop the degree-1 sites. Note that the default "
                   "cutoff of 1.0 is the canonical-lattice sentinel rather than a range, "
                   "so on Diamond the wider setting is the smaller number: 0.71 cell "
                   "units admits the second shell (z = 16) where 1.0 gives the canonical 4.\n");
    } else {
        fprintf(f, "  fix       : give the search more room -- a larger lattice, a larger "
                   "neighbour_cutoff, or fewer sites at the ceiling -- or lower "
                   "min_giant_fraction if the giant component is what failed.\n");
    }
}

/* The scaffold's sites with only the sculpted edges, in the scaffold's
 * own neighbour order, so the files read like a strict sculpt's. */
static Graph* exact_to_graph(const ExactSearch* w, const Graph* base) {
    Graph* g = createGraph(w->n);
    memcpy(g->coords, base->coords, (size_t)w->n * sizeof(Coord));
    for (int x = 0; x < w->n; ++x) {
        for (int q = w->nb_off[x]; q < w->nb_off[x + 1]; ++q) {
            int y = w->nb[q];
            if (x < y && w->present[q]) addEdge(g, x, y);
        }
    }
    return g;
}

// --- END: EXACT DEGREE MATCHING ---


// --- Main ---

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s <dims_str> <periodicity_str> <max_func> <max_trials> <max_saves> \"<degree_dist_string>\" <extensive_logging> <lattice_type> [neighbour_cutoff] [--search=strict|exact] [--min-giant-fraction=F]\n", prog);
    fprintf(stderr, "Example (Legacy): %s 8x8x8 111 4 1000 1 \"0:0,1:0,2:100,3:312,4:100\" 1 FCC\n", prog);
    fprintf(stderr, "Example (New 'e'): %s 8x6x8 110 6 1000 1 \"0:0,1:0,2:100,e:450\" 1 SC\n", prog);
    fprintf(stderr, "  <dims_str>: Dimensions in NxN_yN_z format (e.g., '8x6x8').\n");
    fprintf(stderr, "  <degree_dist_string>: \"d0:N0,d1:N1,e:TotalEdges\"\n");
    fprintf(stderr, "  <lattice_type>: SC, BCC, FCC, Diamond, or MIX:<sc>,<bcc>,<fcc>[,<cutoff>]\n");
    fprintf(stderr, "  [neighbour_cutoff]: candidate-edge range in cell units for SC/BCC/FCC/Diamond (default 1.0 = nearest neighbours)\n");
    fprintf(stderr, "  --search=strict: prune the lattice edge by edge (the default)\n");
    fprintf(stderr, "  --search=exact: exact degree matching; needs a count for every degree 0..max_func, max_trials bounds the attempts\n");
    fprintf(stderr, "  --min-giant-fraction=F: exact search only, the share of active sites the largest component must hold (default 0.99)\n");
    fprintf(stderr, "Example (Mixed):  %s 6x6x6 111 4 1000 1 \"0:0,1:0\" 0 MIX:0.2,0.4,0.4\n", prog);
    fprintf(stderr, "Example (Shells): %s 6x6x6 111 4 1000 1 \"0:0,1:0\" 0 SC 1.74\n", prog);
    fprintf(stderr, "Example (Exact):  %s 6x6x6 111 6 100 1 \"0:10,1:0,2:26,3:75,4:43,5:53,6:9\" 0 SC --search=exact\n", prog);
}

int main(int argc, char *argv[]) {
    /* Eight positional arguments, plus an optional ninth: the neighbour
     * cutoff for the pure lattices. Named flags (--search=..., and
     * --min-giant-fraction=... for the exact search) may sit anywhere in
     * argv and do not count as positions, so every existing eight- or
     * nine-argument call keeps its meaning. */
    int search_exact = 0;
    double min_giant = EXACT_DEFAULT_MIN_GIANT_FRACTION;
    int min_giant_given = 0;
    char* pos[10] = {0};
    int npos = 0;
    for (int i = 1; i < argc; ++i) {
        char* a = argv[i];
        if (strncmp(a, "--", 2) != 0) {
            if (npos < 10) pos[npos] = a;
            npos++;
            continue;
        }
        if (strncmp(a, "--search=", 9) == 0) {
            if (strcmp(a + 9, "exact") == 0) search_exact = 1;
            else if (strcmp(a + 9, "strict") == 0) search_exact = 0;
            else {
                fprintf(stderr, "Error: --search must be 'strict' or 'exact', got '%s'.\n", a + 9);
                return 1;
            }
        } else if (strncmp(a, "--min-giant-fraction=", 21) == 0) {
            char* end = NULL;
            min_giant = strtod(a + 21, &end);
            if (end == a + 21 || *end != '\0' || !(min_giant > 0.0 && min_giant <= 1.0)) {
                fprintf(stderr, "Error: --min-giant-fraction must be in (0, 1], got '%s'.\n", a + 21);
                return 1;
            }
            min_giant_given = 1;
        } else {
            fprintf(stderr, "Error: unknown option '%s'.\n", a);
            print_usage(argv[0]);
            return 1;
        }
    }
    if (npos != 8 && npos != 9) {
        print_usage(argv[0]);
        return 1;
    }
    if (min_giant_given && !search_exact) {
        fprintf(stderr, "Warning: --min-giant-fraction applies to --search=exact only; the "
                        "strict search keeps every active site connected.\n");
    }

    // MODIFIED: Parse dims_str instead of N
    char* dims_str = pos[0];
    int Nx, Ny, Nz;
    if (sscanf(dims_str, "%dx%dx%d", &Nx, &Ny, &Nz) != 3) {
        fprintf(stderr, "Error: Invalid dimensions string '%s'. Expected format: NxN_yN_z (e.g., '8x8x8' or '8x6x8').\n", dims_str);
        return 1;
    }

    char* periodicity_str = pos[1];
    int max_func = atoi(pos[2]);
    int max_trials = atoi(pos[3]);
    int max_saves = atoi(pos[4]);
    char* degree_dist_string_arg = pos[5];
    int extensive_logging = atoi(pos[6]);
    // --- NEW: Parse lattice type ---
    char* lattice_type = pos[7];
    char* cutoff_arg = (npos == 9) ? pos[8] : NULL;

    /* Optional ninth argument: the neighbour cutoff in cell units. MIX
     * may carry its own inside the lattice argument; giving both is
     * refused below rather than letting one silently win. */
    double neighbour_cutoff = 1.0;
    int cutoff_given = 0;
    if (cutoff_arg) {
        char* end = NULL;
        neighbour_cutoff = strtod(cutoff_arg, &end);
        if (end == cutoff_arg || *end != '\0' || !(neighbour_cutoff > 0.0)) {
            fprintf(stderr, "Error: neighbour_cutoff must be a positive number of cell units, got '%s'.\n", cutoff_arg);
            return 1;
        }
        cutoff_given = 1;
    }

    int p_dims[3];
    p_dims[0] = (periodicity_str[0] == '1');
    p_dims[1] = (periodicity_str[1] == '1');
    p_dims[2] = (periodicity_str[2] == '1');
    /* Mirror into the file-scope copy the .nodes writer reads. */
    g_periodicity[0] = p_dims[0];
    g_periodicity[1] = p_dims[1];
    g_periodicity[2] = p_dims[2];
    
    char degree_dist_string[1024];
    strncpy(degree_dist_string, degree_dist_string_arg, sizeof(degree_dist_string) - 1);
    degree_dist_string[sizeof(degree_dist_string) - 1] = '\0';

    if (search_exact) {
        printf("SIMULATION: Starting serial execution with the Exact Degree-Matching Search.\n");
    } else {
        printf("SIMULATION: Starting serial execution with Strict Sculpting Algorithm.\n");
    }
    // MODIFIED: Updated info print
    printf("INFO: Dims=%s, periodicity=[%d,%d,%d], max_func=%d, trials=%d, max_saves=%d, logging=%d, lattice=%s\n",
           dims_str, p_dims[0], p_dims[1], p_dims[2], max_func, max_trials, max_saves, extensive_logging, lattice_type);
    if (search_exact) {
        printf("INFO: search=exact, min_giant_fraction=%g, one attempt per trial.\n", min_giant);
    }
    mkdir("output", 0755);

    /* Seed before building the lattice: MIX draws its sites from the
     * stream, unlike the pure builders, which consume no randomness while
     * being built.
     *
     * The pid is mixed in because time(NULL) only advances once a second:
     * a script looping this executable to collect N networks used to get
     * byte-identical output from every run that started within the same
     * second. Verified before the fix -- three back-to-back runs produced
     * the same file. clock() and a stack address add what they can on top.
     * TOPON_SEED overrides for reproducible runs, which the Python
     * generator gets from seeding `random` directly. */
    unsigned long long seed;
    const char* seed_env = getenv("TOPON_SEED");
    if (seed_env && *seed_env) {
        seed = strtoull(seed_env, NULL, 10);
        printf("INFO: Using TOPON_SEED=%llu (reproducible run).\n", seed);
    } else {
        uint64_t mix = (uint64_t)time(NULL);
        mix ^= (uint64_t)topon_getpid() << 32;
        mix ^= (uint64_t)clock() << 16;
        mix ^= (uint64_t)(uintptr_t)&mix;
        seed = (unsigned long long)splitmix64(&mix);
        printf("INFO: Seed %llu from the clock (set TOPON_SEED to replay this run).\n", seed);
    }
    rng_seed((uint64_t)seed);

    // --- MODIFIED: Create base_graph based on lattice_type input using Nx, Ny, Nz ---
    Graph* base_graph = NULL;
    int is_mix = 0;
    double nearest = 1.0;   /* nearest-neighbour distance of the pure lattice */
    /* "<dims> <TYPE>" for messages, the same tag the Python generator
     * prints (PythonTopologyGenerator._lattice_label). */
    char label[256];
    snprintf(label, sizeof(label), "%s %s", dims_str, lattice_type);
    if (strcmp(lattice_type, "SC") == 0) {
        base_graph = create_sc_lattice(Nx, Ny, Nz, p_dims);
    } else if (strcmp(lattice_type, "BCC") == 0) {
        base_graph = create_bcc_lattice(Nx, Ny, Nz, p_dims);
        nearest = sqrt(3.0) / 2.0;
    } else if (strcmp(lattice_type, "FCC") == 0) {
        base_graph = create_fcc_lattice(Nx, Ny, Nz, p_dims);
        nearest = sqrt(2.0) / 2.0;
    } else if (strcmp(lattice_type, "Diamond") == 0 || strcmp(lattice_type, "DIAMOND") == 0) {
        /* Both spellings: "Diamond" is what generator_python_diamond.py
         * uses, "DIAMOND" matches the shouting style of the other three. */
        base_graph = create_diamond_lattice(Nx, Ny, Nz, p_dims);
        if (!base_graph) return 1;
        nearest = sqrt(3.0) / 4.0;
    } else if (strncmp(lattice_type, "MIX", 3) == 0 &&
               (lattice_type[3] == '\0' || lattice_type[3] == ':')) {
        /* The trailing check matters: a bare strncmp would also swallow
         * "MIXED", "MIXTURE" and any other typo starting with MIX, and
         * silently build a pure-SC lattice instead of rejecting it. */
        /* "MIX:<sc>,<bcc>,<fcc>[,<cutoff>]" -- the fractions ride inside
         * argv[8] so the positional CLI stays the shape every existing
         * caller and SLURM script already writes. */
        is_mix = 1;
        double f_sc = 1.0, f_bcc = 0.0, f_fcc = 0.0, cutoff = 1.0;
        const char* spec = lattice_type + 3;
        int got = 0;
        if (*spec == ':') {
            got = sscanf(spec + 1, "%lf,%lf,%lf,%lf",
                         &f_sc, &f_bcc, &f_fcc, &cutoff);
            if (got < 3) {
                fprintf(stderr, "Error: MIX needs three fractions, e.g. MIX:0.2,0.4,0.4 "
                                "(optionally MIX:0.2,0.4,0.4,1.0 to set the cutoff). Got '%s'.\n",
                        lattice_type);
                return 1;
            }
        }
        if (cutoff_given) {
            /* The cutoff may ride inside the MIX argument or trail as the
             * ninth argument, but not both. */
            if (got == 4) {
                fprintf(stderr, "Error: MIX cutoff given twice, inside '%s' and as the ninth "
                                "argument '%s'. Give it once.\n", lattice_type, cutoff_arg);
                return 1;
            }
            cutoff = neighbour_cutoff;
        }
        double total = f_sc + f_bcc + f_fcc;
        if (f_sc < 0.0 || f_bcc < 0.0 || f_fcc < 0.0) {
            fprintf(stderr, "Error: MIX fractions must be non-negative (got %g,%g,%g).\n",
                    f_sc, f_bcc, f_fcc);
            return 1;
        }
        if (fabs(total - 1.0) > 1e-6) {
            fprintf(stderr, "Error: MIX fractions must sum to 1, got %g from %g,%g,%g. "
                            "They partition the crosslinker population, so a sum below 1 "
                            "would thin the lattice and above 1 would over-fill it.\n",
                    total, f_sc, f_bcc, f_fcc);
            return 1;
        }
        if (cutoff <= 0.0) {
            fprintf(stderr, "Error: MIX cutoff must be positive (got %g).\n", cutoff);
            return 1;
        }
        printf("INFO: Mixed lattice fractions SC=%g BCC=%g FCC=%g, cutoff=%g\n",
               f_sc, f_bcc, f_fcc, cutoff);
        snprintf(label, sizeof(label), "%s MIX (SC:%g,BCC:%g,FCC:%g)%s",
                 dims_str, f_sc, f_bcc, f_fcc, cutoff != 1.0 ? " cutoff " : "");
        if (cutoff != 1.0) {
            size_t used = strlen(label);
            snprintf(label + used, sizeof(label) - used, "%g", cutoff);
        }
        base_graph = create_mixed_lattice(Nx, Ny, Nz, p_dims, f_bcc, f_fcc, cutoff);
        if (!base_graph) return 1;
    } else {
        fprintf(stderr, "Error: Invalid lattice type '%s'. Must be SC, BCC, FCC, Diamond, or MIX:<sc>,<bcc>,<fcc>.\n", lattice_type);
        return 1;
    }

    /* A cutoff on a pure lattice: keep the sites and their numbering and
     * rebuild the edges by the minimum-image search. At the default the
     * canonical pattern built above is the answer, so an eight-argument
     * invocation builds exactly what it always did. Below the lattice's
     * nearest-neighbour distance no site would have a neighbour. */
    if (cutoff_given && !is_mix) {
        if (neighbour_cutoff < nearest - 1e-9) {
            fprintf(stderr, "Error: neighbour_cutoff %g is below the %s nearest-neighbour "
                            "distance of %.4g cell units, so no site would have a neighbour.\n",
                    neighbour_cutoff, lattice_type, nearest);
            freeGraph(base_graph);
            return 1;
        }
        if (neighbour_cutoff != 1.0) {
            Graph* sites = sites_only(base_graph);
            freeGraph(base_graph);
            base_graph = sites;
            double box[3] = {(double)Nx, (double)Ny, (double)Nz};
            connect_within_cutoff(base_graph, p_dims, box, neighbour_cutoff);
        }
        printf("INFO: Neighbour cutoff %g cell units on %s.\n", neighbour_cutoff, lattice_type);
        if (neighbour_cutoff != 1.0) {
            size_t used = strlen(label);
            snprintf(label + used, sizeof(label) - used, " cutoff %g", neighbour_cutoff);
        }
    }

    /* Size target_counts from the graph that was actually built, not from
     * a per-lattice constant. run_single_trial indexes this array by node
     * degree (target_counts[v_new_degree]), so a ceiling below the real
     * maximum is an out-of-bounds read. The pure lattices top out at a
     * known 6/8/12, but a MIX at the default cutoff already reaches 20,
     * and neighbour_cutoff is user-settable, so no constant is safe. The floor
     * of 12 preserves the array size the pure lattices always had. */
    int base_max_degree = 0;
    for (int i = 0; i < base_graph->V; ++i) {
        if (base_graph->degrees[i] > base_max_degree) base_max_degree = base_graph->degrees[i];
    }
    int max_possible_degree = max_func;
    if (base_max_degree > max_possible_degree) max_possible_degree = base_max_degree;
    if (max_possible_degree < 12) max_possible_degree = 12;

    int* target_counts = (int*)malloc((max_possible_degree + 1) * sizeof(int));
    int target_edge_count = -1; // --- NEW: For e:N ---

    char* str_copy = strdup(degree_dist_string);
    // --- MODIFIED: Pass target_edge_count pointer
    if (!parse_degree_distribution(str_copy, target_counts, max_possible_degree, &target_edge_count)) {
         fprintf(stderr, "Error parsing degree distribution string.\n");
         free(str_copy);
         free(target_counts);
         freeGraph(base_graph);
         return 1;
    }
    free(str_copy);

    long long node_sum = 0, explicit_degree_sum = 0; // --- MODIFIED: Renamed degree_sum
    int has_unconstrained = 0;
    for(int i=0; i <= max_func; ++i) {
        if(target_counts[i] >= 0) node_sum += target_counts[i];
        if(target_counts[i] == -1) has_unconstrained = 1;
        if(target_counts[i] > 0) explicit_degree_sum += (long long)i * target_counts[i];
    }

    /* The exact search pins the whole degree sequence, so it needs a count
     * for every degree up to max_func, and an e:N term, if given, has to
     * agree with them. Same rule and wording as
     * degree_matching.resolve_search. */
    if (search_exact) {
        char missing[512] = "";
        int n_missing = 0;
        for (int d = 0; d <= max_func; ++d) {
            if (target_counts[d] >= 0) continue;
            size_t used = strlen(missing);
            snprintf(missing + used, sizeof(missing) - used, "%s%d", n_missing ? ", " : "", d);
            n_missing++;
        }
        if (n_missing) {
            fprintf(stderr,
                    "Error: search 'exact' needs a count for every degree from 0 to "
                    "max_functionality (%d); degree_distribution leaves %s unspecified. "
                    "Give them explicitly (0:N for the vacancies, d:0 for a degree that "
                    "must not occur), or use search 'strict'.\n", max_func, missing);
            free(target_counts);
            freeGraph(base_graph);
            return 1;
        }
        if (target_edge_count >= 0) {
            long long implied = 0;
            for (int d = 1; d <= max_func; ++d) implied += (long long)d * target_counts[d];
            if (implied != 2LL * target_edge_count) {
                fprintf(stderr,
                        "Error: degree_distribution e:%d contradicts its own per-degree "
                        "counts, which need %lld edges (degree sum %lld). Drop the e: term; "
                        "the exact search derives the edge count from the degree counts.\n",
                        target_edge_count, implied / 2, implied);
                free(target_counts);
                freeGraph(base_graph);
                return 1;
            }
        }
    }

    /* Reject explicit targets no site can reach, in the order the Python
     * generator's _validate_targets_reachable checks them.
     *
     * The lattice bound first: no site can finish with more partners than
     * the scaffold gives it, whichever search runs.
     *
     * Then max_func. A node's final degree can never exceed max_func:
     * stage 3 prunes to it, and stage 4 refuses to finish while any
     * ACTIVE node sits above it. But the completion check only scans
     * i <= max_func, so a target like "7:5" with max_func=4 was parsed,
     * stored, then never looked at -- and the run printed "SUCCESS:
     * Target distribution met!" over a network containing no degree-7
     * nodes at all. Failing here makes the request's impossibility
     * visible instead of silently dropping part of it. */
    int lattice_max_degree = simple_max_degree(base_graph);
    for (int i = 0; i <= max_possible_degree; ++i) {
        if (target_counts[i] <= 0) continue;
        const char* why = NULL;
        char msg[512];
        if (search_exact && target_counts[i] > base_graph->V) {
            snprintf(msg, sizeof(msg),
                     "Error: degree_distribution %d:%d exceeds the %d nodes of a %s "
                     "lattice; a lattice cannot hold more nodes of degree %d than it has "
                     "nodes, so this target is unreachable.\n",
                     i, target_counts[i], base_graph->V, label, i);
            why = msg;
        } else if (i > lattice_max_degree) {
            snprintf(msg, sizeof(msg),
                     "Error: degree_distribution %d:%d requires degree-%d nodes, but the "
                     "maximum degree in a %s lattice is %d; no site has more candidate "
                     "partners than that, so this target is unreachable.\n",
                     i, target_counts[i], i, label, lattice_max_degree);
            why = msg;
        } else if (i > max_func) {
            snprintf(msg, sizeof(msg),
                     "Error: degree_distribution %d:%d is unreachable. Sculpting "
                     "enforces max_func=%d, so no node can finish with degree %d.\n",
                     i, target_counts[i], max_func, i);
            why = msg;
        }
        if (why) {
            fputs(why, stderr);
            free(target_counts);
            freeGraph(base_graph);
            return 1;
        }
    }

    if (target_edge_count != -1) { // --- NEW ---
        printf("INFO: Target total edge count 'e' is set to %d.\n", target_edge_count);
    }

    if (search_exact) {
        /* The exact search places every requested site itself, so an
         * over-full request is plain arithmetic, and an odd degree sum
         * belongs to no graph at all (every edge adds 2). Degree 0 is not
         * counted in: the vacancies are whatever the scaffold has left. */
        long long n_active = 0, degree_sum = 0;
        for (int d = 1; d <= max_func; ++d) {
            n_active += target_counts[d];
            degree_sum += (long long)d * target_counts[d];
        }
        if (n_active > (long long)base_graph->V) {
            fprintf(stderr,
                    "Error: degree_distribution places %lld active sites but a %s lattice "
                    "has only %d; enlarge lattice_size, or lower the per-degree counts. "
                    "(Degree 0 is the leftover sites, so it does not have to be counted "
                    "in.)\n", n_active, label, base_graph->V);
            free(target_counts);
            freeGraph(base_graph);
            return 1;
        }
        if (degree_sum % 2 != 0) {
            fprintf(stderr,
                    "Error: degree_distribution has an odd degree sum (%lld); every edge "
                    "contributes 2, so no graph has one. Move one site between two odd "
                    "degrees (e.g. one fewer at degree 1 and one more at degree 2), or "
                    "add one site of odd degree.\n", degree_sum);
            free(target_counts);
            freeGraph(base_graph);
            return 1;
        }
    }

    // --- NEW: Validation for e:N ---
    if (!search_exact && target_edge_count != -1) {
        long long target_degree_sum = (long long)target_edge_count * 2;
        if (explicit_degree_sum > target_degree_sum) {
            fprintf(stderr, "Error: The sum of degrees from explicit targets (%lld) is already greater than the target total degree sum (%lld from e:%d).\n",
                    explicit_degree_sum, target_degree_sum, target_edge_count);
            free(target_counts);
            freeGraph(base_graph);
            return 1;
        }
        if (has_unconstrained) {
            // 'd:*' is the old "unconstrained" syntax, which is ambiguous with e:N
            fprintf(stderr, "Warning: Using e:%d (target edge count) overrides any 'd:*' (unconstrained) targets.\n", target_edge_count);
        }
    }
    // --- END NEW VALIDATION ---
    
    if (!search_exact && node_sum > (long long)base_graph->V) {
        fprintf(stderr, "Error: Sum of specified node counts in distribution (%lld) exceeds total nodes in %s lattice (%d).\n", node_sum, lattice_type, base_graph->V);
        free(target_counts);
        freeGraph(base_graph);
        return 1;
    }
    
    // --- MODIFIED: Only run handshake check if e:N is NOT set
    if (!search_exact && target_edge_count == -1 && !has_unconstrained && (explicit_degree_sum % 2 != 0)) {
        fprintf(stderr, "Error (Handshake Lemma): Sum of specified degrees (%lld) is odd.\n", explicit_degree_sum);
        free(target_counts);
        freeGraph(base_graph);
        return 1;
    }
    
    /* The stream was seeded above the lattice build so MIX can draw its sites. */
    long long success_count = 0;

    printf("\n--- Initial State ---\n");
    print_distribution("Initial Lattice", base_graph, target_counts, -1, 0, max_func, extensive_logging);

    if (search_exact) {
        /* One trial is one exact attempt. Unlike the Python generator,
         * which gives each network DEFAULT_ATTEMPTS seeds, max_trials is
         * the bound here, so a large one keeps retrying until an attempt
         * lands or the run is killed. The one early stop is Python's: a
         * shortfall on a scaffold with no slack, where a fresh draw meets
         * the same forced assignment. */
        ExactSearch* w = exact_create(base_graph, max_func, target_counts, min_giant);
        int no_slack = exact_no_slack(w);
        long long n_active = 0;
        for (int d = 1; d <= max_func; ++d) n_active += target_counts[d];
        long long n_vacancies = (long long)base_graph->V - n_active;
        ExactOutcome best;
        int have_best = 0;
        long long attempts = 0;
        memset(&best, 0, sizeof(best));

        for (long long trial = 0; trial < max_trials; ++trial) {
            if (success_count >= max_saves) break;
            ExactOutcome o;
            exact_attempt(w, &o);
            attempts++;
            if (o.ok) {
                success_count++;
                printf("  [exact] attempt %lld: reached in %.2fs (%lld augmentations, %lld repairs)\n",
                       trial, o.seconds, o.augmentations, o.swaps);
                Graph* g = exact_to_graph(w, base_graph);
                print_distribution("Final Distribution", g, target_counts, trial, 0, max_func, extensive_logging);
                printf("[Trial %lld | SUCCESS] Target distribution met!\n", trial);
                save_graph_to_file(g, dims_str, trial);
                freeGraph(g);
                if (n_vacancies != target_counts[0]) {
                    printf("  [exact] note: %s left %lld sites empty where degree_distribution "
                           "asked for %d. Degree 0 is whatever the scaffold has over, so the "
                           "active counts are still exact; resize the cell if the site density "
                           "was meant to match.\n", label, n_vacancies, target_counts[0]);
                }
                continue;
            }
            char err[256];
            exact_describe_error(w, &o, err, sizeof(err));
            printf("  [exact] attempt %lld: failed in %.2fs (%lld augmentations, %lld repairs) -- %s\n",
                   trial, o.seconds, o.augmentations, o.swaps, err);
            printf("[Trial %lld | FAILED] Could not find a valid network.\n", trial);
            /* Keep the attempt Python's _failure_message would quote: the
             * smallest residual, then the largest giant component. */
            double g_new = o.giant_known ? o.giant_frac : 0.0;
            double g_best = best.giant_known ? best.giant_frac : 0.0;
            if (!have_best || o.residual < best.residual
                || (o.residual == best.residual && g_new > g_best)) {
                best = o;
                have_best = 1;
            }
            if (o.residual && no_slack) {
                printf("  [exact] stopping: the scaffold has no slack, so retrying cannot "
                       "find room that does not exist.\n");
                break;
            }
        }

        printf("\n\nSIMULATION FINISHED.\n");
        if (success_count >= max_saves) {
            printf("Termination condition met: Found %lld networks (target was %d).\n", success_count, max_saves);
        } else {
            printf("All trials completed: Found %lld networks (target was %d).\n", success_count, max_saves);
            if (success_count == 0) {
                exact_failure_message(stderr, w, label, attempts, have_best ? &best : NULL);
            } else {
                /* Some networks were written: not an error, but short. */
                char err[256] = "";
                if (have_best) exact_describe_error(w, &best, err, sizeof(err));
                fprintf(stderr, "Warning: the exact search found %lld of the %d networks asked "
                                "for in %lld attempt(s)%s%s; raise max_trials for the rest.\n",
                        success_count, max_saves, attempts,
                        have_best ? ", best failed attempt: " : "", err);
            }
        }
        exact_free(w);
        freeGraph(base_graph);
        free(target_counts);
        return success_count > 0 ? 0 : 1;
    }

    for (long long trial = 0; trial < max_trials; ++trial) {
        if (success_count >= max_saves) break;

        printf("\n--- Starting Trial %lld / %d (Found %lld so far) ---\n", trial, max_trials, success_count);

        // --- MODIFIED: Pass target_edge_count ---
        Graph* result_graph = run_single_trial(base_graph, max_func, target_counts, target_edge_count, trial, extensive_logging, dims_str, lattice_type);

        if (result_graph != NULL) {
            success_count++;
            printf("[Trial %lld | SUCCESS] Target distribution met!\n", trial);
            // MODIFIED: Pass dims_str (argv[1]) instead of N
            save_graph_to_file(result_graph, dims_str, trial);
            freeGraph(result_graph);
        } else {
            printf("[Trial %lld | FAILED] Could not find a valid network.\n", trial);
        }
    }
    
    printf("\n\nSIMULATION FINISHED.\n");
    if(success_count >= max_saves) {
        printf("Termination condition met: Found %lld networks (target was %d).\n", success_count, max_saves);
    } else {
        printf("All trials completed: Found %lld networks (target was %d).\n", success_count, max_saves);
    }
    
    freeGraph(base_graph);
    free(target_counts);
    return 0;
}