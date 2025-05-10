#include "graph.h"

int compare_entries(const void *a, const void *b) {
  Entry *ea = (Entry *)a;
  Entry *eb = (Entry *)b;
  if (ea->row != eb->row)
    return ea->row - eb->row;
  return ea->col - eb->col;
}

GraphCSR *import_mtx(char *filename) {
  FILE *f = fopen(filename, "r");

  if (!f) {
    printf("Could not open file [%s].\n", filename);
    return NULL;
  }
  MM_typecode matcode;
  if (mm_read_banner(f, &matcode) != 0) {
    printf("Could not process Matrix Market banner in file [%s].\n", filename);
    return NULL;
  }

  if (mm_is_complex(matcode)) {
    printf("Cannot parse complex-valued matrices.\n");
    return NULL;
  }
  if (mm_is_array(matcode)) {
    printf("Cannot parse array matrices.\n");
    return NULL;
  }

  uint32_t M, N, nz;
  if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0) {
    printf("Could not parse matrix size.\n");
    return NULL;
  }

  GraphCSR *graph = (GraphCSR *)malloc(sizeof(GraphCSR));

  graph->num_vertices = M;

  if (mm_is_general(matcode)) {
    graph->num_edges = nz;
  } else {
    graph->num_edges = nz * 2; // For symmetric matrices
  }

  printf("Matrix size: %u x %u, nnz: %u\n", M, N, graph->num_edges);

  Entry *entries = (Entry *)malloc(graph->num_edges * sizeof(Entry));
  if (mm_read_mtx_crd_data(f, nz, entries, matcode) != 0) {
    printf("Could not parse matrix data.\n");
    free(entries);
    return NULL;
  }
  fclose(f);

  if (!mm_is_general(matcode))
    // Duplicate the entries for symmetric matrices
    for (uint32_t i = 0; i < nz; i++) {
      entries[i + nz].row = entries[i].col;
      entries[i + nz].col = entries[i].row;
      entries[i + nz].val = entries[i].val;
    }

  qsort(entries, graph->num_edges, sizeof(Entry), compare_entries);

  graph->row_ptr = (uint32_t *)malloc((graph->num_vertices + 1) * sizeof(uint32_t));
  graph->col_idx = (uint32_t *)malloc(graph->num_edges * sizeof(uint32_t));

  uint32_t i = 0;
  for (uint32_t vertex = 0; vertex < graph->num_vertices; vertex++) {
    graph->row_ptr[vertex] = i;
    while (i < graph->num_edges && vertex == (entries[i].row - 1)) {
      // Matrix Market format is 1-indexed, convert to 0-indexed
      graph->col_idx[i] = entries[i].col - 1;
      i++;
    }
  }
  graph->row_ptr[graph->num_vertices] = graph->num_edges;

  free(entries);

  return graph;
}

void print_graph_csr(GraphCSR *graph) {
  printf("Graph CSR representation:\n");
  printf("Number of vertices: %u\n", graph->num_vertices);
  printf("Number of edges: %u\n", graph->num_edges);

  printf("Row ptr:\n");
  for (uint32_t i = 0; i <= graph->num_vertices; ++i) {
    printf("%u ", graph->row_ptr[i]);
  }
  printf("\n");

  printf("Col idx:\n");
  for (uint32_t i = 0; i < graph->num_edges; ++i) {
    printf("%u ", graph->col_idx[i]);
  }
  printf("\n");
}