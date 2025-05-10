#ifndef GRAPH_H
#define GRAPH_H

#include "mmio.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  uint32_t num_vertices;
  uint32_t num_edges;
  uint32_t *row_ptr;
  uint32_t *col_idx;
} GraphCSR;

GraphCSR *import_mtx(char *filename);
void print_graph_csr(GraphCSR *graph);

#endif