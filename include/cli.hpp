#ifndef COMMAND_LINE_H_
#define COMMAND_LINE_H_

#include <assert.h>
#include <ctype.h>
#include <getopt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define HELP_LEN 1000
char help_string[HELP_LEN];

#define HELP_PRINTED 11
#define MISSING_PARAMETER 12

typedef struct {
  char filename[100];
  int runs;
  uint32_t source;
  bool check;
} Cli_Args;

int str_empty(const char *str) {
  if (str == NULL || str[0] == '\0')
    return 1;
  else
    return 0;
}

void safe_strcat(char *str1, char *str2, size_t len){
  assert(strlen(str1) + strlen(str2) < len &&
         "String overflow");
  strcat(str1, str2);
  
}

void add_help_line(char opt, const char *opt_arg, const char *text,
                   const char *def) {
  const int buf_len = 100;
  char buf[buf_len];

  if (!str_empty(opt_arg))
    snprintf(buf, buf_len, "\n -%c <%s>: %-54s", opt, opt_arg, text);
  else
    snprintf(buf, buf_len, "\n -%c %s: %-54s", opt, "", text);
  
  safe_strcat(help_string, buf, HELP_LEN);
  if (!str_empty(def))
    snprintf(buf, buf_len, " [default: %s]", def);
  safe_strcat(help_string, buf, HELP_LEN);
}

int parse_integer(char *optarg, int* dest) {
  char *endptr;
  *dest = strtol(optarg, &endptr, 10);
  if (*endptr != '\0') {
    fprintf(stderr, "Invalid number for -n: %s\n", optarg);
    return -1;
  }
  return 0;
}

int parse_args(int argc, char **argv, Cli_Args *args) {
  int c_opt;
  int filename_flag = 0, runs_flag = 0, source_flag = 0;
  args->check = false;
  while ((c_opt = getopt(argc, argv, "f:n:s:ch")) != -1) {
    switch (c_opt) {
    case 'f':
      strncpy(args->filename, optarg, sizeof(args->filename) - 1);
      filename_flag = 1;
      break;
    case 'n':
      if (parse_integer(optarg, &args->runs) < 0) {
        fprintf(stderr, "Invalid parameter passed to -%c.\n", optopt);
        return MISSING_PARAMETER;
      }
      runs_flag = 1;
      break;
    case 's':
      if (parse_integer(optarg, &args->runs) < 0) {
        fprintf(stderr, "Invalid parameter passed to -%c.\n", optopt);
        return MISSING_PARAMETER;
      }
      source_flag = 1;
      break;
    case 'c':
      args->check = true;
      break;
    case '?':
      if (optopt == 'f' || optopt == 'n' || optopt == 's') {
        fprintf(stderr, "Option -%c requires an argument.\n", optopt);
        return MISSING_PARAMETER;
      } else if (isprint(optopt)) {
        fprintf(stderr, "Unknown option `-%c'.\n", optopt);
      } else {
        fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
      }
      break;
    case 'h':
      printf("Help: \n%s\n", help_string);
      return HELP_PRINTED;
      break;
    }
  }
  if (filename_flag == 0) {
    fprintf(stderr, "Filename not specified. Specify it with -f.\n");
    return MISSING_PARAMETER;
  }
  if (runs_flag == 0) {
    args->runs = 1;
  }
  if (source_flag == 0) {
    args->source = UINT32_MAX;
  }
  return 0;
}

void init_cli() {
  add_help_line('f', "file", "load graph from file", NULL);
  add_help_line('n', "runs", "number of runs", "1");
  add_help_line('s', "source", "ID of source vertex", "rand");
  add_help_line('c', "", "Checks BFS correctness", NULL);
  add_help_line('h', "", "print this help message", NULL);
}

#endif