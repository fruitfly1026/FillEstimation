#pragma once

#include <stdlib.h>


typedef unsigned int IndexType;
typedef float ValueType;

typedef enum
{
  CSF_SORTED_SMALLFIRST, /** sort the modes in non-decreasing order */
  CSF_SORTED_BIGFIRST,   /** sort the modes in non-increasing order */
  CSF_MODE_CUSTOM        /** custom mode ordering. dim_perm must be set! */
} csf_mode_type;


#define MAX_NMODES 8

#ifndef max
    #define max(a,b) ((a) > (b) ? (a) : (b))
#endif

#ifndef min
    #define min(a,b) ((a) < (b) ? (a) : (b))
#endif

//lo is inclusive, hi is exclusive

typedef char hash_t;
hash_t *hash_create ();
void hash_set (hash_t *table, size_t key, size_t value);
size_t hash_get (hash_t *table, size_t key, size_t value);
void hash_clear (hash_t *table);
void hash_destroy (hash_t *table);
size_t random_range (size_t lo, size_t hi);
double random_uniform ();
void random_choose (size_t *samples, size_t n, size_t lo, size_t hi);
void sort (size_t *stuff, size_t n);
size_t search (const size_t *stuff, size_t lo, size_t hi, size_t key);
size_t search_strict (const size_t *stuff, size_t lo, size_t hi, size_t key);


void * aligned_malloc(size_t const bytes);