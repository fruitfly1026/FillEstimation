#ifndef SORT_H
#define SORT_H

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include <string.h>
#include <omp.h>
#include "util.h"
#include "sparse_formats.h"



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

/**
* @brief Sort a tensor using a permutation of its modes. Sorting uses dim_perm
*        to order modes by decreasing priority. If dim_perm = {1, 0, 2} then
*        nonzeros will be ordered by ind[1], with ties broken by ind[0], and
*        finally deferring to ind[2].
*
* @param tt The tensor to sort.
* @param mode The primary for sorting.
* @param dim_perm An permutation array that defines sorting priority. If NULL,
*                 a default ordering of {0, 1, ..., m} is used.
*/
void tt_sort(
  coo_tensor * const tt,
  IndexType const mode,
  IndexType * dim_perm);


/**
* @brief Sort a tensor using tt_sort on only a range of the nonzero elements.
*        Nonzeros in the range [start, end) will be sorted.
*
* @param tt The tensor to sort.
* @param mode The primary for sorting.
* @param dim_perm An permutation array that defines sorting priority. If NULL,
*                 a default ordering of {0, 1, ..., m} is used.
* @param start The first nonzero to include in the sorting.
* @param end The end of the nonzeros to sort (exclusive).
*/
void tt_sort_range(
  coo_tensor * const tt,
  IndexType const mode,
  IndexType * dim_perm,
  IndexType const start,
  IndexType const end);


/**
* @brief An in-place insertion sort implementation for IndexType's.
*
* @param[out] a The array to sort.
* @param n The number of items to sort.
*/
void insertion_sort(
  IndexType * const a,
  IndexType const n);


/**
* @brief An in-place quicksort implementation for IndexType's.
*
* @param[out] a The array to sort.
* @param n The number of items to sort.
*/
void quicksort(
  IndexType * const a,
  IndexType const n);


/**
* @brief An in-place insertion sort implementation for IndexType's that tracks the
*        resulting permutation of elements.
*
* @param[out] a The array to sort.
* @param[out] perm A permutation array.
* @param n The number of items to sort.
*/
void insertion_sort_perm(
  IndexType * const restrict a,
  IndexType * const restrict perm,
  IndexType const n);


/**
* @brief An in-place quicksort implementation for IndexType's that tracks the
*        resulting permutation of elements.
*
* @param[out] a The array to sort.
* @param[out] perm A permutation array.
* @param n The number of items to sort.
*/
void quicksort_perm(
  IndexType * const restrict a,
  IndexType * const restrict perm,
  IndexType const n);

#endif
