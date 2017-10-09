#pragma once

#include "util.h"
#include <assert.h>

////////////////////////////////////////////////////////////////////////////////
//! Defines the following sparse matrix formats
//
// CSR - Compressed Sparse Row
// COO - Coordinate
////////////////////////////////////////////////////////////////////////////////

typedef struct 
{
    IndexType num_rows, num_cols, num_nonzeros;
    double time, gflops;
    int tag;
} matrix_shape;


// COOrdinate matrix (aka IJV or Triplet format)
typedef struct
{
    IndexType num_rows, num_cols, num_nonzeros;
    double time, gflops;

    IndexType * I;  //row indices
    IndexType * J;  //column indices
    ValueType * V;  //nonzero values
} coo_matrix;

/*
 *  Compressed Sparse Row matrix (aka CRS)
 */
typedef struct 
{
    IndexType num_rows, num_cols, num_nonzeros;
    double time, gflops;

    IndexType * Ap;  //row pointer
    IndexType * Aj;  //column indices
    ValueType * Ax;  //nonzeros
} csr_matrix;


/**
* @brief The main data structure for representing sparse tensors in
*        coordinate format.
*/
typedef struct 
{
  IndexType nmodes;   /** The number of modes in the tensor, denoted 'm'. */
  IndexType nnz;      /** The number of nonzeros in the tensor. */
  IndexType dims[MAX_NMODES];   /** An array containing the dimension of each mode. */
  IndexType * ind[MAX_NMODES];   /** An m x nnz matrix containing the coordinates of each
                      nonzero. The nth nnz is accessed via ind[0][n], ind[1][n],
                      ..., ind[m][n]. */
  ValueType * vals;   /** An array containing the values of each nonzero. */
} coo_tensor;




