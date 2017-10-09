#pragma once

#include "util.h"

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


/**
* @brief The sparsity pattern of a CSF (sub-)tensor.
*/
typedef struct csf_sparsity
{
  /** @brief The size of each fptr and fids array. */
  IndexType nfibs[MAX_NMODES];

  /** @brief The pointer structure for each sub-tree. fptr[f] marks the start
   *         of the children of node 'f'. This structure is a generalization of
   *         the 'rowptr' array used for CSR matrices. */
  IndexType * fptr[MAX_NMODES];

  /** @brief The index of each node. These map nodes back to the original
   *         tensor nonzeros. */
  IndexType * fids[MAX_NMODES];

  /** @brief The actual nonzero values. This array is of length
   *         nfibs[nmodes-1]. */
  ValueType * vals;
} csf_sparsity;



/**
* @brief CSF tensors are the compressed storage format for performing fast
*        tensor computations in the SPLATT library.
*/
typedef struct csf_tensor
{
  /** @brief The number of nonzeros. */
  IndexType nnz;

  /** @brief The number of modes. */
  IndexType nmodes;

  /** @brief The dimension of each mode. */
  IndexType dims[MAX_NMODES];

  /** @brief This maps levels in the tensor to actual tensor modes.
   *         dim_perm[0] is the mode stored at the root level and so on.
   *         NOTE: do not use directly; use`csf_depth_to_mode()` instead.
   */
  IndexType dim_perm[MAX_NMODES];

  /** @brief Sparsity structures -- one for each tile. */
  csf_sparsity * pt;
} csf_tensor;




/**** COO tensor functions ****/
coo_tensor * tt_read(char const * const ifname);
coo_tensor * tt_read_file(char const * const fname);
IndexType tt_remove_empty(coo_tensor * const tt);
coo_tensor * tt_alloc(IndexType const nnz, IndexType const nmodes);
void tt_free(coo_tensor * tt);


/**** CSF tensor functions ****/
int csf_load(
    char const * const fname,
    IndexType * nmodes,
    csf_tensor ** tensors,
    double const * const options);
void csf_free(
    csf_tensor * tensors,
    double const * const options);
