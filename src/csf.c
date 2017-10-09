
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "csf.h"
#include "sort.h"
#include "util.h"
#include "sparse_formats.h"



int csf_load(
    char const * const fname,
    IndexType * nmodes,
    csf_tensor ** tensors,
    double const * const options)
{
  sptensor_t * tt = tt_read(fname);
  if(tt == NULL) {
    return SPLATT_ERROR_BADINPUT;
  }

  tt_remove_empty(tt);

  *tensors = csf_alloc(tt, options);
  *nmodes = tt->nmodes;

  tt_free(tt);

  return SPLATT_SUCCESS;
}




void csf_free(
    splatt_csf * tensors,
    double const * const options)
{
  csf_free(tensors, options);
}





/**
* @brief Find a permutation of modes that results in non-increasing mode size.
*
* @param dims The tensor dimensions.
* @param nmodes The number of modes.
* @param perm_dims The resulting permutation.
*/
static void p_order_dims_small(
  idx_t const * const dims,
  idx_t const nmodes,
  idx_t * const perm_dims)
{
  idx_t sorted[MAX_NMODES];
  idx_t matched[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    sorted[m] = dims[m];
    matched[m] = 0;
  }
  quicksort(sorted, nmodes);

  /* silly n^2 comparison to grab modes from sorted dimensions.
   * TODO: make a key/val sort...*/
  for(idx_t mfind=0; mfind < nmodes; ++mfind) {
    for(idx_t mcheck=0; mcheck < nmodes; ++mcheck) {
      if(sorted[mfind] == dims[mcheck] && !matched[mcheck]) {
        perm_dims[mfind] = mcheck;
        matched[mcheck] = 1;
        break;
      }
    }
  }
}


/**
* @brief Find a permutation of modes that results in non-decreasing mode size.
*
* @param dims The tensor dimensions.
* @param nmodes The number of modes.
* @param perm_dims The resulting permutation.
*/
static void p_order_dims_large(
  idx_t const * const dims,
  idx_t const nmodes,
  idx_t * const perm_dims)
{
  idx_t sorted[MAX_NMODES];
  idx_t matched[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    sorted[m] = dims[m];
    matched[m] = 0;
  }
  /* sort small -> large */
  quicksort(sorted, nmodes);

  /* reverse list */
  for(idx_t m=0; m < nmodes/2; ++m) {
    idx_t tmp = sorted[nmodes-m-1];
    sorted[nmodes-m-1] = sorted[m];
    sorted[m] = tmp;
  }

  /* silly n^2 comparison to grab modes from sorted dimensions.
   * TODO: make a key/val sort...*/
  for(idx_t mfind=0; mfind < nmodes; ++mfind) {
    for(idx_t mcheck=0; mcheck < nmodes; ++mcheck) {
      if(sorted[mfind] == dims[mcheck] && !matched[mcheck]) {
        perm_dims[mfind] = mcheck;
        matched[mcheck] = 1;
        break;
      }
    }
  }

}


/**
* @brief Construct the sparsity structure of the outer-mode of a CSF tensor.
*
* @param ct The CSF tensor to construct.
* @param tt The coordinate tensor to construct from. Assumed to be already
*            sorted.
* @param tile_id The ID of the tile to construct.
* @param nnztile_ptr A pointer into 'tt' that marks the start of each tile.
*/
static void p_mk_outerptr(
  csf_tensor * const ct,
  coo_tensor const * const tt,
  IndexType const tile_id,
  IndexType const * const nnztile_ptr)
{
  IndexType const nnzstart = nnztile_ptr[tile_id];
  IndexType const nnzend   = nnztile_ptr[tile_id+1];
  IndexType const nnz = nnzend - nnzstart;

  assert(nnzstart < nnzend);

  /* grap top-level indices */
  IndexType const * const restrict ttind =
      nnzstart + tt->ind[csf_depth_to_mode(ct, 0)];

  /* count fibers */
  IndexType nfibs = 1;
  for(IndexType x=1; x < nnz; ++x) {
    assert(ttind[x-1] <= ttind[x]);
    if(ttind[x] != ttind[x-1]) {
      ++nfibs;
    }
  }
  ct->pt[tile_id].nfibs[0] = nfibs;
  assert(nfibs <= ct->dims[csf_depth_to_mode(ct, 0)]);

  /* grab sparsity pattern */
  csf_sparsity * const pt = ct->pt + tile_id;

  pt->fptr[0] = splatt_malloc((nfibs+1) * sizeof(**(pt->fptr)));
  if(ct->ntiles > 1) {
    pt->fids[0] = splatt_malloc(nfibs * sizeof(**(pt->fids)));
  } else {
    pt->fids[0] = NULL;
  }

  IndexType  * const restrict fp = pt->fptr[0];
  IndexType  * const restrict fi = pt->fids[0];
  fp[0] = 0;
  if(fi != NULL) {
    fi[0] = ttind[0];
  }

  IndexType nfound = 1;
  for(idx_t n=1; n < nnz; ++n) {
    /* check for end of outer index */
    if(ttind[n] != ttind[n-1]) {
      if(fi != NULL) {
        fi[nfound] = ttind[n];
      }
      fp[nfound++] = n;
    }
  }

  fp[nfibs] = nnz;
}








/**
* @brief Construct dim_iperm, which is the inverse of dim_perm.
*
* @param ct The CSF tensor.
*/
static void p_fill_dim_iperm(
    splatt_csf * const ct)
{
  for(idx_t level=0; level < ct->nmodes; ++level) {
    ct->dim_iperm[ct->dim_perm[level]] = level;
  }
}



/**
* @brief Construct the sparsity structure of any mode but the last. The first
*        (root) mode is handled by p_mk_outerptr and the first is simply a copy
*        of the nonzeros.
*
* @param ct The CSF tensor to construct.
* @param tt The coordinate tensor to construct from. Assumed to be already
*            sorted.
* @param tile_id The ID of the tile to construct.
* @param nnztile_ptr A pointer into 'tt' that marks the start of each tile.
* @param mode Which mode we are constructing.
*/
static void p_mk_fptr(
  csf_tensor * const ct,
  coo_tensor const * const tt,
  IndexType const tile_id,
  IndexType const * const nnztile_ptr,
  IndexType const mode)
{
  assert(mode < ct->nmodes);

  IndexType const nnzstart = nnztile_ptr[tile_id];
  IndexType const nnzend   = nnztile_ptr[tile_id+1];
  IndexType const nnz = nnzend - nnzstart;

  /* outer mode is easy; just look at outer indices */
  if(mode == 0) {
    p_mk_outerptr(ct, tt, tile_id, nnztile_ptr);
    return;
  }
  /* the mode after accounting for dim_perm */
  IndexType const * const restrict ttind =
      nnzstart + tt->ind[csf_depth_to_mode(ct, mode)];

  csf_sparsity * const pt = ct->pt + tile_id;

  /* we will edit this to point to the new fiber idxs instead of nnz */
  IndexType * const restrict fprev = pt->fptr[mode-1];

  /* first count nfibers */
  IndexType nfibs = 0;
  /* foreach 'slice' in the previous dimension */
  for(IndexType s=0; s < pt->nfibs[mode-1]; ++s) {
    ++nfibs; /* one by default per 'slice' */
    /* count fibers in current hyperplane*/
    for(IndexType f=fprev[s]+1; f < fprev[s+1]; ++f) {
      if(ttind[f] != ttind[f-1]) {
        ++nfibs;
      }
    }
  }
  pt->nfibs[mode] = nfibs;


  pt->fptr[mode] = splatt_malloc((nfibs+1) * sizeof(**(pt->fptr)));
  pt->fids[mode] = splatt_malloc(nfibs * sizeof(**(pt->fids)));
  IndexType * const restrict fp = pt->fptr[mode];
  IndexType * const restrict fi = pt->fids[mode];
  fp[0] = 0;

  /* now fill in fiber info */
  IndexType nfound = 0;
  for(IndexType s=0; s < pt->nfibs[mode-1]; ++s) {
    IndexType const start = fprev[s]+1;
    IndexType const end = fprev[s+1];

    /* mark start of subtree */
    fprev[s] = nfound;
    fi[nfound] = ttind[start-1];
    fp[nfound++] = start-1;

    /* mark fibers in current hyperplane */
    for(IndexType f=start; f < end; ++f) {
      if(ttind[f] != ttind[f-1]) {
        fi[nfound] = ttind[f];
        fp[nfound++] = f;
      }
    }
  }

  /* mark end of last hyperplane */
  fprev[pt->nfibs[mode-1]] = nfibs;
  fp[nfibs] = nnz;
}


void csf_find_mode_order(
  IndexType const * const dims,
  IndexType const nmodes,
  csf_mode_type which,
  IndexType const mode,
  IndexType * const perm_dims)
{
  switch(which) {
  case CSF_SORTED_SMALLFIRST:
    p_order_dims_small(dims, nmodes, perm_dims);
    break;

  case CSF_SORTED_BIGFIRST:
    p_order_dims_large(dims, nmodes, perm_dims);
    break;

  /* no-op, perm_dims better be set... */
  case CSF_MODE_CUSTOM:
    break;

  default:
    fprintf(stderr, "csf_mode_type '%d' not recognized.\n", which);
    break;
  }
}


/**
* @brief Allocate and fill a CSF tensor from a coordinate tensor without
*        tiling.
*
* @param ct The CSF tensor to fill out.
* @param tt The sparse tensor to start from.
*/
static void p_csf_alloc_untiled(
  csf_tensor * const ct,
  coo_tensor * const tt)
{
  IndexType const nmodes = tt->nmodes;
  tt_sort(tt, ct->dim_perm[0], ct->dim_perm);

  ct->pt = custom_malloc(sizeof(*(ct->pt)));

  csf_sparsity * const pt = ct->pt;

  /* last row of fptr is just nonzero inds */
  pt->nfibs[nmodes-1] = ct->nnz;
  pt->fids[nmodes-1] = custom_malloc(ct->nnz * sizeof(**(pt->fids)));
  pt->vals           = custom_malloc(ct->nnz * sizeof(*(pt->vals)));
  memcpy(pt->fids[nmodes-1], tt->ind[csf_depth_to_mode(ct, nmodes-1)],
      ct->nnz * sizeof(**(pt->fids)));
  memcpy(pt->vals, tt->vals, ct->nnz * sizeof(*(pt->vals)));

  /* setup a basic tile ptr for one tile */
  IndexType nnz_ptr[2];
  nnz_ptr[0] = 0;
  nnz_ptr[1] = tt->nnz;

  /* create fptr entries for the rest of the modes, working down from roots.
   * Skip the bottom level (nnz) */
  for(IndexType m=0; m < tt->nmodes-1; ++m) {
    p_mk_fptr(ct, tt, 0, nnz_ptr, m);
  }
}


/**
* @brief Allocate and fill a CSF tensor.
*
* @param ct The CSF tensor to fill.
* @param tt The coordinate tensor to work from.
* @param mode_type The allocation scheme for the CSF tensor.
* @param mode Which mode we are converting for (if applicable).
* @param splatt_opts Used to determine tiling scheme.
*/
static void p_mk_csf(
  csf_tensor * const ct,
  coo_tensor * const tt,
  csf_mode_type mode_type,
  IndexType const mode,
  double const * const splatt_opts)
{
  ct->nnz = tt->nnz;
  ct->nmodes = tt->nmodes;

  for(IndexType m=0; m < tt->nmodes; ++m) {
    ct->dims[m] = tt->dims[m];
  }

  /* get the indices in order */
  csf_find_mode_order(tt->dims, tt->nmodes, mode_type, mode, ct->dim_perm);

  p_csf_alloc_untiled(ct, tt);

}



void csf_free_mode(
    csf_tensor * const csf)
{
  free(csf->pt.vals);
  free(csf->pt.fids[csf->nmodes-1]);
  for(idx_t m=0; m < csf->nmodes-1; ++m) {
    free(csf->pt.fptr[m]);
    free(csf->pt.fids[m]);
  }
  free(csf->pt);
}



void csf_free(
  csf_tensor * const csf,
  double const * const opts)
{
  csf_free_mode(csf);
  free(csf);
}


csf_tensor * csf_alloc(
  coo_tensor * const tt,
  double const * const opts)
{
  csf_tensor * ret = NULL;

  double * tmp_opts = NULL;
  IndexType last_mode = 0;

  ret = custom_malloc(sizeof(*ret));
  p_mk_csf(ret, tt, CSF_SORTED_SMALLFIRST, 0, opts);

  return ret;
}





