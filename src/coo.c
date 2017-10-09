#include "sparse_formats.h"
#include "util.h"
#include <math.h>

IndexType tt_remove_empty(
  coo_tensor * const tt)
{
  IndexType dim_sizes[MAX_NMODES];

  IndexType nremoved = 0;

  /* Allocate indmap */
  IndexType const nmodes = tt->nmodes;
  IndexType const nnz = tt->nnz;

  IndexType maxdim = 0;
  for(IndexType m=0; m < tt->nmodes; ++m) {
    maxdim = tt->dims[m] > maxdim ? tt->dims[m] : maxdim;
  }
  /* slice counts */
  IndexType * scounts = custom_malloc(maxdim * sizeof(*scounts));

  for(IndexType m=0; m < nmodes; ++m) {
    dim_sizes[m] = 0;
    memset(scounts, 0, maxdim * sizeof(*scounts));

    /* Fill in indmap */
    for(IndexType n=0; n < tt->nnz; ++n) {
      /* keep track of #unique slices */
      if(scounts[tt->ind[m][n]] == 0) {
        scounts[tt->ind[m][n]] = 1;
        ++dim_sizes[m];
      }
    }

    /* move on if no remapping is necessary */
    if(dim_sizes[m] == tt->dims[m]) {
      tt->indmap[m] = NULL;
      continue;
    }

    nremoved += tt->dims[m] - dim_sizes[m];

    /* Now scan to remove empty slices */
    IndexType ptr = 0;
    for(IndexType i=0; i < tt->dims[m]; ++i) {
      if(scounts[i] == 1) {
        scounts[i] = ptr++;
      }
    }

    tt->indmap[m] = custom_malloc(dim_sizes[m] * sizeof(**tt->indmap));

    /* relabel all indices in mode m */
    tt->dims[m] = dim_sizes[m];
    for(IndexType n=0; n < tt->nnz; ++n) {
      IndexType const global = tt->ind[m][n];
      IndexType const local = scounts[global];
      assert(local < dim_sizes[m]);
      tt->indmap[m][local] = global; /* store local -> global mapping */
      tt->ind[m][n] = local;
    }
  }

  free(scounts);
  return nremoved;
}


void tt_get_dims(
    FILE * fin,
    IndexType * const outnmodes,
    IndexType * const outnnz,
    IndexType * outdims,
    IndexType * offset)
{
  char * ptr = NULL;
  IndexType nnz = 0;
  char * line = NULL;
  ssize_t read;
  size_t len = 0;

  /* first count modes in tensor */
  IndexType nmodes = 0;
  while((read = getline(&line, &len, fin)) != -1) {
    if(read > 1 && line[0] != '#') {
      /* get nmodes from first nnz line */
      ptr = strtok(line, " \t");
      while(ptr != NULL) {
        ++nmodes;
        ptr = strtok(NULL, " \t");
      }
      break;
    }
  }
  --nmodes;

  for(IndexType m=0; m < nmodes; ++m) {
    outdims[m] = 0;
    offset[m] = 1;
  }

  /* fill in tensor dimensions */
  rewind(fin);
  while((read = getline(&line, &len, fin)) != -1) {
    /* skip empty and commented lines */
    if(read > 1 && line[0] != '#') {
      ptr = line;
      for(IndexType m=0; m < nmodes; ++m) {
        IndexType ind = strtoull(ptr, &ptr, 10);

        /* outdim is maximum */
        outdims[m] = (ind > outdims[m]) ? ind : outdims[m];

        /* offset is minimum */
        offset[m] = (ind < offset[m]) ? ind : offset[m];
      }
      /* skip over tensor val */
      strtod(ptr, &ptr);
      ++nnz;
    }
  }
  *outnnz = nnz;
  *outnmodes = nmodes;

  /* only support 0 or 1 indexing */
  for(IndexType m=0; m < nmodes; ++m) {
    if(offset[m] != 0 && offset[m] != 1) {
      fprintf(stderr, "SPLATT: ERROR tensors must be 0 or 1 indexed. "
                      "Mode %"SPLATT_PF_IDX" is %"SPLATT_PF_IDX" indexed.\n",
          m, offset[m]);
      exit(1);
    }
  }

  /* adjust dims when zero-indexing */
  for(IndexType m=0; m < nmodes; ++m) {
    if(offset[m] == 0) {
      ++outdims[m];
    }
  }

  rewind(fin);
  free(line);
}




coo_tensor * tt_read(FILE * fin)
{
  char * ptr = NULL;

  /* first count nnz in tensor */
  IndexType nnz = 0;
  IndexType nmodes = 0;

  IndexType dims[MAX_NMODES];
  IndexType offsets[MAX_NMODES];
  tt_get_dims(fin, &nmodes, &nnz, dims, offsets);

  if(nmodes > MAX_NMODES) {
    fprintf(stderr, "ERROR: maximum %"SPLATT_PF_IDX" modes supported. "
                    "Found %"SPLATT_PF_IDX". Please recompile with "
                    "MAX_NMODES=%"SPLATT_PF_IDX".\n",
            (IndexType) MAX_NMODES, nmodes, nmodes);
    return NULL;
  }

  /* allocate structures */
  coo_tensor * tt = tt_alloc(nnz, nmodes);
  memcpy(tt->dims, dims, nmodes * sizeof(*dims));

  char * line = NULL;
  int64_t read;
  size_t len = 0;

  /* fill in tensor data */
  rewind(fin);
  nnz = 0;
  while((read = getline(&line, &len, fin)) != -1) {
    /* skip empty and commented lines */
    if(read > 1 && line[0] != '#') {
      ptr = line;
      for(IndexType m=0; m < nmodes; ++m) {
        tt->ind[m][nnz] = strtoull(ptr, &ptr, 10) - offsets[m];
      }
      tt->vals[nnz++] = strtod(ptr, &ptr);
    }
  }

  free(line);

  return tt;
}



coo_tensor * tt_alloc(
  IndexType const nnz,
  IndexType const nmodes)
{
  coo_tensor * tt = (coo_tensor*) splatt_malloc(sizeof(*tt));

  tt->nnz = nnz;
  tt->vals = custom_malloc(nnz * sizeof(*tt->vals));
  
  tt->nmodes = nmodes;

  tt->dims = custom_malloc(nmodes * sizeof(*tt->dims));
  tt->ind  = custom_malloc(nmodes * sizeof(*tt->ind));
  for(IndexType m=0; m < nmodes; ++m) {
    tt->ind[m] = custom_malloc(nnz * sizeof(**tt->ind));
  }

  return tt;
}


void tt_free(
  coo_tensor * tt)
{
  tt->nnz = 0;
  for(IndexType m=0; m < tt->nmodes; ++m) {
    free(tt->ind[m]);
  }
  tt->nmodes = 0;
  free(tt->dims);
  free(tt->ind);
  free(tt->vals);
  free(tt);
}