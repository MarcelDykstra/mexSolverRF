#include <cuda.h>
#include <cusolverRf.h>
#include <mex.h>
#include <stdbool.h>
#include <stdio.h>
#include <gpu/mxGPUArray.h>
#define EXPORT_FCNS
#include "mex_export.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int n;
  int nnz;
  int *row_idx;
  int *col_idx;
  double *val;
} csrMtx;

typedef struct {
  int n;
  int *val;
} intVct;

void check_rf_status(cusolverStatus_t status);
void check_cuda_status(cudaError_t status);
void create_csr_matrix(const mxArray *mxA, csrMtx *csrA);
void delete_csr_matrix(csrMtx csrA);
void create_int_vector(const mxArray *mxA, intVct *vctA);
void delete_int_vector(intVct vctA);

//------------------------------------------------------------------------------
// Library handles.
cusolverRfHandle_t cuRFh = NULL;

//------------------------------------------------------------------------------
EXPORTED_FUNCTION void mexRfInitialize(const mxArray *mxA, const mxArray *mxL,
                                       const mxArray *mxU, const mxArray *mxP,
                                       const mxArray *mxQ)
{
  csrMtx csrA, csrL, csrU;
  intVct vctP, vctQ;

#ifdef _WIN32
  mxInitGPU();
#endif

  create_csr_matrix(mxA, &csrA);
  create_csr_matrix(mxL, &csrL);
  create_csr_matrix(mxU, &csrU);
  create_int_vector(mxP, &vctP);
  create_int_vector(mxQ, &vctQ);

  check_rf_status(cusolverRfCreate(&cuRFh));

  check_rf_status(cusolverRfSetupHost(csrA.n,
                 csrA.nnz, csrA.row_idx, csrA.col_idx, csrA.val,
                 csrL.nnz, csrL.row_idx, csrL.col_idx, csrL.val,
                 csrU.nnz, csrU.row_idx, csrU.col_idx, csrL.val,
                 vctP.val, vctQ.val, cuRFh));

  delete_csr_matrix(csrA);
  delete_csr_matrix(csrL);
  delete_csr_matrix(csrU);
  delete_int_vector(vctP);
  delete_int_vector(vctQ);
}


//------------------------------------------------------------------------------
void check_rf_status(cusolverStatus_t status)
{
  switch (status) {
    case CUSOLVER_STATUS_NOT_INITIALIZED:
      mexErrMsgTxt("mexRF: Library not initialised.\n");
    case CUSOLVER_STATUS_INVALID_VALUE:
      mexErrMsgTxt("mexRF: Unsupported value or parameter was passed.\n");
    case CUSOLVER_STATUS_ALLOC_FAILED:
      mexErrMsgTxt("mexRF: Allocation of memory failed.\n");
    case CUSOLVER_STATUS_EXECUTION_FAILED:
      mexErrMsgTxt("mexRF: Kernel failed to launch on the GPU.\n");
    case CUSOLVER_STATUS_INTERNAL_ERROR:
      mexErrMsgTxt("mexRF: Internal operation failed.\n");
    case CUSOLVER_STATUS_ARCH_MISMATCH:
      mexErrMsgTxt("mexRF: Device compute capability mismatch.\n");
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      mexErrMsgTxt("mexRF: Matrix type not supported.\n");
  }
}

//------------------------------------------------------------------------------
void check_cuda_status(cudaError_t status)
{
  char msg[4096];

  if (status != cudaSuccess) {
    sprintf(msg, "mexRF: CUDA: %s\n", cudaGetErrorString(cudaGetLastError()));
    mexErrMsgTxt(msg);
  }
}

//------------------------------------------------------------------------------
void create_csr_matrix(const mxArray *mxA, csrMtx *csrA)
{
  mwIndex *jc;
  mwIndex *ir;

  csrA->n = mxGetM(mxA);
  if (!mxIsSparse(mxA) || mxGetN(mxA) != mxGetM(mxA) ||
      csrA->n == 0 || mxIsComplex(mxA) || !mxIsDouble(mxA)) {
    mexErrMsgTxt("mexRF: Bad matrix.");
  }

  jc = mxGetJc(mxA);
  ir = mxGetIr(mxA);
  csrA->val = mxGetPr(mxA);
  csrA->nnz = jc[csrA->n];

  // Cast array from 'mwIndex' to 'int'.
  csrA->row_idx = new int[csrA->nnz];
  for (int j = 0; j <= csrA->n; j++) {
    csrA->row_idx[j] = (int) jc[j];
  }

  // Cast array from 'mwIndex' to 'int'.
  csrA->col_idx = new int[csrA->nnz];
  for (int j = 0; j < csrA->nnz; j++) {
    csrA->col_idx[j] = (int) ir[j];
  }
}

//------------------------------------------------------------------------------
void delete_csr_matrix(csrMtx csrA)
{
  delete csrA.row_idx;
  delete csrA.col_idx;
}

//------------------------------------------------------------------------------
void create_int_vector(const mxArray *mxA, intVct *vctA)
{
  double *pr;

  vctA->n = mxGetM(mxA);

  if (mxIsSparse(mxA) || mxGetN(mxA) != 1 || vctA->n == 0 ||
      mxIsComplex(mxA) ||!mxIsDouble(mxA)) {
    mexErrMsgTxt("mexRF: Bad vector.");
  }

  pr = mxGetPr(mxA);

  // Cast 'double array to 'int'.
  vctA->val = new int[vctA->n];
  for (int j = 0; j < vctA->n; j++) {
    vctA->val[j] = (int) pr[j];
  }

}

//------------------------------------------------------------------------------
void delete_int_vector(intVct vctA)
{
  delete vctA.val;
}

//------------------------------------------------------------------------------
void mexFunction(int nlhs, mxArray * plhs[],
                 int nrhs,const mxArray * prhs[])
{
  mexPrintf("mexRF: Use loadlibrary.\n");
  return;
}

#ifdef __cplusplus
}
#endif
