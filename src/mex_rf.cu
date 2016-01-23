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
void gpu_create_csr_matrix(csrMtx csrA, csrMtx *gpu_csrA);
void delete_csr_matrix(csrMtx csrA);
void gpu_delete_csr_matrix(csrMtx gpu_csrA);
void create_int_vector(const mxArray *mxA, intVct *vctA);
void gpu_create_int_vector(intVct vctA, intVct *gpu_vctA);
void delete_int_vector(intVct vctA);
void gpu_delete_int_vector(intVct gpu_vctA);

//------------------------------------------------------------------------------
// Library handles.
cusolverRfHandle_t cuRFh = NULL;

// GPU permutation vectors.
intVct gpu_vctP = {0, NULL}, gpu_vctQ = {0, NULL};

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
  gpu_create_int_vector(vctP, &gpu_vctP);
  gpu_create_int_vector(vctQ, &gpu_vctQ);

  check_rf_status(cusolverRfCreate(&cuRFh));

  check_rf_status(cusolverRfSetupHost(csrA.n,
                 csrA.nnz, csrA.row_idx, csrA.col_idx, csrA.val,
                 csrL.nnz, csrL.row_idx, csrL.col_idx, csrL.val,
                 csrU.nnz, csrU.row_idx, csrU.col_idx, csrL.val,
                 vctP.val, vctQ.val, cuRFh));
  check_cuda_status(cudaDeviceSynchronize());

  check_rf_status(cusolverRfAnalyze(cuRFh));
  check_cuda_status(cudaDeviceSynchronize());

  delete_csr_matrix(csrA);
  delete_csr_matrix(csrL);
  delete_csr_matrix(csrU);
  delete_int_vector(vctP);
  delete_int_vector(vctQ);
}

//------------------------------------------------------------------------------
EXPORTED_FUNCTION void mexRfRefactor(const mxArray *mxA)
{
  csrMtx csrA, gpu_csrA;

  create_csr_matrix(mxA, &csrA);
  gpu_create_csr_matrix(csrA, &gpu_csrA);

  check_rf_status(cusolverRfResetValues(csrA.n, csrA.nnz,
                  gpu_csrA.row_idx, gpu_csrA.col_idx, gpu_csrA.val,
                  gpu_vctP.val, gpu_vctQ.val, cuRFh));
  check_cuda_status(cudaDeviceSynchronize());

  check_rf_status(cusolverRfRefactor(cuRFh));
  check_cuda_status(cudaDeviceSynchronize());

  gpu_delete_csr_matrix(gpu_csrA);
  delete_csr_matrix(csrA);
}

//------------------------------------------------------------------------------
EXPORTED_FUNCTION void mexRfDestroy(void)
{
  check_rf_status(cusolverRfDestroy(cuRFh));
  gpu_delete_int_vector(gpu_vctP);
  gpu_delete_int_vector(gpu_vctQ);
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
  csrA->row_idx = new int[csrA->n + 1];
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
void gpu_create_csr_matrix(csrMtx csrA, csrMtx *gpu_csrA)
{
  gpu_csrA->n = csrA.n;
  gpu_csrA->nnz = csrA.nnz;
  cudaMalloc((void **) &(gpu_csrA->row_idx), sizeof(int) * (gpu_csrA->n + 1));
  cudaMemcpy(csrA.row_idx, gpu_csrA->row_idx, sizeof(int) * (gpu_csrA->n + 1),
             cudaMemcpyDeviceToHost);
  cudaMalloc((void **) &(gpu_csrA->col_idx), sizeof(int) * gpu_csrA->nnz);
  cudaMemcpy(csrA.col_idx, gpu_csrA->col_idx, sizeof(int) * gpu_csrA->nnz,
             cudaMemcpyDeviceToHost);
  cudaMalloc((void **) &(gpu_csrA->val), sizeof(double) * gpu_csrA->nnz);
  cudaMemcpy(csrA.val, gpu_csrA->val, sizeof(double) * gpu_csrA->nnz,
             cudaMemcpyDeviceToHost);
}

//------------------------------------------------------------------------------
void delete_csr_matrix(csrMtx csrA)
{
  delete csrA.row_idx;
  delete csrA.col_idx;
}

//------------------------------------------------------------------------------
void gpu_delete_csr_matrix(csrMtx gpu_csrA)
{
  cudaFree(gpu_csrA.row_idx);
  cudaFree(gpu_csrA.col_idx);
  cudaFree(gpu_csrA.val);
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
void gpu_create_int_vector(intVct vctA, intVct *gpu_vctA)
{
  gpu_vctA->n = vctA.n;
  cudaMalloc((void **) &(gpu_vctA->val), sizeof(int) * gpu_vctA->n);
  cudaMemcpy(vctA.val, gpu_vctA->val, sizeof(int) * gpu_vctA->n,
             cudaMemcpyDeviceToHost);
}

//------------------------------------------------------------------------------
void delete_int_vector(intVct vctA)
{
  delete vctA.val;
}

//------------------------------------------------------------------------------
void gpu_delete_int_vector(intVct gpu_vctA)
{
  cudaFree(gpu_vctA.val);
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
