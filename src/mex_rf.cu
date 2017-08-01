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
} cscMtx;

typedef struct {
    int n;
    int *val;
} intVct;

typedef struct {
    int n;
    double *val;
} doubleVct;

void check_rf_status(cusolverStatus_t status);
void check_cuda_status(cudaError_t status);
void create_csc_matrix(const mxArray *mxA, cscMtx *cscA);
void gpu_create_csc_matrix(cscMtx cscA, cscMtx *gpu_cscA);
void delete_csc_matrix(cscMtx cscA);
void gpu_delete_csc_matrix(cscMtx gpu_cscA);
void create_idx_vector(const mxArray *mxA, intVct *vctA);
void gpu_create_idx_vector(intVct vctA, intVct *gpu_vctA);
void delete_int_vector(intVct vctA);
void gpu_delete_int_vector(intVct gpu_vctA);
void gpu_create_double_vector(const mxArray *mxA, doubleVct *vctA);
void gpu_delete_double_vector(doubleVct gpu_vctA);
void mex_create_double_vector(doubleVct gpu_vctA, mxArray **mxA);

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
    cscMtx cscA, cscL, cscU;
    intVct vctP, vctQ;

#ifdef _WIN32
    mxInitGPU();
#endif

    create_csc_matrix(mxA, &cscA);
    create_csc_matrix(mxL, &cscL);
    create_csc_matrix(mxU, &cscU);
    create_idx_vector(mxP, &vctP);
    create_idx_vector(mxQ, &vctQ);
    gpu_create_idx_vector(vctP, &gpu_vctP);
    gpu_create_idx_vector(vctQ, &gpu_vctQ);

    check_rf_status(cusolverRfCreate(&cuRFh));
    check_rf_status(cusolverRfSetMatrixFormat(cuRFh,
                    CUSOLVERRF_MATRIX_FORMAT_CSC,
                    CUSOLVERRF_UNIT_DIAGONAL_STORED_L));
    check_rf_status(cusolverRfSetResetValuesFastMode(cuRFh,
                    CUSOLVERRF_RESET_VALUES_FAST_MODE_ON));

    check_rf_status(cusolverRfSetupHost(cscA.n,
                    cscA.nnz, cscA.row_idx, cscA.col_idx, cscA.val,
                    cscL.nnz, cscL.row_idx, cscL.col_idx, cscL.val,
                    cscU.nnz, cscU.row_idx, cscU.col_idx, cscU.val,
                    vctP.val, vctQ.val, cuRFh));
    check_cuda_status(cudaDeviceSynchronize());

    check_rf_status(cusolverRfAnalyze(cuRFh));
    check_cuda_status(cudaDeviceSynchronize());

    delete_csc_matrix(cscA);
    delete_csc_matrix(cscL);
    delete_csc_matrix(cscU);
    delete_int_vector(vctP);
    delete_int_vector(vctQ);
}

//------------------------------------------------------------------------------
EXPORTED_FUNCTION void mexRfRefactor(const mxArray *mxA)
{
    cscMtx cscA, gpu_cscA;

    create_csc_matrix(mxA, &cscA);
    gpu_create_csc_matrix(cscA, &gpu_cscA);

    check_rf_status(cusolverRfResetValues(cscA.n, cscA.nnz,
                    gpu_cscA.row_idx, gpu_cscA.col_idx, gpu_cscA.val,
                    gpu_vctP.val, gpu_vctQ.val, cuRFh));
    check_cuda_status(cudaDeviceSynchronize());

    check_rf_status(cusolverRfRefactor(cuRFh));
    check_cuda_status(cudaDeviceSynchronize());

    gpu_delete_csc_matrix(gpu_cscA);
    delete_csc_matrix(cscA);
}

//------------------------------------------------------------------------------
EXPORTED_FUNCTION mxArray *mexRfSolve(const mxArray *mxA)
{
    doubleVct gpu_vctXF;
    double *gpu_temp;
    mxArray *mxX;

    gpu_create_double_vector(mxA, &gpu_vctXF);
    check_cuda_status(cudaMalloc((void **) &gpu_temp,
                      sizeof(double) * gpu_vctXF.n));

    check_rf_status(cusolverRfSolve(cuRFh, gpu_vctP.val, gpu_vctQ.val, 1,
                    gpu_temp, gpu_vctXF.n, gpu_vctXF.val, gpu_vctXF.n));
    check_cuda_status(cudaDeviceSynchronize());

    mex_create_double_vector(gpu_vctXF, &mxX);

    check_cuda_status(cudaFree(gpu_temp));
    gpu_delete_double_vector(gpu_vctXF);
    return mxX;
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
        break;
        case CUSOLVER_STATUS_INVALID_VALUE:
            mexErrMsgTxt("mexRF: Unsupported value or parameter was passed.\n");
        break;
        case CUSOLVER_STATUS_ALLOC_FAILED:
            mexErrMsgTxt("mexRF: Allocation of memory failed.\n");
        break;
        case CUSOLVER_STATUS_EXECUTION_FAILED:
            mexErrMsgTxt("mexRF: Kernel failed to launch on the GPU.\n");
        break;
        case CUSOLVER_STATUS_INTERNAL_ERROR:
            mexErrMsgTxt("mexRF: Internal operation failed.\n");
        break;
        case CUSOLVER_STATUS_ARCH_MISMATCH:
            mexErrMsgTxt("mexRF: Device compute capability mismatch.\n");
        break;
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            mexErrMsgTxt("mexRF: Matrix type not supported.\n");
        break;
    }
}

//------------------------------------------------------------------------------
void check_cuda_status(cudaError_t status)
{
    char msg[4096];

    if (status != cudaSuccess) {
        sprintf(msg, "mexRF: CUDA: %s\n",
                cudaGetErrorString(cudaGetLastError()));
        mexErrMsgTxt(msg);
    }
}

//------------------------------------------------------------------------------
void create_csc_matrix(const mxArray *mxA, cscMtx *cscA)
{
    mwIndex *jc;
    mwIndex *ir;

    cscA->n = mxGetM(mxA);
    if (!mxIsSparse(mxA) || mxGetN(mxA) != mxGetM(mxA) ||
        cscA->n == 0 || mxIsComplex(mxA) || !mxIsDouble(mxA)) {
        mexErrMsgTxt("mexRF: Matrix dimensions and type must agree.");
    }

    jc = mxGetJc(mxA);
    ir = mxGetIr(mxA);
    cscA->val = mxGetPr(mxA);
    cscA->nnz = (int) jc[cscA->n];

    // Cast array from 'mwIndex' to 'int'.
    cscA->row_idx = new int[cscA->n + 1];
    for (int j = 0; j <= cscA->n; j++) {
        cscA->row_idx[j] = (int) jc[j];
    }

    // Cast array from 'mwIndex' to 'int'.
    cscA->col_idx = new int[cscA->nnz];
    for (int j = 0; j < cscA->nnz; j++) {
        cscA->col_idx[j] = (int) ir[j];
    }
}

//------------------------------------------------------------------------------
void gpu_create_csc_matrix(cscMtx cscA, cscMtx *gpu_cscA)
{
    gpu_cscA->n = cscA.n;
    gpu_cscA->nnz = cscA.nnz;
    check_cuda_status(cudaMalloc((void **) &(gpu_cscA->row_idx),
                      sizeof(int) * (gpu_cscA->n + 1)));
    check_cuda_status(cudaMemcpy(gpu_cscA->row_idx, cscA.row_idx,
                      sizeof(int) * (gpu_cscA->n + 1),
                      cudaMemcpyHostToDevice));
    check_cuda_status(cudaMalloc((void **) &(gpu_cscA->col_idx),
                      sizeof(int) * gpu_cscA->nnz));
    check_cuda_status(cudaMemcpy(gpu_cscA->col_idx, cscA.col_idx,
                      sizeof(int) * gpu_cscA->nnz,
                      cudaMemcpyHostToDevice));
    check_cuda_status(cudaMalloc((void **) &(gpu_cscA->val),
                      sizeof(double) * gpu_cscA->nnz));
    check_cuda_status(cudaMemcpy(gpu_cscA->val, cscA.val,
                      sizeof(double) * gpu_cscA->nnz,
                      cudaMemcpyHostToDevice));
}

//------------------------------------------------------------------------------
void delete_csc_matrix(cscMtx cscA)
{
    delete cscA.row_idx;
    delete cscA.col_idx;
}

//------------------------------------------------------------------------------
void gpu_delete_csc_matrix(cscMtx gpu_cscA)
{
    check_cuda_status(cudaFree(gpu_cscA.row_idx));
    check_cuda_status(cudaFree(gpu_cscA.col_idx));
    check_cuda_status(cudaFree(gpu_cscA.val));
}

//------------------------------------------------------------------------------
void create_idx_vector(const mxArray *mxA, intVct *vctA)
{
    double *pr;

    vctA->n = mxGetM(mxA);

    if (mxIsSparse(mxA) || mxGetN(mxA) != 1 || vctA->n == 0 ||
        mxIsComplex(mxA) ||!mxIsDouble(mxA)) {
        mexErrMsgTxt("mexRF: Vector dimensions and type must agree.");
    }

    pr = mxGetPr(mxA);

    // Cast 'double' array to 'int'.
    vctA->val = new int[vctA->n];
    for (int j = 0; j < vctA->n; j++) {
        // Decrease by one for zero-based indexing.
        vctA->val[j] = (int) --pr[j];
    }

}

//------------------------------------------------------------------------------
void gpu_create_idx_vector(intVct vctA, intVct *gpu_vctA)
{
    gpu_vctA->n = vctA.n;
    check_cuda_status(cudaMalloc((void **) &(gpu_vctA->val),
                      sizeof(int) * gpu_vctA->n));
    check_cuda_status(cudaMemcpy(gpu_vctA->val, vctA.val,
                      sizeof(int) * gpu_vctA->n, cudaMemcpyHostToDevice));
}

//------------------------------------------------------------------------------
void delete_int_vector(intVct vctA)
{
    delete vctA.val;
}

//------------------------------------------------------------------------------
void gpu_delete_int_vector(intVct gpu_vctA)
{
    check_cuda_status(cudaFree(gpu_vctA.val));
}

//------------------------------------------------------------------------------
void gpu_create_double_vector(const mxArray *mxA, doubleVct *gpu_vctA)
{
    doubleVct vctA;

    vctA.n = mxGetM(mxA);
    vctA.val = mxGetPr(mxA);

    if (mxIsSparse(mxA) || mxGetN(mxA) != 1 || vctA.n == 0 ||
        mxIsComplex(mxA) ||!mxIsDouble(mxA)) {
      mexErrMsgTxt("mexRF: Vector dimensions and type must agree.");
    }

    gpu_vctA->n = vctA.n;
    check_cuda_status(cudaMalloc((void **) &(gpu_vctA->val),
                      sizeof(double) * gpu_vctA->n));
    check_cuda_status(cudaMemcpy(gpu_vctA->val, vctA.val,
                      sizeof(double) * gpu_vctA->n, cudaMemcpyHostToDevice));
}

//------------------------------------------------------------------------------
void gpu_delete_double_vector(doubleVct gpu_vctA)
{
    check_cuda_status(cudaFree(gpu_vctA.val));
}

//------------------------------------------------------------------------------
void mex_create_double_vector(doubleVct gpu_vctA, mxArray **mxA)
{
    double *val;

    *mxA = mxCreateNumericMatrix(gpu_vctA.n, 1, mxDOUBLE_CLASS, mxREAL);
    val = (double *) mxMalloc(sizeof(double) * gpu_vctA.n);
    check_cuda_status(cudaMemcpy(val, gpu_vctA.val,
                      sizeof(double) * gpu_vctA.n, cudaMemcpyDeviceToHost));

    mxSetPr(*mxA, val);
}

//------------------------------------------------------------------------------
void mexFunction(int nlhs, mxArray * plhs[],
                 int nrhs, const mxArray * prhs[])
{
    mexPrintf("mexRF: Use loadlibrary.\n");
    return;
}

#ifdef __cplusplus
}
#endif
