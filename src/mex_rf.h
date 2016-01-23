#include "mex_export.h"
#define EXTERN_C EXPORTED_FUNCTION
#include <mex.h>

#ifdef __cplusplus
extern "C" {
#endif

EXPORTED_FUNCTION void mexRfInitialize(const mxArray *mxA, const mxArray *mxL,
                                       const mxArray *mxU, const mxArray *mxP,
                                       const mxArray *mxQ);

EXPORTED_FUNCTION void mexRfRefactor(const mxArray *mxA);

EXPORTED_FUNCTION mxArray *mexRfSolve(const mxArray *mxA);

EXPORTED_FUNCTION void mexRfDestroy(void);

EXPORTED_FUNCTION void mexFunction(int nlhs, mxArray *plhs[],
                                   int nrhs, const mxArray *prhs[]);

#ifdef __cplusplus
}
#endif
