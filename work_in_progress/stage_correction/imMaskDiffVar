#include <mex.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    uint8_T *image1 = (uint8_T *)mxGetData(prhs[0]);
    uint8_T *image2 = (uint8_T *)mxGetData(prhs[1]);
    size_t tot_pix = mxGetNumberOfElements(prhs[0]);
    
    plhs[0] = mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL);
    
    double* mean_diff = mxGetPr(plhs[0]);
    double tot_valid = 0;
    
    for (size_t i = 0; i<tot_pix; i++){
        if (image1[i] != 0 && image2[i] != 0)
        {
            mean_diff[0] += abs(double(image1[i]) - double(image2[i]));
            tot_valid++;
        }
    }
    mean_diff[0] /= tot_valid;
}