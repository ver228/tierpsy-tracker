#include <mex.h>
#include <math.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    
    /*
%CHAINCODELENGTH2INDEX Translate a length into an index. The index
%   represents the numerically-closest element to the desired length in
%   an ascending array of chain code lengths.
%
%   INDICES = CHAINCODELENGTH2INDEX(LENGTHS, CHAINCODELENGTHS)
%
%   Inputs:
%       lengths          - the lengths to translate into indices
%       chainCodeLengths - an ascending array of chain code lengths
%                          Note: the chain code lengths must increase at
%                          every successive index
%
%   Output:
%       indices - the indices for the elements closest to the desired
%                 lengths
%
% See also COMPUTECHAINCODELENGTHS, CIRCCOMPUTECHAINCODELENGTHS
%
%
% © Medical Research Council 2012
% You will not remove any copyright or other notices from the Software;
% you must reproduce all copyright notices and other proprietary
% notices on any copies of the Software.
     */
    
    /*
 
% Check the lengths.
% Note: circular chain-code lengths are minimally bounded at 0 and
% maximally bounded at the first + last lengths.
if any(lengths < 0)
    error('chainCodeLength2Index:TooShort', ...
        'The lengths cannot be negative');
end
 
if any(lengths > chainCodeLengths(1) + chainCodeLengths(end))
 
    error('chainCodeLength2Index:TooLong', ...
        ['The lengths cannot be greater than ', ...
        num2str(chainCodeLengths(end))]);
end
 
if any(isnan(chainCodeLengths)) || any(isnan(lengths))
     error('chainCodeLength2Index:NaN', ...
         'chainCodeLengths or lenghts contain NaN elements');
end
     */
    //input
    double *lengths = (double *)mxGetData(prhs[0]); 
    double *chainCodeLengths = (double *)mxGetData(prhs[1]);
    
    int numberOfChainCodes = mxGetN(prhs[1])*mxGetM(prhs[1]);
    int lastIndexChain = numberOfChainCodes-1;
    
    int numberOfPoints = mxGetN(prhs[0])*mxGetM(prhs[0]);
    
    //output
    plhs[0] = mxCreateNumericMatrix(mxGetM(prhs[0]), mxGetN(prhs[0]),mxDOUBLE_CLASS,mxREAL);
    double *indices = mxGetPr(plhs[0]);
    
    
//% Go through the lengths.
    
    double distJ, distNextJ;
    int j;
    for(int i =0;  i<numberOfPoints; i++)
    {
        //% Is the length too small?
        if(lengths[i] < chainCodeLengths[0])
        {
            //% Find the closest index.
            if (lengths[i] / chainCodeLengths[0] < 0.5)
                indices[i] = numberOfChainCodes;
            else
                indices[i] = 1;
        }
        //% Is the length too big?
        else
        {
            if (lengths[i] > chainCodeLengths[lastIndexChain])
            {
                //% Find the closest index.
                if ((lengths[i] - chainCodeLengths[lastIndexChain]) / chainCodeLengths[0] < 0.5)
                    indices[i] = numberOfChainCodes;
                else
                    indices[i] = 1;
            }
            //% Find the closest index.
            else
            {
                //% Try jumping to just before the requested length.
                //% Note: most chain-code lengths advance by at most sqrt(2) at each
                //% index. But I don't trust IEEE division so I use 1.5 instead.
                j = int(round(lengths[i] / 1.5));
                //% Did we jump past the requested length?
                if (j > lastIndexChain || lengths[i] < chainCodeLengths[j])
                        j = 0;
                
                //% find the closest index.
                distJ = fabs(lengths[i] - chainCodeLengths[j]); //important use fabs, abs will cast the value to integer
                //mexPrintf("D = %i, %i\n", i, j);
                //mexPrintf("%f, %f, %f\n", lengths[i], distJ, chainCodeLengths[j]);
                while (j < lastIndexChain)
                {
                    //% Is this index closer than the next one?
                    //% Note: overlapping points have equal distances. Therefore, if
                    //% the distances are equal, we advance.
                    distNextJ = fabs(lengths[i] - chainCodeLengths[j + 1]);
                    if (distJ < distNextJ)
                        break;
                    
                    //% Advance.
                    distJ = distNextJ;
                    j = j + 1;
                }
                
                //% Record the closest index.
                indices[i] = double(j+1); //shift one index due to MATLAB indexing
            }
        }
    }
     
}
