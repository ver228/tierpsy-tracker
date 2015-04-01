#include <mex.h>
#include <math.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /*%CIRCCOMPUTECHAINCODELENGTHS Compute the chain-code length, at each point,
%   for a circularly-connected, continuous line of points.
%
%   LENGTHS = CIRCCOMPUTECHAINCODELENGTHS(POINTS)
%
%   Input:
%       points - the circularly-connected, continuous line of points on
%                which to measure the chain-code length
%
%   Output:
%       lengths - the chain-code length at each point
%
% See also CHAINCODELENGTH2INDEX, COMPUTECHAINCODELENGTHS
%
%
% © Medical Research Council 2012
% You will not remove any copyright or other notices from the Software;
% you must reproduce all copyright notices and other proprietary
% notices on any copies of the Software.
     */
    double *points[2];
    points[0] = (double *)mxGetData(mxDuplicateArray(prhs[0]));
    int numberOfPoints = int(mxGetM(prhs[0]));
    points[1] = points[0] + numberOfPoints;
    
    //output
    //% Pre-allocate memory.
    plhs[0] = mxCreateNumericMatrix(numberOfPoints,1,mxDOUBLE_CLASS,mxREAL);
    double *lengths = mxGetPr(plhs[0]);
    
    /*
//% Are the points 2 dimensional?
if ndims(points) ~=2 || mxGetM(prhs[0]) != 2 && mxGetN(prhs[0]) != 2)
    error('circComputeChainCodeLengths:PointsNot2D', ...
        'The matrix of points must be 2 dimensional');
end
 
% Orient the points as a N-by-2 matrix.
isTransposed = false;
if size(points, 2) ~= 2
    points = points';
    isTransposed = true;
end
     */
    
//% Pre-compute values.
    const double sqrt2 = sqrt(2);
    int lastIndex = numberOfPoints-1;
//% Measure the difference between subsequent points.
    double dPoints[2];
    
    dPoints[0] = abs(points[0][0] - points[0][lastIndex]);
    dPoints[1] = abs(points[1][0] - points[1][lastIndex]);
    
//% No change or we walked in a straight line.
    if ((dPoints[0]  == 0) || (dPoints[1]  == 0))
        lengths[0] = abs(dPoints[0]) + abs(dPoints[1]);
//% We walked one point diagonally.
    else
    {
        if ((dPoints[0]  == 1) && (dPoints[1]  == 1))
            lengths[0] = sqrt2;
        //% We walked fractionally or more than one point.
        else
            lengths[0] = sqrt(dPoints[0]*dPoints[0] + dPoints[1]*dPoints[1]);
    }
    
//% Measure the chain code length.
    for (int i = 1; i < numberOfPoints; i++)
    {
        //% Measure the difference between subsequent points.
        dPoints[0] = abs(points[0][i] - points[0][i-1]);
        dPoints[1] = abs(points[1][i] - points[1][i-1]);
        
        //% No change or we walked in a straight line.
        if ((dPoints[0]  == 0) || (dPoints[1]  == 0))
            lengths[i] = lengths[i-1] + abs(dPoints[0]) + abs(dPoints[1]);
        //% We walked one point diagonally.
        else
        {
            if ((dPoints[0]  == 1) && (dPoints[1]  == 1))
                lengths[i] = lengths[i-1] + sqrt2;
            //% We walked fractionally or more than one point.
            else
                lengths[0] = lengths[i-1] + sqrt(dPoints[0]*dPoints[0] + dPoints[1]*dPoints[1]);
        }
    }
}