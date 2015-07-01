#include <mex.h>
#include <cmath>
using namespace std;
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /*%COMPUTECHAINCODELENGTHS Compute the chain-code length, at each point, for
     * %   a continuous line of points.
     * %
     * %   LENGTHS = COMPUTECHAINCODELENGTHS(POINTS)
     * %
     * %   Input:
     * %       points - the continuous line of points on which to measure the
     * %                chain-code length
     * %
     * %   Output:
     * %       lengths - the chain-code length at each point
     * %
     * % See also CHAINCODELENGTH2INDEX, CIRCCOMPUTECHAINCODELENGTHS
     * %
     * %
     * % © Medical Research Council 2012
     * % You will not remove any copyright or other notices from the Software;
     * % you must reproduce all copyright notices and other proprietary
     * % notices on any copies of the Software.
     */
    double *points[2];
    points[0] = (double *)mxGetData(mxDuplicateArray(prhs[0]));
    int numberOfPoints = int(mxGetM(prhs[0]));
    points[1] = points[0] + numberOfPoints;
    
    //output
    //% Pre-allocate memory.
    plhs[0] = mxCreateNumericMatrix(numberOfPoints,1,mxDOUBLE_CLASS,mxREAL);
    double *lengths = mxGetPr(plhs[0]);
    
//% Pre-compute values.
    const double sqrt2 = sqrt(2);
    int lastIndex = numberOfPoints-1;
//% Measure the difference between subsequent points.
    double dPoints[2];
    
    dPoints[0] = 0;
    dPoints[1] = 0;
    
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


