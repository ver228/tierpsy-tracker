#include <mex.h>
#include <cmath>
using namespace std;
const double sqrt2 = sqrt(2);

inline double getSign(double x) {
    return double((0 < x) - (x < 0));
}

void calculateFracPix(int p1I, double de1, double dp1[2], double *points[2], double p1[2])
{
    double dy1, dx1, dum;
    
    if (dp1[0] == 0 || dp1[1] == 0)
    {
        p1[0] = de1*getSign(dp1[0]) + points[0][p1I];
        p1[1] = de1*getSign(dp1[1]) + points[1][p1I];
    }
    else
    {
        if(abs(dp1[0]) ==1 && abs(dp1[1]) ==1)
        {
            p1[0] = de1/sqrt2*dp1[0] + points[0][p1I];
            p1[1] = de1/sqrt2*dp1[1] + points[1][p1I];
        }
        else
        {
            dum = (dp1[1] / dp1[0]);
            dy1 = de1 / sqrt(1 +  dum*dum);
            dum = (dp1[0] / dp1[1]);
            dx1 = de1 / sqrt(1 + dum*dum);
            p1[0] = dy1*getSign(dp1[0]) + points[0][p1I];
            p1[1] = dx1*getSign(dp1[1]) + points[1][p1I];
        }
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /*
     * %CURVATURE Compute the curvature for a vector of points.
     * %
     * %   ANGLES = CURVATURE(POINTS, EDGELENGTH, CHAINCODELENGTHS)
     * %
     * %   Inputs:
     * %       points           - the vector of points ((x,y) pairs).
     * %       edgeLength       - the length of edges from the angle vertex.
     * %       chainCodeLengths - the chain-code length at each point;
     * %                          if empty, the array indices are used instead
     * %
     * %   Output:
     * %       angles - the angles of curvature per point (0 = none to +-180 =
     * %                maximum curvature). The sign represents whether the angle
     * %                points left or right. Vertices with insufficient edges are
     * %                labeled NaN.
     * %
     * % See also CIRCCURVATURE, COMPUTECHAINCODELENGTHS
     * %
     * %
     * % © Medical Research Council 2012
     * % You will not remove any copyright or other notices from the Software;
     * % you must reproduce all copyright notices and other proprietary
     * % notices on any copies of the Software.
     */
    
    
    //input
    double *points[2], *chainCodeLengths;
    double edgeLength;
    
    points[0] = (double *)mxGetData(prhs[0]);
    int numberOfPoints = mxGetM(prhs[0]);
    points[1] = points[0] + numberOfPoints;
    
    edgeLength = double(mxGetScalar(prhs[1]));
    
    chainCodeLengths = (double *)mxGetData(prhs[2]);
    
    //output
    plhs[0] = mxCreateNumericMatrix(numberOfPoints,1,mxDOUBLE_CLASS,mxREAL);
    double *angles = mxGetPr(plhs[0]);
    for (int k = 0; k<numberOfPoints; k++)
        angles[k] = mxGetNaN();
    
    if (numberOfPoints < 2 * edgeLength + 1 || mxGetM(prhs[2])!= numberOfPoints)
    {
        mexWarnMsgTxt("The length of the edges from the vertex exceeds the number of points");
        return;
    }
    
    double p1[2], p2[2];
    double de1, de2, dp1[2], dp2[2];
    
    int p1I, pvI, p2I;
//% Compute the curvature using the chain-code lengths.
    //% Initialize the first edge.
    p1I = 0;
    pvI = 0;
    
    //mexPrintf("/%1.1f\n", edgeLength);
    while (pvI < numberOfPoints-1 && chainCodeLengths[pvI] - chainCodeLengths[p1I] < edgeLength)
        pvI++;
    
    
    //% Compute the angles.
    
    p2I = pvI;
    while (p2I < numberOfPoints)
    {
        //% Find the second edge.
        while (p2I < numberOfPoints && chainCodeLengths[p2I] - chainCodeLengths[pvI] < edgeLength)
            p2I++;
        
        //mexPrintf("|%i, %i, %i|\n", pvI+1, p1I+1, p2I+1);
        
        //% Compute the angle.
        if(p2I < numberOfPoints)
        {
            /*% Compute fractional pixels for the first edge.
             * % Note: the first edge is equal to or just over the requested
             * % edge length. Therefore, the fractional pixels for the
             * % requested length lie on the line separating point 1 (index =
             * % p1I) from the next closest point to the vertex (index = p1I +
             * % 1). Now, we need to add the difference between the requested
             * % and real distance (de1) to point p1I, going in a line towards
             * % p1I + 1. Therefore, we need to solve the differences between
             * % the requested and real x & y (dx1 & dy1). Remember the
             * % requested x & y lie on the slope between point p1I and p1I +
             * % 1. Therefore, dy1 = m * dx1 where m is the slope. We want to
             * % solve de1 = sqrt(dx1^2 + dy1^2). Plugging in m, we get de1 =
             * % sqrt(dx1^2 + (m*dx1)^2). Then re-arrange the equality to
             * % solve:
             * %
             * % dx1 = de1/sqrt(1 + m^2) and dy1 = de1/sqrt(1 + (1/m)^2)
             * %
             * % But, Matlab uses (r,c) = (y,x), so x & y are reversed.
             */
            
            de1 = chainCodeLengths[pvI] - chainCodeLengths[p1I] - edgeLength;
            dp1[0] = points[0][p1I + 1] - points[0][p1I];
            dp1[1] = points[1][p1I + 1] - points[1][p1I];
            
            calculateFracPix(p1I, de1, dp1, points, p1);
            
            //% Compute fractional pixels for the second edge.
            de2 = chainCodeLengths[p2I] - chainCodeLengths[pvI] - edgeLength;
            dp2[0] = points[0][p2I - 1] - points[0][p2I];
            dp2[1] = points[1][p2I - 1] - points[1][p2I];
            
            calculateFracPix(p2I, de2, dp2, points, p2);
            //mexPrintf("%i, %1.1f, %1.1f, %1.1f, %1.1f, %1.1f\n", p2I, de2, dp2[0], dp2[1], p2[0] , p2[1]);
            //% Use the difference in tangents to measure the angle.
            double a1, a2;
            a2 = atan2(points[0][pvI] - p2[0], points[1][pvI] - p2[1]);
            a1 = atan2(p1[0] - points[0][pvI], p1[1] - points[1][pvI]);
            angles[pvI] = a2-a1;
            
            if (angles[pvI] > M_PI)
                angles[pvI] = angles[pvI] - 2 * M_PI;
            else
            {
                if (angles[pvI] < -M_PI)
                    angles[pvI] = angles[pvI] + 2 * M_PI;
            }
            angles[pvI] = angles[pvI] * 180 / M_PI;
            
            //% Advance.
            pvI++;
            
            //% Find the first edge.
            while(p1I < numberOfPoints-1 && chainCodeLengths[pvI] - chainCodeLengths[p1I + 1] > edgeLength)
                p1I++;
            
        }
    }
}

