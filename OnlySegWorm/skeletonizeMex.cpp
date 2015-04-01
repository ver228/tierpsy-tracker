#include <mex.h>
#include <math.h>

void getDistances(double **c1, double **c2, int j1, int j2, int nextJ1, int nextJ2, double *dnj1, double *dnj2, double &d12, double &d1, double &d2)
{
    double dum;
    
    d12 = 0;
    d1 = 0;
    d2 = 0;
    for (int i = 0; i<2; i++)
    {
        dnj1[i] = c1[i][nextJ1]-c1[i][j1];
        dnj2[i] = c1[i][nextJ2]-c1[i][j2];
        
        dum = c1[i][nextJ1]-c2[i][nextJ2];
        d12 += dum*dum;
        
        dum = c1[i][nextJ1]-c2[i][j2];
        d1 += dum*dum;
        
        dum = c1[i][j1]-c2[i][nextJ2];
        d2 += dum*dum;
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /*
%SKELETONIZE Skeletonize takes the 2 pairs of start and end points on a
%contour(s), then traces the skeleton between them using the specified
%increments.
%
%   [SKELETON] = SKELETONIZE(S1, E1, I1, S2, E2, I2, C1, C2)
%
%   Inputs:
%       s1       - The starting index for the first contour segment.
%       e1       - The ending index for the first contour segment.
%       i1       - The increment to walk along the first contour segment.
%                  Note: a negative increment means walk backwards.
%                  Contours are circular, hitting an edge wraps around.
%       s2       - The starting index for the second contour segment.
%       e2       - The ending index for the second contour segment.
%       i2       - The increment to walk along the second contour segment.
%                  Note: a negative increment means walk backwards.
%                  Contours are circular, hitting an edge wraps around.
%       c1       - The contour for the first segment.
%       c2       - The contour for the second segment.
%       isAcross - SET TO FALSE Does the skeleton cut across, connecting s1 with e2?
%                  Otherwise, the skeleton simply traces the midline
%                  between both contour segments.
%
%   Output:
%       skeleton - the skeleton traced between the 2 sets of contour points.
%       cWidths  - the widths between the 2 sets of contour points.
%                  Note: there are no widths when cutting across.
%
%
% © Medical Research Council 2012
% You will not remove any copyright or other notices from the Software;
% you must reproduce all copyright notices and other proprietary
% notices on any copies of the Software.
     */
    
//input
    int s1 = int(mxGetScalar(prhs[0]))-1; //decrease for C indexing
    int e1 = int(mxGetScalar(prhs[1]))-1;
    int i1 = int(mxGetScalar(prhs[2]));
    int s2 = int(mxGetScalar(prhs[3]))-1;
    int e2 = int(mxGetScalar(prhs[4]))-1;
    int i2 = int(mxGetScalar(prhs[5]));
    
    double *c1[2], *c2[2];
    c1[0] = (double *)mxGetData(mxDuplicateArray(prhs[6]));
    int c1Size = int(mxGetM(prhs[6]));
    c1[1] = c1[0] + c1Size;
    
    c2[0]  = (double *)mxGetData(mxDuplicateArray(prhs[7]));
    int c2Size = int(mxGetM(prhs[7]));
    c2[1] = c2[0] + c2Size;
    
    plhs[0] = mxCreateNumericMatrix(1,2,mxDOUBLE_CLASS,mxREAL);
    plhs[1] = mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL);
    
//% The first starting index is before the ending one.
    int we1, ws1, we2, ws2;
    double size1, size2;
    double dum;
    
    if (s1 <= e1)
        //% We are going forward.
        if (i1 > 0)
            size1 = (e1 - s1 + 1) / i1;
    //% We are wrapping backward.
        else
        {
            we1 = 0;
            ws1 = c1Size-1;
            size1 = (s1 + c1Size - e1 + 1) / (-1*i1);
        }
    //% The first starting index is after the ending one.
    else
    {
        //% We are going backward.
        if (i1 < 0)
            size1 = (s1 - e1 + 1) / (-1*i1);
        
        //% We are wrapping forward.
        else
        {
            we1 = c1Size-1;
            ws1 = 0;
            size1 = (c1Size - s1 + 1 + e1) / i1;
        }
    }
    
//% The second starting index is before the ending one.
    if (s2 <= e2)
    {
        //% We are going forward.
        if (i2 > 0)
            size2 = (e2 - s2 + 1) / i2;
        
        //% We are wrapping backward.
        else
        {
            we2 = 0;
            ws2 = c2Size-1;
            size2 = (s2 + c2Size - e2 + 1) / (-1*i2);
        }
    }
    //% The second starting index is after the ending one.
    else
    {
        //% We are going backward.
        if (i2 < 0)
            size2 = (s2 - e2 + 1) / (-1*i2);
        
        //% We are wrapping forward.
        else
        {
            we2 = c2Size-1;
            ws2 = 0;
            size2 = (c2Size - s2 + 1 + e2) / i2;
        }
    }
    
//% Trace the midline between the contour segments.
//% Note: the next few pages of code represent multiple, nearly identical
//% algorithms. The reason they are inlined as separate instances is to
//% mildly speed up one of the tightest loops in our program.
    
// % pre-allocate memory
    double *skeleton[2], *cWidths;
    int numberOfPoints = 2*int(floor(size1 + size2));
    skeleton[0] = new double [2*numberOfPoints];
    skeleton[1] = skeleton[0] + numberOfPoints;
    cWidths = new double [numberOfPoints];
    
    int j1 = s1;
    int j2 = s2;
    int nextJ1, nextJ2;
    double d1, d2, d12, prevWidth;
    double dnj1[2], dnj2[2];
    
    getDistances(c1, c2, j1, j2, nextJ1, nextJ2, dnj1, dnj2, d12, d1, d2);
    
//% Initialize the skeleton and contour widths.
    skeleton[0][0] = round((c1[0][j1] + c2[0][j2])/ 2);
    skeleton[1][0] = round((c1[1][j1] + c2[1][j2])/ 2);
    
    cWidths[0] = 0;
    for (int i = 0; i<2; i++)
    {
        dum = c1[i][j1]-c2[i][j2];
        cWidths[0] += dum*dum;
    }
    cWidths[0] = sqrt(cWidths[0]);
    
    if (j1 == we1) //% wrap
        j1 = ws1;
    
    if (j2 == we2) //% wrap
        j2 = ws2;
    
    int sLength = 1;
    //mexPrintf("}%i,%i,%i,%i\n", we1+1,ws1+1,we2+1, ws2+1);
//% Skeletonize the contour segments and measure the width.
    while ((j1 != e1) && (j2 != e2))
    {
        
        //% Compute the widths.
        nextJ1 = j1 + i1;
        if (nextJ1 == we1) //% wrap
            nextJ1 = ws1;
        
        nextJ2 = j2 + i2;
        if (nextJ2 == we2) //% wrap
            nextJ2 = ws2;
        //mexPrintf("|%i, %i, %i, %i, %i|", j1+1, j2+1, nextJ1+1, nextJ2+1, sLength+1);
        
        getDistances(c1, c2, j1, j2, nextJ1, nextJ2, dnj1, dnj2, d12, d1, d2);
        
        
        //% Advance along both contours.
        if ((d12 <= d1 && d12 <= d2) || d1 == d2)
        {
            j1 = nextJ1;
            j2 = nextJ2;
            cWidths[sLength] = sqrt(d12);
        }
        //% The contours go in similar directions.
        //% Follow the smallest width.
        else
        {
            if ((dnj1[0]*dnj2[0]> -1) && (dnj1[1]*dnj2[1]> -1))
            {
                //% Advance along the the first contour.
                if (d1 <= d2)
                {
                    j1 = nextJ1;
                    cWidths[sLength] = sqrt(d1);
                }
                //% Advance along the the second contour.
                else
                {
                    j2 = nextJ2;
                    cWidths[sLength] = sqrt(d2);
                }
            }
            //% The contours go in opposite directions.
            //% Follow decreasing widths or walk along both contours.
            //% In other words, catch up both contours, then walk along both.
            //% Note: this step negotiates hairpin turns and bulges.
            else
            {
                //% Advance along both contours.
                prevWidth = cWidths[sLength - 1];
                prevWidth = prevWidth*prevWidth;
                if ((d12 <= d1 && d12 <= d2) || d1 == d2 || (d1 > prevWidth && d2 > prevWidth ))
                {
                    j1 = nextJ1;
                    j2 = nextJ2;
                    cWidths[sLength] = sqrt(d12);
                }
            //% Advance along the the first contour.
                else
                {
                    if (d1 < d2)
                    {
                        j1 = nextJ1;
                        cWidths[sLength] = sqrt(d1);
                    }
                    //% Advance along the the second contour.
                    else
                    {
                        j2 = nextJ2;
                        cWidths[sLength] = sqrt(d2);
                    }
                }
            }
        }
        
        //% Compute the skeleton.
        for(int i=0; i<2; i++)
            skeleton[i][sLength] = round((c1[i][j1] + c2[i][j2]) / 2);
        
        sLength ++;
    }
            
    //% Add the last point.
    if (j1 != e1 || j2 != e2)
    {
        //mexPrintf("-|%i, %i|", j1+1, sLength+1);
        cWidths[sLength] = 0;
        
        for(int i=0; i<2; i++)
        {
            skeleton[i][sLength] = round((c1[i][e1] + c2[i][e2]) / 2);
            dum = (c1[i][e1] - c2[i][e2]);
            cWidths[sLength] += dum*dum;
        }
        cWidths[sLength] = sqrt(cWidths[sLength]);
        sLength++;
    }
    
    plhs[0] = mxCreateNumericMatrix(sLength,2,mxDOUBLE_CLASS,mxREAL);
    plhs[1] = mxCreateNumericMatrix(sLength,1,mxDOUBLE_CLASS,mxREAL);
    
    double *dumPtr; //dum pointer to save the output data
    
    dumPtr = mxGetPr(plhs[0]);
    for (int i = 0; i<sLength; i++)
    {
        dumPtr[i] = skeleton[0][i];
        dumPtr[i+sLength] = skeleton[1][i];
    }
    
    dumPtr = mxGetPr(plhs[1]);
    for (int i = 0; i<sLength; i++)
        dumPtr[i] = cWidths[i];
          
    delete [] cWidths;
    delete [] skeleton[0];
    //mexPrintf("|%i, %i|\n", j1+1, sLength+1);
     
}