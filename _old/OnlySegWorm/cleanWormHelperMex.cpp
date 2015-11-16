#include <mex.h>
#include <cmath>
using namespace std;
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    //input
    double *contour[2];
    contour[0] = (double *)mxGetData(mxDuplicateArray(prhs[0]));
    int numberOfPoints = int(mxGetM(prhs[0]));
    contour[1] = contour[0] + numberOfPoints;
//% Remove small overlapping segments and anti alias the contour.
//% Note: we don't remove loops. Removing loops may, for example, incorrectly
//% clean up a collapsed contour and/or remove a tail whip thereby leading to
//% false positives and/or false negatives, respectively.
    
    int lastIndex = numberOfPoints-1;
    bool *keep;
    keep = new bool [numberOfPoints];
    for(int k =0; k<numberOfPoints; k++)
        keep[k] = true;
    
//% Remove the first point.
//keep = 1:size(contour, 1); % points to keep
    
    if (contour[0][0] == contour[0][lastIndex] && contour[1][0] == contour[1][lastIndex])
    {
        keep[0] = false;
    }
    
    int i, nextI, next2I;
    double dContour[2];
//% Remove small overlapping segments and anti alias the contour.
    i = 0;
    while (i <= lastIndex)
    {
        //% Initialize the next 2 indices.
        if (i < lastIndex - 1)
        {
            nextI = i + 1;
            next2I = i + 2;
        }
        //% The second index wraps.
        else
        {
            if (i < lastIndex)
            {
                nextI = lastIndex;
                next2I = 0;
                
                //% Find the next kept point.
                while (!keep[next2I])
                    next2I++;
                
                //% The are no more kept points.
                if (i == next2I)
                    break;
            }
            //% Both indices wrap.
            else
            {
                //% Find the next kept point.
                nextI = 0;
                while (!keep[nextI])
                    nextI++;
                
                //% The are no more kept points.
                if(i == nextI)
                    break;
                
                
                //% Find the next kept point.
                next2I = nextI + 1;
                while (!keep[next2I])
                    next2I++;
                
                //% The are no more kept points.
                if (i == next2I)
                    break;
                
            }
        }
        //% Remove any overlap.
        dContour[0] = abs(contour[0][i] - contour[0][next2I]);
        dContour[1] = abs(contour[1][i] - contour[1][next2I]);
        if (dContour[0] == 0 && dContour[1] == 0)
        {
            keep[i] = false;
            keep[nextI] = false;
            
            //% Advance.
            i += 2;
        }
        //% Smooth any stairs.
        else
        {
            if (dContour[0] <= 1 && dContour[1] <= 1)
            {
                keep[nextI] = false;
                //% Advance.
                i += 2;
            }
            //% Advance.
            else
                i++;
            
        }
        
    }
    
    int tot_valid = 0;
    for(int k = 0; k<numberOfPoints; k++)
    {
        if(keep[k])
            tot_valid++;
        
    }
    
    //mexPrintf("D1= %i, %i\n", tot_valid, numberOfPoints);
    
    //output
    plhs[0] = mxCreateNumericMatrix(double(tot_valid),2,mxDOUBLE_CLASS,mxREAL);
    
    double *dumPtr; //dum pointer to save the output data
    dumPtr = mxGetPr(plhs[0]);
    
    int valid_ind = 0;
    for (int i = 0; i<numberOfPoints; i++)
    {
        
        if(keep[i])
        {
            //mexPrintf("%i\n", i+1);
            dumPtr[valid_ind] = contour[0][i];
            dumPtr[valid_ind+tot_valid] = contour[1][i];
            valid_ind++;
        }
    }
    
    delete []keep;
}
    
    
