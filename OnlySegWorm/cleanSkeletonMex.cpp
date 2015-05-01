#include <mex.h>
#include <cmath>
using namespace std;
inline void getDSkeleton(double skeleton[], int s1I, int s2I, int numberOfPoints, double dSkeleton[]) {
    for (int i = 0; i<2; i++)
        dSkeleton[i] = abs(skeleton[s1I + i*numberOfPoints] - skeleton[s2I + i*numberOfPoints]);
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /*
     * %CLEANSKELETON Clean an 8-connected skeleton by removing any overlap and
     * %interpolating any missing points.
     * %
     * %   [CSKELETON] = CLEANSKELETON(SKELETON)
     * %
     * %   Note: the worm's skeleton is still rough. Therefore, index lengths, as
     * %         opposed to chain-code lengths, are used as the distance metric
     * %         over the worm's skeleton.
     * %
     * %   Input:
     * %       skeleton    - the 8-connected skeleton to clean
     * %       widths      - the worm's contour widths at each skeleton point
     * %       wormSegSize - the size (in contour points) of a worm segment.
     * %                     Note: the worm is roughly divided into 24 segments
     * %                     of musculature (i.e., hinges that represent degrees
     * %                     of freedom) on each side. Therefore, 48 segments
     * %                     around a 2-D contour.
     * %                     Note 2: "In C. elegans the 95 rhomboid-shaped body
     * %                     wall muscle cells are arranged as staggered pairs in
     * %                     four longitudinal bundles located in four quadrants.
     * %                     Three of these bundles (DL, DR, VR) contain 24 cells
     * %                     each, whereas VL bundle contains 23 cells." -
     * %                     www.wormatlas.org
     * %
     * %   Output:
     * %       cSkeleton - the cleaned skeleton (no overlap & no missing points)
     * %       cWidths   - the cleaned contour widths at each skeleton point
     * %
     * %
     * % © Medical Research Council 2012
     * % You will not remove any copyright or other notices from the Software;
     * % you must reproduce all copyright notices and other proprietary
     * % notices on any copies of the Software.
     *
     * % If a worm touches itself, the cuticle prevents the worm from folding and
     * % touching adjacent pairs of muscle segments; therefore, the distance
     * % between touching segments must be, at least, the length of 2 muscle
     * % segments.
     */
    const double FLAG_MAX = 999999;
    
    double *skeleton = (double *)mxGetData(mxDuplicateArray(prhs[0]));
    mwSize numberOfPoints = mxGetM(prhs[0]);
    //mexPrintf("I = %i\n", numberOfPoints);
    
    double *widths = (double *)mxGetData(mxDuplicateArray(prhs[1]));
    double *wormSegSize = (double *)mxGetData(prhs[2]);
    int maxSkeletonOverlap = int(ceil(2 * wormSegSize[0]));
    
    double *iSort, *pSort;
    int *iSortC, *pSortC;
    mxArray * sort_outputs[2];
    mxArray * sort_input[1];
    sort_input[0] = mxDuplicateArray(prhs[0]); //copy skeleton
    //mxArray * sort_outputs2[2];
    
    
    mexCallMATLAB(2, sort_outputs, 1, sort_input, "sortrows");
    pSort = (double *)mxGetData(sort_outputs[1]); //% the sorted points
    
    sort_input[0] = mxDuplicateArray(sort_outputs[1]);
    mexCallMATLAB(2, sort_outputs, 1, sort_input, "sort");
    iSort = (double *)mxGetData(sort_outputs[1]); //% index -> sorted point index
    
    //mxDestroyArray(sort_input[0]);
    //mxDestroyArray(sort_outputs[0]);
    //mxDestroyArray(sort_outputs[1]);
    
    iSortC = new int [numberOfPoints];
    pSortC = new int [numberOfPoints];
    for(int i =0; i<numberOfPoints; i++)
    {
        iSortC[i] = int(iSort[i])-1;
        pSortC[i] = int(pSort[i])-1;
    }
            
// Remove small loops.
    int *keep;
    keep = new int [numberOfPoints];
    for(int i = 0; i<numberOfPoints; i++)
        keep[i] = i;
    
    int minI, maxI;
    int s1I = 0; // % the first index for the skeleton loop
    int s2I, pI;
    double dSkeleton[2];
    while (s1I < numberOfPoints-1)
    {
        //mexPrintf("%i|", s1I+1);
        //% Find small loops.
        // % Note: distal, looped sections are most likely touching;
        // % therefore, we don't remove these.
        if (keep[s1I] != FLAG_MAX)
        {
            minI = s1I; //% the minimum index for the loop
            maxI = s1I; //% the maximum index for the loop
            
            
            //% Search backwards.
            if (iSortC[s1I] > 0)
            {
                int pI = iSortC[s1I] - 1; //% the index for the sorted points
                int s2I = pSortC[pI]; // % the second index for the skeleton loop
                
                getDSkeleton(skeleton, s1I, s2I, numberOfPoints, dSkeleton);
                //mexPrintf("|%1.1f,%1.1f|", dSkeleton[0], dSkeleton[1]);
                while ((dSkeleton[0]<=1) || (dSkeleton[1]<=1))
                {
                    if ((s2I > s1I) && (keep[s2I]!=FLAG_MAX) && (dSkeleton[0]<=1) && (dSkeleton[1]<=1) && abs(s1I - s2I) < maxSkeletonOverlap)
                    {
                        minI = fmin(minI, s2I);
                        maxI = fmax(maxI, s2I);
                    }
                    
                    
                    // Advance the second index for the skeleton loop.
                    pI = pI - 1;
                    if(pI < 1)
                            break;
                    
                    s2I = pSortC[pI];
                    getDSkeleton(skeleton, s1I, s2I, numberOfPoints, dSkeleton);
                    //mexPrintf("-%i,%i-", pI+1, s2I+1);
                }
            }
            //mexPrintf("|%i,%i|", minI+1, maxI+1);
            //% Search forwards.
            if (iSortC[s1I]< numberOfPoints-1)
            {
                pI = iSortC[s1I] + 1; //% the index for the sorted points
                s2I = pSortC[pI]; //% the second index for the skeleton loop
                getDSkeleton(skeleton, s1I, s2I, numberOfPoints, dSkeleton);
                while ((dSkeleton[0]<=1) || (dSkeleton[1]<=1))
                {
                    if ((s2I > s1I) && (keep[s2I]!=FLAG_MAX) && (dSkeleton[0]<=1) && (dSkeleton[1]<=1) && abs(s1I - s2I) < maxSkeletonOverlap)
                    {
                        minI = fmin(minI, s2I);
                        maxI = fmax(maxI, s2I);
                    }
                    
                    // Advance the second index for the skeleton loop.
                    pI = pI + 1;
                    if (pI > numberOfPoints-1)
                        break;
                    
                    s2I = pSortC[pI];
                    getDSkeleton(skeleton, s1I, s2I, numberOfPoints, dSkeleton);
                }
            }
            
            //% Remove small loops.
            if (minI < maxI)
            {
                //% Remove the overlap.
                if ((skeleton[minI] == skeleton[maxI]) &&  (skeleton[minI + numberOfPoints] == skeleton[maxI + numberOfPoints]))
                {
                    for(int i = minI+1; i<=maxI; i++)
                    {
                        keep[i] = FLAG_MAX;
                        widths[minI] = fmin(widths[minI], widths[i]);
                    }
                }
                //% Remove the loop.
                else
                {
                    if(minI < maxI - 1)
                    {
                        //mexPrintf("{%i-%1.1f}", minI+1, widths[minI]);
                        for(int i = minI+1; i<=maxI- 1; i++)
                        {
                            keep[i] = FLAG_MAX;
                            widths[minI] = fmin(widths[minI], widths[i]);
                            widths[maxI] = fmin(widths[maxI], widths[i]);
                        }
                        //mexPrintf("-%1.1f-", widths[minI]);
                    }
                }
            }
            
            //% Advance the first index for the skeleton loop.
            s1I = s1I < maxI ? maxI : s1I + 1;
        }
        //% Advance the first index for the skeleton loop.
        else
            s1I = s1I + 1;
    }
    //mexPrintf("\n");
    
    int newTotal = 0;
    for (int i = 0; i<numberOfPoints; i++)
    {
        if (keep[i] != FLAG_MAX)
        {
            skeleton[newTotal] = skeleton[i];
            skeleton[newTotal+numberOfPoints] = skeleton[i+numberOfPoints];
            widths[newTotal] = widths[i];
            newTotal++;
        }
    }
    //correct for the data in the second dimension
    for (int i = 0; i<newTotal; i++)
        skeleton[i+newTotal] = skeleton[i+numberOfPoints];
    //% The head and tail have no width.
    widths[0] = 0;
    widths[newTotal-1] = 0;
    numberOfPoints = newTotal;
    
    //mexPrintf("II = %i\n", numberOfPoints);
    //for (int i=0; i<numberOfPoints; i++)
    //    mexPrintf("|%1.1f|", widths[i]);
    //mexPrintf("\n");
    
    delete[] keep;
    delete[] iSortC;
    delete[] pSortC;
    
//% Heal the skeleton by interpolating missing points.
    double *cSkeleton, *cWidths;
    int buffSize = 2*numberOfPoints;
    cSkeleton = new double [2*buffSize]; //% pre-allocate memory
    cWidths = new double [buffSize]; //% pre-allocate memory
            
    int j = 0;
    float x,y, x1,x2, y1, y2, delY, delX, delW;
    int points;
    for(int i = 0; i<numberOfPoints-1; i++)
    {
        //% Initialize the point differences.
        y = abs(skeleton[i + 1] - skeleton[i]);
        x = abs(skeleton[i + 1 + numberOfPoints] - skeleton[i + numberOfPoints]);
        //mexPrintf("|x:%i,y:%i|", x,y);
        //mexPrintf("|x:%1.1f,y:%1.1f|", x,y);
        //% Add the point.
        if ((y == 0 || y == 1) && (x == 0 || x == 1))
        {
            cSkeleton[j] = skeleton[i];
            cSkeleton[j + buffSize] = skeleton[i + numberOfPoints];
            
            cWidths[j] = widths[i];
            j++;
        }
        //% Interpolate the missing points.
        else
        {
            points = fmax(y, x);
            y1 = skeleton[i];
            y2 = skeleton[i + 1];
            delY = (y2-y1)/points;
            x1 = skeleton[i+ numberOfPoints];
            x2 = skeleton[i + 1+ numberOfPoints];
            delX = (x2-x1)/points;
            delW = (widths[i + 1] - widths[i])/points;
            for(int m = 0; m <= points; m++)
            {
                cSkeleton[j+m] = round(y1 + double(m)*delY);
                cSkeleton[j+m + buffSize] = round(x1 + double(m)*delX);
                cWidths[j+m] = round(widths[i] + double(m)*delW);
            }
            j += points;
        }
    }
    
//% Add the last point.
    if ((cSkeleton[0] != skeleton[numberOfPoints-1]) || (cSkeleton[buffSize] != skeleton[2*numberOfPoints-1]))
    {
        cSkeleton[j] = skeleton[numberOfPoints-1];
        cSkeleton[buffSize+j] = skeleton[2*numberOfPoints-1];
        cWidths[j] = widths[numberOfPoints-1];
        j++;
    }
    numberOfPoints = j;
    //mexPrintf("III = %i\n", numberOfPoints);
//% Collapse any extra memory.
//cSkeleton(j:end,:) = [];
//cWidths(j:end) = [];
    
//% Anti alias.
    keep = new int [numberOfPoints];
    for(int i = 0; i<numberOfPoints; i++)
        keep[i] = i;
    int nextI, i = 0;
    while (i < numberOfPoints - 2)
    {
        //% Smooth any stairs.
        nextI = i + 2;
        if ((abs(cSkeleton[i] - cSkeleton[nextI])<=1) && (abs(cSkeleton[i+buffSize] - cSkeleton[nextI+buffSize])<=1))
        {
            keep[i + 1] = FLAG_MAX;
            
            //% Advance.
            i = nextI;
            //mexPrintf("|%i|", i);
        }
        //% Advance.
        else
            i++;
    }
     
    newTotal = 0;
    for (int i = 0; i<numberOfPoints; i++)
    {
        if (keep[i] != FLAG_MAX)
        {
            cSkeleton[newTotal] = cSkeleton[i];
            cSkeleton[newTotal+buffSize] = cSkeleton[i+buffSize];
            cWidths[newTotal] = cWidths[i];
            newTotal++;
        }
        
    }
    //correct for the data in the second dimension
    for (int i = 0; i<newTotal; i++)
        cSkeleton[i+newTotal] = cSkeleton[i+buffSize];
    
    numberOfPoints = newTotal;
    //mexPrintf("IV = %i\n", numberOfPoints);
    delete[] keep;
    
//output
    plhs[0] = mxCreateNumericMatrix(numberOfPoints,2,mxDOUBLE_CLASS,mxREAL);
    plhs[1] = mxCreateNumericMatrix(numberOfPoints,1,mxDOUBLE_CLASS,mxREAL);
    
    double *dum; //dum pointer to save the output data
    
    dum = mxGetPr(plhs[0]);
    for (int i = 0; i<2*numberOfPoints; i++)
        dum[i] = cSkeleton[i];
    
    dum = mxGetPr(plhs[1]);
    for (int i = 0; i<numberOfPoints; i++)
        dum[i] = cWidths[i];
    
    delete[] cSkeleton;
    delete[] cWidths;
    
}

    