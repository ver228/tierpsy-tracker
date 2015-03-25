#include <mex.h>
#include <math.h>
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
/*
%MINPEAKSCIRCDIST Find the minimum peaks in a circular vector. The peaks
%are separated by, at least, the given distance.
%
%   [PEAKS INDICES] = MINPEAKSCIRCDIST(X, DIST)
%
%   [PEAKS INDICES] = MINPEAKSCIRCDIST(X, DIST, CHAINCODELENGTHS)
%
%   Inputs:
%       x                - the vector of values
%       dist             - the minimum distance between peaks
%       chainCodeLengths - the chain-code length at each index;
%                          if empty, the array indices are used instead
%
%   Outputs:
%       peaks   - the maximum peaks
%       indices - the indices for the peaks
%
%   See also MAXPEAKSCIRCDIST, CIRCCOMPUTECHAINCODELENGTHS
%
%
% © Medical Research Council 2012
% You will not remove any copyright or other notices from the Software; 
% you must reproduce all copyright notices and other proprietary 
% notices on any copies of the Software.
*/

//input
double *x = (double *)mxGetData(prhs[0]);
double dist = mxGetScalar(prhs[1]);
int numberOfPoints = mxGetN(prhs[0])*mxGetM(prhs[0]);

//% Are there chain-code lengths?
double *chainCodeLengths;
int numberOfChainCodes;
if (nrhs==3)
{
    chainCodeLengths = (double *)mxGetData(prhs[2]);
    numberOfChainCodes = mxGetN(prhs[2])*mxGetM(prhs[2]);
}
else
    numberOfChainCodes = 0;

//% Use the array indices for length.
bool cleanChain = false;
if (numberOfChainCodes==0)
{
    cleanChain = true;
    chainCodeLengths = new double [numberOfPoints];
    numberOfChainCodes = numberOfPoints;
    for (int i = 1; i<=numberOfPoints; i++)
        chainCodeLengths[i-1] = double(i);
    
}


int lastIndexChain = numberOfChainCodes-1;


//% Is the vector larger than the search window?
double winSize = 2 * dist + 1;
if (chainCodeLengths[lastIndexChain] < winSize)
{
    mxArray * max_input[1] = {mxDuplicateArray(prhs[0])};
    mexCallMATLAB(2, plhs, 1, max_input, "min");
    return;
}


//% Initialize the peaks and indices.
int indexSize = int(ceil(numberOfPoints / winSize));
double *peaks;
int *indicesI;
peaks = new double [2*numberOfPoints]; //reserve enough memory even if indexSize is smaller in theory
indicesI = new int [2*numberOfPoints];



//% Search for peaks.
int im = -1; //% the last maxima index
int ie = -1; //% the end index for the last maxima's search window
int ip = 0; //% the current, potential, max peak index
double p = x[ip]; //% the current, potential, max peak value
int i = 1; //% the vector index
int j = 0; //% the recorded, maximal peaks index
int k;      
bool isMin;
while (i < numberOfPoints)
{
    //% Found a potential peak.
    if (x[i] < p)
    {
        ip = i;
        p = x[i];
    }
    
    //% Test the potential peak.
    if (chainCodeLengths[i] - chainCodeLengths[ip] >= dist || i == numberOfPoints-1)
    {
        //% Check the untested values next to the previous maxima.
        if (im >= 0 && chainCodeLengths[ip] - chainCodeLengths[im]<= 2 * dist)
        {
            //% Check the untested values next to the previous maxima. 
            isMin = true;
            k = ie;
            while (isMin && k >= 0 && chainCodeLengths[ip] - chainCodeLengths[k] < dist)
            {
                //% Is the previous peak larger?
                if (x[ip] >= x[k])
                    isMin = false;
                
                //% Advance.
                k--;
            }
            //% Record the peak.
            if (isMin)
            {
                indicesI[j] = ip;
                peaks[j] = p;
                j = j + 1;
            }
            
            //% Record the maxima.
            im = ip;
            ie = i;
            ip = i;
            p = x[ip];
        }  
        //% Record the peak.
        else
        {
            indicesI[j] = ip;
            peaks[j] = p;
            j = j + 1;
            im = ip;
            ie = i;
            ip = i;
            p = x[ip];
        }
    }
        
    //% Advance.
    i++;
}

//mexPrintf("%i, %i\n", indexSize,j);
indexSize = j;

//% Collapse any extra memory.
//indices(j:end) = [];
//peaks(j:end) = [];
int indexStart = 0;
int indexEnd = indexSize-1;


//% If we have two or more peaks, we have to check the start and end for mistakes.
if(indexSize > 2)
{
    //% If the peaks at the start and end are too close, keep the largest or
    //% the earliest one.
    if ((chainCodeLengths[indicesI[indexStart]] + chainCodeLengths[lastIndexChain] - chainCodeLengths[indicesI[indexEnd]]) < dist)
    {
        if (peaks[indexStart] >= peaks[indexEnd])
            indexStart++;
        else
            indexEnd--;
            
    }  
    //% Otherwise, check any peaks that are too close to the start and end.
    else
    {
        //% If we have a peak at the start, check the wrapping portion just
        //% before the end.
        k = numberOfPoints-1;
        
        while ((chainCodeLengths[indicesI[indexStart]] + chainCodeLengths[lastIndexChain] - chainCodeLengths[k]) < dist)
            //% Remove the peak.
        {
            if (peaks[0] >= x[k])
            {
                indexStart++;
                break;
            }
            //% Advance.
            k--;
        }
            
        //% If we have a peak at the end, check the wrapping portion just
        //% before the start.
        k = 0;
        while (chainCodeLengths[lastIndexChain] - chainCodeLengths[indicesI[indexEnd]] + chainCodeLengths[k] < dist)
        {
            //% Remove the peak.
            if (peaks[indexEnd] > x[k])
            {
                indexEnd--;
                break;
            }
            //% Advance.
            k++;
        }
        
    }
}

//output

indexSize = indexEnd-indexStart+1;
plhs[0] = mxCreateNumericMatrix(indexSize,1,mxDOUBLE_CLASS,mxREAL);
plhs[1] = mxCreateNumericMatrix(indexSize,1,mxDOUBLE_CLASS,mxREAL);

//mexPrintf("%i, %i, %i, %i, %i\n", indexStart, indexEnd, indexSize, mxGetM(plhs[0]), mxGetN(plhs[0]));
//mexPrintf("%1.1f\n", peaks[j-1]);

double *dum; //dum pointer to save the output data
dum = mxGetPr(plhs[0]);
for (int i = indexStart; i<=indexEnd; i++)
    dum[i-indexStart] = peaks[i];
    //mexPrintf("%1.1f\n", peaks[i]);
    
dum = mxGetPr(plhs[1]);
for (int i = indexStart; i<=indexEnd; i++)
    dum[i-indexStart] = double(indicesI[i]+1); //add one for MATLAB indexing 

delete[] peaks;
delete[] indicesI;

if (cleanChain)
    delete[] chainCodeLengths;
}
