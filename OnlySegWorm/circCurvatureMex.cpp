#include <mex.h>
#include <math.h>

inline double getSign(double x) {
    return double((0 < x) - (x < 0));
}

void computeFractionalPixel(double *points, int numberOfPoints, double de1, int nextP1I, int p1I, double *p1)
{
    double dp1[2];
    const double SQRT2 = 1.414213562373095;
    for(int j = 0; j<2; j++){
        //mexPrintf("%f, %f\n", points[nextP1I + j*numberOfPoints], points[p1I + j*numberOfPoints]);
        dp1[j] = points[nextP1I + j*numberOfPoints] - points[p1I + j*numberOfPoints];
    }
    //mexPrintf("%f, %f\n", dp1[0], dp1[1]);
    
    if ((dp1[0] == 0) || (dp1[1] == 0))
    {
        for(int j = 0; j<2; j++)
            p1[j] = de1*getSign(dp1[j]) + points[p1I + j*numberOfPoints];
        //mexPrintf("1) %f, %f \n", p1[0], p1[1]);
        //mexPrintf("%f, %f, %f, %f\n", de1, points[p1I + 0*numberOfPoints], dp1[0], getSign(dp1[0]));//de1, , );
    }
    else
    {
        if ((abs(dp1[0]) == 1) && (abs(dp1[1]) == 1))
        {
            for(int j = 0; j<2; j++)
                p1[j] = points[p1I + j*numberOfPoints] + (dp1[j] * de1 / SQRT2);
            //mexPrintf("%f, %f, %f, %f\n", de1, points[p1I + 0*numberOfPoints], dp1[0], getSign(dp1[0]));//de1, , );
            //mexPrintf("2) %f, %f\n", p1[0], p1[1]);
        }
        else
        {
            double r;
            r = (dp1[1] / dp1[0]);
            double dy1 = de1 / sqrt(1 +  r*r);
            r = (dp1[0] / dp1[1]);
            double dx1 = de1 / sqrt(1 +  r*r);
            p1[0] = dy1 * getSign(dp1[0]) + points[p1I]; //points(p1I,0)
            p1[1] = dx1 * getSign(dp1[1]) + points[p1I + numberOfPoints]; //points(p1I,1)
            
            //mexPrintf("3) %f, %f\n", p1[0], p1[1]);
        }
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
//   Inputs:
//       points          - the vector of clockwise, circularly-connected
//                          points ((x,y) pairs).
//       edgeLength       - the length of edges from the angle vertex.
//       chainCodeLengths - the chain-code length at each point;
//                          if empty, the array indices are used instead
//   Output:
//       angles - the angles of curvature per point (0 = none to +-180 =
//                maximum curvature). The sign represents whether the angle
//                is convex (+) or concave (-).
    
    // input. note: (double *)mxGetData() and mxGetPr() are equivalent
    
    double *points = (double *)mxGetData(prhs[0]);
    mwSize numberOfPoints = mxGetM(prhs[0]);
    int lastArrayIndex = (numberOfPoints-1);
    double *ptrEdgeLength = (double *)mxGetData(prhs[1]);
    double edgeLength = ptrEdgeLength[0];
    
    
    //output
    plhs[0] = mxCreateNumericMatrix(numberOfPoints,1,mxDOUBLE_CLASS,mxREAL);
    double *angles = mxGetPr(plhs[0]);
    
    int p1I, p2I;
    double a1, a2;
    //% Compute the curvature using the array indices for length.
    if (true)//(mxGetM(prhs[2]) == 0 && mxGetN(prhs[2]) == 0)
    {
        //% Initialize the edges.
        edgeLength = round(edgeLength);
        p1I = int(numberOfPoints - edgeLength);
        p2I = int(edgeLength);
        
        for(int kk=0; kk<numberOfPoints; kk++, p1I++, p2I++)
        {
            if(p1I == numberOfPoints)
                p1I = 0;
            
            if(p2I == numberOfPoints)
                p2I = 0;
            
            //% Use the difference in tangents to measure the angle.
            a2 = atan2(points[kk] - points[p2I], points[kk+numberOfPoints] - points[p2I+numberOfPoints]);
            a1 = atan2(points[p1I] - points[kk], points[p1I+numberOfPoints] - points[kk+numberOfPoints]);
            angles[kk] = a2-a1;
            
            if(angles[kk] > M_PI)
                    angles[kk] = angles[kk]- 2 * M_PI;
            else
            {
                if(angles[kk] < -M_PI)
                        angles[kk] = angles[kk] + 2 * M_PI;
            }
            angles[kk] = angles[kk] * 180 / M_PI;
        }
        
    }
    else
    {
        //read chainCodeLenghts if it is not empty
        double *chainCodeLengths = (double *)mxGetData(prhs[2]);
        // Compute the curvature using the chain-code lengths.
        int p1I = lastArrayIndex;
        int pvI = 0;
        double pvLength = chainCodeLengths[p1I] + chainCodeLengths[pvI];
        double e1 = pvLength - chainCodeLengths[p1I];
        while ((p1I > 0) && (e1 < edgeLength))
        {
            p1I = p1I - 1;
            e1 = pvLength - chainCodeLengths[p1I];
        }
        
        //Compute the angles.
        int p2I = pvI;
        double e2;
        
        int nextP1I, prevP2I;
        double de1, de2, nextE1;
        double p1[2], p2[2];
        double a1,a2;
        
        int tot = 0;
        while (pvI < numberOfPoints)
        {
            //tot++;
            //if (tot > 2)
            //        return;
            // Compute the second edge length.
            if (p2I >= pvI)
                e2 = chainCodeLengths[p2I] - chainCodeLengths[pvI];
            // Compute the wrapped, second edge length.
            else
                e2 = chainCodeLengths[numberOfPoints-1] + chainCodeLengths[p2I] - chainCodeLengths[pvI];
            
            // Find the second edge.
            while (e2 < edgeLength)
            {
                p2I = p2I + 1;
                // Wrap.
                if (p2I > lastArrayIndex)
                    p2I = p2I - numberOfPoints;
                
                // Compute the second edge length.
                if (p2I >= pvI)
                    e2 = chainCodeLengths[p2I] - chainCodeLengths[pvI];
                // Compute the wrapped, second edge length.
                else
                    e2 = chainCodeLengths[numberOfPoints-1] + chainCodeLengths[p2I] - chainCodeLengths[pvI];
            }
            
            // *% Compute fractional pixels for the first edge.
            //% Note: the first edge is equal to or just over the requested edge
            //% length. Therefore, the fractional pixels for the requested length
            //% lie on the line separating point 1 (index = p1I) from the next
            //% closest point to the vertex (index = p1I + 1). Now, we need to
            //% add the difference between the requested and real distance (de1)
            //% to point p1I, going in a line towards p1I + 1. Therefore, we need
            //% to solve the differences between the requested and real x & y
            //% (dx1 & dy1). Remember the requested x & y lie on the slope
            //% between point p1I and p1I + 1. Therefore, dy1 = m * dx1 where m
            //% is the slope. We want to solve de1 = sqrt(dx1^2 + dy1^2).
            //% Plugging in m, we get de1 = sqrt(dx1^2 + (m*dx1)^2). Then
            //% re-arrange the equality to solve:
            //%
            //% dx1 = de1/sqrt(1 + m^2) and dy1 = de1/sqrt(1 + (1/m)^2)
            //%
            //% But, Matlab uses (r,c) = (y,x), so x & y are reversed.
            
            
            de1 = e1 - edgeLength;
            nextP1I =  (p1I < lastArrayIndex) ? p1I + 1 : p1I - lastArrayIndex;
            computeFractionalPixel(points, numberOfPoints, de1, nextP1I, p1I, p1);
            
            
            // Compute fractional pixels for the second edge (uses the previous pixel).
            de2 = e2 - edgeLength;
            prevP2I =  (p2I > 0) ? p2I - 1 : p2I + lastArrayIndex;
            computeFractionalPixel(points, numberOfPoints, de2, prevP2I, p2I, p2);
            
            // Use the difference in tangents to measure the angle.
            a2 = atan2(points[pvI] - p2[0], points[pvI + numberOfPoints] - p2[1]);
            a1 = atan2(p1[0] - points[pvI], p1[1] - points[pvI + numberOfPoints]);
            angles[pvI] = a2-a1;
            
            if (angles[pvI] > M_PI)
                angles[pvI] = angles[pvI] - 2 * M_PI;
            else
                if (angles[pvI] < -1*M_PI)
                    angles[pvI] = angles[pvI] + 2 * M_PI;
            
            angles[pvI] = angles[pvI] * 180 / M_PI;
            
            // Advance.
            pvI = pvI + 1;
            
            // Compute the first edge length.
            if (pvI <= lastArrayIndex)
            {
                if (p1I <= pvI)
                    e1 = chainCodeLengths[pvI] - chainCodeLengths[p1I];
                else  // Compute the wrapped, second edge length.
                    e1 = chainCodeLengths[lastArrayIndex] + chainCodeLengths[pvI] - chainCodeLengths[p1I];
                
                
                // Find the first edge.
                nextE1 = e1;
                nextP1I = p1I;
                
                while (nextE1 > edgeLength)
                {
                    // Advance.
                    e1 = nextE1;
                    p1I = nextP1I;
                    nextP1I = p1I + 1;
                    
                    // Wrap.
                    if (nextP1I > lastArrayIndex)
                        nextP1I = nextP1I - numberOfPoints;
                    
                    // Compute the first edge length.
                    if (nextP1I <= pvI)
                        nextE1 = chainCodeLengths[pvI] - chainCodeLengths[nextP1I];
                    else // Compute the wrapped, second edge length.
                        nextE1 = chainCodeLengths[lastArrayIndex] + chainCodeLengths[pvI] - chainCodeLengths[nextP1I];
                    
                }
            }
        }
    }
}