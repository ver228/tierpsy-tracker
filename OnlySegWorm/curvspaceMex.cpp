#include <mex.h>
#include <math.h>

double distance(double *x, double *y, int p_dim);
void interpintv(double *pt1, double *pt2, int p_dim, double intv, double *newpt);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    
    /*
% CURVSPACE Evenly spaced points along an existing curve in 2D or 3D.
%   CURVSPACE(P,N) generates N points that interpolates a curve
%   (represented by a set of points) with an equal spacing. Each
%   row of P defines a point, which means that P should be a n x 2
%   (2D)
%
%   (Example)
%   x = -2*pi:0.5:2*pi;
%   y = 10*sin(x);
%   z = linspace(0,10,length(x));
%   N = 50;
%   p = [x',y',z'];
%   q = curvspace(p,N);
%   figure;
%   plot3(p(:,1),p(:,2),p(:,3),'*b',q(:,1),q(:,2),q(:,3),'.r');
%   axis equal;
%   legend('Original Points','Interpolated Points');
%
%   See also LINSPACE.
%
 
%   25/03/15 AEJ adapted from Yo Fukushima
     */
    
//%% initial settings %%
    //input
    int p_size = int(mxGetM(prhs[0])); //extract number of points
    int p_dim = int(mxGetN(prhs[0]));//extract the dimension of points (can be more than 2D)
    
    double *p[p_dim]; //a pointers for each coordinate
    p[0] = (double *)mxGetData(prhs[0]); //get a pointer to the data
    for (int k = 1; k<p_dim; k++)
        p[k] = p[0] + k*p_size;
    
    double N = mxGetScalar(prhs[1]); //get the number of resampling points
    
    //output
    plhs[0] = mxCreateNumericMatrix(N, double(p_dim),mxDOUBLE_CLASS,mxREAL);
    double **q;
    q = new double* [p_dim];
    q[0] = mxGetPr(plhs[0]);
    for (int k = 1; k<p_dim; k++)
        q[k] = q[0] + k*int(N);
    
    //helper variables
    double totaldist, intv, remainder, distsum, disttmp;
    double dum, R;
    int p_ind_first, kk;
    
    double *ptnow, *newpt, *pttarget;
    ptnow = new double [p_dim];
    newpt = new double [p_dim];
    pttarget = new double [p_dim];
    
    //%% distance between points in p %%
    totaldist = 0;
    for(int k0 = 0; k0<p_size-1; k0++)
    {
        R =0 ;
        for(int k = 0; k<p_dim; k++)
        {
            dum = p[k][k0]-p[k][k0+1];
            R += dum*dum;
        }
        totaldist += sqrt(R);
    }
    //%% interval %%
    intv = totaldist/(N-1);
    q[0][0] = totaldist;
    
    for (int k = 0; k<p_dim; k++)
    {
        q[k][0] = p[k][0]; //% copy the first output point
        ptnow[k] = p[k][0]; //initialize the current point
        
    }
    
    //%% iteration %%
    p_ind_first = 1;
    for (int q_ind = 1; q_ind < N; q_ind++)
    {
        distsum = 0;
        remainder = intv; //% remainder of distance that should be accumulated
        kk = 0;
        
        while(true)
        {
            for(int k=0; k<p_dim; k++)
                pttarget[k] = p[k][p_ind_first+kk];
            
            
            //% calculate the distance from active point to the closest point in p
            disttmp = distance(ptnow, pttarget, p_dim);
            
            //add that distance to the total distance from the resampled point
            distsum += disttmp;
            
            //% if distance is enough, generate newpt else, accumulate distance
            if (distsum >= intv)
            {
                interpintv(ptnow, pttarget, p_dim, remainder, newpt);
                break;
            }
            else
            {
                remainder -= disttmp;
                kk++;
                for(int k=0; k<p_dim; k++)
                    ptnow[k] = pttarget[k];
                
                if (p_ind_first+kk == p_size)
                {
                    for(int k=0; k<p_dim; k++)
                        newpt[k] = p[k][p_size-1];
                    break;
                }
            }
        }
        
        p_ind_first += kk;
        //% add to the new resampled point
        for(int k=0; k<p_dim; k++)
        {
            q[k][q_ind] = newpt[k];
            ptnow[k] = newpt[k]; //% update current point
        }
        
   
    }
    
    delete [] ptnow;
    delete [] newpt;
    delete [] q;
    delete [] pttarget;
}

double distance(double *x, double *y, int p_dim)
{
//%% calculate distance %%
    double dum;
    double R = 0;
    for(int k = 0; k<p_dim; k++)
    {
        dum = x[k]-y[k];
        R+= dum*dum;
    }
    R = sqrt(R);
    return R;
}

void interpintv(double *pt1, double *pt2, int p_dim, double intv, double *newpt)
{
    double dum;
    double normR = 0;
    for(int k = 0; k<p_dim; k++)
    {
        dum = pt2[k]-pt1[k];
        normR += dum*dum;
    }
    normR = sqrt(normR);
    
    for(int k = 0; k<p_dim; k++)
    {
        newpt[k] = intv*(pt2[k]-pt1[k])/normR + pt1[k];
    }
            
}