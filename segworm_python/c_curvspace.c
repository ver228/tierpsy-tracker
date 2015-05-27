#include <math.h>
#include <stdio.h>
double distance(double *x, double *y, int p_dim);
void interpintv(double *pt1, double *pt2, int p_dim, double intv, double *newpt);

inline int ind(m, d, ndim) {
    return m*ndim + d; 
}
double c_curvspace(double *points, int p_size, int p_dim, int N, double *output)
{
    
    /*
% CURVSPACE Evenly spaced points along an existing curve in 2D or 3D.
%   CURVSPACE(points,N) generates N points that interpolates a curve
%   (represented by a set of points) with an equal spacing. Each
%   row of P defines a point, which means that P should be a n x 2
%   (2D)
%
%   (Example)
%   x = -2*pi:0.5:2*pi;
%   y = 10*sin(x);
%   z = linspace(0,10,length(x));
%   N = 50;
%   points = [x',y',z'];
%   q = curvspace(points,N);
%   figure;
%   plot3(points(:,1),points(:,2),points(:,3),'*b',q(:,1),q(:,2),q(:,3),'.r');
%   axis equal;
%   legend('Original Points','Interpolated Points');
%
%   See also LINSPACE.
%
 
%   25/03/15 AEJ adapted from Yo Fukushima
     */
    
//%% initial settings %%
    //input   
    //helper variables
    double totaldist, intv, remainder, distsum, disttmp;
    double dum, R;
    int p_ind_first, kk;
    
    double ptnow[p_dim], newpt[p_dim], pttarget[p_dim];
    
    
    //%% distance between points in p %%
    totaldist = 0;
    for(int k0 = 0; k0<p_size-1; k0++)
    {
        R = 0;
        for(int k = 0; k<p_dim; k++)
        {
            dum = points[ind(k0,k, p_dim)]-points[ind(k0+1,k, p_dim)];
            R += dum*dum;
        }
        
        totaldist += sqrt(R);
    }
    //%% interval %%
    intv = totaldist/(N-1);
    
    
    for (int k = 0; k<p_dim; k++)
    {
        output[k] = points[ind(0,k, p_dim)]; //% copy the first output point
        ptnow[k] = points[ind(0,k, p_dim)]; //initialize the current point
        
    }
    
    //%% iteration %%
    p_ind_first = 1;
    for (int q_ind = 1; q_ind < N; q_ind++)
    {
        distsum = 0;
        remainder = intv; //% remainder of distance that should be accumulated
        kk = 0;
        
        while(1)
        {
            for(int k=0; k<p_dim; k++)
                pttarget[k] = points[ind(p_ind_first+kk,k, p_dim)];
            
            
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
                        newpt[k] = points[ind(p_size-1,k, p_dim)];
                    break;
                }
            }
        }
        
        p_ind_first += kk;
        //% add to the new resampled point
        for(int k=0; k<p_dim; k++)
        {
            output[ind(q_ind,k, p_dim)] = newpt[k];
            ptnow[k] = newpt[k]; //% update current point
        }
        
   
    }
    //put total distance as function output
    
    return totaldist;
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