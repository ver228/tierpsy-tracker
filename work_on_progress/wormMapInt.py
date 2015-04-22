# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:42:09 2015

@author: ajaver
"""
from scipy.interpolate import RectBivariateSpline

plt.figure()
plt.imshow(worm_img, cmap = 'gray', interpolation = 'none')

plt.plot(dat['skeleton'][1],dat['skeleton'][0], '.-g')
plt.plot(dat['contour_ventral'][1],dat['contour_ventral'][0], '.-k')
plt.plot(dat['contour_dorsal'][1],dat['contour_dorsal'][0], '.-k')

mid = int(np.round(dat['contour_ventral'].shape[1]/2.))
xx = [dat['contour_ventral'][1][mid], dat['contour_dorsal'][1][mid]]
yy = [dat['contour_ventral'][0][mid], dat['contour_dorsal'][0][mid]]
plt.plot(xx,yy, 'r')


worm_width = np.sqrt((dat['contour_ventral'][1]-dat['contour_dorsal'][1])**2 +\
 (dat['contour_ventral'][0]-dat['contour_dorsal'][0])**2)

med_width = np.median(worm_width)




delF = np.arange(-5,6);
x_grid = np.zeros((delF.size, dat['skeleton'].shape[1]-2))
y_grid = np.zeros((delF.size, dat['skeleton'].shape[1]-2))


for ind in range(1,dat['skeleton'].shape[1]-1):
    dx = dat['skeleton'][1][ind+1] - dat['skeleton'][1][ind-1]
    dy = dat['skeleton'][0][ind+1] - dat['skeleton'][0][ind-1]

    if np.abs(dx) < np.abs(dy):
        x_per = dat['skeleton'][1][ind] + delF;
        y_per = -dx/dy*(delF) + dat['skeleton'][0][ind] 
    else:
        y_per = dat['skeleton'][0][ind] + delF;
        x_per = -dy/dx*(delF) + dat['skeleton'][1][ind] 

    for ii in range(x_per.size):
        x_grid[ii, ind-1] = dat['skeleton'][1][ind] - dat['skeleton'][1][ind] + x_per[ii]
        y_grid[ii, ind-1]  = dat['skeleton'][0][ind] - dat['skeleton'][0][ind] + y_per[ii]


    #plt.plot(xx,yy)
    
#if np.abs(dx) > np.abs(dy):
#    xx = dat['skeleton'][0][mid] + [-5,5];
#    yy = dy/dx*(xx-dat['skeleton'][0][mid]) + dat['skeleton'][1][mid] 
#else:
#    yy = dat['skeleton'][0][mid] + [-5,5];
#    xx = dx/dy*(yy-dat['skeleton'][0][mid]) + dat['skeleton'][0][mid] 
#plt.plot(xx,yy, 'b')

plt.figure()
plt.imshow(worm_img, interpolation='none', cmap='gray')
plt.plot(x_grid,y_grid, '.b')


f = RectBivariateSpline(np.arange(worm_img.shape[0]), np.arange(worm_img.shape[1]), worm_img)
plt.figure()
plt.imshow(f.ev(y_grid, x_grid), interpolation='none')

