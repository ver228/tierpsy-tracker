# -*- coding: utf-8 -*-
"""
Created on Thu May  7 14:29:15 2015

@author: ajaver
"""

plt.imshow(worm_img, interpolation='none', cmap = 'gray')
for key in ['contour_ventral', 'skeleton', 'contour_dorsal']:
    plt.plot(dat[key][1,:], dat[key][0,:], '.-')

contours['worm_V_all'] = np.hstack((dat['contour_ventral'][::-1,:], \
    dat['skeleton'][::-1,::-1]));
            
contours['worm_D_all'] = np.hstack((dat['contour_dorsal'][::-1,:], \
    dat['skeleton'][::-1,::-1]));
            
nsegments = dat['skeleton'].shape[1];

for key in ['worm_D_all', 'worm_V_all']:
    worm_mask = np.zeros(worm_img.shape)
    cc = [contours[key].astype(np.int32).T];
    cv2.drawContours(worm_mask, cc, 0, 1, -1)
    pix_list = np.where(worm_mask==1);
    pix_val = worm_img[pix_list].astype(np.int);
    
    pix_coord = np.array(pix_list);
    ske_ind = np.argmin(cdist(pix_coord.T, dat['skeleton'].T), axis=1)
    
    ske_avg = np.zeros(nsegments)
    ske_num =  np.zeros(nsegments)
    for pix_i, ske_i in enumerate(ske_ind):
        ske_avg[ske_i] += pix_val[pix_i]
        ske_num[ske_i] += 1
    plt.plot(ske_avg/ske_num)    
    
#plt.imshow(worm_mask)



#pix_dat = np.array((pix_list[0], pix_list[1], pix_val))