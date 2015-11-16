function [worm_img, roi_corner] = getWormROI(img, CMx, CMy, roi_size )
    %%
    %%Extract a square Region Of Interest (ROI)
    %%img - 2D  array containing the data to be extracted
    %%CMx, CMy - coordinates of the center of the ROI
    %%roi_size - side size in pixels of the ROI
    %%
    
    roi_center = roi_size/2;
    roi_range = [-roi_center, roi_center];

    %obtain bounding box from the trajectories
    range_x = round(CMx) + roi_range;
    range_y = round(CMy) + roi_range;
    
    if range_x(1)<=0, range_x = range_x - range_x(0); end
    if range_y(1)<=0, range_y = range_y - range_y(0); end
    
    if range_x(2) > size(img, 2), range_x = range_x + (size(img, 2) - range_x(2)); end
    if range_y(2) > size(img, 1), range_y = range_y + (size(img, 1) - range_y(1)); end
    
    worm_img = img(range_y(1):range_y(2), range_x(1):range_x(2));
    roi_corner = [range_x(1), range_y(1)];
end
