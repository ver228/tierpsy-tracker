\\\\\\\\\\\\\\\indexes2check = [10, 11, 679];
file_name = '/Volumes/behavgenom$/GeckoVideo/MaskedVideos/20151001_1525/CSTCTest_Ch1_01102015_152413.hdf5';%'/Volumes/behavgenom$/GeckoVideo/MaskedVideos/Chris_20150803/CSTCTest_Ch2_03082015_113211.hdf5';
save_dir = '/Users/ajaver/CSTCTest_Ch1_01102015_152413/';

if ~exist(save_dir, 'dir')
    mkdir(save_dir)
end

%% read file data 
ii = find(file_name == filesep, 1, 'last');
mask_dir = file_name(1:ii);

file_name_short = file_name(ii+1:end);
results_dir = strrep(mask_dir , 'MaskedVideos', 'Results');

ii = find(file_name_short == '.', 1, 'last');
base_name = file_name_short(1:ii-1);


skeletons_file = [results_dir, base_name, '_skeletons.hdf5'];
trajectories_data = h5read(skeletons_file, '/trajectories_data');


if isfield(trajectories_data, 'worm_index_N')
    worm_indexes = trajectories_data.worm_index_N;
else
    worm_indexes = trajectories_data.worm_index_joined;
end

%%
for ind = indexes2check;
    good = worm_indexes == ind;
    
    frameS = double(trajectories_data.frame_number(good)) + 1; %pass from numpy to matlab indexing
    skel_idS = double(trajectories_data.skeleton_id(good)) + 1; %pass from numpy to matlab indexing
    coord_x = double(trajectories_data.coord_x(good)) + 1; %pass from numpy to matlab indexing
    coord_y = double(trajectories_data.coord_y(good)) + 1; %pass from numpy to matlab indexing
    
    threshS = trajectories_data.threshold(good);
    roi_size = max(trajectories_data.roi_size(good));
    
    worm_maskS = zeros(roi_size, roi_size, numel(frameS), 'uint8');
    worm_skeletonS = zeros(2, 49, numel(frameS));
    
    tot_skel = numel(skel_idS);
    tic
    for ii = 1:tot_skel
        fprintf('worm index %i - %i of %i \n', ind, ii, tot_skel);
        
        %get parameters from the work in a given frame
        frame = frameS(ii);
        threshold = threshS(ii);
        CMx = coord_x(ii);
        CMy = coord_y(ii);
        
        %read frame
        img = h5read(file_name, '/mask', [1, 1, frame], [2048, 2048, 1]);
        
        %the roi containing the worm
        roi_center = roi_size/2;
        roi_range = [-roi_center+1, roi_center];
        
        range_x = round(CMx) + roi_range;
        range_y = round(CMy) + roi_range;
        
        if range_x(1)<=0, range_x = range_x - range_x(0); end
        if range_y(1)<=0, range_y = range_y - range_y(0); end
        
        if range_x(2) > size(img, 2), range_x = range_x + (size(img, 2) - range_x(2)); end
        if range_y(2) > size(img, 1), range_y = range_y + (size(img, 1) - range_y(1)); end
        
        range_x = int64(range_x);
        range_y = int64(range_y);
        
        worm_img = img(range_x(1):range_x(2),range_y(1):range_y(2));
        
        %calculate mask
        worm_img = medfilt2(worm_img);
        worm_mask = ((worm_img < threshold) & (worm_img~=0));
        worm_mask = imclose(worm_mask, ones(3));
        
        %select the region in the center, in the multiworm tracker the worm
        %should be in the center of the ROI
        props = bwconncomp(worm_mask);
        worm_maskD = zeros(size(worm_mask));
        if props.NumObjects > 1
            center = regionprops(props, 'centroid');
            delC = [center.Centroid] - roi_center;
            [~, imin] = min(delC(1:2:end).^2 + delC(2:2:end).^2);
            
            worm_maskD(props.PixelIdxList{imin}) = worm_img(props.PixelIdxList{imin});
        else
            worm_maskD(worm_mask) = worm_img(worm_mask);
        end
        
        %store worm mask
        worm_maskS(:,:,ii) = worm_maskD;
        
        %store skeleton centered in the mask
        
        skel_id = skel_idS(ii);
        skeleton = h5read(skeletons_file, '/skeleton', [1,1,skel_id], [2,49,1]);
        
        skel_roi = zeros(size(skeleton));
        skel_roi(2,:) = skeleton(2,:) - CMy + roi_center + 2;
        skel_roi(1,:) = skeleton(1,:) - CMx + roi_center + 2;
        worm_skeletonS(:,:,ii) = skel_roi;
        
    end
    save(sprintf('%c%sworm_%i.mat', filesep, save_dir, ind), 'worm_maskS', 'coord_x', 'coord_y', 'frameS',  'worm_skeletonS')
    
    toc
end

%{
frame = 1;
figure
imshow(worm_maskS(:,:, frame))
hold on
plot(worm_skeletonS(2,:,frame), worm_skeletonS(1,:,frame))
%}
%%
%xx = bsxfun(@plus, coord_x', squeeze(worm_skeletonS(1,:,:)));
%yy = bsxfun(@plus, coord_y', squeeze(worm_skeletonS(2,:,:)));
%plot(xx,yy)
