
kezhi_file = '/Users/ajaver/Desktop/Gecko_compressed/20150511/kezhi_format/Capture_Ch5_11052015_195105_kezhi.hdf5';
worm_name = 'worm_1656.hdf5'; 

%RESAMPLE_SIZE = 50;
addpath('/Users/ajaver/GitHub_repositories/Multiworm_Tracking/OnlySegWorm')
tic
masks = h5read(kezhi_file, ['/' worm_name '/masks']);
for frame = 1:1000%size(masks,3)
    disp(frame)
    %for frame = 1:100
    
    worm_mask = masks(:,:,frame) ~= 0;
    worm = segWormBWimgSimpleM(worm_mask, frame, 0.1, false);
    %worm_results = getWormSkeletonM(worm_mask, 1, [], RESAMPLE_SIZE);
    %sWormSegs = 24;
    %cWormSegs = 2 * sWormSegs;
    %wormSegSize = size(contour, 1) / cWormSegs;
    
    %contour = double(h5read(filename, sprintf('/%i', frame)))';
    %segworm_reduced
    %figure, hold on
    %plot(contour(:,1), contour(:,2), '-xb')
    %plot(skeleton(:,1), skeleton(:,2), '-or')
end
toc