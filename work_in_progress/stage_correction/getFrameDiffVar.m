function  img_abs_diff = getFrameDiffVar(masked_image_file)
%% mask information
mask_info = h5info(masked_image_file, '/mask');
% size of each frame
frame_size = mask_info.Dataspace.Size(1:2);
frame_total = mask_info.Dataspace.Size(3);

%% Calculate the variance between consecutive images

img_abs_diff = zeros(frame_total-1,1);

frame_prev =  h5read(masked_image_file, '/mask', [1,1,1], [frame_size(1),frame_size(2), 1]);
for ii = 2:frame_total
    if mod(ii, 100) == 0
        %progress
        fprintf('%2.2f%%\n', (ii+1)/frame_total*100)
    end
    
    frame_current = h5read(masked_image_file, '/mask', [1,1,ii], [frame_size(1),frame_size(2), 1]);
    
    % calculate the absolute difference between frames
    %img_abs_diff(ii-timeDiff) = sum(sum(abs( frame_bef - frame_aft)));
    
    good = (frame_prev.*frame_current) >0;
    img_abs_diff(ii-1) = var(single(frame_prev(good)-frame_current(good)));
    
    %buffer to the previous frame
    frame_prev = frame_current;
end