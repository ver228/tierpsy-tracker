list_of_masks_file = '/Users/ajaver/Documents/GitHub/Multiworm_Tracking/work_in_progress/stage_correction/masks_agar_1.txt';
assert(exist(list_of_masks_file, 'file') > 0);

fid = fopen(list_of_masks_file, 'r');
list_of_masks = textscan(fid,'%s','Delimiter','\n');
list_of_masks = list_of_masks{1};
fclose(fid);

for ind_mask = 1:numel(list_of_masks)
    masked_image_file = list_of_masks{ind_mask};
    dd = strrep(masked_image_file, 'MaskedVideos', 'Results');
    skeletons_file = strrep(dd, '.hdf5', '_skeletons.hdf5');
    
    disp(masked_image_file)
    try
        has_finished = h5readatt(skeletons_file, '/stage_movement', 'has_finished');
    catch ME
        has_finished = 0;
    end
    
    if has_finished == 0
        alignStageMotionSegwormFun(masked_image_file,skeletons_file)
    else
        disp('The file was processed before.')
    end
    
    
end