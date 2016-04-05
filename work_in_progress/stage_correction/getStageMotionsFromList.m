list_of_masks_file = '/Users/ajaver/Documents/GitHub/Multiworm_Tracking/work_in_progress/stage_correction/masks_agar_2.txt';
assert(exist(list_of_masks_file, 'file') > 0);

fid = fopen(list_of_masks_file, 'r');
list_of_masks = textscan(fid,'%s','Delimiter','\n');
list_of_masks = list_of_masks{1};
fclose(fid);

tot_masks = numel(list_of_masks);
for ind_mask = 1:tot_masks
    masked_image_file = list_of_masks{ind_mask};
    dd = strrep(masked_image_file, 'MaskedVideos', 'Results');
    skeletons_file = strrep(dd, '.hdf5', '_skeletons.hdf5');
    
    fprintf('%i/%i) %s\n', ind_mask, tot_masks, masked_image_file)
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


alignStageMotionSegwormFun('/Users/ajaver/Desktop/Videos/single_worm/agar_2/MaskedVideos/unc-75 (e950)I on food L_2010_09_23__11_55___3___4.hdf5', '/Users/ajaver/Desktop/Videos/single_worm/agar_2/Results/unc-75 (e950)I on food L_2010_09_23__11_55___3___4_skeletons.hdf5');