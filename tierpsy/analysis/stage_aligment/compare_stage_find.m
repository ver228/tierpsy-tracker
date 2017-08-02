clc
clear all
masked_image_file = '/Users/ajaver/Documents/GitHub/tierpsy-tracker/tests/data/SCHAFER_LAB_SINGLE_WORM/MaskedVideos/L4_19C_1_R_2015_06_24__16_40_14__.hdf5';
skeletons_file = '/Users/ajaver/Documents/GitHub/tierpsy-tracker/tests/data/SCHAFER_LAB_SINGLE_WORM/Results/L4_19C_1_R_2015_06_24__16_40_14___skeletons.hdf5';

masked_image_file = '/Users/ajaver/Tmp/MaskedVideos/SCHAFER_LAB_SINGLE_WORM.hdf5';
skeletons_file = '/Users/ajaver/Tmp/Results/SCHAFER_LAB_SINGLE_WORM_skeletons.hdf5';
%masked_image_file = '/Volumes/behavgenom_archive$/single_worm/finished/mutants/snf-11(ok156)V@RM2710/food_OP50/XX/30m_wait/clockwise/snf-11 (ok156)V on food L_2009_11_18__11_54_28___6___5.hdf5';
%skeletons_file = '/Volumes/behavgenom_archive$/single_worm/finished/mutants/snf-11(ok156)V@RM2710/food_OP50/XX/30m_wait/clockwise/snf-11 (ok156)V on food L_2009_11_18__11_54_28___6___5_skeletons.hdf5';

%%
frame_diffs_d = h5read(skeletons_file, '/stage_movement/frame_diffs');
video_timestamp_ind = h5read(skeletons_file, '/timestamp/raw');
%%
fps = h5readatt(skeletons_file, '/stage_movement', 'fps');
delay_frames = h5readatt(skeletons_file, '/stage_movement', 'delay_frames');
stage_log = h5read(masked_image_file, '/stage_log');
mediaTimes = stage_log.stage_time';
locations = [stage_log.stage_x , stage_log.stage_y];
    
%%
video_timestamp_ind = video_timestamp_ind + 1; %correct for python indexing
if isnan(video_timestamp_ind(end))
    video_timestamp_ind(end) = video_timestamp_ind(end-1);
end

if numel(video_timestamp_ind) > numel(frame_diffs_d) + 1
    video_timestamp_ind = video_timestamp_ind(1:numel(frame_diffs_d)+1);
end
frame_diffs = nan(1, max(video_timestamp_ind)-1);
dd = video_timestamp_ind - min(video_timestamp_ind);
dd = dd(dd>0);
frame_diffs(dd) = frame_diffs_d;
%%
frameDiffs = frame_diffs;
delayFrames = delay_frames;
verbose = false;
%%
[is_stage_move, movesI, stage_locations] = findStageMovement_ver2(frame_diffs, mediaTimes, locations, delay_frames, fps);
