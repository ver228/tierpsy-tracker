%function alignStageMotion(masked_image_file,skeletons_file, is_swimming)

main_dir = '/Users/ajaver/Desktop/Videos/single_worm/agar_2/MaskedVideos/';
results_dir = strrep(main_dir, 'MaskedVideos', 'Results');
feat_dir = strrep(main_dir, 'MaskedVideos', 'Features');

is_swimming = false;

files = dir(main_dir);
for iif = 32:numel(files)
    file = files(iif);
    
    if ~isempty(regexp(file.name, '\w*.hdf5', 'ONCE'))
        
        clear is_stage_move movesI stage_locations
        %{
        fid = H5F.open(skeletons_file,'H5F_ACC_RDWR','H5P_DEFAULT');
        %is_finished = H5L.exists(fid,'stage_vec','H5P_DEFAULT');
        H5F.close(fid);
        %}
        %if ~is_finished &&
        
        fprintf('%i) %s\n', iif, file.name)
        masked_image_file = fullfile(main_dir, file.name);
        skeletons_file = fullfile(results_dir, strrep(file.name, '.hdf5', '_skeletons.hdf5'));
        features_mat = fullfile(feat_dir, strrep(file.name, '.hdf5', '_features.mat'));
        
        %% read time stamps. I should put this data into the masked files dir
        video_timestamp_ind = h5read(skeletons_file, '/timestamp/raw');
        video_timestamp_ind = video_timestamp_ind + 1; %correct for python indexing
        
        if any(isnan(video_timestamp_ind))
            disp('The timestamp is corrupt or do not exist')
            continue
        end
        
        video_timestamp_time = h5read(skeletons_file, '/timestamp/time');
        fps = 1/median(diff(video_timestamp_time));
        
        %% Open the information file and read the tracking delay time.
        
        xml_info = h5read(masked_image_file, '/xml_info');
        %this is not the cleaneast but matlab does not have a xml parser from
        %text string
        dd = strsplit(xml_info, '<delay>');
        dd = strsplit(dd{2}, '</delay>');
        delay_str = dd{1};
        delay_time = str2double(delay_str) / 1000;
        delay_frames = ceil(delay_time * fps);
        
        %% Read the media times and locations from the log file.
        %from the .log.csv file
        stage_data = h5read(masked_image_file, '/stage_data');
        mediaTimes = stage_data.stage_time';%*60;
        locations = [stage_data.stage_x , stage_data.stage_y];
        
        %% calculate the variance of the difference between frames
        % Ev's code uses the full vectors without dropping frames
        frame_diffs_d = getFrameDiffVar(masked_image_file);
        %% The shift makes everything a bit more complicated. I have to remove the first frame, before resizing the array considering the dropping frames.
        frame_diffs = nan(1, max(video_timestamp_ind)-1);
        dd = video_timestamp_ind-min(video_timestamp_ind);
        dd = dd(dd>0);
        frame_diffs(dd) = frame_diffs_d;
        %%
        
        try
            clear is_stage_move movesI stage_locations
            [is_stage_move, movesI, stage_locations] = findStageMovement_ver2(frame_diffs, mediaTimes, locations, delay_frames, fps);
            stage_vec = nan(numel(is_stage_move),2);
        catch M
            fprintf('%i) %s\n', iif, file.name)
            disp(ME)
            continue
            
        end
        %convert output into a vector that can be added to the skeletons file to obtain the real worm displacements
        for kk = 1:size(stage_locations,1)
            bot = max(1, movesI(kk,2)+1);
            top = min(numel(is_stage_move), movesI(kk+1,1)-1);
            stage_vec(bot:top, 1) = stage_locations(kk,1);
            stage_vec(bot:top, 2) = stage_locations(kk,2);
        end
        
        %the nan values must match the spected video motions
        assert(all(isnan(stage_vec(:,1)) == is_stage_move))
        
        %prepare vectors to save into the hdf5 file.
        %Go back to the original movie indexing. I do not want to include the missing frames at this point.
        frame_diffs_d = frame_diffs_d';
        is_stage_move_d = int8(is_stage_move(video_timestamp_ind))';
        
        
        %% change into a format that i can add directly to the skeletons in skeletons_file
        stage_vec_d = stage_vec(video_timestamp_ind, :);
        
        pixels2microns_x = h5readatt(masked_image_file, '/mask', 'pixels2microns_x');
        pixels2microns_y = h5readatt(masked_image_file, '/mask', 'pixels2microns_y');
        
        stage_vec_d(:,1) = stage_vec_d(:,1)*pixels2microns_y;
        stage_vec_d(:,2) = stage_vec_d(:,2)*pixels2microns_x;
        stage_vec_d = stage_vec_d';
        
        
        
        %{
        load(features_mat)
        seg_motion = info.video.annotations.frames==2;
        
        if (all(seg_motion==is_stage_motion))
            disp('Segworm and this code have the same frame aligment.')
        end
        
        plot(worm.posture.skeleton.x(:, 1:15:end), worm.posture.skeleton.y(:, 1:15:end))
        
        skeletons = h5read(skeletons_file, '/skeleton');
        
        skel_x = squeeze(skeletons(1,:,:)) + ones(49,1)*stage_vec_d(1,:);
        skel_y = squeeze(skeletons(2,:,:)) + ones(49,1)*stage_vec_d(2,:);
        
        figure, hold on
        %plot(worm.posture.skeleton.x(25,:))
        plot(skel_x(25,1:400))
        
        figure
        plot(skel_x(:, 1:15:end), skel_y(:, 1:15:end))
        
        figure
        plot(squeeze(skel_x(1,:)))
        %}
        
        %%
        %this removes crap from previous analysis
        %%save stage vector
        fid = H5F.open(skeletons_file,'H5F_ACC_RDWR','H5P_DEFAULT');
        if H5L.exists(fid,'/stage_vec','H5P_DEFAULT')
            H5L.delete(fid,'/stage_vec','H5P_DEFAULT');
        end
        
        if H5L.exists(fid,'/is_stage_move','H5P_DEFAULT')
            H5L.delete(fid,'/is_stage_move','H5P_DEFAULT');
        end
        H5F.close(fid);
        
        
        %% delete data from previous analysis if any
        fid = H5F.open(skeletons_file,'H5F_ACC_RDWR','H5P_DEFAULT');
        if H5L.exists(fid,'/stage_movement','H5P_DEFAULT')
            gid = H5G.open(fid, '/stage_movement');
            if H5L.exists(gid,'stage_vec','H5P_DEFAULT')
                H5L.delete(gid,'stage_vec','H5P_DEFAULT');
            end
            
            if H5L.exists(gid,'is_stage_move','H5P_DEFAULT')
                H5L.delete(gid,'is_stage_move','H5P_DEFAULT');
            end
            
            if H5L.exists(gid,'frame_diff','H5P_DEFAULT')
                H5L.delete(gid,'frame_diff','H5P_DEFAULT');
            end
            H5L.delete(gid,'/stage_movement','H5P_DEFAULT');
        end
        H5F.close(fid);
        
        
        %% save stage vector
        
        h5create(skeletons_file, '/stage_movement/stage_vec', size(stage_vec_d), 'Datatype', 'double', ...
            'Chunksize', size(stage_vec_d), 'Deflate', 5, 'Fletcher32', true, 'Shuffle', true)
        h5write(skeletons_file, '/stage_movement/stage_vec', stage_vec_d);
        
        h5create(skeletons_file, '/stage_movement/is_stage_move', size(is_stage_move_d), 'Datatype', 'int8', ...
            'Chunksize', size(is_stage_move_d), 'Deflate', 5, 'Fletcher32', true, 'Shuffle', true)
        h5write(skeletons_file, '/stage_movement/is_stage_move', is_stage_move_d);
        
        h5create(skeletons_file, '/stage_movement/frame_diffs', size(frame_diffs_d), 'Datatype', 'double', ...
            'Chunksize', size(frame_diffs_d), 'Deflate', 5, 'Fletcher32', true, 'Shuffle', true)
        h5write(skeletons_file, '/stage_movement/frame_diffs', frame_diffs_d);
        
        h5writeatt(skeletons_file, '/stage_movement', 'fps', fps)
        h5writeatt(skeletons_file, '/stage_movement', 'delay_frames', delay_frames)
        
    end
    
    
end

%masked_image_file = '/Users/ajaver/Desktop/Videos/single_worm/agar_goa/MaskedVideos/goa-1 (sa734)I on food L_2010_03_04__10_44_32___8___6.hdf5';
%skeletons_file = '/Users/ajaver/Desktop/Videos/single_worm/agar_goa/Results/goa-1 (sa734)I on food L_2010_03_04__10_44_32___8___6_skeletons.hdf5';

