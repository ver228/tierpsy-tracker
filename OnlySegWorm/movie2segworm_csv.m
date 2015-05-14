function movie2segworm_csv(save_csv_name, masked_image_file, trajectories_file)
%{
save_csv_name = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/tmp_CaptureTest_90pc_Ch1_02022015_141431.csv';
save_file = save_csv_name;
masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/CaptureTest_90pc_Ch1_02022015_141431.hdf5';
trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/CaptureTest_90pc_Ch1_02022015_141431_segworm.hdf5';
%}
%read data from worm trajectories
disp(save_csv_name)
disp(masked_image_file)
disp(trajectories_file)
AA = importdata(save_csv_name);

data = [];
AA.colheaders{1} = 'plate_worms_id';
for ii = 1:numel(AA.colheaders)
    data.(AA.colheaders{ii}) = AA.data(:,ii);
end
clear AA

[~,base_name,~] = fileparts(save_csv_name);
movie2segwormfun(data, masked_image_file, trajectories_file, base_name);

delete(save_csv_name) %free space and use it as a flag that the excecution was successful