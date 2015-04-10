%masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/prueba/CaptureTest_90pc_Ch1_02022015_141431.hdf5';
%save_csv_name = '/Users/ajaver/Desktop/Gecko_compressed/prueba/trajectories/CaptureTest_90pc_Ch1_02022015_141431.csv';
%save_file = '/Users/ajaver/Desktop/Gecko_compressed/prueba/trajectories/bCaptureTest_90pc_Ch1_02022015_141431_segworm.hdf5';

masked_image_file = '/Users/ajaver/Desktop/sygenta/Compressed/data_20150114/control_9_fri_12th_dec_2.hdf5';
trajectories_file = '/Users/ajaver/Desktop/sygenta/Trajectories/data_20150114/control_9_fri_12th_dec_2_trajectories.hdf5';
save_csv_name = '/Users/ajaver/Desktop/sygenta/Trajectories/control_9_fri_12th_dec_2.csv';
save_file = '/Users/ajaver/Desktop/sygenta/Trajectories/dum.hdf5';
    
%read data from worm trajectories

AA = importdata(save_csv_name);
data = [];
AA.colheaders{1} = 'plate_worms_id';
for ii = 1:numel(AA.colheaders)
    data.(AA.colheaders{ii}) = AA.data(:,ii);
end
clear AA

movie2segwormfun(data, masked_image_file, trajectories_file);