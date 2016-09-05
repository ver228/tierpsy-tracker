mask_file = '/Users/ajaver/Desktop/Videos/single_worm/switched_sample/unc-104 (e1265)III on food L_2011_10_18__11_29_31__1.hdf5';


timestamp_time = h5read(mask_file, '/timestamp/time');
timestamp_raw = h5read(mask_file, '/timestamp/raw');

stage_data = h5read(mask_file, '/stage_data');

%read frame 10 (image)
frame_number = 10;
image = h5read(mask_file, '/mask', [1,1,frame_number], [640, 480,1])';


h5read(mask_file, '/xml_info')