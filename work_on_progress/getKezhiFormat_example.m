filename = '/Users/ajaver/Desktop/Gecko_compressed/20150511/kezhi_format/Capture_Ch1_11052015_195105/worm_2502.hdf5';

masks = h5read(filename, '/masks');
frames = h5read(filename, '/frames');
CMs = h5read(filename, '/CMs');
