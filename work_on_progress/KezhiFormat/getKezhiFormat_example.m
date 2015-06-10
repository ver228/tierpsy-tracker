movies_file = '/Volumes/behavgenom$/GeckoVideo/Invidual_videos/20150512/Capture_Ch1_12052015_194303/';
%kezhi_file = '/Volumes/behavgenom$/GeckoVideo/kezhi_format/20150512/Capture_Ch1_12052015_194303/';
kezhi_file = '/Users/ajaver/Desktop/Gecko_compressed/20150511/kezhi_format/Capture_Ch1_11052015_195105_kezhi.hdf5';

%filename = [kezhi_file, 'worm_12.hdf5'];
worm_id = '/worm_1.hdf5';


masks = h5read(kezhi_file, [worm_id, '/masks']);
frames = h5read(kezhi_file, [worm_id, '/frames']);
CMs = h5read(kezhi_file, [worm_id, '/CMs']);

%plot movement of the center of mass
figure
plot(CMs(1,:), CMs(2,:))

%select a frame
frame = 10;
imshow(masks(:,:,10),[])

