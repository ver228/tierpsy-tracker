movies_file = '/Volumes/behavgenom$/GeckoVideo/Invidual_videos/20150512/Capture_Ch1_12052015_194303/';
kezhi_file = '/Volumes/behavgenom$/GeckoVideo/kezhi_format/20150512/Capture_Ch1_12052015_194303/';


filename = [kezhi_file, 'worm_12.hdf5'];



masks = h5read(filename, '/masks');
frames = h5read(filename, '/frames');
CMs = h5read(filename, '/CMs');

%plot movement of the center of mass
figure
plot(CMs(1,:), CMs(2,:))

%select a frame
frame = 10;
imshow(masks(:,:,10),[])