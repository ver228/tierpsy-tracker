
import matplotlib.pylab as plt
import numpy as np
import tables
import pandas as pd

trajectories_file = '/Volumes/behavgenom$/GeckoVideo/Trajectories/20150218/Trajectory_CaptureTest_90pc_Ch2_18022015_230213.hdf5'
feature_fid = tables.open_file(trajectories_file, mode = 'r')

df = pd.DataFrame.from_records(feature_fid.root.plate_worms[:])
#df = df[df.worm_index_joined>0]
grouped = df.groupby('worm_index_joined')

#%%

mean_values = grouped.aggregate(np.mean)
std_values = grouped.aggregate(np.std)
mean_values['track_length'] = grouped.size();
#%%
#mean_values.plot(y='area', x='intensity_mean', kind='scatter')
mean_values.plot(y='intensity_mean', x='track_length', kind='scatter')
#%%
#filter_data = mean_values[(mean_values.area>450) & (mean_values.area<1000) \
#&(mean_values.intensity_mean>135) & (mean_values.intensity_mean<155)]
filter_data = mean_values[mean_values.track_length > 900]
filter_data.plot(y='intensity_mean', x='track_length', kind='scatter')
#%%

dat_hist = np.histogram2d(df.coord_x, df.coord_y, bins=2048)
plt.imshow(np.log10(dat_hist[0]+1))

#%%
import os
save_dir = '/Users/ajaver/Desktop/manual_join/Trajectory_CaptureTest_90pc_Ch2_18022015_230213/';
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    
for ind in filter_data.index:
    largest_track = df[df.worm_index_joined == ind]
    frame_range = (largest_track.frame_number.min(), largest_track.frame_number.max());
    title_str = '%i - %i' % frame_range
    largest_track.plot(x = 'coord_x', y = 'coord_y', \
                       xlim = (0, 2048), ylim = (0, 2048), \
                       legend = False, title = title_str)
                       
    save_name = save_dir + 'F=%06i - I=%i'  % (frame_range[0], ind) 
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()
#%%
import h5py
masked_image_file = '/Volumes/behavgenom$/GeckoVideo/Compressed/20150218/CaptureTest_90pc_Ch2_18022015_230213.hdf5'
mask_fid = h5py.File(masked_image_file, "r");
I_worms = mask_fid["/mask"]

plt.figure()
plt.imshow(I_worms[0,:,:], cmap='gray', interpolation='none')


ini_trajectories = df[df.frame_number < 1500]    
grouped_ini = ini_trajectories.groupby('worm_index')

for trajectory in grouped_ini:
    xx = np.array(trajectory[1]['coord_x'])
    yy = np.array(trajectory[1]['coord_y'])
    plt.plot(xx,yy)
    
plt.xlim((0,2048))
plt.ylim((0,2048))
#%%

mean_area = df.groupby('worm_index')['area'].aggregate(['mean', 'count'])

good_index = mean_area[(mean_area['mean']>500) & (mean_area['count']>5)].index
df_good = df[df.worm_index.isin(good_index)]

#df_good = df_good[mean_area> 500]

#%%
#ini_trajectories = df_good#[df.frame_number < 1500]
df_good = df_good.sort('frame_number')
#%%
tracks_data = df_good[['worm_index', 'frame_number', 'coord_x', 'coord_y', 'area', 'intensity_mean', 'speed']].groupby('worm_index').aggregate(['mean', 'max', 'min', 'first', 'last'])
#%%
MAX_DELTA = 100
MAX_MOV = 50;

index2check = list(tracks_data.index);
join_index_list = [];
#while index2check:
for kk in range(1):    
    ind_list = [min(index2check)];
    while 1:
        next_ind = ind_list[-1]
        try:
            index2check.remove(next_ind)
        except ValueError:
            break;
        
        if next_ind == 1978:
            print 'hola'
            break;
        
        final_frame = tracks_data['frame_number']['last'][next_ind]
        final_x = tracks_data['coord_x']['last'][next_ind]
        final_y = tracks_data['coord_y']['last'][next_ind]
        mean_area = tracks_data['area']['mean'][next_ind]
        
        
        
        tracks2check = tracks_data[\
            (tracks_data['frame_number']['first']>final_frame) & \
            (tracks_data['frame_number']['first']<=(final_frame+MAX_DELTA))]
        
        
        R = np.sqrt((tracks2check['coord_x']['first']-final_x)**2 + (tracks2check['coord_y']['first']-final_y)**2)
                
        deltaT = tracks2check['frame_number']['first'] - final_frame;
        area_ratio = tracks2check['coord_x']['first']/mean_area
        try:
            good = (area_ratio<2) & (area_ratio>0.5) & (R<MAX_MOV);
            #select the smallest one in relation to the time vs real distance
            ind_list.append((R*deltaT)[good].argmin());
        except ValueError:
            break;
    join_index_list.append(ind_list)
    print len(index2check), ind_list
#%%
df_good['worm_index_joined'] = 0
for (new_index, index_list) in enumerate(join_index_list):
    print new_index
    rows2change = []
    for ind in index_list:
        rows2change = []
        if type(rows2change) is list:
            rows2change = df_good.worm_index == ind;
        else:
            rows2change |= (df_good.worm_index == ind)
    df_good.loc[rows2change,'worm_index_joined'] = new_index;
#df_good['worm_index_joined'] = 0
#for (new_index, index_list) in enumerate(join_index_list):
#    print new_index
#    rows2change = []
#    for ind in index_list:
#        rows2change = []
#        if type(rows2change) is list:
#            rows2change = df_good.worm_index == ind;
#        else:
#            rows2change |= (df_good.worm_index == ind)
#    df_good.loc[rows2change,'worm_index_joined'] = new_index;
