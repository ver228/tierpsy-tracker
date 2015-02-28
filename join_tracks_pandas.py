
import matplotlib.pylab as plt
import numpy as np
import tables
import pandas as pd

trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/Features_CaptureTest_90pc_Ch2_18022015_230213.hdf5';
#trajectories_file = '/Volumes/behavgenom$/GeckoVideo/Trajectories/20150218/Trajectory_CaptureTest_90pc_Ch2_18022015_230213.hdf5'
trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/Features_Mask_short_CaptureTest_90pc_Ch2_18022015_230213.hdf5'
feature_fid = tables.open_file(trajectories_file, mode = 'r')

df = pd.DataFrame.from_records(feature_fid.root.plate_worms[:])

#%%
df_good = df#[df['frame_number']<(25*3600)]

#mean_area = df_good.groupby('worm_index')['area'].aggregate(['mean', 'count'])
#valid = (mean_area['mean']>300) & (mean_area['count']>25)
#good_index = mean_area[valid].index
#
#df_good = df_good[df_good.worm_index.isin(good_index)]
#N_lim = 40
#valid = (df_good['coord_x']>N_lim) & (df_good['coord_x']<2048-N_lim) & \
# (df_good['coord_y']>N_lim) & (df_good['coord_y']<2048-N_lim) 
#df_good = df_good[valid]
##df_good = df[df['frame_number']<(25*3600) & df.worm_index.isin(good_index)]
#df_good = df_good.sort('frame_number')
tracks_data = df_good[['worm_index_joined', 'frame_number', 'coord_x', \
                         'coord_y', 'area', 'major_axis', \
                         'intensity_mean', 'speed']].groupby('worm_index_joined').aggregate(['mean', 'max', 'min', 'first', 'last'])
#%%

MAX_DELTA = 25;
#MAX_MOV = 50;
AREA_RATIO_LIM = (0.67, 1.5)
index2check = list(tracks_data.index);
join_index_list = [];
while index2check:
    ind_list = [min(index2check)];
    print len(index2check), ind_list[-1]
    while 1:
        next_ind = ind_list[-1]
        try:
            index2check.remove(next_ind)
        except ValueError:
            ind_list = ind_list[:-1]
            break;
        final_frame = tracks_data['frame_number']['last'][next_ind]
        final_x = tracks_data['coord_x']['last'][next_ind]
        final_y = tracks_data['coord_y']['last'][next_ind]
        mean_area = tracks_data['area']['mean'][next_ind]
        max_major_axis = tracks_data['area']['max'][next_ind]
        tracks2check = tracks_data[\
            (tracks_data['frame_number']['first']>final_frame) & \
            (tracks_data['frame_number']['first']<=(final_frame+MAX_DELTA))]
        
        R = np.sqrt((tracks2check['coord_x']['first']-final_x)**2 + (tracks2check['coord_y']['first']-final_y)**2)
        deltaT = tracks2check['frame_number']['first'] - final_frame;
        area_ratio = tracks2check['coord_x']['first']/mean_area
        try:
            good = (area_ratio<AREA_RATIO_LIM[1]) & (area_ratio>AREA_RATIO_LIM[0]) 
            good &= (R<max_major_axis);#only join trajectories that move at most one worm body
            #select the smallest one in relation to the time vs real distance
            ind_list.append((R*deltaT)[good].argmin());
        except ValueError:
            break;
    join_index_list.append(ind_list)
#%%
df_good['worm_index_joined'] = -1
for (new_index, index_list) in enumerate(join_index_list):
    rows2change = []
    for ind in index_list:
        #if ind == 1217055:
        print ind, new_index+1
        if type(rows2change) is list:
            rows2change = df_good.worm_index == ind;
        else:
            rows2change |= (df_good.worm_index == ind)
    df_good.loc[rows2change,'worm_index_joined'] = new_index+1;   
#    print new_index#, np.sum(rows2change), np.sum(df_good['worm_index_joined']>0)

#%%
joined_tracks = df_good[df_good['worm_index_joined'] > 0]
track_joined = joined_tracks['worm_index_joined'].value_counts()
#%%
joined_tracks = df_good
fig = []
fig.append(plt.figure())
fig.append(plt.figure())
fig.append(plt.figure())

for ind in track_joined.index[2:]:
    coord = joined_tracks[joined_tracks['worm_index_joined'] == ind]
    xx = np.array(coord['coord_x'])
    yy = np.array(coord['coord_y'])
    tt = np.array(coord['frame_number'])
    
    
#    plt.figure(fig[0].number)
#    plt.plot(tt,xx)
#    plt.figure(fig[1].number)
#    plt.plot(tt,yy)
#    plt.figure(fig[2].number)
#    plt.plot(xx,yy)
#
#    w_ind_list = coord['worm_index'].unique()
#    for w_ind in w_ind_list:
#        plt.plot(tracks_data['coord_x']['last'][w_ind], \
#        tracks_data['coord_y']['last'][w_ind], 'r.')
#        plt.plot(tracks_data['coord_x']['first'][w_ind], \
#        tracks_data['coord_y']['first'][w_ind], 'g.')