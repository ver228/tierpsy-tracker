
import matplotlib.pylab as plt
import numpy as np
import tables
import pandas as pd

trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/Features_CaptureTest_90pc_Ch2_18022015_230213.hdf5'
#trajectories_file = '/Volumes/behavgenom$/GeckoVideo/Trajectories/20150218/Trajectory_CaptureTest_90pc_Ch2_18022015_230213.hdf5'
trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/Features_Mask_short_CaptureTest_90pc_Ch2_18022015_230213.hdf5'
feature_fid = tables.open_file(trajectories_file, mode='r')

df = pd.DataFrame.from_records(feature_fid.root.plate_worms[:])

#%%
joined_tracks = df[df['worm_index_joined'] > 0]


#mean_area = df_good.groupby('worm_index')['area'].aggregate(['mean', 'count'])
#valid = (mean_area['mean']>300) & (mean_area['count']>25)
#good_index = mean_area[valid].index
#
#df_good = df_good[df_good.worm_index.isin(good_index)]
#N_lim = 40
# valid = (df_good['coord_x']>N_lim) & (df_good['coord_x']<2048-N_lim) & \
# (df_good['coord_y']>N_lim) & (df_good['coord_y']<2048-N_lim)
#df_good = df_good[valid]
##df_good = df[df['frame_number']<(25*3600) & df.worm_index.isin(good_index)]
#df_good = df_good.sort('frame_number')
tracks_data = joined_tracks[['worm_index_joined',
                             'frame_number',
                             'coord_x',
                             'coord_y',
                             'area',
                             'major_axis',
                             'intensity_mean',
                             'speed']].groupby('worm_index_joined').aggregate(['mean',
                                                                               'max',
                                                                               'min',
                                                                               'first',
                                                                               'last',
                                                                               'count'])

tracks_dataQ = joined_tracks[['worm_index_joined',
                              'frame_number',
                              'coord_x',
                              'coord_y',
                              'area',
                              'major_axis',
                              'intensity_mean',
                              'speed']].groupby('worm_index_joined').quantile(0.9)
#%%

track_joined = joined_tracks['worm_index_joined'].value_counts()
#%%
fig = []
fig.append(plt.figure())
fig.append(plt.figure())
fig.append(plt.figure())

for ind in track_joined.index[3:4]:
    if tracks_dataQ['speed'][ind] > 2.2:
        coord = joined_tracks[joined_tracks['worm_index_joined'] == ind]
        xx = np.array(coord['coord_x'])
        yy = np.array(coord['coord_y'])
        tt = np.array(coord['frame_number'])

        plt.figure(fig[0].number)
        plt.plot(tt, xx)
        plt.figure(fig[1].number)
        plt.plot(tt, yy)
        plt.figure(fig[2].number)
        plt.plot(xx, yy)

#    w_ind_list = coord['worm_index'].unique()
#    for w_ind in w_ind_list:
#        plt.plot(tracks_data['coord_x']['last'][w_ind], \
#        tracks_data['coord_y']['last'][w_ind], 'r.')
#        plt.plot(tracks_data['coord_x']['first'][w_ind], \
#        tracks_data['coord_y']['first'][w_ind], 'g.')
