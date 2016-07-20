import tables
import itertools
import numpy as np
import matplotlib.pylab as plt
import time

featuresFile = '/Volumes/ajaver$/DinoLite/Results/Exp5-20150116-3/A002 - 20150116_140923H_features.hdf5'


feature_fid = tables.open_file(featuresFile, mode='r+')
feature_table = feature_fid.get_node('/plate_worms')


if not feature_table.cols.frame_number.is_indexed:
    feature_table.cols.frame_number.create_csindex()
    print 'Index created'

max_frame = feature_table.cols.frame_number[-1]

stats_str = ['median', 'mean', 'N', 'std', 'max', 'min']

area_stats = {}
for stat in stats_str:
    area_stats[stat] = np.zeros((max_frame + 1, 2))


tic = time.time()


def frame_selector(row):
    return row['frame_number']
for frame_number, rows in itertools.groupby(feature_table, frame_selector):
    if frame_number % 1000 == 0:
        toc = time.time()
        print frame_number, toc - tic
        tic = toc

    dat_list = [r['area'] for r in rows]

    for ii in range(2):
        if ii == 1:
            dat = np.array([x for x in dat_list if x > 40])
        else:
            dat = np.array(dat_list)
        area_stats['N'][frame_number, ii] = len(dat)
        area_stats['median'][frame_number, ii] = np.median(dat)
        area_stats['mean'][frame_number, ii] = np.mean(dat)
        area_stats['max'][frame_number, ii] = np.max(dat)
        area_stats['min'][frame_number, ii] = np.min(dat)
        area_stats['std'][frame_number, ii] = np.std(dat)


for key in area_stats.keys():
    fig = plt.figure()
    fig.suptitle(key)
    plt.plot(area_stats[key][:, 0])

for key in area_stats.keys():
    fig = plt.figure()
    fig.suptitle(key)
    plt.plot(area_stats[key][:, 1])


feature_fid.close()
