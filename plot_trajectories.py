
import matplotlib.pylab as plt
import numpy as np
import tables

trajectories_list = ['/Users/ajaver/Desktop/Gecko_compressed/Features_Mask_short_CaptureTest_90pc_Ch2_18022015_230213.hdf5', 
                     '/Users/ajaver/Desktop/Gecko_compressed/Features_CaptureTest_90pc_Ch4_16022015_174636.hdf5'];
for trajectories_file in trajectories_list[0:1]:

    N = 20
    #plot top 20 largest trajectories
    feature_fid = tables.open_file(trajectories_file, mode = 'r')
    feature_table = feature_fid.get_node('/plate_worms')
    dum = np.array(feature_table.cols.worm_index_joined);
    dum[dum<0]=0
    track_size = np.bincount(dum)
    track_size[0] = 0
    indexes = np.argsort(track_size)[::-1]
    print track_size[indexes][0:N]
    
    fig = plt.figure()
      
    for ii in indexes[0:N]:
        
        coord = [(row['coord_x'], row['coord_y'], row['frame_number']) \
        for row in feature_table.where('worm_index_joined == %i'% ii)]

        coord = np.array(coord).T
        plt.plot(coord[2,:], coord[1,:], '-')
        
    feature_fid.close()
    
#%%
pix2um = 2e-2/1e-6/2048;

feature_fid = tables.open_file(trajectories_file, mode = 'r')
feature_table = feature_fid.get_node('/plate_worms')
strC = 'brg'

fig = []
for ii in range(3):
    fig.append(plt.figure())

for cc, ind in enumerate(indexes[17:20]):#:3]:
    coord = [(row['coord_x'], row['coord_y'], row['frame_number']) \
            for row in feature_table.where('worm_index_joined == %i'% ind)]
    coord = np.array(coord).T
    #ax = plt.gca()
    for deltaT in [15]:#[1, 3]+range(5,51,5):# + [50, 100, 500, 1000]:
        delX = coord[0,deltaT:] - coord[0,0:-deltaT] 
        delY = coord[1,deltaT:] - coord[1,0:-deltaT]
        delT = coord[2,deltaT:] - coord[2,0:-deltaT] 
        good = delT == deltaT        
        delX = delX[good]
        delY = delY[good]
        
        plt.figure(fig[2].number)
        plt.subplot(1, 3, cc)
        xx = coord[0,:]-np.mean(coord[0,:]);
        yy = coord[1,:]-np.mean(coord[1,:]);
        plt.plot(xx, yy, strC[cc], label='%i' % (cc+1))
    
        
        
        R = np.sqrt(delX*delX + delY*delY)*pix2um;
        [counts, edges] = np.histogram(R, 100)
        bin_delta = edges[1]-edges[0];
        bins = edges[0:-1] + bin_delta/2;
        bins = bins/(deltaT/25.0)
        
        plt.figure(fig[0].number)
        plt.plot(bins, counts, strC[cc], label='%i' % (cc+1))
    
        #plt.plot(bins, counts, label='%i' % deltaT)
        
        plt.figure(fig[1].number)
        plt.subplot(1, 3, cc)
        plt.plot(R/(deltaT/25.0), strC[cc], label='%i' % (cc+1))
        plt.ylim([0, 600])
        
    #plt.xscale('log')
    #plt.xlim((0,500))

plt.figure(fig[0].number)
plt.legend()

feature_fid.close()

#plt.xscale('log')
    
#%%
coord = [(row['coord_x'], row['coord_y'], row['frame_number'], row['worm_index_joined']) for row in feature_table.where('frame_number <45000 and worm_index_joined != -1')]    
coord = np.array(coord)
plt.plot(coord[:,0],coord[:,1], '.')   

#R = (np.sqrt()**2+np.diff(coord[1,:])**2));
#%%
#'/Users/ajaver/Desktop/Gecko_compressed/Features_CaptureTest_90pc_Ch2_18022015_230213.hdf5'
