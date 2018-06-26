# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 13:50:57 2018

@author: lferiani
"""

import tables
import time
import os

import numpy as np

from tierpsy.helper.misc import TABLE_FILTERS


## constants
frame_size = (2048, 2048)
n_frames = 600
stack_size = (n_frames,) + frame_size
buffer_size = 64

efile1 = 'estack1.hdf5'
efile2 = 'estack2.hdf5'
cfile1 = 'cstack1.hdf5'
cfile2 = 'cstack2.hdf5'

#%

# create stack
frame_stack = np.random.randint(0, high=256, size=(n_frames,)+frame_size, dtype=np.uint8)
np.multiply(frame_stack*10, frame_stack < 20, out=frame_stack)
buff = np.zeros((buffer_size,) + frame_size, dtype=np.uint8)

# write expandable array with chunksize=1
tic = time.time()

# delete any previous  if it existed
with tables.File(efile1, "w") as fid:
    pass

with tables.File(efile1, "r+") as fid:
    
    # initialise dataset
    estack = fid.create_earray('/',
                           'frame_stack',
                           atom=tables.UInt8Atom(),
                           shape=((0,) + frame_size),
                           chunkshape=((1,) + frame_size),
                           expectedrows=n_frames,
                           filters=TABLE_FILTERS)
    
    for i in range(0, n_frames):
        
        # index in buffer
        j = i % buffer_size
        
        # if first frame of new buffer, zero the buffer
        if j == 0:
            buff *= 0
        # if
        
        # fill up buffer
        buff[j,:,:] = frame_stack[i,:,:]
        
        # if buffer full, write it to file
        if j == buffer_size-1:
            estack.append(buff)
        # if
    # for
# with
print(time.time() - tic)

"""
    
# write expandable array with chunksize matching buffer
tic = time.time()

with tables.File(efile2, "w") as fid:
    pass

with tables.File(efile2, "r+") as fid:
    
    # initialise dataset
    estack = fid.create_earray('/',
                           'frame_stack',
                           atom=tables.UInt8Atom(),
                           shape=((0,) + frame_size),
                           chunkshape=((buffer_size,) + frame_size),
                           expectedrows=n_frames,
                           filters=TABLE_FILTERS)
    
    for i in range(0, n_frames//buffer_size):
        
        estack.append(frame_stack[i*buffer_size : (i+1)*buffer_size,:,:])
    # for
# with
print(time.time() - tic)

# write chunked array as big as final file in bits
tic = time.time()

with tables.File(cfile1, "w") as fid:
    pass

with tables.File(cfile1, "r+") as fid:
    
    # initialise dataset
    cstack = fid.create_carray('/',
                           'frame_stack',
                           atom=tables.UInt8Atom(),
                           shape=((n_frames,) + frame_size),
                           filters=TABLE_FILTERS)
    
    for i in range(0, n_frames//buffer_size):
        
        cstack[i*buffer_size : (i+1)*buffer_size,:,:] = frame_stack[i*buffer_size : (i+1)*buffer_size,:,:]
    # for
    
# with
print(time.time() - tic)
"""

# write chunked array as big as final file
tic = time.time()

with tables.File(cfile2, "w") as fid:
    pass

with tables.File(cfile2, "r+") as fid:
    
    # initialise dataset
    cstack = fid.create_carray('/',
                           'frame_stack',
                           atom=tables.UInt8Atom(),
                           shape=((n_frames,) + frame_size),
                           filters=TABLE_FILTERS)
    
    cstack[:] = frame_stack;
# with
print(time.time() - tic)


#%% now test speed vs compression size

compression_level = range(0,8)

cfilesize_MB = np.zeros(len(compression_level))
celapsed = np.zeros(len(compression_level))
efilesize_MB = np.zeros(len(compression_level))
eelapsed = np.zeros(len(compression_level))

for cl in compression_level:
    
    print(cl)
    
    TABLE_FILTERS.complevel = cl;
    
    with tables.File(cfile2, "w") as fid:
        pass
    
    tic = time.time()
    
    with tables.File(cfile2, "r+") as fid:
        
        # initialise dataset
        cstack = fid.create_carray('/',
                               'frame_stack',
                               atom=tables.UInt8Atom(),
                               shape=((n_frames,) + frame_size),
                               filters=TABLE_FILTERS)
        
        cstack[:] = frame_stack;
    # with
    celapsed[cl] = time.time() - tic
    cfilesize_MB[cl] = os.stat(cfile2).st_size/1024**2
    
    print("time: %s, size: %s MB" % (celapsed[cl], cfilesize_MB[cl]))
    
# for
    
    

for cl in compression_level:
    
    print(cl)
    
    TABLE_FILTERS.complevel = cl;
    with tables.File(efile1, "w") as fid:
        pass

    tic = time.time()
    with tables.File(efile1, "r+") as fid:
        
        # initialise dataset
        estack = fid.create_earray('/',
                               'frame_stack',
                               atom=tables.UInt8Atom(),
                               shape=((0,) + frame_size),
                               chunkshape=((1,) + frame_size),
                               expectedrows=n_frames,
                               filters=TABLE_FILTERS)
        
        for i in range(0, n_frames):
            
            # index in buffer
            j = i % buffer_size
            
            # if first frame of new buffer, zero the buffer
            if j == 0:
                buff *= 0
            # if
            
            # fill up buffer
            buff[j,:,:] = frame_stack[i,:,:]
            
            # if buffer full, write it to file
            if j == buffer_size-1:
                estack.append(buff)
            # if
        # for
    # with
    eelapsed[cl] = time.time() - tic
    efilesize_MB[cl] = os.stat(efile1).st_size/1024**2
    
    print("time: %s, size: %s MB" % (eelapsed[cl], efilesize_MB[cl]))
    
# for
    
    
#%% plots
    
import matplotlib.pyplot as plt


ecompression_ratio = np.divide(efilesize_MB, efilesize_MB[0])
espace_savings = 1 - ecompression_ratio

ccompression_ratio = np.divide(cfilesize_MB, cfilesize_MB[0])
cspace_savings = 1 - ccompression_ratio

plt.close('all')

# plot raw data
fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel('Compression setting')
ax1.set_ylabel('File size, [MB]', color=color)
#ax1.plot(compression_level, efilesize_MB, '-o', color=color)
ax1.plot(compression_level, cfilesize_MB, '-o', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
ax2.set_ylabel('time, [s]', color=color)  # we already handled the x-label with ax1
#ax2.plot(compression_level, eelapsed, '-o', color=color)
ax2.plot(compression_level, celapsed, '-o', color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
    
    
# plot space saving
fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel('Compression setting')
ax1.set_ylabel('Space savings $(1-\\frac{compressed size}{original size})$', color=color)  # we already handled the x-label with ax1
#ax1.plot(compression_level, espace_savings, '-o', color=color)
ax1.plot(compression_level, cspace_savings, '-o', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
ax2.set_ylabel('Space savings $(1-\\frac{compressed size}{original size})$', color=color)  # we already handled the x-label with ax1
ax2.set_ylabel('time, [s]', color=color)
#ax2.plot(compression_level, eelapsed, '-o', color=color)
ax2.plot(compression_level, celapsed, '-o', color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show() 


# plot compression ratio vs time
#x_plot = eelapsed
#y_plot = ecompression_ratio
x_plot = celapsed
y_plot = ccompression_ratio

fig = plt.figure()
ax = plt.axes()
ax.set_xlabel('time, [s]')
ax.set_ylabel('Compression ratio')
line = ax.plot(x_plot, y_plot, '-o')
plt.setp(line, markerfacecolor = 'w')
ax.set_ylim(0, 1.1)

for i,txt in enumerate(compression_level):
    ax.annotate(txt, xy=(x_plot[i], y_plot[i]), xytext=(x_plot[i]-2, y_plot[i]-0.05))
# for
fig.show()