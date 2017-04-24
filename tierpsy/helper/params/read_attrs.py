import tables
import h5py
import numpy as np


class AttrReader():
    def __init__(self, file_name, dflt=1):
        self.file_name = file_name
        self.dflt = dflt
        self.field =  self._find_field()

    def _find_field(self):
        valid_fields = ['/mask', '/trajectories_data', '/features_timeseries']
        with tables.File(self.file_name, 'r') as fid:
            for field in valid_fields:
                if field in fid:
                    return field
        raise KeyError("Not valid field {} found in {}".format(valid_fields, fname)) 


    def _read_attr(self, attr_name, dflt=None):
        if dflt is None:
            dflt = self.dflt

        with tables.File(self.file_name, 'r') as fid:
            node = fid.get_node(self.field)

            #print([(k,node._v_attrs[k]) for k in node._v_attrs.keys()])


            if attr_name in node._v_attrs:
                attr = node._v_attrs[attr_name]
            else:
                attr = self.dflt #default in old videos
            return attr

    def get_fps(self):
        expected_fps = self._read_attr('expected_fps', dflt=1)
        try:
            #try to calculate the frames per second from the timestamp
            with tables.File(self.file_name, 'r') as fid:
                timestamp_time = fid.get_node('/timestamp/time')[:]

                if np.all(np.isnan(timestamp_time)):
                    raise ValueError
                fps = 1 / np.nanmedian(np.diff(timestamp_time))

                if np.isnan(fps) or fps < 1:
                    raise ValueError

                time_units = 'seconds'
                is_user_fps = 0

        except (tables.exceptions.NoSuchNodeError, IOError, ValueError):
            #read the user defined timestamp
            fps = expected_fps
            time_units = self._read_attr('time_units', dflt=None)
            if not isinstance(time_units, str):
                if fps == 1:
                    time_units = 'frames'
                else:
                    time_units = 'seconds'
            
            is_user_fps = 1
        
        self._fps = fps
        self._expected_fps = expected_fps
        self._is_user_fps = is_user_fps
        self._time_units = time_units

        return self._fps, self._expected_fps, self._time_units, self._is_user_fps

    @property
    def fps(self):
        try:
            return self._fps
        except:
            self.get_fps()
            return self._fps

    @property
    def time_units(self):
        try:
            return self._time_units
        except:
            self.get_fps()
            return self._time_units


    @property
    def is_user_fps(self):
        try:
            return self._is_user_fps
        except:
            self.get_fps()
            return self._is_user_fps

    def get_microns_per_pixel(self):
        try:
            return self._microns_per_pixel
        except:
            try:
                #this for the shaffer's lab single worm case...
                with tables.File(self.file_name, 'r') as fid:
                    microns_per_pixel_scale = fid.get_node('/stage_movement')._v_attrs['microns_per_pixel_scale']
                    if microns_per_pixel_scale.size == 2:
                        assert np.abs(
                            microns_per_pixel_scale[0]) == np.abs(
                            microns_per_pixel_scale[1])
                        microns_per_pixel = np.abs(microns_per_pixel_scale[0])
                xy_units = 'micrometers'

            except (KeyError, tables.exceptions.NoSuchNodeError):
                microns_per_pixel = self._read_attr('microns_per_pixel', dflt = 1)
                xy_units = self._read_attr('xy_units', dflt = None)
                if xy_units is None:
                    if microns_per_pixel == 1:
                        xy_units = 'pixels'
                    else:
                        xy_units = 'micrometers'
 

            
            self._microns_per_pixel = microns_per_pixel
            self._xy_units = xy_units
            return self._microns_per_pixel, self._xy_units

    @property
    def microns_per_pixel(self):
        try:
            return self._microns_per_pixel
        except:
            self.get_microns_per_pixel()
            return self._microns_per_pixel
    
    @property
    def xy_units(self):
        try:
            return self._xy_units
        except:
            self.get_microns_per_pixel()
            return self._xy_units

def read_fps(fname, dflt=1):
    reader = AttrReader(fname, dflt)
    return reader.fps

def read_microns_per_pixel(fname, dflt=1):
    reader = AttrReader(fname, dflt)
    return reader.microns_per_pixel

def read_unit_conversions(fname, dflt=1):
    reader = AttrReader(fname, dflt)
    fps_out = reader.get_fps()
    
    microns_per_pixel_out = reader.get_microns_per_pixel()
    is_light_background = reader._read_attr('is_light_background', 1)
    
    print(fps_out, microns_per_pixel_out, is_light_background)
    return fps_out, microns_per_pixel_out, is_light_background


def copy_unit_conversions(group_to_save, original_file, dflt=1):
    fps_out, microns_per_pixel_out, is_light_background = \
    read_unit_conversions(original_file, dflt)

    #expected_fps and fps will be the same if is_user_fps is True.
    fps, expected_fps, is_user_fps, time_units = fps_out
    microns_per_pixel, xy_units = microns_per_pixel_out

    # save some data used in the calculation as attributes
    group_to_save._v_attrs['microns_per_pixel'] = microns_per_pixel
    group_to_save._v_attrs['xy_units'] = xy_units 
    
    group_to_save._v_attrs['fps'] = fps
    group_to_save._v_attrs['expected_fps'] = expected_fps 
    group_to_save._v_attrs['is_user_fps'] = is_user_fps
    group_to_save._v_attrs['time_units'] = time_units

    group_to_save._v_attrs['is_light_background'] = is_light_background

    return fps, microns_per_pixel, is_light_background

def set_unit_conversions(group_to_save, expected_fps=None, microns_per_pixel=None, is_light_background=1):


    #this is a not so pretty hack to be able to deal with h5py library that the compressVideo file uses
    if isinstance(group_to_save, h5py._hl.dataset.Dataset):
        attr_writer = getattr(group_to_save, 'attrs')
    else:
        attr_writer = getattr(group_to_save, '_v_attrs')


    # save some data used in the calculation as attributes
    if microns_per_pixel is None:
        attr_writer['microns_per_pixel'] = 1
        attr_writer['xy_units'] = 'pixels'
    else: 
        attr_writer['microns_per_pixel'] = microns_per_pixel
        attr_writer['xy_units'] = 'micrometers'

    # save some data used in the calculation as attributes
    if expected_fps is None:
        attr_writer['expected_fps'] = 1
        attr_writer['time_units'] = 'frames'
    else: 
        attr_writer['expected_fps'] = expected_fps
        attr_writer['time_units'] = 'seconds'
    attr_writer['is_light_background'] = is_light_background

