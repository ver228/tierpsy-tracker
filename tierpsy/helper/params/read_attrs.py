import tables
import numpy as np
import json
import os

VALID_FIELDS = ['/mask', '/trajectories_data', '/features_timeseries']
class AttrReader():
    def __init__(self, file_name, dflt=1):
        self.file_name = file_name
        self.dflt = dflt
        self.field =  self._find_field()


    def _find_field(self):
        if os.path.exists(self.file_name):
            with tables.File(self.file_name, 'r') as fid:
                for field in VALID_FIELDS:
                    if field in fid:
                        return field
        #raise KeyError("Not valid field {} found in {}".format(VALID_FIELDS, self.file_name)) 
        return ''

    def _read_attr(self, attr_name, dflt=None):
        
        if dflt is None:
            dflt = self.dflt

        attr = dflt
        if self.field:
            with tables.File(self.file_name, 'r') as fid:
                node = fid.get_node(self.field)
                
                if attr_name in node._v_attrs:
                    attr = node._v_attrs[attr_name] 
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

        except:# (tables.exceptions.NoSuchNodeError, IOError, ValueError):
            fps = self._read_attr('fps', dflt=-1)
            if fps < 0:
                #read the user defined timestamp
                fps = expected_fps

            time_units = self._read_attr('time_units', dflt=None)
            if not isinstance(time_units, str):
                if fps == 1:
                    time_units = 'frames'
                else:
                    time_units = 'seconds'
            
            
        
        self._fps = float(fps)
        self._expected_fps = float(expected_fps)
        self._time_units = time_units

        return self._fps, self._expected_fps, self._time_units

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


    def get_microns_per_pixel(self):
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
 

            
        self._microns_per_pixel = float(microns_per_pixel)
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

    def get_ventral_side(self):
        try:

            with tables.File(self.file_name, 'r') as fid:
                exp_info_b = fid.get_node('/experiment_info').read()
                exp_info = json.loads(exp_info_b.decode("utf-8"))
                ventral_side = exp_info['ventral_side']
        
        except:
            ventral_side = self._read_attr('ventral_side', dflt = "")
        
        return ventral_side


def read_ventral_side(fname):
    reader = AttrReader(fname)
    return reader.get_ventral_side()
    
def read_fps(fname, dflt=1):
    assert isinstance(fname, str)
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

    #expected_fps and fps will be the same if it was defined by the user.
    fps, expected_fps, time_units = fps_out
    microns_per_pixel, xy_units = microns_per_pixel_out

    # save some data used in the calculation as attributes
    group_to_save._v_attrs['microns_per_pixel'] = microns_per_pixel
    group_to_save._v_attrs['xy_units'] = xy_units 
    
    group_to_save._v_attrs['fps'] = fps
    group_to_save._v_attrs['expected_fps'] = expected_fps 
    group_to_save._v_attrs['time_units'] = time_units

    group_to_save._v_attrs['is_light_background'] = is_light_background

    return fps, microns_per_pixel, is_light_background

def set_unit_conversions(group_to_save, expected_fps=None, microns_per_pixel=None, is_light_background=1):

    attr_writer = getattr(group_to_save, '_v_attrs')
    # save some data used in the calculation as attributes
    if microns_per_pixel is None or microns_per_pixel<=0:
        attr_writer['microns_per_pixel'] = 1
        attr_writer['xy_units'] = 'pixels'
    else: 
        attr_writer['microns_per_pixel'] = microns_per_pixel
        attr_writer['xy_units'] = 'micrometers'

    # save some data used in the calculation as attributes
    if expected_fps is None or expected_fps<=0:
        attr_writer['expected_fps'] = 1
        attr_writer['time_units'] = 'frames'
    else: 
        attr_writer['expected_fps'] = expected_fps
        attr_writer['time_units'] = 'seconds'
    attr_writer['is_light_background'] = int(is_light_background) #bool is not be supported by hdf5



